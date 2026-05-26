import gc
import json
import math
from collections.abc import Callable
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
import sentencepiece
from jaxtyping import Array, PyTree

from kairos.nn.layers import Block, RMSNorm, dense_init, dkwargs
from kairos.nn.rope import compute_rope_freqs


def emb_init(key: Array, shape: tuple[int, ...], dtype) -> Array:
    return jax.random.uniform(key, shape, minval=-0.1, maxval=0.1)


class Transformer(nn.Module):
    dim: int
    mlp_dim: int
    n_heads: int
    n_layers: int
    vocab_size: int
    theta_rope: float
    use_rope: bool = False
    use_interleaved_rope: bool = False
    flash_attention: Callable | bool | None = None
    context_size: int = 2048
    pad_id: int = -1
    qk_norm: bool = False
    activation: str = "swiglu"
    gqa: int = 1
    head_dim: int = -1

    @nn.compact
    def __call__(
        self,
        x: Array,
        cache: dict[str, jax.Array | tuple[jax.Array, jax.Array]] | None = None,
        generate: bool = False,
    ) -> tuple[Array, dict[str, jax.Array]]:
        bsz, n = x.shape
        new_cache = {}

        if cache:
            offset = cache["tokens"].shape[1]
            new_cache["tokens"] = jnp.concatenate([cache["tokens"], x], axis=1)
        else:
            offset = 0
            new_cache["tokens"] = x

        mask_padding = new_cache["tokens"] == self.pad_id
        mask_padding = jnp.expand_dims(mask_padding, 1).repeat(n, axis=1)
        mask_padding = ~jnp.tril(mask_padding, k=offset - 1)

        mask_causal = jnp.expand_dims(jnp.ones((n, n + offset)), 0)
        mask_causal = jnp.tril(mask_causal, k=offset)

        mask = jnp.log(mask_padding * mask_causal)
        mask = jnp.expand_dims(mask, 1)

        h = nn.Embed(
            self.vocab_size,
            self.dim,
            embedding_init=emb_init,
            dtype=jax.dtypes.bfloat16,
        )(x)

        if self.use_rope and self.theta_rope > 0:
            head_dim = self.head_dim if self.head_dim > 0 else self.dim // self.n_heads
            freqs = compute_rope_freqs(head_dim, n + offset, self.theta_rope)
            freqs = jnp.expand_dims(freqs, 1)
        if not self.use_rope:
            idx = jnp.repeat(jnp.arange(offset, offset + n).reshape(1, -1), bsz, axis=0)
            h = h + nn.Embed(
                self.context_size,
                self.dim,
                embedding_init=emb_init,
                dtype=jax.dtypes.bfloat16,
            )(idx)
            freqs = None

        CBlock = nn.checkpoint(Block, policy=jax.checkpoint_policies.nothing_saveable())
        for i in range(self.n_layers):
            if self.use_interleaved_rope:
                freqs_i = None if (i % 4 == 2) else freqs
            else:
                freqs_i = freqs
            block_i = CBlock(
                self.mlp_dim,
                self.n_heads,
                i,
                self.flash_attention,
                self.qk_norm,
                self.activation,
                self.gqa,
                self.head_dim,
            )
            cache_i = cache[f"layer_{i}"] if cache else None
            h, cache_i = block_i(h, mask, freqs_i, cache_i)
            new_cache[f"layer_{i}"] = cache_i

        h = RMSNorm()(h)
        if generate:
            h = h[:, -1:, :]
        h = nn.Dense(self.vocab_size, kernel_init=dense_init(self.n_layers), **dkwargs)(
            h
        )
        return h.astype("float32"), new_cache


def create_model(
    args,
    mesh: jax.sharding.Mesh | None = None,
    partition_spec: jax.sharding.PartitionSpec | None = None,
) -> tuple[nn.Module, sentencepiece.SentencePieceProcessor]:
    flash_attention = None
    if args.use_flash_attention:
        from functools import partial

        from flash_attn3_jax import flash_mha

        sm_scale = 1.0 / math.sqrt(args.model_dim // args.n_heads)
        flash_attention = jax.shard_map(
            partial(flash_mha, is_causal=True, softmax_scale=sm_scale),
            mesh=mesh,
            in_specs=(partition_spec, partition_spec, partition_spec),
            out_specs=(partition_spec),
            check_vma=False,
        )

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    model = Transformer(
        args.model_dim,
        args.mlp_dim,
        args.n_heads,
        args.n_layers,
        tokenizer.vocab_size(),
        args.theta_rope,
        use_rope=args.use_rope,
        use_interleaved_rope=args.use_interleaved_rope,
        flash_attention=flash_attention,
        qk_norm=args.qk_norm,
        activation=args.activation,
        gqa=args.grouped_query_attention,
        head_dim=args.head_dim,
    )
    return model, tokenizer


def load_model(
    filename: Path,
    step: int = -1,
    flash_attention: Callable | bool | None = None,
    shape_to_sharding: Callable | None = None,
    dtype: jnp.dtype = jnp.dtype("bfloat16"),
) -> tuple[nn.Module, PyTree, sentencepiece.SentencePieceProcessor]:
    # Convert to absolute path to satisfy orbax requirements
    filename = filename.resolve()
    with (filename / "args.json").open() as f:
        ckpt_args = json.load(f)

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=ckpt_args["tokenizer"])

    pad_id = tokenizer.pad_id()

    dkwargs["param_dtype"] = jnp.dtype("bfloat16")
    model = Transformer(
        ckpt_args["model_dim"],
        ckpt_args["mlp_dim"],
        ckpt_args["n_heads"],
        ckpt_args["n_layers"],
        tokenizer.vocab_size(),
        ckpt_args.get("theta_rope", 10000.0),
        use_rope=ckpt_args["use_rope"],
        use_interleaved_rope=ckpt_args.get("use_interleaved_rope", False),
        flash_attention=flash_attention,
        pad_id=pad_id,
        qk_norm=ckpt_args.get("qk_norm", False),
        activation=ckpt_args.get("activation", "swiglu"),
        gqa=ckpt_args.get("grouped_query_attention", 1),
        head_dim=ckpt_args.get("head_dim", -1),
    )

    def init_params() -> PyTree:
        x = jnp.ones((jax.process_count() * ckpt_args["bsz"], ckpt_args["csz"])).astype(
            "int32"
        )
        params = model.init(jax.random.PRNGKey(0), x)
        return params

    if shape_to_sharding is None:
        shape_to_sharding = lambda x: jax.sharding.SingleDeviceSharding(
            jax.devices()[0]
        )

    def add_sharding(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=shape_to_sharding(x))

    params_shape = jax.eval_shape(init_params)
    params_shape = jax.tree_util.tree_map(add_sharding, params_shape)

    params = None
    if not (filename / str(step) / "default").exists():
        handler = orbax.CompositeCheckpointHandler("params_avg")
        checkpointer = orbax.Checkpointer(handler)
        params = checkpointer.restore(
            f"{filename}/{step}",
            args=orbax.args.Composite(
                params_avg=orbax.args.StandardRestore(params_shape)
            ),
        )["params_avg"]
    else:
        restore_args = jax.tree_util.tree_map(
            lambda x: orbax.ArrayRestoreArgs(
                restore_type=jax.Array, dtype=dtype, sharding=shape_to_sharding(x)
            ),
            {"params_avg": params_shape},
        )
        ckptr = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
        params = ckptr.restore(
            f"{filename}/{step}/default/",
            args=orbax.args.PyTreeRestore(
                {"params_avg": params_shape}, restore_args=restore_args, transforms={}
            ),
        )["params_avg"]
    dkwargs["param_dtype"] = jnp.dtype("float32")
    return model, params, tokenizer


def load_soup(
    checkpoints: list[tuple[Path | str, int, float]],
    flash_attention: Callable | bool | None = None,
    shape_to_sharding: Callable | None = None,
    dtype: jnp.dtype = jnp.dtype("bfloat16"),
    mlp_soup: bool = False,
) -> tuple[nn.Module, PyTree, sentencepiece.SentencePieceProcessor]:
    path, step, weight = checkpoints[-1]
    print(f"Loading checkpoint {path} (step={step}, weight={weight})")
    model, params, tokenizer = load_model(
        Path(path), step, flash_attention, shape_to_sharding, dtype
    )

    if mlp_soup:
        soup = params

        # -------------------------
        # 2. Build MLP-only soup
        # -------------------------
        mlp_accumulator = None
        total_weight = 0.0

        for path, step, weight in checkpoints:
            print(f"Loading checkpoint {path} (step={step}, weight={weight})")
            _, params, _ = load_model(
                Path(path),
                step,
                flash_attention,
                shape_to_sharding,
                dtype,
            )

            def select_mlp(path, value):
                return value if "MLP_0" in "/".join(map(str, path)) else None

            mlp_params = jax.tree_util.tree_map_with_path(select_mlp, params)

            if mlp_accumulator is None:
                mlp_accumulator = jax.tree.map(
                    lambda x: x * weight if x is not None else None,
                    mlp_params,
                )
            else:
                mlp_accumulator = jax.tree.map(
                    lambda acc, x: acc + x * weight if x is not None else acc,
                    mlp_accumulator,
                    mlp_params,
                )

            total_weight += weight
            del params
            gc.collect()

        # -------------------------
        # 3. Normalize MLP soup
        # -------------------------
        mlp_accumulator = jax.tree.map(
            lambda x: x / total_weight if x is not None else None,
            mlp_accumulator,
        )

        # -------------------------
        # 4. Merge back into base params
        # -------------------------
        def merge_mlp(path, base_value, mlp_value):
            if "MLP_0" in "/".join(map(str, path)):
                return mlp_value
            return base_value

        soup = jax.tree_util.tree_map_with_path(merge_mlp, soup, mlp_accumulator)

        return model, soup, tokenizer
    else:
        soup = jax.tree.map(lambda x: x * weight, params)
        total_weight = weight
        del params
        gc.collect()

        for path, step, weight in checkpoints[1:]:
            print(f"Loading checkpoint {path} (step={step}, weight={weight})")
            _, params, _ = load_model(
                Path(path), step, flash_attention, shape_to_sharding, dtype
            )
            soup = jax.tree.map(lambda s, p: s + p * weight, soup, params)
            total_weight += weight
            del params
            gc.collect()

        print(f"Normalizing soup (total_weight={total_weight})")
        soup = jax.tree.map(lambda x: x / total_weight, soup)

        return model, soup, tokenizer
