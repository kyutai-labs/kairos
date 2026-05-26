import json
import math
import re
import sys
from collections.abc import Generator as ABCGenerator
from collections.abc import Iterable
from pathlib import Path
from timeit import default_timer as timer

import flax.linen as nn
import jax
import jax.numpy as jnp
import sentencepiece
from jax.sharding import NamedSharding
from jaxtyping import Array, PyTree
from pydantic_settings import CliApp

from kairos.data.tokenization import detokenize, tokenize
from kairos.nn import load_model
from kairos.utils import ArgsParser, cast_bf16, pad_left


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def sample_top_n(
    logits: Array, key: Array, n: int = 25, temperature: float = 0.8
) -> Array:
    idx = jnp.argsort(-logits, axis=-1)
    top_n_logits = jnp.take_along_axis(logits, idx[..., :n], axis=-1)
    pred = jax.random.categorical(key, top_n_logits / temperature)
    pred = jnp.expand_dims(pred, axis=-1)
    pred = jnp.take_along_axis(idx, pred, axis=-1)
    return jnp.squeeze(pred, axis=-1)


def trim_cache(
    cache: dict[str, jax.Array | tuple[jax.Array, jax.Array]],
) -> dict[str, jax.Array | tuple[jax.Array, jax.Array]]:
    new_cache = {}
    for k, v in cache.items():
        if k == "tokens":
            new_cache[k] = v[:, 1:]
        else:
            new_cache[k] = v[0][:, 1:, ...], v[1][:, 1:, ...]
    return new_cache


class Generator:
    def __init__(
        self,
        model: nn.Module,
        params: PyTree,
        tokenizer: sentencepiece.SentencePieceProcessor,
        topn: int = 25,
        temp: float = 0.8,
        max_model_length: int | None = None,
        data_sharding: NamedSharding | None = None,
    ) -> None:
        def _model_apply(params, x, cache=None):
            return model.apply(params, x, cache, generate=True)

        def _generate_loop(i, state):
            tokens, cache, key = state
            key, subkey = jax.random.split(key)
            logits, cache = model.apply(params, tokens, cache, generate=True)
            tokens = sample_top_n(logits, subkey, topn, temp)
            cache = trim_cache(cache)
            return (tokens, cache, key)

        self.topn = topn
        self.temp = temp
        self.params = params
        self.tokenizer = tokenizer
        self.model_apply = jax.tree_util.Partial(_model_apply)
        self.generate_loop = jax.tree_util.Partial(_generate_loop)
        self.max_model_length = max_model_length
        self.data_sharding = data_sharding

    def generate_fast(self, x: Array, key: Array, length: int) -> Array:
        logits, cache = self.model_apply(self.params, x)
        key, subkey = jax.random.split(key)
        tokens = sample_top_n(logits, subkey, self.topn, self.temp)[:, -1:]
        x, cache, _ = jax.lax.fori_loop(
            0, length - 1, self.generate_loop, (tokens, cache, key)
        )
        return cache["tokens"]

    def generate(
        self,
        prompts: list[str],
        length: int = 16,
        key: Array | None = None,
        only_return_generated: bool = False,
    ) -> list[str]:
        key = jax.random.PRNGKey(1234) if key is None else key
        tokens = [tokenize(s, self.tokenizer) for s in prompts]
        max_length = 0
        for s in tokens:
            max_length = max(len(s), max_length)
        total_length = max_length + length
        total_length = math.ceil(total_length / 64) * 64
        if self.max_model_length is not None:
            total_length = min(total_length, self.max_model_length)
        tokens = [pad_left(s, self.tokenizer.pad_id(), total_length) for s in tokens]
        tokens = jnp.asarray(tokens)
        if self.data_sharding is not None:
            tokens = jax.device_put(tokens, self.data_sharding)
        generations = self.generate_fast(tokens, key, length)
        text_generations = [
            detokenize(generations[i, :].tolist(), self.tokenizer, True, True)
            for i in range(len(prompts))
        ]
        if only_return_generated:
            outputs = []
            for tg, p in zip(text_generations, prompts):
                output = tg[len(p) :].strip()
                outputs.append(output)
            return outputs
        return text_generations


def batching(
    iterator: Iterable, batch_size: int = 16
) -> ABCGenerator[list, None, None]:
    batch = []
    for example in iterator:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


class GenerateArgs(ArgsParser):
    model: Path | None = None
    train: str | None = None
    step: int = -1
    prompts: Path | None = None


if __name__ == "__main__":
    args = CliApp.run(GenerateArgs)

    model, params, tokenizer = load_model(args.model, args.step, False)
    params = cast_bf16(params)

    def generate_loop_slow(i: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        tokens, key = state
        key, subkey = jax.random.split(key)
        logits, _ = model.apply(params, tokens, generate=True)
        pred = sample_top_n(logits, subkey)[:, i]
        tmp = jnp.where(tokens[:, i + 1] == tokenizer.pad_id(), pred, tokens[:, i + 1])
        return tokens.at[:, i + 1].set(tmp), key

    def generate_slow(x: Array, key: Array) -> Array:
        x, _ = jax.lax.fori_loop(0, 32, generate_loop_slow, (x, key))
        return x

    key = jax.random.PRNGKey(1234)
    generator = Generator(model, params, tokenizer, topn=10, temp=0.8)

    if args.prompts:
        dataset = json.load(args.prompts.open())
        outputs = []
        for batch in batching(dataset):
            prompts = [data["instruction"] for data in batch]
            prompts = [
                f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
                for p in prompts
            ]
            key, subkey = jax.random.split(key)
            generations = generator.generate(prompts, length=1024, key=subkey)
            generations = [
                g[len(p) :].split("<|im_end|>")[0].strip()
                for g, p in zip(generations, prompts)
            ]
            for data, generation in zip(batch, generations):
                generation = re.sub(r"<\|im_start\|>[a-z]*", "", generation)
                data["output"] = generation
                data["generator"] = f"{args.model.name}@{args.step}"
                outputs.append(data)
                sys.stderr.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(json.dumps(outputs, ensure_ascii=False))
    else:
        user_str = f"{color.BOLD}{color.DARKCYAN} user >{color.END} "
        while True:
            s = input(user_str)
            prompt = []
            while len(s) > 0:
                prompt.append(s)
                s = input(user_str)

            prompt = "\n".join(prompt)
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            t0 = timer()
            key, subkey = jax.random.split(key)
            generation = generator.generate([prompt], length=1024, key=subkey)[0]
            dt = timer() - t0
            generation = generation[len(prompt) :].split("<|im_end|>")[0].strip()
            print(f"{color.BOLD}{color.RED}helium>{color.END} {generation}")
            print(f"{round(dt, 3)} sec, {1024 / dt:.1f} tok/sec.")

    exit(0)
