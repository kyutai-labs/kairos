import datetime

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jaxtyping import Array, PyTree


def sync(name: str) -> None:
    multihost_utils.sync_global_devices(name)


def cast_bf16(params: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x: x.astype("bfloat16"), params)


def cast_fp32(params: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x: x.astype("float32"), params)


def global_array(
    local_array: Array | np.ndarray,
    mesh: jax.sharding.Mesh,
    sharding: jax.sharding.Sharding,
) -> Array:
    bsz, csz = local_array.shape
    global_shape = (jax.process_count() * bsz, csz)
    arrays = jax.device_put(
        jnp.split(local_array, len(mesh.local_devices), axis=0), mesh.local_devices
    )
    return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)


def params_count(params: PyTree) -> int:
    n_params = 0
    for e in jax.tree_util.tree_flatten(params)[0]:
        if isinstance(e, jax.Array):
            n_params += e.size
    return n_params


def compute_grad_norm(grads: PyTree) -> Array:
    res = 0.0
    for x in jax.tree_util.tree_flatten(grads)[0]:
        res += jnp.linalg.norm(x) ** 2
    return jnp.sqrt(res)


def format_number(n: int) -> str:
    if n < 1000:
        return f"{n}"
    if n < 1000000:
        return f"{n / 1000:.2f}K"
    if n < 1000000000:
        return f"{n / 1000000:.2f}M"
    if n < 1000000000000:
        return f"{n / 1000000000:.2f}B"
    return f"{n / 1000000000000:.2f}T"


def get_date() -> str:
    return datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")


def pad(tokens: list[int], pad_id: int, length: int) -> list[int]:
    if len(tokens) < length:
        return tokens + [pad_id] * (length - len(tokens))
    return tokens[:length]


def pad_left(tokens: list[int], pad_id: int, length: int) -> list[int]:
    if len(tokens) < length:
        return [pad_id] * (length - len(tokens)) + tokens
    return tokens[:length]
