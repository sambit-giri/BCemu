"""
BCemu backend module — shared numpy/jax/torch dispatch helpers.

A tiny array-library-agnostic layer: pick which backend's primitives
(exp/log/sqrt/trapz/...) to use at call time, so the same formula works
under plain numpy, a JAX-differentiable pipeline, or torch autograd.

Used internally by BCemu.cosmology; import it directly if another module
needs the same multi-backend support.

Install optional backends with:
    pip install jax jaxlib      # backend='jax'
    pip install torch           # backend='torch'
"""
import numpy as np

JAX_AVAILABLE = False
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    pass

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

SUPPORTED_BACKENDS = ('numpy', 'jax', 'torch')


def check_backend(backend):
    """Raise ValueError/ImportError if `backend` isn't usable; no-op otherwise."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}"
        )
    if backend == 'jax' and not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for backend='jax'.\n"
            "Install with: pip install jax jaxlib\n"
            "Or use backend='numpy' or backend='torch'."
        )
    if backend == 'torch' and not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for backend='torch'.\n"
            "Install with: pip install torch\n"
            "Or use backend='numpy' or backend='jax'."
        )


def as_torch(x):
    """Wrap a value as a torch.Tensor without breaking an existing autograd
    graph (torch.tensor(...) would copy-and-detach; torch.as_tensor doesn't).

    Non-tensor inputs (plain floats/ints -- internal constants, not
    user-controlled leaves) are wrapped as float64 regardless of torch's
    global default dtype, so precision doesn't silently degrade when mixed
    with a caller's float64 leaf tensors; the global default dtype is left
    untouched for everything else."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.float64)


def exp(x, backend):
    if backend == 'jax':
        return jnp.exp(x)
    if backend == 'torch':
        return torch.exp(as_torch(x))
    return np.exp(x)


def log(x, backend):
    if backend == 'jax':
        return jnp.log(x)
    if backend == 'torch':
        return torch.log(as_torch(x))
    return np.log(x)


def log10(x, backend):
    if backend == 'jax':
        return jnp.log10(x)
    if backend == 'torch':
        return torch.log10(as_torch(x))
    return np.log10(x)


def sqrt(x, backend):
    if backend == 'jax':
        return jnp.sqrt(x)
    if backend == 'torch':
        return torch.sqrt(as_torch(x))
    return np.sqrt(x)


def linspace(start, stop, num, backend):
    if backend == 'jax':
        return jnp.linspace(start, stop, num)
    if backend == 'torch':
        return torch.linspace(start, stop, num, dtype=torch.float64)
    return np.linspace(start, stop, num)


def trapz(y, x, backend, axis=-1):
    if backend == 'jax':
        if hasattr(jnp, 'trapezoid'):
            return jnp.trapezoid(y, x, axis=axis)
        return jnp.trapz(y, x, axis=axis)
    if backend == 'torch':
        return torch.trapezoid(as_torch(y), as_torch(x), dim=axis)
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)
