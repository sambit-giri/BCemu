import numpy as np
import os, json
from tqdm.auto import tqdm

# --- Optional backend imports ---

_JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from flax.serialization import from_bytes
    _JAX_AVAILABLE = True
except ImportError:
    pass

_MSGPACK_AVAILABLE = False
try:
    import msgpack as _msgpack_lib
    _MSGPACK_AVAILABLE = True
except ImportError:
    pass

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from .download import get_package_resource_path, download_emulators
except (ImportError, ModuleNotFoundError):
    from BCemu import get_package_resource_path, download_emulators

if _JAX_AVAILABLE:
    class FlaxBCemuNet(nn.Module):
        """
        A JAX/Flax based network whose architecture is defined by a list of layer sizes.
        This internal class is used to load the dynamically-sized models
        created by the Optuna training script.
        """
        n_output_pca: int
        hidden_layers: list[int]

        @nn.compact
        def __call__(self, x):
            for i, out_features in enumerate(self.hidden_layers):
                x = nn.Dense(features=out_features, name=f'hidden_{i}')(x)
                x = nn.relu(x)
            x = nn.Dense(features=self.n_output_pca, name='output')(x)
            return x


def _decode_flax_msgpack(filepath):
    """
    Decode a Flax msgpack parameter file to a nested dict of numpy arrays,
    without requiring JAX or Flax to be installed.

    BCemu2025 model files store each numpy array as msgpack ExtType 1 where
    the extension data is itself a complete msgpack-encoded 3-element array::

        [shape_list, dtype_string, raw_bytes_as_bin]

    e.g. [[8, 256], 'float32', b'...'].  The whole ExtType payload is a
    valid msgpack value, so a plain ``unpackb(data)`` is sufficient.
    """
    if not _MSGPACK_AVAILABLE:
        raise ImportError(
            "The 'msgpack' package is required for the numpy/torch backends.\n"
            "Install it with: pip install msgpack"
        )

    def _ext_hook(code, data):
        if code == 1:
            obj = _msgpack_lib.unpackb(data, raw=False)
            # obj = [shape_list, dtype_str, raw_bytes]
            shape, dtype_str, raw_bytes = obj[0], obj[1], obj[2]
            return np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
        return _msgpack_lib.ExtType(code, data)

    def _decode_keys(obj):
        """Recursively convert byte-string keys to unicode strings."""
        if isinstance(obj, dict):
            return {
                (k.decode('utf-8') if isinstance(k, bytes) else k): _decode_keys(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_decode_keys(item) for item in obj]
        return obj

    with open(filepath, 'rb') as f:
        raw = _msgpack_lib.unpackb(f.read(), ext_hook=_ext_hook, raw=True)
    return _decode_keys(raw)


def _numpy_forward(params, hidden_layers, x):
    """
    Pure-numpy forward pass matching the FlaxBCemuNet architecture.

    Parameters
    ----------
    params : dict
        Nested dict with keys 'hidden_0', 'hidden_1', ..., 'output',
        each containing 'kernel' and 'bias' numpy arrays.
    hidden_layers : list of int
        Number of units per hidden layer.
    x : np.ndarray, shape (..., n_input)
    """
    for i in range(len(hidden_layers)):
        layer = params[f'hidden_{i}']
        x = x @ layer['kernel'] + layer['bias']
        x = np.maximum(0.0, x)  # ReLU
    out = params['output']
    return x @ out['kernel'] + out['bias']


def _params_numpy_to_torch(params_np):
    """Recursively convert a nested dict of numpy arrays to float32 torch tensors."""
    if isinstance(params_np, np.ndarray):
        return torch.tensor(params_np, dtype=torch.float32)
    if isinstance(params_np, dict):
        return {k: _params_numpy_to_torch(v) for k, v in params_np.items()}
    return params_np


def _torch_forward(params, hidden_layers, x):
    """
    PyTorch forward pass matching the FlaxBCemuNet architecture.
    Supports autograd — gradients flow through x while params are treated
    as fixed constants (requires_grad=False by default).

    Parameters
    ----------
    params : dict
        Nested dict with keys 'hidden_0', ..., 'output',
        each containing 'kernel' and 'bias' torch tensors.
    hidden_layers : list of int
        Number of units per hidden layer.
    x : torch.Tensor, shape (..., n_input)
    """
    for i in range(len(hidden_layers)):
        layer = params[f'hidden_{i}']
        x = x @ layer['kernel'] + layer['bias']
        x = torch.relu(x)
    out = params['output']
    return x @ out['kernel'] + out['bias']


class BCemu2025:
    """
    Emulator for baryonic effects on the matter power spectrum (BCemu 2025).

    Loads natively-trained Flax models and supports multiple backends:

    * ``'numpy'`` – Pure-numpy backend (default).  Requires only ``msgpack``.
                    Fast on all platforms, including Apple Silicon.
                    No differentiability.
    * ``'torch'`` – PyTorch backend.  Requires ``torch`` and ``msgpack``.
                    Supports autograd via ``get_boost_differentiable``.
    * ``'jax'``   – JAX/Flax backend.  Provides JIT-compiled,
                    fully-differentiable inference via
                    ``get_boost_differentiable``.  Best on TPUs / NVIDIA GPUs.

    Parameters
    ----------
    model_dir : str or None
        Path to the directory containing the BCemu2025 model files.
        Defaults to the package's ``input_data`` directory, downloading
        files automatically when absent.
    backend : {'jax', 'numpy', 'torch'}
        Computational backend to use for inference.
    """

    def __init__(self, model_dir=None, backend='numpy'):
        # --- Validate backend ---
        supported_backends = ('jax', 'numpy', 'torch')
        if backend not in supported_backends:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Supported backends: {supported_backends}."
            )
        if backend == 'jax' and not _JAX_AVAILABLE:
            raise ImportError(
                "JAX and Flax are required for backend='jax'.\n"
                "Install them with:  pip install jax flax\n"
                "Or use backend='numpy' or backend='torch'."
            )
        if backend in ('numpy', 'torch') and not _MSGPACK_AVAILABLE:
            raise ImportError(
                "The 'msgpack' package is required for the numpy/torch backends.\n"
                "Install it with:  pip install msgpack"
            )
        if backend == 'torch' and not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for backend='torch'.\n"
                "Install it from https://pytorch.org or with:  pip install torch\n"
                "Or use backend='numpy' for a lighter-weight alternative."
            )

        self.backend = backend

        # --- Resolve model directory ---
        if model_dir is None:
            package_name = "BCemu"
            data_dir_name = "input_data"
            default_model_dir = get_package_resource_path(package_name, data_dir_name)

            meta_path_check = os.path.join(default_model_dir, 'BCemu2025_meta.json')
            if not os.path.exists(meta_path_check):
                print(f"Emulator files not found in default directory: {default_model_dir}")
                print("Calling downloader...")
                download_emulators(model_name='BCemu2025')
                if not os.path.exists(meta_path_check):
                    raise FileNotFoundError("Download failed or did not create 'BCemu2025_meta.json'.")

            self.model_dir = default_model_dir
        else:
            self.model_dir = model_dir

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: '{self.model_dir}'")

        meta_path = os.path.join(self.model_dir, 'BCemu2025_meta.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: '{meta_path}'")

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.k = np.array(meta['k_values'])
        self.param_names = meta['param_names']
        self.z_grid = np.array(meta['z_grid'])
        self.q2_grid = np.array(meta['q2_grid'])
        self.cosmology = meta['cosmology']

        arch = meta['architecture']
        self.hidden_layers = arch['hidden_layers']
        self.n_output_pca = arch['n_output_pca']
        self.n_input_features = arch['n_input']

        # --- Backend-specific one-time setup ---
        if backend == 'jax':
            self.z_grid_jax = jnp.array(self.z_grid)
            self.q2_grid_jax = jnp.array(self.q2_grid)
            self.flax_model = FlaxBCemuNet(
                n_output_pca=self.n_output_pca,
                hidden_layers=self.hidden_layers,
            )
            # JIT-compile the apply function once; architecture is unified
            jit_apply = jax.jit(self.flax_model.apply)

        # --- Load per-grid-point weights and transforms ---
        self.emulators = {}
        print(f"Loading BCemu2025 models (backend='{backend}')...")
        grid_points = [(z, q2) for z in self.z_grid for q2 in self.q2_grid]

        for z, q2 in tqdm(grid_points, desc="Loading model weights"):
            key = (z, q2)
            model_path = os.path.join(
                self.model_dir,
                f'BCemu2025_emulator_z{z:.2f}_q2_{q2:.2f}.msgpack',
            )
            transform_path = os.path.join(
                self.model_dir,
                f'BCemu2025_transforms_z{z:.2f}_q2_{q2:.2f}.npz',
            )

            transforms = np.load(transform_path)

            if backend == 'jax':
                with open(model_path, 'rb') as f:
                    dummy_params = self.flax_model.init(
                        jax.random.PRNGKey(0),
                        jnp.ones((1, self.n_input_features)),
                    )['params']
                    loaded_params = from_bytes(dummy_params, f.read())
                self.emulators[key] = {
                    'apply_fn': jit_apply,
                    'params': loaded_params,
                    'scaler_mean': jnp.array(transforms['scaler_mean']),
                    'scaler_scale': jnp.array(transforms['scaler_scale']),
                    'pca_mean': jnp.array(transforms['pca_mean']),
                    'pca_components': jnp.array(transforms['pca_components']),
                }
            elif backend == 'numpy':
                loaded_params = _decode_flax_msgpack(model_path)
                self.emulators[key] = {
                    'params': loaded_params,
                    'scaler_mean': np.array(transforms['scaler_mean']),
                    'scaler_scale': np.array(transforms['scaler_scale']),
                    'pca_mean': np.array(transforms['pca_mean']),
                    'pca_components': np.array(transforms['pca_components']),
                }
            else:  # torch
                loaded_params_np = _decode_flax_msgpack(model_path)
                self.emulators[key] = {
                    'params': _params_numpy_to_torch(loaded_params_np),
                    'scaler_mean': torch.tensor(transforms['scaler_mean'], dtype=torch.float32),
                    'scaler_scale': torch.tensor(transforms['scaler_scale'], dtype=torch.float32),
                    'pca_mean': torch.tensor(transforms['pca_mean'], dtype=torch.float32),
                    'pca_components': torch.tensor(transforms['pca_components'], dtype=torch.float32),
                }

        # --- Wire up get_boost_differentiable ---
        self.get_boost_differentiable = None
        if backend == 'jax':
            print("JIT compiling the differentiable boost function...")
            self.get_boost_differentiable = jax.jit(self._get_boost_differentiable)
            print("...JIT compilation complete.")
        elif backend == 'torch':
            self.get_boost_differentiable = self._get_boost_differentiable_torch

        print("...BCemu2025 emulator is ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_reconstructed_Sk_numpy(self, params_np, z, q2):
        """Run one grid-point forward pass; always returns a numpy array."""
        key = (z, q2)
        c = self.emulators[key]

        if self.backend == 'torch':
            x = torch.tensor(params_np, dtype=torch.float32)
            x = (x - c['scaler_mean']) / c['scaler_scale']
            amplitudes = _torch_forward(c['params'], self.hidden_layers, x)
            result = amplitudes @ c['pca_components'] + c['pca_mean']
            return result.detach().numpy().flatten()

        # numpy and jax share the input-scaling step
        scaled = (params_np - np.array(c['scaler_mean'])) / np.array(c['scaler_scale'])

        if self.backend == 'jax':
            pca_amplitudes = np.array(
                c['apply_fn']({'params': c['params']}, jnp.array(scaled))
            )
        else:  # numpy
            pca_amplitudes = _numpy_forward(c['params'], self.hidden_layers, scaled)

        return (
            np.dot(pca_amplitudes, np.array(c['pca_components'])) + np.array(c['pca_mean'])
        ).flatten()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bcm_param_info(self):
        """
        Defines, displays, and returns the fixed and fiducial free parameters
        for the baryonification model.
        """
        print("--- Baryonification Model Parameters ---")

        fixed_params = {
            'ciga': 0.1,
            'gamma': 1.5,
            'Mhalo_min': 2.5e11,
            'eps1': 0.5,
            'halo_excl': 0.4,
            'q1': 0.25,
        }

        free_params_fiducial = {
            'Theta_co': 0.3,
            'log10Mc': 13.1,
            'mu': 1.0,
            'delta': 6.0,
            'eta': 0.10,
            'deta': 0.22,
            'Nstar': 0.028,
            'fb': 0.0486 / 0.306,
        }

        def format_params(params_dict):
            items = [f"{name}={value:.3g}" for name, value in params_dict.items()]
            return ", ".join(items)

        print(f"Fixed parameters: {format_params(fixed_params)}")
        print(f"Fiducial free parameters: {format_params(free_params_fiducial)}")

        return fixed_params, free_params_fiducial

    def get_boost(self, bcmdict, z, q2=0.70):
        """Return (k, S(k)) using bilinear interpolation over the (z, q2) grid.
        Always returns numpy arrays regardless of backend."""
        from scipy.interpolate import interp1d
        params_np = np.array([[bcmdict[key] for key in self.param_names]])

        if not (self.z_grid.min() <= z <= self.z_grid.max() and
                self.q2_grid.min() <= q2 <= self.q2_grid.max()):
            raise ValueError(f"Query point (z={z}, q2={q2}) is outside the emulator's grid.")

        z_idx = np.searchsorted(self.z_grid, z)
        z1, z2 = self.z_grid[max(0, z_idx - 1)], self.z_grid[min(len(self.z_grid) - 1, z_idx)]
        q2_idx = np.searchsorted(self.q2_grid, q2)
        q2_1, q2_2 = self.q2_grid[max(0, q2_idx - 1)], self.q2_grid[min(len(self.q2_grid) - 1, q2_idx)]

        if z1 == z2 and q2_1 == q2_2:
            return self.k, self._get_reconstructed_Sk_numpy(params_np, z, q2)

        S11 = self._get_reconstructed_Sk_numpy(params_np, z1, q2_1)
        S12 = self._get_reconstructed_Sk_numpy(params_np, z1, q2_2)
        S21 = self._get_reconstructed_Sk_numpy(params_np, z2, q2_1)
        S22 = self._get_reconstructed_Sk_numpy(params_np, z2, q2_2)

        if q2_1 == q2_2:
            f_interp_z = interp1d([z1, z2], np.vstack([S11, S21]), axis=0)
            S_final = f_interp_z(z)
        elif z1 == z2:
            f_interp_q2 = interp1d([q2_1, q2_2], np.vstack([S11, S12]), axis=0)
            S_final = f_interp_q2(q2)
        else:
            f_interp_q2_at_z1 = interp1d([q2_1, q2_2], np.vstack([S11, S12]), axis=0)
            S_z1 = f_interp_q2_at_z1(q2)
            f_interp_q2_at_z2 = interp1d([q2_1, q2_2], np.vstack([S21, S22]), axis=0)
            S_z2 = f_interp_q2_at_z2(q2)
            f_interp_z = interp1d([z1, z2], np.vstack([S_z1, S_z2]), axis=0)
            S_final = f_interp_z(z)

        return self.k, S_final.flatten()

    def _get_boost_differentiable(self, params_jnp, z, q2=0.70):
        """
        JAX-native differentiable boost (backend='jax' only).

        Parameters
        ----------
        params_jnp : jnp.ndarray, shape (n_params,)
            BCM parameters as a JAX array.
        z, q2 : float
            Redshift and q2 value.

        Returns
        -------
        jnp.ndarray, shape (n_k,)
            Compatible with jax.jacfwd / jax.grad.
        """
        z_idx = jnp.searchsorted(self.z_grid_jax, z)
        z_idx_low = jnp.maximum(0, z_idx - 1)
        z_idx_high = jnp.minimum(len(self.z_grid_jax) - 1, z_idx)

        q2_idx = jnp.searchsorted(self.q2_grid_jax, q2)
        q2_idx_low = jnp.maximum(0, q2_idx - 1)
        q2_idx_high = jnp.minimum(len(self.q2_grid_jax) - 1, q2_idx)

        z1, z2 = self.z_grid_jax[z_idx_low], self.z_grid_jax[z_idx_high]
        q2_1, q2_2 = self.q2_grid_jax[q2_idx_low], self.q2_grid_jax[q2_idx_high]

        # Evaluate all grid points and stack; JAX needs to trace all branches
        all_S = []
        for z_val in self.z_grid:
            for q2_val in self.q2_grid:
                c = self.emulators[(z_val, q2_val)]
                scaled_p = (params_jnp - c['scaler_mean']) / c['scaler_scale']
                amplitudes = c['apply_fn']({'params': c['params']}, scaled_p)
                all_S.append((jnp.dot(amplitudes, c['pca_components']) + c['pca_mean']).flatten())

        all_S = jnp.stack(all_S, axis=0)

        def _get(zi, q2i):
            return all_S[zi * len(self.q2_grid) + q2i]

        S11 = _get(z_idx_low, q2_idx_low)
        S12 = _get(z_idx_low, q2_idx_high)
        S21 = _get(z_idx_high, q2_idx_low)
        S22 = _get(z_idx_high, q2_idx_high)

        w_z  = (z  - z1)  / jnp.maximum(1e-9, z2  - z1)
        w_q2 = (q2 - q2_1) / jnp.maximum(1e-9, q2_2 - q2_1)

        S_z1 = (1 - w_q2) * S11 + w_q2 * S12
        S_z2 = (1 - w_q2) * S21 + w_q2 * S22
        S_final = (1 - w_z) * S_z1 + w_z * S_z2

        S_final = jnp.where(z1 == z2,    S_z1, S_final)
        S_final = jnp.where(q2_1 == q2_2, (1 - w_z) * S11 + w_z * S21, S_final)
        return S_final

    def _get_boost_differentiable_torch(self, params_t, z, q2=0.70):
        """
        PyTorch-native differentiable boost (backend='torch' only).

        Parameters
        ----------
        params_t : torch.Tensor, shape (n_params,)
            BCM parameters as a torch tensor (set requires_grad=True for autograd).
        z, q2 : float
            Redshift and q2 value.

        Returns
        -------
        torch.Tensor, shape (n_k,)
            Compatible with torch.autograd.functional.jacobian / .backward().

        Examples
        --------
        >>> params_t = torch.tensor([...], requires_grad=True)
        >>> S = emu.get_boost_differentiable(params_t, z=0.5)
        >>> J = torch.autograd.functional.jacobian(
        ...         lambda p: emu.get_boost_differentiable(p, z=0.5), params_t)
        """
        # Grid index finding uses numpy — no gradients needed w.r.t. z/q2
        z_idx   = np.searchsorted(self.z_grid, z)
        z_low   = self.z_grid[max(0, z_idx - 1)]
        z_high  = self.z_grid[min(len(self.z_grid) - 1, z_idx)]
        q2_idx  = np.searchsorted(self.q2_grid, q2)
        q2_low  = self.q2_grid[max(0, q2_idx - 1)]
        q2_high = self.q2_grid[min(len(self.q2_grid) - 1, q2_idx)]

        def _eval(z_val, q2_val):
            c = self.emulators[(z_val, q2_val)]
            scaled = (params_t - c['scaler_mean']) / c['scaler_scale']
            amplitudes = _torch_forward(c['params'], self.hidden_layers, scaled)
            return (amplitudes @ c['pca_components'] + c['pca_mean']).flatten()

        if z_low == z_high and q2_low == q2_high:
            return _eval(z_low, q2_low)

        S11 = _eval(z_low,  q2_low)
        S12 = _eval(z_low,  q2_high)
        S21 = _eval(z_high, q2_low)
        S22 = _eval(z_high, q2_high)

        if z_low == z_high:
            w_q2 = (q2 - q2_low) / max(1e-9, q2_high - q2_low)
            return (1 - w_q2) * S11 + w_q2 * S12

        if q2_low == q2_high:
            w_z = (z - z_low) / max(1e-9, z_high - z_low)
            return (1 - w_z) * S11 + w_z * S21

        w_z  = (z  - z_low)  / (z_high  - z_low)
        w_q2 = (q2 - q2_low) / (q2_high - q2_low)
        S_z1 = (1 - w_q2) * S11 + w_q2 * S12
        S_z2 = (1 - w_q2) * S21 + w_q2 * S22
        return (1 - w_z) * S_z1 + w_z * S_z2
