import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.serialization import from_bytes
import numpy as np
import os, json
from tqdm.auto import tqdm

try:
    from .download import get_package_resource_path, download_emulators
except (ImportError, ModuleNotFoundError):
    from BCemu import get_package_resource_path, download_emulators

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

class BCemu2025:
    """
    A JAX-based class that loads natively-trained Flax models for fast, 
    version-agnostic, and optionally differentiable inference.
    """
    def __init__(self, model_dir=None, differentiable=False):
        """
        Initializes the BCemu2025 object by loading JAX-native models.

        Args:
            model_dir (str, optional): Path to the directory containing emulator files.
                If None, it uses the default package location and downloads if necessary.
            differentiable (bool, optional): If True, the differentiable method
                `get_boost_differentiable` will be Just-In-Time (JIT) compiled
                during initialization. This adds to the startup time but makes
                subsequent calls to the differentiable function extremely fast.
                Defaults to False.
        """
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
        self.emulators = {}

        print("Loading BCemu2025 JAX models...")
        grid_points = [(z, q2) for z in self.z_grid for q2 in self.q2_grid]
        for z, q2 in tqdm(grid_points, desc="Compiling individual models"):
            key = (z, q2)
            model_path = os.path.join(self.model_dir, f'BCemu2025_emulator_z{z:.2f}_q2_{q2:.2f}.msgpack')
            arch_path = os.path.join(self.model_dir, f'BCemu2025_arch_z{z:.2f}_q2_{q2:.2f}.json')
            transform_path = os.path.join(self.model_dir, f'BCemu2025_transforms_z{z:.2f}_q2_{q2:.2f}.npz')

            with open(arch_path, 'r') as f: arch = json.load(f)
            flax_model = FlaxBCemuNet(n_output_pca=arch['n_output_pca'], hidden_layers=arch['hidden_layers'])
            
            with open(model_path, 'rb') as f:
                dummy_params = flax_model.init(jax.random.PRNGKey(0), jnp.ones((1, arch['n_input'])))['params']
                loaded_params = from_bytes(dummy_params, f.read())

            transforms = np.load(transform_path)
            self.emulators[key] = {
                'apply_fn': flax_model.apply,
                'params': loaded_params,
                'scaler_mean': jnp.array(transforms['scaler_mean']),
                'scaler_scale': jnp.array(transforms['scaler_scale']),
                'pca_mean': jnp.array(transforms['pca_mean']),
                'pca_components': jnp.array(transforms['pca_components'])
            }
        
        # --- Optionally pre-compile the differentiable function for performance ---
        self.get_boost_differentiable = None
        if differentiable:
            print("JIT compiling the differentiable boost function...")
            self.get_boost_differentiable = jax.jit(self._get_boost_differentiable)
            print("...JIT compilation complete.")
        
        print("...BCemu2025 emulator is ready.")

    def _get_reconstructed_Sk_numpy(self, params_np, z, q2):
        """Internal method for on-grid prediction using NumPy (non-differentiable)."""
        key = (z, q2)
        components = self.emulators[key]
        
        scaled_params = (params_np - np.array(components['scaler_mean'])) / np.array(components['scaler_scale'])
        
        pca_amplitudes_jax = components['apply_fn']({'params': components['params']}, jnp.array(scaled_params))
        pca_amplitudes = np.array(pca_amplitudes_jax)
        
        reconstructed_sk = np.dot(pca_amplitudes, np.array(components['pca_components'])) + np.array(components['pca_mean'])
        
        return reconstructed_sk.flatten()

    def get_boost(self, bcmdict, z, q2=0.70):
        """
        Calculates S(k) using NumPy/SciPy for fast, non-differentiable inference.
        """
        from scipy.interpolate import interp1d
        params_np = np.array([[bcmdict[key] for key in self.param_names]])

        if not (self.z_grid.min() <= z <= self.z_grid.max() and \
                self.q2_grid.min() <= q2 <= self.q2_grid.max()):
            raise ValueError(f"Query point (z={z}, q2={q2}) is outside the emulator's grid.")

        z_idx = np.searchsorted(self.z_grid, z); z1, z2 = self.z_grid[max(0, z_idx - 1)], self.z_grid[min(len(self.z_grid)-1, z_idx)]
        q2_idx = np.searchsorted(self.q2_grid, q2); q2_1, q2_2 = self.q2_grid[max(0, q2_idx - 1)], self.q2_grid[min(len(self.q2_grid)-1, q2_idx)]
        
        if z1 == z2 and q2_1 == q2_2: return self.k, self._get_reconstructed_Sk_numpy(params_np, z, q2)
        
        S11 = self._get_reconstructed_Sk_numpy(params_np, z1, q2_1); S12 = self._get_reconstructed_Sk_numpy(params_np, z1, q2_2)
        S21 = self._get_reconstructed_Sk_numpy(params_np, z2, q2_1); S22 = self._get_reconstructed_Sk_numpy(params_np, z2, q2_2)
        
        if q2_1 == q2_2: f_interp_z = interp1d([z1, z2], np.vstack([S11, S21]), axis=0); S_final = f_interp_z(z)
        elif z1 == z2: f_interp_q2 = interp1d([q2_1, q2_2], np.vstack([S11, S12]), axis=0); S_final = f_interp_q2(q2)
        else:
            f_interp_q2_at_z1 = interp1d([q2_1, q2_2], np.vstack([S11, S12]), axis=0); S_z1 = f_interp_q2_at_z1(q2)
            f_interp_q2_at_z2 = interp1d([q2_1, q2_2], np.vstack([S21, S22]), axis=0); S_z2 = f_interp_q2_at_z2(q2)
            f_interp_z = interp1d([z1, z2], np.vstack([S_z1, S_z2]), axis=0); S_final = f_interp_z(z)

        return self.k, S_final.flatten()

    def _get_boost_differentiable(self, params_jnp, z, q2=0.70):
        """
        Calculates the baryonic suppression S(k) using only JAX operations.
        This pure function is designed to be JIT-compiled for performance and
        is fully differentiable with respect to its inputs.

        Args:
            params_jnp (jax.numpy.ndarray): A JAX array of baryonic parameters
                with shape (1, n_params) or (n_params,). The order must match
                the internal `self.param_names` list.
            z (float): The redshift at which to calculate the suppression.
            q2 (float, optional): The integrated quenching value. Defaults to 0.70.

        Returns:
            jax.numpy.ndarray: The predicted baryonic suppression S(k), a JAX
                               array of shape (n_k_bins,).
        """
        def _get_reconstructed_Sk_jax(p_jnp, z_val, q2_val):
            components = self.emulators[(z_val, q2_val)]
            scaled_p = (p_jnp - components['scaler_mean']) / components['scaler_scale']
            amplitudes = components['apply_fn']({'params': components['params']}, scaled_p)
            return jnp.dot(amplitudes, components['pca_components']) + components['pca_mean']

        z_idx = jnp.searchsorted(self.z_grid, z)
        z1, z2 = self.z_grid[jnp.maximum(0, z_idx - 1)], self.z_grid[jnp.minimum(len(self.z_grid)-1, z_idx)]
        
        q2_idx = jnp.searchsorted(self.q2_grid, q2)
        q2_1, q2_2 = self.q2_grid[jnp.maximum(0, q2_idx - 1)], self.q2_grid[jnp.minimum(len(self.q2_grid)-1, q2_idx)]

        S11 = _get_reconstructed_Sk_jax(params_jnp, z1.item(), q2_1.item())
        S12 = _get_reconstructed_Sk_jax(params_jnp, z1.item(), q2_2.item())
        S21 = _get_reconstructed_Sk_jax(params_jnp, z2.item(), q2_1.item())
        S22 = _get_reconstructed_Sk_jax(params_jnp, z2.item(), q2_2.item())

        w_z = (z - z1) / jnp.maximum(1e-9, z2 - z1)
        w_q2 = (q2 - q2_1) / jnp.maximum(1e-9, q2_2 - q2_1)
        
        S_z1 = (1 - w_q2) * S11 + w_q2 * S12
        S_z2 = (1 - w_q2) * S21 + w_q2 * S22
        S_final = (1 - w_z) * S_z1 + w_z * S_z2
        
        # Handle cases where z or q2 are on the grid to avoid NaN gradients
        S_final = jnp.where(z1 == z2, S_z1, S_final)
        S_final = jnp.where(q2_1 == q2_2, (1 - w_z) * S11 + w_z * S21, S_final)
        
        return S_final.flatten()