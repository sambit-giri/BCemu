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
    version-agnostic, and optionally differentiable inference. This version
    assumes a single, unified architecture for all emulator grid points.
    """
    def __init__(self, model_dir=None, differentiable=False):
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
        
        # Load the single, unified architecture from the metadata
        arch = meta['architecture']
        self.flax_model = FlaxBCemuNet(
            n_output_pca=arch['n_output_pca'],
            hidden_layers=arch['hidden_layers']
        )
        self.n_input_features = arch['n_input']

        self.emulators = {}
        print("Loading BCemu2025 JAX models with unified architecture...")
        grid_points = [(z, q2) for z in self.z_grid for q2 in self.q2_grid]
        
        # JIT compile the model's apply function once, as the architecture is the same for all
        jit_apply = jax.jit(self.flax_model.apply)

        for z, q2 in tqdm(grid_points, desc="Loading model weights"):
            key = (z, q2)
            model_path = os.path.join(self.model_dir, f'BCemu2025_emulator_z{z:.2f}_q2_{q2:.2f}.msgpack')
            transform_path = os.path.join(self.model_dir, f'BCemu2025_transforms_z{z:.2f}_q2_{q2:.2f}.npz')

            with open(model_path, 'rb') as f:
                # Initialize with a dummy key, then load the actual params
                dummy_params = self.flax_model.init(jax.random.PRNGKey(0), jnp.ones((1, self.n_input_features)))['params']
                loaded_params = from_bytes(dummy_params, f.read())

            transforms = np.load(transform_path)
            self.emulators[key] = {
                'apply_fn': jit_apply, # Use the same JIT'd function for all
                'params': loaded_params,
                'scaler_mean': jnp.array(transforms['scaler_mean']),
                'scaler_scale': jnp.array(transforms['scaler_scale']),
                'pca_mean': jnp.array(transforms['pca_mean']),
                'pca_components': jnp.array(transforms['pca_components'])
            }
        
        self.get_boost_differentiable = None
        if differentiable:
            print("JIT compiling the differentiable boost function...")
            self.get_boost_differentiable = jax.jit(self._get_boost_differentiable)
            print("...JIT compilation complete.")
        
        print("...BCemu2025 emulator is ready.")

    def _get_reconstructed_Sk_numpy(self, params_np, z, q2):
        key = (z, q2)
        components = self.emulators[key]
        scaled_params = (params_np - np.array(components['scaler_mean'])) / np.array(components['scaler_scale'])
        pca_amplitudes_jax = components['apply_fn']({'params': components['params']}, jnp.array(scaled_params))
        pca_amplitudes = np.array(pca_amplitudes_jax)
        reconstructed_sk = np.dot(pca_amplitudes, np.array(components['pca_components'])) + np.array(components['pca_mean'])
        return reconstructed_sk.flatten()

    def get_boost(self, bcmdict, z, q2=0.70):
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
        
        S_final = jnp.where(z1 == z2, S_z1, S_final)
        S_final = jnp.where(q2_1 == q2_2, (1 - w_z) * S11 + w_z * S21, S_final)
        
        return S_final.flatten()