import numpy as np
import jax.numpy as jnp
import jax
import pickle
import BCemu
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- 1. Initialize the Emulator ---
bfcemu = BCemu.BCemu2025(differentiable=True)

# --- 2. Define Baryonic and Cosmological Parameters ---
# These are the 8 parameters the emulator expects.
Ob, Om = 0.0486, 0.306
bcmdict = {
    'Theta_co': 0.3,
    'log10Mc': 13.1,
    'mu': 1.0,
    'delta': 6.0,
    'eta': 0.08,
    'deta': 0.23,
    'Nstar': 0.025,
    'fb': Ob / Om,
}
bcmdict_jnp = jnp.array([bcmdict[key] for key in bfcemu.param_names])

# The new emulator is also a function of the quenching parameter q2.
# We will use a typical value for this example.
q2_val = 0.70

# Define the k-values where we want to evaluate the suppression for our plot
k_eval = 10**np.linspace(-1, 1.08, 100)

# --- 3. Get Emulated Predictions ---
# We call the emulator for each redshift and then interpolate the result
# onto our desired k_eval grid.
print("\nRunning emulator predictions...")
predictions = {}
for z_val in [0.0, 0.5, 1.0, 1.5, 2.0]:
    # The get_boost function returns the emulator's k-grid and the S(k) on that grid
    # Sk_emu = bfcemu.get_boost(bcmdict, z=z_val, q2=q2_val)
    Sk_emu = bfcemu.get_boost_differentiable(bcmdict_jnp, z=z_val, q2=q2_val)
    k_emu = bfcemu.k
    # We interpolate the result onto our specific k_eval grid for plotting
    p_interp = np.interp(k_eval, k_emu, Sk_emu)
    predictions[z_val] = p_interp
    print(f" - Prediction for z={z_val} complete.")

# --- 4. Read the BAHAMAS data for comparison ---
try:
    BAH = pickle.load(open('BAHAMAS_data.pkl', 'rb'))
except FileNotFoundError:
    print("\nWarning: 'BAHAMAS_data.pkl' not found. Plot will only show emulated results.")
    BAH = None

# --- 5. Compute Derivatives for z=0.5, q2=0.7 ---
print("\nComputing derivatives...")

# Define a function that takes parameters and returns the boost
def boost_function(params):
    return bfcemu.get_boost_differentiable(params, z=0.5, q2=0.7)

# Compute the Jacobian (derivatives with respect to all parameters)
jacobian_fn = jax.jacfwd(boost_function)
derivatives = jacobian_fn(bcmdict_jnp)

print("Derivatives computed successfully.")

# --- 6. Plot the Results ---
rcParams['font.family'] = 'sans-serif'
rcParams['axes.labelsize'] = 14
rcParams['font.size'] = 14
rcParams['axes.linewidth'] = 1.6

# First figure: Original boost predictions
plt.figure(figsize=(15, 9))
axes_map = {
    0.0: plt.subplot2grid((2, 6), (0, 0), colspan=2),
    0.5: plt.subplot2grid((2, 6), (0, 2), colspan=2),
    1.0: plt.subplot2grid((2, 6), (0, 4), colspan=2),
    1.5: plt.subplot2grid((2, 6), (1, 1), colspan=2),
    2.0: plt.subplot2grid((2, 6), (1, 3), colspan=2),
}

for z, ax in axes_map.items():
    ax.set_title(f'$z={z}$')
    if BAH and f'z={int(z) if z%1==0 else z}' in BAH:
        ax.semilogx(BAH[f'z={int(z) if z%1==0 else z}']['k'], 
                   BAH[f'z={int(z) if z%1==0 else z}']['S'], 
                   '-', c='C0', lw=5, alpha=0.2, label='BAHAMAS')
    ax.semilogx(k_eval, predictions[z], '--', c='b', lw=3, label='Emulated')
    ax.set_xlim(0.09, 12)
    ax.set_ylim(0.7, 1.08)
    ax.set_xlabel('$k$ ($h$ Mpc$^{-1}$)')
    ax.set_ylabel('$\mathcal{S}(k)$')

axes_map[0.0].legend()
plt.suptitle('Baryonic Boost Predictions', fontsize=24, y=0.98)
plt.tight_layout()
plt.show()

# Second figure: Derivatives at z=0.5, q2=0.7
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

k_emu = bfcemu.k
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

for i, param_name in enumerate(bfcemu.param_names):
    ax = axes[i]
    
    # Get the derivative for this parameter
    param_derivative = derivatives[:,i]
    
    # Interpolate onto k_eval for smoother plotting
    deriv_interp = np.interp(k_eval, k_emu, param_derivative)
    
    ax.semilogx(k_eval, deriv_interp, '-', color=colors[i], lw=3, 
                label=f'd$\mathcal{{S}}$/d{param_name}')
    ax.set_xlabel('$k$ ($h$ Mpc$^{-1}$)')
    ax.set_ylabel(f'd$\mathcal{{S}}$/d{param_name}')
    ax.set_title(f'Derivative w.r.t. {param_name}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.09, 12)
    
    # Add a horizontal line at zero for reference
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

plt.suptitle('Derivatives of Baryonic Boost at z=0.5, q2=0.7', fontsize=24, y=0.98)
plt.tight_layout()
plt.show()

# --- 7. Print some summary statistics ---
print("\n--- Derivative Summary ---")
print(f"Parameter values used:")
for i, param_name in enumerate(bfcemu.param_names):
    print(f"  {param_name}: {bcmdict[param_name]:.4f}")

print(f"\nDerivative statistics at z=0.5, q2=0.7:")
for i, param_name in enumerate(bfcemu.param_names):
    param_derivative = derivatives[:,i]
    deriv_interp = np.interp(k_eval, k_emu, param_derivative)
    max_abs_deriv = np.max(np.abs(deriv_interp))
    print(f"  {param_name}: max |dS/d{param_name}| = {max_abs_deriv:.4f}")

# --- 8. Optional: Show parameter sensitivity at a specific k-value ---
k_specific = 1.0  # h/Mpc
k_idx = np.argmin(np.abs(k_eval - k_specific))

print(f"\nDerivatives at k = {k_specific} h/Mpc:")
for i, param_name in enumerate(bfcemu.param_names):
    param_derivative = np.array(derivatives[i])
    k_emu_np = np.array(k_emu)
    deriv_interp = np.interp(k_eval, k_emu_np, param_derivative)
    deriv_at_k = deriv_interp[k_idx]
    print(f"  d$\mathcal{{S}}$/d{param_name} = {deriv_at_k:.6f}")

print("\nAnalysis complete!")