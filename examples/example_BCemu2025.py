import numpy as np
import pickle
import BCemu
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- 1. Initialize the Emulator ---
bfcemu = BCemu.BCemu2025()

# --- 2. Define Baryonic and Cosmological Parameters ---
# These are the 8 parameters the emulator expects.
Ob, Om = 0.0463, 0.2793
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
    k_emu, Sk_emu = bfcemu.get_boost(bcmdict, z=z_val, q2=q2_val)
    
    # We interpolate the result onto our specific k_eval grid for plotting
    p_interp = np.interp(k_eval, k_emu, Sk_emu)
    predictions[z_val] = p_interp
    print(f"  - Prediction for z={z_val} complete.")

# --- 4. Read the BAHAMAS data for comparison ---
try:
    BAH = pickle.load(open('BAHAMAS_data.pkl', 'rb'))
except FileNotFoundError:
    print("\nWarning: 'BAHAMAS_data.pkl' not found. Plot will only show emulated results.")
    BAH = None

# --- 5. Plot the Results ---
rcParams['font.family'] = 'sans-serif'
rcParams['axes.labelsize']  = 20
rcParams['font.size']       = 20 
rcParams['axes.linewidth']  = 1.6

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
        ax.semilogx(BAH[f'z={int(z) if z%1==0 else z}']['k'], BAH[f'z={int(z) if z%1==0 else z}']['S'], '-', c='C0', lw=5, alpha=0.2, label='BAHAMAS')

    ax.semilogx(k_eval, predictions[z], '--', c='b', lw=3, label='Emulated')

    ax.set_xlim(0.09, 12)
    ax.set_ylim(0.7, 1.08)
    ax.set_xlabel('$k$ ($h$ Mpc$^{-1}$)')
    ax.set_ylabel('$\mathcal{S}(k)$')

axes_map[0.0].legend()

plt.tight_layout()
plt.show()
