import numpy as np
import jax.numpy as jnp
import jax
import pickle
import BCemu
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- 1. Initialize the Emulator ---
bfcemu = BCemu.BCemu2025(backend='jax')

# --- 2. Define Baryonic and Cosmological Parameters ---
Ob, Om = 0.0486, 0.306
bcmdict = {
    'Theta_co': 0.3,
    'log10Mc': 13.1,
    'mu': 1.0,
    'delta': 6.0,
    'eta': 0.10,
    'deta': 0.22,
    'Nstar': 0.028,
    'fb': Ob / Om,
}
bcmdict_jnp = jnp.array([bcmdict[key] for key in bfcemu.param_names])

q2_val = 0.70
k_eval = 10**np.linspace(-1, 1.08, 200)

# LaTeX labels for each BCM parameter
PARAM_LATEX = {
    'Theta_co': r'$\Theta_{\rm co}$',
    'log10Mc':  r'$\log_{10}M_c$',
    'mu':       r'$\mu$',
    'delta':    r'$\delta$',
    'eta':      r'$\eta$',
    'deta':     r'$\Delta\eta$',
    'Nstar':    r'$N_\star$',
    'fb':       r'$f_b$',
}

# --- 3. Get Emulated Predictions ---
print("\nRunning emulator predictions...")
z_list = [0.0, 0.5, 1.0, 1.5, 2.0]
predictions = {}
for z_val in z_list:
    Sk_emu = bfcemu.get_boost_differentiable(bcmdict_jnp, z=z_val, q2=q2_val)
    p_interp = np.interp(k_eval, bfcemu.k, np.array(Sk_emu))
    predictions[z_val] = p_interp
    print(f"  z={z_val:.1f} done.")

# --- 4. Read the BAHAMAS data for comparison ---
try:
    BAH = pickle.load(open('BAHAMAS_data.pkl', 'rb'))
except FileNotFoundError:
    print("\nWarning: 'BAHAMAS_data.pkl' not found.")
    BAH = None

# --- 5. Compute Derivatives for z=0.5, q2=0.7 ---
print("\nComputing Jacobian...")
derivatives = jax.jacfwd(
    lambda p: bfcemu.get_boost_differentiable(p, z=0.5, q2=0.7)
)(bcmdict_jnp)
print("  Done.")

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
rcParams.update({
    'font.family':          'sans-serif',
    'font.size':            12,
    'axes.labelsize':       13,
    'axes.titlesize':       12,
    'axes.linewidth':       1.2,
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'xtick.top':            True,
    'ytick.right':          True,
    'xtick.minor.visible':  True,
    'ytick.minor.visible':  True,
    'legend.framealpha':    0.9,
    'legend.edgecolor':     '0.75',
    'legend.fontsize':      11,
})

# Colormap for redshift sequence
cmap = plt.get_cmap('plasma')
z_colors = {z: cmap(i / (len(z_list) - 1)) for i, z in enumerate(z_list)}

# ---------------------------------------------------------------------------
# Figure 1: S(k) predictions
# ---------------------------------------------------------------------------
fig1, ax = plt.subplots(figsize=(6, 4.5))

for z in z_list:
    col = z_colors[z]
    # BAHAMAS comparison (thick, semi-transparent background line)
    if BAH is not None:
        bah_key = f'z={int(z) if z % 1 == 0 else z}'
        if bah_key in BAH:
            ax.semilogx(BAH[bah_key]['k'], BAH[bah_key]['S'],
                        color=col, lw=6, alpha=0.2, solid_capstyle='round')
    ax.semilogx(k_eval, predictions[z],
                color=col, lw=2, label=f'$z = {z:.1f}$')

ax.axhline(1.0, ls=':', color='k', lw=0.8, alpha=0.4)
ax.set_xscale('log')
ax.set_xlim(0.09, 12)
ax.set_ylim(0.68, 1.05)
ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
ax.set_ylabel(r'$\mathcal{S}(k) = P_{\rm hydro}/P_{\rm DMO}$')
ax.legend(loc='lower left', ncol=2)
ax.grid(True, which='both', alpha=0.15, lw=0.6)
ax.set_title(r'Baryonic suppression — BCemu2025', pad=8)

fig1.tight_layout()
# fig1.savefig('BCemu2025_boost.pdf', bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# Figure 2: Jacobian ∂S/∂θ at z=0.5, q2=0.7
# ---------------------------------------------------------------------------
param_names = bfcemu.param_names
n_params = len(param_names)           # 8
n_cols = 4
n_rows = int(np.ceil(n_params / n_cols))

tab_colors = plt.get_cmap('tab10').colors

fig2, axes2 = plt.subplots(n_rows, n_cols,
                            figsize=(14, 3.6 * n_rows),
                            sharex=True)
axes2_flat = axes2.flatten()

k_emu = np.array(bfcemu.k)

for i, pname in enumerate(param_names):
    ax = axes2_flat[i]
    col = tab_colors[i % 10]

    deriv = np.interp(k_eval, k_emu, np.array(derivatives[:, i]))

    # Filled area (split by sign for clarity)
    ax.fill_between(k_eval, 0, deriv,
                    where=(deriv >= 0), color=col, alpha=0.18, lw=0)
    ax.fill_between(k_eval, 0, deriv,
                    where=(deriv <  0), color=col, alpha=0.18, lw=0)
    ax.semilogx(k_eval, deriv, color=col, lw=2)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.35)

    ax.set_xlim(0.09, 12)
    ax.grid(True, which='both', alpha=0.15, lw=0.6)

    # Panel label and parameter name
    ax.set_title(PARAM_LATEX.get(pname, pname), pad=5)
    ax.text(0.04, 0.95, f'({chr(ord("a") + i)})',
            transform=ax.transAxes, fontsize=10,
            va='top', ha='left', color='0.35')

    # y-label only on leftmost column
    if i % n_cols == 0:
        ax.set_ylabel(r'$\partial\mathcal{S}/\partial\theta$')

# x-labels only on bottom row
for ax in axes2[n_rows - 1]:
    ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')

# Hide unused subplots (if any)
for ax in axes2_flat[n_params:]:
    ax.set_visible(False)

fig2.suptitle(
    r'Jacobian $\partial\mathcal{S}/\partial\theta$ at $z=0.5,\;q_2=0.7$',
    y=1.01, fontsize=13,
)
fig2.tight_layout()
# fig2.savefig('BCemu2025_derivatives.pdf', bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("\n--- Parameter values ---")
for pname in param_names:
    print(f"  {pname:12s} = {bcmdict[pname]:.4f}")

print("\n--- max |∂S/∂θ| at z=0.5, q2=0.7 ---")
for i, pname in enumerate(param_names):
    deriv = np.interp(k_eval, k_emu, np.array(derivatives[:, i]))
    print(f"  {pname:12s}   {np.max(np.abs(deriv)):.5f}")

print("\nDone.")
