"""
Compare BCemu2025 backends: jax, numpy, torch
=============================================
This script loads the BCemu2025 emulator with each available backend and:
  1. Runs get_boost for a range of redshifts and checks that outputs agree.
  2. Times inference across backends.
  3. For differentiable backends (jax, torch), computes dS/d(param) and
     compares the resulting Jacobians.

Any backend whose required packages are missing is skipped gracefully.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import BCemu

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
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
Z_VALUES = [0.0, 0.5, 1.0]
Q2_VAL = 0.70

# ---------------------------------------------------------------------------
# Load each backend (skip silently if packages are missing)
# ---------------------------------------------------------------------------
emulators = {}
for backend in ('jax', 'numpy', 'torch'):
    try:
        print(f"\n{'='*60}")
        print(f"Loading backend='{backend}'...")
        emulators[backend] = BCemu.BCemu2025(backend=backend)
        print(f"Backend '{backend}' loaded successfully.")
    except ImportError as e:
        print(f"Skipping backend='{backend}': {e}")

if not emulators:
    raise RuntimeError("No backends could be loaded. Install at least one of: jax+flax, msgpack, torch.")

# Use any available emulator to get shared metadata
_any_emu = next(iter(emulators.values()))
param_names = _any_emu.param_names
k_grid = _any_emu.k

# ---------------------------------------------------------------------------
# 1. Run get_boost for each backend and collect results
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Running get_boost for each backend...")

boost_results = {}   # {backend: {z: S(k)}}
timing = {}          # {backend: seconds per call}

for backend, emu in emulators.items():
    boost_results[backend] = {}
    times = []
    for z in Z_VALUES:
        t0 = time.perf_counter()
        k, S = emu.get_boost(bcmdict, z=z, q2=Q2_VAL)
        times.append(time.perf_counter() - t0)
        boost_results[backend][z] = S
    timing[backend] = np.mean(times)
    print(f"  {backend:6s}  mean time per call: {timing[backend]*1e3:.1f} ms")

# ---------------------------------------------------------------------------
# 2. Numerical agreement check
# ---------------------------------------------------------------------------
print("\nMax absolute difference between backends:")
backend_list = list(boost_results.keys())
for i in range(len(backend_list)):
    for j in range(i + 1, len(backend_list)):
        b1, b2 = backend_list[i], backend_list[j]
        diffs = [np.max(np.abs(boost_results[b1][z] - boost_results[b2][z]))
                 for z in Z_VALUES]
        print(f"  |{b1} - {b2}|_max = {max(diffs):.2e}")

# ---------------------------------------------------------------------------
# 3. Derivatives via get_boost_differentiable
# ---------------------------------------------------------------------------
jacobians = {}

if 'jax' in emulators:
    try:
        import jax
        import jax.numpy as jnp
        emu_jax = emulators['jax']
        params_jnp = jnp.array([bcmdict[k] for k in param_names])
        J_jax = jax.jacfwd(emu_jax.get_boost_differentiable)(params_jnp, z=0.5, q2=Q2_VAL)
        jacobians['jax'] = np.array(J_jax)
        print("\nJAX Jacobian shape:", jacobians['jax'].shape)
    except Exception as e:
        print(f"\nJAX Jacobian failed: {e}")

if 'torch' in emulators:
    try:
        import torch
        emu_torch = emulators['torch']
        params_t = torch.tensor([bcmdict[k] for k in param_names],
                                 dtype=torch.float32, requires_grad=False)
        J_torch = torch.autograd.functional.jacobian(
            lambda p: emu_torch.get_boost_differentiable(p, z=0.5, q2=Q2_VAL),
            params_t,
        )
        jacobians['torch'] = J_torch.detach().numpy()
        print("Torch Jacobian shape:", jacobians['torch'].shape)
    except Exception as e:
        print(f"\nTorch Jacobian failed: {e}")

if 'jax' in jacobians and 'torch' in jacobians:
    max_diff = np.max(np.abs(jacobians['jax'] - jacobians['torch']))
    print(f"\nMax |J_jax - J_torch| = {max_diff:.2e}")

# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
rcParams['font.family'] = 'sans-serif'
rcParams['axes.labelsize'] = 13
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.4

colors = {'jax': 'C0', 'numpy': 'C1', 'torch': 'C2'}
linestyles = {'jax': '-', 'numpy': '--', 'torch': ':'}

# --- Figure 1: S(k) per backend ---
fig, axes = plt.subplots(1, len(Z_VALUES), figsize=(5 * len(Z_VALUES), 4.5), sharey=True)
if len(Z_VALUES) == 1:
    axes = [axes]

for ax, z in zip(axes, Z_VALUES):
    ax.set_title(f'$z = {z}$')
    for backend, results in boost_results.items():
        ax.semilogx(k_grid, results[z],
                    color=colors[backend],
                    ls=linestyles[backend],
                    lw=2.5,
                    label=backend)
    ax.set_xlim(0.09, 12)
    ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    ax.grid(True, which='both', alpha=0.25)

axes[0].set_ylabel(r'$\mathcal{S}(k) = P_\mathrm{hydro}/P_\mathrm{DMO}$')
axes[0].legend(title='backend', framealpha=0.9)
plt.suptitle('BCemu2025: backend comparison — $\\mathcal{S}(k)$')
plt.tight_layout(rect=[0, 0, 1, 0.93])
# plt.savefig('BCemu2025_backend_boost.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 2: Residuals relative to first available backend ---
if len(backend_list) > 1:
    ref = backend_list[0]
    others = backend_list[1:]
    fig2, axes2 = plt.subplots(1, len(Z_VALUES), figsize=(5 * len(Z_VALUES), 4.0), sharey=True)
    if len(Z_VALUES) == 1:
        axes2 = [axes2]

    for ax, z in zip(axes2, Z_VALUES):
        ax.set_title(f'$z = {z}$')
        for backend in others:
            residual = boost_results[backend][z] - boost_results[ref][z]
            ax.semilogx(k_grid, residual,
                        color=colors[backend],
                        ls=linestyles[backend],
                        lw=2,
                        label=f'{backend} − {ref}')
        ax.axhline(0, color='k', lw=1, ls='--', alpha=0.5)
        ax.set_xlim(0.09, 12)
        ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
        ax.grid(True, which='both', alpha=0.25)

    axes2[0].set_ylabel(r'$\Delta\mathcal{S}(k)$')
    axes2[0].legend(framealpha=0.9)
    plt.suptitle(f'BCemu2025: residuals vs {ref} backend')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    # plt.savefig('BCemu2025_backend_residuals.png', dpi=150, bbox_inches='tight')
    plt.show()

# --- Figure 3: Jacobians (if at least one differentiable backend ran) ---
if jacobians:
    n_params = len(param_names)
    fig3, axes3 = plt.subplots(2, (n_params + 1) // 2,
                                figsize=(4 * ((n_params + 1) // 2), 7))
    axes3 = axes3.flatten()

    for i, name in enumerate(param_names):
        ax = axes3[i]
        for backend, J in jacobians.items():
            ax.semilogx(k_grid, J[:, i],
                        color=colors[backend],
                        ls=linestyles[backend],
                        lw=2,
                        label=backend)
        ax.axhline(0, color='k', lw=1, ls='--', alpha=0.5)
        ax.set_title(rf'd$\mathcal{{S}}$/d({name})')
        ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
        ax.grid(True, which='both', alpha=0.25)
        ax.set_xlim(0.09, 12)

    # Hide unused subplots
    for ax in axes3[n_params:]:
        ax.set_visible(False)

    axes3[0].legend(title='backend', framealpha=0.9)
    plt.suptitle(r'BCemu2025: Jacobian $\partial\mathcal{S}/\partial\theta$ at $z=0.5$',
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    # plt.savefig('BCemu2025_backend_jacobians.png', dpi=150, bbox_inches='tight')
    plt.show()

# --- Figure 4: Timing bar chart ---
fig4, ax4 = plt.subplots(figsize=(5, 3.5))
bars = ax4.bar(list(timing.keys()),
               [v * 1e3 for v in timing.values()],
               color=[colors[b] for b in timing],
               width=0.5,
               edgecolor='k',
               linewidth=0.8)
ax4.bar_label(bars, fmt='%.1f ms', padding=3)
ax4.set_ylabel('Mean time per call (ms)')
ax4.set_title('BCemu2025: inference time by backend')
ax4.set_ylim(0, max(timing.values()) * 1.4 * 1e3)
ax4.grid(axis='y', alpha=0.3)
plt.tight_layout()
# plt.savefig('BCemu2025_backend_timing.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nDone.")
