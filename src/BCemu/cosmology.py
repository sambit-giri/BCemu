"""
BCemu cosmology module — generic, backend-agnostic background-cosmology
functions: distances, sound horizon fitting formulas, linear growth, sigma8.

All functions assume a **flat** universe. Dark energy follows the CPL
parametrization ``w(z) = w0 + wa * z/(1+z)``; the default ``w0=-1, wa=0``
reproduces exact flat LCDM, so nothing changes for callers that don't pass
w0/wa.

Every function takes a ``backend`` argument, ``'numpy'`` (default), ``'jax'``
or ``'torch'``, selecting which array library's primitives (exp/log/sqrt/
trapz/...) are used internally. This is what makes the module usable inside a
JAX-differentiable (or torch-autograd) inference pipeline: pass jax/torch
tracer values in for the cosmological parameters and get gradients out.

Install optional backends with:
    pip install jax jaxlib      # backend='jax'
    pip install torch           # backend='torch'
"""
import numpy as np

from . import backend as bk

_SCIPY_AVAILABLE = False
try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    _SCIPY_AVAILABLE = True
except ImportError:
    pass

C_KM_S = 299792.458          # speed of light [km/s]
MNU_EV_DEFAULT = 0.06        # CAMB/CosmoMC default: one massive neutrino eigenstate [eV]
OMEGA_NU_H2_DEFAULT = MNU_EV_DEFAULT / 93.14   # standard relation, Omega_nu * h^2


def _warn_camb_ignores_backend(backend, func_name):
    """method='camb' branches always run on real CAMB (numpy-only, not
    differentiable); let the caller know their backend choice is a no-op
    here rather than silently downgrading it."""
    if backend != 'numpy':
        print(
            f"{func_name}(method='camb'): CAMB is always numpy-only and non-differentiable; "
            f"backend={backend!r} is ignored for this method."
        )


# ---------------------------------------------------------------------------
# Background expansion (flat universe, CPL dark energy)
# ---------------------------------------------------------------------------

def dark_energy_density(z, w0=-1.0, wa=0.0, backend='numpy'):
    """
    CPL dark-energy density relative to today, rho_DE(z)/rho_DE(0).

    ``w(z) = w0 + wa * z/(1+z)``. Default ``w0=-1, wa=0`` gives exactly 1
    (cosmological constant).
    """
    bk.check_backend(backend)
    return (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * bk.exp(-3.0 * wa * z / (1.0 + z), backend)


def Ez(z, Om, w0=-1.0, wa=0.0, backend='numpy'):
    """
    Dimensionless Hubble parameter E(z) = H(z)/H0 for a flat universe with
    matter density Om and CPL dark energy (default w0=-1, wa=0 -> flat LCDM).

    Parameters
    ----------
    z : redshift (scalar or array)
    Om : matter density parameter at z=0
    w0, wa : CPL dark-energy equation-of-state parameters
    backend : {'numpy', 'jax', 'torch'}
    """
    bk.check_backend(backend)
    fDE = dark_energy_density(z, w0=w0, wa=wa, backend=backend)
    E2 = Om * (1.0 + z) ** 3 + (1.0 - Om) * fDE
    return bk.sqrt(E2, backend)


def omega_m_z(z, Om, w0=-1.0, wa=0.0, backend='numpy'):
    """Matter density parameter at redshift z, Omega_m(z), for a flat universe."""
    bk.check_backend(backend)
    return Om * (1.0 + z) ** 3 / Ez(z, Om, w0=w0, wa=wa, backend=backend) ** 2


def omega_de_z(z, Om, w0=-1.0, wa=0.0, backend='numpy'):
    """Dark-energy density parameter at redshift z (flat universe: 1 - Omega_m(z))."""
    bk.check_backend(backend)
    return 1.0 - omega_m_z(z, Om, w0=w0, wa=wa, backend=backend)


def comoving_distance(z, H0, Om, w0=-1.0, wa=0.0, backend='numpy', n_grid=4000):
    """
    Flat-universe line-of-sight comoving distance D_C(z) [Mpc], CPL dark
    energy (default w0=-1, wa=0 -> flat LCDM). Radiation is neglected
    (negligible at the BAO/BAO-like redshifts this module targets).

    z must be a scalar (or 0-d array/tensor) redshift.
    """
    bk.check_backend(backend)
    grid = bk.linspace(0.0, 1.0, n_grid, backend)
    zz = grid * z
    integrand = C_KM_S / (H0 * Ez(zz, Om, w0=w0, wa=wa, backend=backend))
    return bk.trapz(integrand, zz, backend)


def luminosity_distance(z, H0, Om, w0=-1.0, wa=0.0, backend='numpy', n_grid=4000):
    """Flat-universe luminosity distance D_L(z) = (1+z) * D_C(z) [Mpc]."""
    bk.check_backend(backend)
    DC = comoving_distance(z, H0, Om, w0=w0, wa=wa, backend=backend, n_grid=n_grid)
    return (1.0 + z) * DC


def distance_modulus(z, H0, Om, w0=-1.0, wa=0.0, backend='numpy', n_grid=4000):
    """
    Distance modulus mu(z) = 5*log10(D_L / 10 pc), for SNe Ia-style
    likelihoods. D_L is computed in Mpc and converted internally.
    """
    bk.check_backend(backend)
    DL_mpc = luminosity_distance(z, H0, Om, w0=w0, wa=wa, backend=backend, n_grid=n_grid)
    return 5.0 * bk.log10(DL_mpc * 1.0e5, backend)   # D_L[Mpc]*1e6 pc / 10 pc = D_L[Mpc]*1e5


# ---------------------------------------------------------------------------
# Omega_m bookkeeping (incl. fixed massive-neutrino contribution)
# ---------------------------------------------------------------------------

def omega_m_total(ombh2, omch2, H0, omega_nu_h2=OMEGA_NU_H2_DEFAULT):
    """
    Omega_m at z=0, including a fixed massive-neutrino contribution
    (Omega_nu*h^2 = omega_nu_h2, default corresponds to CAMB/CosmoMC's fixed
    Sigma m_nu = 0.06 eV, one massive eigenstate). Standard CAMB/CosmoMC
    convention -- a bare (ombh2+omch2)/h^2 undercounts Omega_m slightly.

    Pure arithmetic: works with plain floats or with any backend's native
    array/tensor type (no explicit backend selection needed).
    """
    return (ombh2 + omch2 + omega_nu_h2) / (H0 / 100.0) ** 2


# ---------------------------------------------------------------------------
# Sound horizon fitting formulas
# ---------------------------------------------------------------------------

def _eh98_sound_horizon(z_at, ombh2, omch2, backend):
    """
    Eisenstein & Hu (1998) sound-horizon formalism, evaluated at an arbitrary
    redshift z_at (the drag redshift for r_drag, or the photon-decoupling
    redshift for r_star).
    """
    om0h2 = ombh2 + omch2
    theta27 = 2.7255 / 2.7
    zeq = 2.5e4 * om0h2 / theta27 ** 4
    keq = 7.46e-2 * om0h2 / theta27 ** 2

    def Rfunc(z):
        return 31.5 * ombh2 / theta27 ** 4 * (z / 1.0e3) ** -1

    Req, Rat = Rfunc(zeq), Rfunc(z_at)
    return (2.0 / (3.0 * keq)) * bk.sqrt(6.0 / Req, backend) * bk.log(
        (bk.sqrt(1.0 + Rat, backend) + bk.sqrt(Rat + Req, backend)) / (1.0 + bk.sqrt(Req, backend)),
        backend
    )


def rdrag_fitting(ombh2, omch2, method='aubourg2015', omega_nu_h2=OMEGA_NU_H2_DEFAULT,
                   backend='numpy'):
    """
    Sound horizon at the baryon drag epoch, r_drag [Mpc], via a choice of
    fast analytic fitting formulas (all just algebra -- no Boltzmann code
    involved, fully differentiable). Accuracy figures are the papers' own
    claims relative to CAMB/CLASS.

    'eh98'
        Eisenstein & Hu (1998). ~2-4% accurate.
    'aubourg2015'
        Aubourg et al. (2015, arXiv:1411.1074) Eq. 16. ~0.021% accurate.
        Default.
    'aizpuru2021_narrow'
        Aizpuru, Arjona & Nesseris (2021, arXiv:2106.00428) Eq. 7.
        ~0.003% accurate, but only valid within ~10-sigma of Planck
        (Omega_m h^2 in [0.13, 0.15], Omega_b h^2 in [0.0214, 0.0234]).
    'aizpuru2021_broad'
        Same paper, Eq. 8. ~0.018% accurate, broader validity range
        (Omega_m h^2 in [0.05, 0.25], Omega_b h^2 in [0.016, 0.03]).
    'aizpuru2021_neutrino'
        Same paper, Eq. 10. ~0.0076% accurate; the only one of these that
        explicitly includes the massive-neutrino dependence.

    This is an early-universe quantity: it does not depend on the late-time
    dark-energy equation of state (no w0/wa dependence).
    """
    bk.check_backend(backend)
    om0h2 = ombh2 + omch2  # Omega_m h^2 (baryons+CDM; neutrino folded in separately below)

    if method == 'eh98':
        theta27 = 2.7255 / 2.7
        om0h2_ = om0h2
        b1 = 0.313 * om0h2_ ** -0.419 * (1.0 + 0.607 * om0h2_ ** 0.674)
        b2 = 0.238 * om0h2_ ** 0.223
        zdrag = 1291.0 * om0h2_ ** 0.251 / (1.0 + 0.659 * om0h2_ ** 0.828) * (1.0 + b1 * ombh2 ** b2)
        return _eh98_sound_horizon(zdrag, ombh2, omch2, backend)

    elif method == 'aubourg2015':
        return 55.154 * bk.exp(-72.3 * (omega_nu_h2 + 0.0006) ** 2, backend) / (
            om0h2 ** 0.25351 * ombh2 ** 0.12807)

    elif method == 'aizpuru2021_narrow':
        a1, a2, a3, a4, a5, a6, a7 = 0.00785436, 0.177084, 0.00912388, 0.618711, 11.9611, 2.81343, 0.784719
        return 1.0 / (a1 * ombh2 ** a2 + a3 * om0h2 ** a4 + a5 * ombh2 ** a6 * om0h2 ** a7)

    elif method == 'aizpuru2021_broad':
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = (0.00257366, 0.05032, 0.013, 0.7720642, 0.24346362,
                                               0.00641072, 0.5350899, 32.7525, 0.315473)
        return 1.0 / (a1 * ombh2 ** a2 + a3 * ombh2 ** a4 * om0h2 ** a5 + a6 * om0h2 ** a7) \
            - a8 / om0h2 ** a9

    elif method == 'aizpuru2021_neutrino':
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = (0.0034917, -19.972694, 0.000336186, 0.0000305, 0.22752,
                                               0.00003142567, 0.5453798, 374.14994, 4.022356899)
        return (a1 * bk.exp(a2 * (a3 + omega_nu_h2) ** 2, backend)) / (
            a4 * ombh2 ** a5 + a6 * om0h2 ** a7 + a8 * (ombh2 * om0h2) ** a9)

    else:
        raise ValueError(f"unknown r_drag fitting method: {method!r}")


_ZSTAR_METHODS = ('aizpuru2021', 'hu_sugiyama1996')


def photon_decoupling_redshift(ombh2, omch2, method='aizpuru2021'):
    """
    Redshift of photon decoupling z_star. Pure algebra -- differentiable
    under any backend without dispatch (only uses **, +, *, /).

    'aizpuru2021' (default)
        Aizpuru, Arjona & Nesseris (2021, arXiv:2106.00428) Appendix A.2,
        Eq. 16 -- the same paper as ``rdrag_fitting``'s aizpuru2021_* fits,
        but this z_star fit is only in the appendix, not the abstract.
        ~0.0005% accurate vs. CLASS+HyRec. Valid within ~10-sigma of Planck
        (Omega_m h^2 in [0.13, 0.15], Omega_b h^2 in [0.0214, 0.0234]) -- the
        same validity range as ``rdrag_fitting(method='aizpuru2021_narrow')``.
    'hu_sugiyama1996'
        Hu & Sugiyama (1996). ~0.3% accurate (per Aizpuru et al. 2021's own
        check against CLASS+HyRec) but no restricted validity range -- use
        this outside the narrow window above.
    """
    if method not in _ZSTAR_METHODS:
        raise ValueError(f"method must be one of {_ZSTAR_METHODS}, got {method!r}")

    om0h2 = ombh2 + omch2

    if method == 'aizpuru2021':
        return (391.672 * om0h2 ** -0.372296 + 937.422 * ombh2 ** -0.97966) / (
            om0h2 ** -0.0192951 * ombh2 ** -0.93681) + om0h2 ** -0.731631

    else:  # 'hu_sugiyama1996'
        g1 = 0.0783 * ombh2 ** -0.238 / (1.0 + 39.5 * ombh2 ** 0.763)
        g2 = 0.560 / (1.0 + 21.1 * ombh2 ** 1.81)
        return 1048.0 * (1.0 + 0.00124 * ombh2 ** -0.738) * (1.0 + g1 * om0h2 ** g2)


def r_star(ombh2, omch2, zstar_method='aizpuru2021', backend='numpy'):
    """
    Sound horizon at photon decoupling, r_star [Mpc] (as opposed to r_drag,
    the sound horizon at the baryon *drag* epoch used for BAO). Uses the
    Eisenstein & Hu (1998) sound-horizon formalism evaluated at z_star (see
    ``photon_decoupling_redshift`` for the ``zstar_method`` choices/accuracy).

    No closed-form fit for r_star itself (as opposed to z_star) is known in
    the literature -- CAMB/CLASS always compute it numerically. This EH98
    sound-horizon formalism, combined with the accurate default z_star fit,
    is the best available closed-form approximation; treat it as good to
    roughly the same accuracy class as ``rdrag_fitting(method='eh98')``
    (~2-4%), since the EH98 integral's own radiation-content approximations
    (not just the z_star endpoint) also contribute error.

    Early-universe quantity: no w0/wa dependence.
    """
    bk.check_backend(backend)
    z_star = photon_decoupling_redshift(ombh2, omch2, method=zstar_method)
    return _eh98_sound_horizon(z_star, ombh2, omch2, backend)


def theta_star(ombh2, omch2, H0, Om, w0=-1.0, wa=0.0, zstar_method='aizpuru2021',
                backend='numpy', n_grid=4000):
    """
    Acoustic angular scale at decoupling, theta_star = r_star / D_M(z_star)
    (flat universe: D_M = D_C). This is the quantity CosmoMC's 100*theta_MC
    approximates -- use this to sample the real theta_MC parameterization
    instead of substituting H0 directly.
    """
    bk.check_backend(backend)
    z_star = photon_decoupling_redshift(ombh2, omch2, method=zstar_method)
    rs = r_star(ombh2, omch2, zstar_method=zstar_method, backend=backend)
    DM = comoving_distance(z_star, H0, Om, w0=w0, wa=wa, backend=backend, n_grid=n_grid)
    return rs / DM


# ---------------------------------------------------------------------------
# BAO distance/sound-horizon bundle
# ---------------------------------------------------------------------------

def bao_kinematics(z, H0, ombh2, omch2, w0=-1.0, wa=0.0, rdrag_method='aubourg2015',
                    omega_nu_h2=OMEGA_NU_H2_DEFAULT, backend='numpy', n_grid=4000):
    """
    Background quantities needed by BAO likelihoods at a tracer redshift z:
    angular diameter distance D_A(z) [Mpc], Hubble rate H(z) [km/s/Mpc], and
    the sound horizon at the drag epoch r_drag [Mpc].

    Returns
    -------
    DA, H, rd
    """
    bk.check_backend(backend)
    Om = omega_m_total(ombh2, omch2, H0, omega_nu_h2=omega_nu_h2)
    DC = comoving_distance(z, H0, Om, w0=w0, wa=wa, backend=backend, n_grid=n_grid)
    DA = DC / (1.0 + z)
    H = H0 * Ez(z, Om, w0=w0, wa=wa, backend=backend)
    rd = rdrag_fitting(ombh2, omch2, method=rdrag_method, omega_nu_h2=omega_nu_h2, backend=backend)
    return DA, H, rd


# ---------------------------------------------------------------------------
# Linear growth: D(z), f(z) = dlnD/dlna
# ---------------------------------------------------------------------------

_GROWTH_METHODS = ('linder2005', 'ode_solve')


def _growth_gamma(w0, wa):
    """Linder & Cahn (2007) generalization of the growth index gamma to CPL
    dark energy; reduces to Linder (2005)'s gamma=0.55 at w0=-1, wa=0."""
    w_eff = w0 + 0.5 * wa
    return 0.55 + 0.05 * (1.0 + w_eff)


def _growth_factor_linder2005(z, Om, w0, wa, backend, n_grid):
    """
    D(z)/D(0) = exp(-integral_0^z f(z')/(1+z') dz'), f(z) = Omega_m(z)^gamma.
    A quadrature over a closed-form fit -- same cost/differentiability class
    as comoving_distance, no ODE solve involved.
    """
    gamma = _growth_gamma(w0, wa)
    grid = bk.linspace(0.0, 1.0, n_grid, backend)
    zz = grid * z
    fz = omega_m_z(zz, Om, w0=w0, wa=wa, backend=backend) ** gamma
    integral = bk.trapz(fz / (1.0 + zz), zz, backend)
    return bk.exp(-integral, backend)


def _growth_rate_linder2005(z, Om, w0, wa, backend):
    gamma = _growth_gamma(w0, wa)
    return omega_m_z(z, Om, w0=w0, wa=wa, backend=backend) ** gamma


def _growth_ode_rhs_factory(Om, w0, wa, backend):
    """
    d(y1,y2)/dlna, y1=D, y2=dD/dlna, for the linear growth ODE
        D'' + (2 + dlnH/dlna) D' - 1.5 Omega_m(a) D = 0
    written as a first-order system in ln(a).
    """
    def rhs(lna, y1, y2):
        a = bk.exp(lna, backend)
        z = 1.0 / a - 1.0
        Omz = omega_m_z(z, Om, w0=w0, wa=wa, backend=backend)
        # dlnH/dlna = -1.5*Omega_m(a) + 1.5*(1+w(a))*Omega_DE(a) ... use the
        # equivalent, numerically simpler form via d(lnE)/dlna:
        Ode_z = 1.0 - Omz
        w_a = w0 + wa * z / (1.0 + z)
        dlnH_dlna = -1.5 * Omz - 1.5 * (1.0 + w_a) * Ode_z
        dy1 = y2
        dy2 = -(2.0 + dlnH_dlna) * y2 + 1.5 * Omz * y1
        return dy1, dy2
    return rhs


def _growth_ode_solve_numpy(a_target, Om, w0, wa, a_init, n_steps):
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for growth_factor(method='ode_solve', backend='numpy').\n"
            "Install with: pip install scipy"
        )
    rhs = _growth_ode_rhs_factory(Om, w0, wa, 'numpy')

    def fun(lna, y):
        dy1, dy2 = rhs(lna, y[0], y[1])
        return [dy1, dy2]

    lna_init = np.log(a_init)
    lna_target = np.log(a_target)
    sol = solve_ivp(fun, [lna_init, lna_target], [a_init, a_init],
                     method='RK45', rtol=1e-8, atol=1e-10, dense_output=False)
    return float(sol.y[0, -1]), float(sol.y[1, -1])


def _growth_ode_solve_fixed_step(a_target, Om, w0, wa, a_init, n_steps, backend):
    """Fixed-step RK4 in ln(a), differentiable under jax/torch autodiff."""
    rhs = _growth_ode_rhs_factory(Om, w0, wa, backend)
    lna_init = np.log(a_init)
    lna_target = bk.log(a_target, backend)   # backend is 'jax' or 'torch' here
    h = (lna_target - lna_init) / n_steps

    y1, y2 = a_init, a_init
    lna = lna_init
    for _ in range(n_steps):
        k1_1, k1_2 = rhs(lna, y1, y2)
        k2_1, k2_2 = rhs(lna + 0.5 * h, y1 + 0.5 * h * k1_1, y2 + 0.5 * h * k1_2)
        k3_1, k3_2 = rhs(lna + 0.5 * h, y1 + 0.5 * h * k2_1, y2 + 0.5 * h * k2_2)
        k4_1, k4_2 = rhs(lna + h, y1 + h * k3_1, y2 + h * k3_2)
        y1 = y1 + (h / 6.0) * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
        y2 = y2 + (h / 6.0) * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)
        lna = lna + h
    return y1, y2


def _growth_ode_solve(a_target, Om, w0, wa, backend, a_init=1.0e-3, n_steps=200):
    if backend == 'numpy':
        return _growth_ode_solve_numpy(a_target, Om, w0, wa, a_init, n_steps)
    return _growth_ode_solve_fixed_step(a_target, Om, w0, wa, a_init, n_steps, backend)


def growth_factor(z, Om, w0=-1.0, wa=0.0, method='linder2005', backend='numpy', n_grid=4000):
    """
    Linear growth factor D(z), normalized to D(z=0) = 1.

    method='linder2005' (default)
        Closed-form growth-index fit, f(z) = Omega_m(z)^gamma with gamma
        generalized for CPL dark energy (Linder & Cahn 2007), integrated by
        quadrature. Fast, fully differentiable under every backend the same
        way the rest of this module is.
    method='ode_solve'
        Numerically integrates the exact linear growth ODE. More accurate
        than the fit, especially away from w0=-1. Differentiability depends
        on backend: 'numpy' uses scipy.integrate.solve_ivp (not
        differentiable); 'jax'/'torch' use a fixed-step RK4 integrator so
        gradients flow through normally.
    """
    bk.check_backend(backend)
    if method not in _GROWTH_METHODS:
        raise ValueError(f"method must be one of {_GROWTH_METHODS}, got {method!r}")

    if method == 'linder2005':
        return _growth_factor_linder2005(z, Om, w0, wa, backend, n_grid)

    a_target = 1.0 / (1.0 + z)
    D_target, _ = _growth_ode_solve(a_target, Om, w0, wa, backend)
    D_today, _ = _growth_ode_solve(1.0, Om, w0, wa, backend)
    return D_target / D_today


def growth_rate(z, Om, w0=-1.0, wa=0.0, method='linder2005', backend='numpy', n_grid=4000):
    """
    Linear growth rate f(z) = dlnD/dlna.

    method options and differentiability notes: see growth_factor.
    """
    bk.check_backend(backend)
    if method not in _GROWTH_METHODS:
        raise ValueError(f"method must be one of {_GROWTH_METHODS}, got {method!r}")

    if method == 'linder2005':
        return _growth_rate_linder2005(z, Om, w0, wa, backend)

    a_target = 1.0 / (1.0 + z)
    D_target, dD_dlna = _growth_ode_solve(a_target, Om, w0, wa, backend)
    return dD_dlna / D_target


def sigma8_z(sigma8_0, z, Om, w0=-1.0, wa=0.0, method='linder2005', backend='numpy', n_grid=4000):
    """sigma8 at redshift z: sigma8(z) = sigma8_0 * D(z)/D(0)."""
    bk.check_backend(backend)
    return sigma8_0 * growth_factor(z, Om, w0=w0, wa=wa, method=method, backend=backend, n_grid=n_grid)


# ---------------------------------------------------------------------------
# sigma8 from the primordial amplitude As
# ---------------------------------------------------------------------------

_BARTLETT2024_COEFFS = (0.51172, 0.04593, 0.73983, 1.56738, 1.16846, 0.59348, 0.19994, 25.09218, 9.36909, 0.00011)
_SYREN_NEW_COEFFS = (0.0187, 2.4891, 12.9495, 0.7527, 2.3685, 1.5062, 1.3057, 0.0885,
                     0.1471, 3.4982, 0.006, 19.2779, 11.1463, 1.5433, 7.0578, 2.0564)
_SIGMA8_METHODS = ('bartlett2024', 'syren_new', 'sui2025', 'camb')


def _bartlett2024_f(Om, Ob, h, ns, backend):
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = _BARTLETT2024_COEFFS
    return (a0 * Om + a1 * h + a2 * (Om - a3 * Ob) * (bk.log(a4 * Om, backend) - a5 * ns)
            * (ns + a6 * h * (a7 * Ob - a8 * ns + bk.log(a9 * h, backend))))


def _syren_new_f(Om, Ob, h, ns, mnu, w0, wa, backend):
    c = _SYREN_NEW_COEFFS
    term1 = c[0] * (-Ob * c[1] + Om * c[2]
                    + bk.log(-c[3] * w0 + bk.log(-c[4] * w0 - c[5] * wa, backend), backend))
    term2 = Om * c[6] + c[7] * mnu + c[8] * ns - bk.log(Om * c[9] - c[10] * wa, backend)
    term3 = Ob * c[11] - Om * c[12] - ns
    term4 = -Om * c[13] - c[14] * h + c[15] * mnu + ns
    return term1 * term2 * term3 * term4


def As_to_sigma8(ombh2, omch2, H0, ns, ln1e10As, tau=0.0544, mnu=MNU_EV_DEFAULT, w0=-1.0, wa=0.0,
                  method='bartlett2024', backend='numpy'):
    """
    sigma8(z=0) from the primordial amplitude As (as ln(10^10 As)) and shape
    parameters.

    method='bartlett2024' (default)
        Bartlett et al. (2024, arXiv:2311.15865, "A precise symbolic emulator
        of the linear matter power spectrum") closed-form symbolic-regression
        fit -- fast, differentiable under any backend. Flat LCDM only (`mnu`,
        `w0`, `wa` are ignored); claimed accuracy is 0.1% RMS vs CAMB (max
        error ~0.25%), checked in practice closer to ~1-1.5% at individual
        points -- expected point-wise variance around an RMS-averaged claim,
        verified to match the authors' reference implementation exactly
        (github.com/DeaglanBartlett/symbolic_pofk).
    method='syren_new' (alias: 'sui2025')
        Sui, Bartlett et al. (2025, arXiv:2410.14623, "syren-new: Precise
        formulae for the linear and nonlinear matter power spectra with
        massive neutrinos and dynamical dark energy") -- generalizes the fit
        to include `mnu` [eV] and CPL `w0`/`wa`. Differentiable under any
        backend. On pure flat-LCDM inputs (default mnu=0.06, w0=-1, wa=0)
        this is slightly *less* accurate than 'bartlett2024' (~0.5% vs ~0.25%
        max error, per the paper's own comparison) -- use 'bartlett2024'
        unless you actually need the mnu/w0/wa dependence. Valid for
        mnu in [0, 0.15] eV, w0 in [-1.3, -0.7], wa in [-0.7, 0.5], over the
        same Omega_m/Omega_b/h/ns/As box as 'bartlett2024'.
    method='camb'
        Real CAMB linear P(k) call at z=0 -- slow (~0.2s/call), exact, but
        always numpy-only and non-differentiable regardless of `backend`.
    """
    bk.check_backend(backend)
    if method not in _SIGMA8_METHODS:
        raise ValueError(f"method must be one of {_SIGMA8_METHODS}, got {method!r}")

    if method == 'bartlett2024':
        h = H0 / 100.0
        Om = omega_m_total(ombh2, omch2, H0)
        Ob = ombh2 / h ** 2
        As_bartlett = bk.exp(ln1e10As, backend) / 10.0   # their convention: As = 10^9 * (true primordial amplitude)
        f = _bartlett2024_f(Om, Ob, h, ns, backend)
        return f * bk.sqrt(As_bartlett, backend)

    elif method in ('syren_new', 'sui2025'):
        h = H0 / 100.0
        Om = omega_m_total(ombh2, omch2, H0, omega_nu_h2=mnu / 93.14)
        Ob = ombh2 / h ** 2
        As_syren = bk.exp(ln1e10As, backend) / 10.0   # same convention: 10^9 * (true primordial amplitude)
        f = _syren_new_f(Om, Ob, h, ns, mnu, w0, wa, backend)
        return f * bk.sqrt(As_syren, backend)

    else:  # 'camb'
        _warn_camb_ignores_backend(backend, 'As_to_sigma8')
        import camb
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=float(H0), ombh2=float(ombh2), omch2=float(omch2), tau=float(tau))
        pars.InitPower.set_params(As=float(np.exp(ln1e10As)) * 1e-10, ns=float(ns))
        pars.set_matter_power(redshifts=[0.0], kmax=2.0)
        pars.NonLinear = camb.model.NonLinear_none
        res = camb.get_results(pars)
        return res.get_sigma8()[-1]


def sigma8_to_As(ombh2, omch2, H0, ns, sigma8, tau=0.0544, mnu=MNU_EV_DEFAULT, w0=-1.0, wa=0.0,
                  method='bartlett2024', backend='numpy', ln1e10As_bracket=(0.5, 5.0)):
    """
    Inverse of As_to_sigma8: ln(10^10 As) that reproduces a target sigma8(z=0),
    given shape parameters. Method options, `mnu`/`w0`/`wa` handling, and
    accuracy caveats: see As_to_sigma8.

    method='bartlett2024'/'syren_new'/'sui2025'
        Analytic inversion of the corresponding closed-form fit -- exact
        given the fit, fully differentiable under any backend.
    method='camb'
        Root-finds (scipy.optimize.brentq) the ln(10^10 As) that makes a real
        CAMB call reproduce sigma8, since CAMB has no closed-form sigma8->As
        map. Always numpy-only and non-differentiable regardless of
        `backend`. `ln1e10As_bracket` sets the search interval passed to
        brentq -- widen it if the solve fails to bracket a root.
    """
    bk.check_backend(backend)
    if method not in _SIGMA8_METHODS:
        raise ValueError(f"method must be one of {_SIGMA8_METHODS}, got {method!r}")

    if method == 'bartlett2024':
        h = H0 / 100.0
        Om = omega_m_total(ombh2, omch2, H0)
        Ob = ombh2 / h ** 2
        f = _bartlett2024_f(Om, Ob, h, ns, backend)
        As_bartlett = (sigma8 / f) ** 2   # invert sigma8 = f * sqrt(As_bartlett)
        return bk.log(10.0 * As_bartlett, backend)

    elif method in ('syren_new', 'sui2025'):
        h = H0 / 100.0
        Om = omega_m_total(ombh2, omch2, H0, omega_nu_h2=mnu / 93.14)
        Ob = ombh2 / h ** 2
        f = _syren_new_f(Om, Ob, h, ns, mnu, w0, wa, backend)
        As_syren = (sigma8 / f) ** 2
        return bk.log(10.0 * As_syren, backend)

    else:  # 'camb'
        _warn_camb_ignores_backend(backend, 'sigma8_to_As')
        if not _SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for sigma8_to_As(method='camb').\n"
                "Install with: pip install scipy"
            )

        def resid(ln1e10As):
            return As_to_sigma8(ombh2, omch2, H0, ns, ln1e10As, tau=tau, method='camb') - sigma8

        return brentq(resid, ln1e10As_bracket[0], ln1e10As_bracket[1], xtol=1e-6)
