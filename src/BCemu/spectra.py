"""
BCemu spectra module — full P(k,z) and CMB C_ell via CAMB + baryonic feedback.

Install the optional CAMB dependency with:
    pip install camb
or
    pip install BCemu[spectra]
"""

try:
    import camb
    import camb.model as camb_model
    HAS_CAMB = True
except ImportError:
    HAS_CAMB = False

import numpy as np
from scipy.interpolate import RectBivariateSpline

try:
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz


class BaryonicCAMB:
    """
    CMB + matter power spectrum simulator backed by the CAMB Boltzmann solver.

    Subclass this and override ``S`` to inject any baryonic feedback model.
    The modified P_dmb = P_NL_dmo * S drives both the CMB lensing convergence
    spectrum C_L^kk (Limber integral) and all lensed primary CMB spectra
    (TT, EE, BB, TE) via CAMB's get_lensed_cls_with_spectrum.

    All power spectra use physical Mpc units (k in Mpc^-1, P in Mpc^3).

    Parameters
    ----------
    cosmo_params : dict, optional
        Cosmological parameters (H0, ombh2, omch2, tau, ns, As).
        Defaults to Planck 2018 TT+TE+EE+lowE+lensing.
    lmax : int
        Maximum multipole for CMB spectra.
    kmax : float
        Maximum wavenumber [Mpc^-1] for matter power spectrum grids.
    """

    PLANCK2018 = dict(
        H0=67.36, ombh2=0.02237, omch2=0.1200,
        tau=0.0544, ns=0.9649, As=np.exp(3.044) * 1e-10
    )
    C_LIGHT = 2.99792458e5  # km/s
    _VALID_K_EXTRAP  = (None, 1, 'fixed', 'same', 'extrapolate')
    _VALID_NL_MODEL  = ('mead2020', 'takahashi', 'halofit')

    def __init__(self, cosmo_params=None, lmax=3000, kmax=50.0,
                 k_extrap=None, nl_model='mead2020'):
        if not HAS_CAMB:
            raise ImportError(
                "camb is required for the spectra module. "
                "Install with: pip install camb  or  pip install BCemu[spectra]"
            )
        if k_extrap not in self._VALID_K_EXTRAP:
            raise ValueError(
                f"k_extrap must be one of {self._VALID_K_EXTRAP}, got {k_extrap!r}"
            )
        if nl_model not in self._VALID_NL_MODEL:
            raise ValueError(
                f"nl_model must be one of {self._VALID_NL_MODEL}, got {nl_model!r}"
            )
        self.cosmo_params = cosmo_params if cosmo_params is not None \
            else self.PLANCK2018.copy()
        self.lmax    = lmax
        self.kmax    = kmax
        self.k_extrap = k_extrap
        self.nl_model = nl_model

        print("Setting up CAMB (background + CMB transfer functions)...")
        self._results_bg = self._run_camb_cmb()
        self._camb_lmax = self._results_bg.Params.max_l

        print("Building matter power spectrum grids...")
        self._setup_pk_grids()
        print("Ready.")

    # ------------------------------------------------------------------
    # Internal CAMB helpers
    # ------------------------------------------------------------------

    def _make_cosmo_pars(self):
        p = self.cosmo_params
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=p['H0'], ombh2=p['ombh2'], omch2=p['omch2'], tau=p['tau']
        )
        pars.InitPower.set_params(As=p['As'], ns=p['ns'])
        return pars

    def _run_camb_cmb(self):
        pars = self._make_cosmo_pars()
        pars.set_for_lmax(self.lmax, lens_potential_accuracy=1)
        return camb.get_results(pars)

    def _setup_pk_grids(self):
        z_nl  = np.concatenate([[0.0], np.geomspace(0.01, 5.0, 40)])
        z_lin = np.concatenate([
            [0.0], np.geomspace(0.01, 5.0, 40),
            np.geomspace(5.5, 1000.0, 25)
        ])

        pars = self._make_cosmo_pars()
        pars.set_matter_power(redshifts=list(z_lin), kmax=self.kmax)
        pars.NonLinear = camb_model.NonLinear_none
        res = camb.get_results(pars)
        k, self._z_lin, self._pk_lin = \
            res.get_linear_matter_power_spectrum(hubble_units=False, k_hunit=False)

        pars = self._make_cosmo_pars()
        pars.set_matter_power(redshifts=list(z_nl), kmax=self.kmax)
        pars.NonLinear = camb_model.NonLinear_both
        pars.NonLinearModel.set_params(self.nl_model)
        res = camb.get_results(pars)
        _, self._z_nl, self._pk_nl_dmo = \
            res.get_nonlinear_matter_power_spectrum(hubble_units=False, k_hunit=False)

        self._k_grid = k
        lk = np.log(self._k_grid)
        self._spline_pk_lin = RectBivariateSpline(
            self._z_lin, lk, np.log(self._pk_lin), kx=3, ky=3
        )
        self._spline_pk_nl = RectBivariateSpline(
            self._z_nl, lk, np.log(self._pk_nl_dmo), kx=3, ky=3
        )

    def _eval_pk_lin(self, k, z):
        lk = np.log(np.clip(k, self._k_grid[0], self._k_grid[-1]))
        zc = np.clip(z, self._z_lin[0], self._z_lin[-1])
        return np.exp(self._spline_pk_lin(zc, lk, grid=True))

    def _eval_pk_nl(self, k, z):
        lk = np.log(np.clip(k, self._k_grid[0], self._k_grid[-1]))
        zc = np.clip(z, self._z_nl[0], self._z_nl[-1])
        return np.exp(self._spline_pk_nl(zc, lk, grid=True))

    def _apply_k_extrap(self, spline, z_vals, k_arr, k_min, k_max):
        """
        Evaluate a 2-D spline S(z, log k) applying the k_extrap boundary policy.

        k_extrap=None / 1       : S=1 for k outside [k_min, k_max] (default)
        k_extrap='fixed'/'same' : edge value (clip k to boundary)
        k_extrap='extrapolate'  : physically motivated boundary extension:
          k > k_max, alpha > 0  — upturn already resolved in the emulator range:
                                  power-law continuation of the upturn.
          k > k_max, alpha <= 0 — S still decreasing at k_max, upturn not yet
                                  reached: return S=1 to avoid artificially
                                  over-suppressing the lensing signal.
          k < k_min             — power-law with the natural slope, which drives
                                  S toward 1 as k -> 0 (large-scale limit).
        """
        policy = self.k_extrap
        n_z, n_k = len(z_vals), len(k_arr)

        if policy == 'extrapolate':
            k_in  = (k_arr >= k_min) & (k_arr <= k_max)
            k_hi  = k_arr > k_max
            k_lo  = k_arr < k_min
            Sk = np.ones((n_z, n_k))

            if k_in.any():
                Sk[:, k_in] = spline(z_vals, np.log(k_arr[k_in]), grid=True)

            if k_hi.any():
                pts = np.log(np.array([k_max * 0.97, k_max]))
                S_e = np.clip(spline(z_vals, pts, grid=True), 1e-10, None)  # (n_z, 2)
                alpha = (np.log(S_e[:, 1]) - np.log(S_e[:, 0])) / (pts[1] - pts[0])
                S_bnd = S_e[:, 1]                                    # (n_z,)
                dlk   = np.log(k_arr[k_hi]) - np.log(k_max)         # (n_hi,)  > 0
                extrap = S_bnd[:, np.newaxis] * np.exp(
                    alpha[:, np.newaxis] * dlk[np.newaxis, :])
                # upturn (alpha > 0): continue power-law; still decreasing: S = 1
                Sk[:, k_hi] = np.where(
                    (alpha <= 0)[:, np.newaxis], 1.0, np.clip(extrap, 0.0, None))

            if k_lo.any():
                pts = np.log(np.array([k_min, k_min * 1.03]))
                S_e = np.clip(spline(z_vals, pts, grid=True), 1e-10, None)
                alpha = (np.log(S_e[:, 1]) - np.log(S_e[:, 0])) / (pts[1] - pts[0])
                S_bnd = S_e[:, 0]
                dlk   = np.log(k_arr[k_lo]) - np.log(k_min)         # (n_lo,)  < 0
                Sk[:, k_lo] = np.clip(
                    S_bnd[:, np.newaxis] * np.exp(
                        alpha[:, np.newaxis] * dlk[np.newaxis, :]),
                    0.0, None)
            return Sk

        elif policy in ('fixed', 'same'):
            k_c = np.clip(k_arr, k_min, k_max)
            Sk = spline(z_vals, np.log(k_c), grid=True)
        else:  # None or 1 — S=1 outside [k_min, k_max]
            Sk = np.ones((n_z, n_k))
            k_in = (k_arr >= k_min) & (k_arr <= k_max)
            if k_in.any():
                Sk[:, k_in] = spline(z_vals, np.log(k_arr[k_in]), grid=True)
        return np.clip(Sk, 0.0, None)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def S(self, k, z_array):
        """
        Baryonic suppression S(k, z):  P_dmb = S * P_NL_dmo.

        Base implementation returns 1 (no feedback). Override in subclasses.

        Parameters
        ----------
        k : array (n_k,)      wavenumbers [Mpc^-1]
        z_array : array (n_z,)  redshifts

        Returns
        -------
        S : array (n_z, n_k)
        """
        k_arr = np.atleast_1d(k)
        z_arr = np.atleast_1d(z_array)
        return np.ones((len(z_arr), len(k_arr)))

    def matter_spectrum(self, k, z_array):
        """
        Compute matter power spectra.

        Returns
        -------
        dict with keys:
          'k'        : wavenumbers [Mpc^-1], shape (n_k,)
          'z'        : redshifts, shape (n_z,)
          'P_L_dmo'  : linear DMO [Mpc^3], shape (n_z, n_k)
          'P_NL_dmo' : nonlinear DMO [Mpc^3], shape (n_z, n_k)
          'P_dmb'    : P_NL_dmo * S(k,z) [Mpc^3], shape (n_z, n_k)
        """
        k_arr = np.atleast_1d(k)
        z_arr = np.atleast_1d(z_array)
        pk_lin = self._eval_pk_lin(k_arr, z_arr)
        pk_nl  = self._eval_pk_nl(k_arr, z_arr)
        Sk     = self.S(k_arr, z_arr)
        return dict(k=k_arr, z=z_arr,
                    P_L_dmo=pk_lin, P_NL_dmo=pk_nl,
                    P_dmb=pk_nl * Sk)

    def _limber_ckk(self, L_array):
        """
        C_L^{kappa kappa} via the Limber approximation.

        Uses P_dmb = P_NL_dmo * S for z <= z_nl_max, linear DMO for z > z_nl_max.
        """
        p   = self.cosmo_params
        H0  = p['H0']
        h   = H0 / 100.0
        Om  = (p['ombh2'] + p['omch2']) / h**2
        H0c = H0 / self.C_LIGHT

        chi_star = self._results_bg.comoving_radial_distance(1100.0)

        z_nl_max  = float(self._z_nl[-1])
        z_lin_max = float(self._z_lin[-1])
        z_int = np.concatenate([
            np.geomspace(0.001, z_nl_max, 900),
            np.geomspace(z_nl_max * 1.01, min(z_lin_max * 0.999, 999.0), 300)
        ])

        chi_int = self._results_bg.comoving_radial_distance(z_int)
        a_int   = 1.0 / (1.0 + z_int)
        W = 1.5 * H0c**2 * Om * chi_int * (chi_star - chi_int) / (chi_star * a_int)

        L = np.asarray(L_array)
        k_lo = max((L.min() + 0.5) / chi_int.max(), self._k_grid[0]  * 1.001)
        k_hi = min((L.max() + 0.5) / chi_int.min(), self._k_grid[-1] * 0.999)
        k_limber = np.geomspace(k_lo, k_hi, 400)

        mask_nl = z_int <= z_nl_max
        pk_dmb  = np.zeros((len(z_int), len(k_limber)))

        if mask_nl.any():
            pk_nl = self._eval_pk_nl(k_limber, z_int[mask_nl])
            Sk    = self.S(k_limber, z_int[mask_nl])
            pk_dmb[mask_nl] = pk_nl * Sk

        if (~mask_nl).any():
            pk_dmb[~mask_nl] = self._eval_pk_lin(k_limber, z_int[~mask_nl])

        k_Lchi   = (L[:, np.newaxis] + 0.5) / chi_int[np.newaxis, :]
        k_Lchi_c = np.clip(k_Lchi, k_limber[0], k_limber[-1])

        lk_limber = np.log(k_limber)
        lpk_dmb   = np.log(np.maximum(pk_dmb, 1e-40))
        lk_query  = np.log(k_Lchi_c)

        idx = np.searchsorted(lk_limber, lk_query, side='right') - 1
        idx = np.clip(idx, 0, len(k_limber) - 2)

        j      = np.arange(len(chi_int))
        lpk_lo = lpk_dmb[j[np.newaxis, :], idx]
        lpk_hi = lpk_dmb[j[np.newaxis, :], idx + 1]
        lk_lo  = lk_limber[idx]
        lk_hi  = lk_limber[idx + 1]
        w      = np.clip((lk_query - lk_lo) / (lk_hi - lk_lo), 0.0, 1.0)

        pk_Lchi   = np.exp(lpk_lo + w * (lpk_hi - lpk_lo))
        kernel    = (W / chi_int)**2
        integrand = pk_Lchi * kernel[np.newaxis, :]
        return _trapz(integrand, chi_int, axis=1)

    def cmb_spectrum(self, lmax=None):
        """
        Compute lensed and unlensed CMB power spectra driven by P_dmb.

        Baryonic feedback enters TT, EE, BB, and TE through the modified
        CMB lensing potential derived from P_dmb via the Limber integral.

        Returns
        -------
        dict with keys:
          'ell', 'C_kappakappa',
          'C_TT', 'C_EE', 'C_BB', 'C_TE'       : lensed [muK^2]
          'C_TT_unlensed', 'C_EE_unlensed',
          'C_TE_unlensed'                        : unlensed [muK^2]
        """
        if lmax is None:
            lmax = self.lmax
        if lmax > self.lmax:
            raise ValueError(
                f"lmax={lmax} exceeds self.lmax={self.lmax} set at init."
            )

        ell = np.arange(lmax + 1)

        print("  Computing C_L^kk via Limber integral...", end=' ', flush=True)
        ckk_vals = self._limber_ckk(np.arange(2, lmax + 1))
        ckk = np.zeros(lmax + 1)
        ckk[2:] = ckk_vals
        print("done.")

        clpp = np.zeros(self._camb_lmax + 1)
        clpp[2:lmax + 1] = 4.0 * ckk[2:] / (2.0 * np.pi)

        lensed = self._results_bg.get_lensed_cls_with_spectrum(
            clpp, lmax=lmax, CMB_unit='muK', raw_cl=True
        )  # (lmax+1, 4): TT, EE, BB, TE

        unlensed = self._results_bg.get_unlensed_scalar_cls(
            lmax=lmax, CMB_unit='muK', raw_cl=True
        )  # (lmax+1, 4): TT, EE, BB, TE

        return dict(
            ell=ell, C_kappakappa=ckk,
            C_TT=lensed[:, 0],    C_EE=lensed[:, 1],
            C_BB=lensed[:, 2],    C_TE=lensed[:, 3],
            C_TT_unlensed=unlensed[:, 0],
            C_EE_unlensed=unlensed[:, 1],
            C_TE_unlensed=unlensed[:, 3],
        )


class HMcodeCAMB(BaryonicCAMB):
    """
    Baryonic feedback via HMcode built into CAMB (Mead et al.).

    S(k, z) = P^NL_{feedback} / P^NL_{DMO}

    Parameters
    ----------
    baryonic_feedback : {'mead2020', 'mead2015'}
        HMcode version. 'mead2020' is more accurate and exposes an explicit
        AGN heating temperature parameter.
    logT_AGN : float
        Log10 of AGN heating temperature [K]. Only used for 'mead2020'.
        Default 7.8.
    A_baryon : float
        HMcode baryon amplitude parameter. Default 3.13.
    eta_baryon : float
        HMcode baryon eta parameter. Default 0.603.
    cosmo_params : dict, optional
    lmax : int
    kmax : float
    """

    _CAMB_MODEL = {
        'mead2020': 'mead2020_feedback',
        'mead2015': 'mead2015',
    }

    def __init__(self, cosmo_params=None, baryonic_feedback='mead2020',
                 logT_AGN=7.8, A_baryon=3.13, eta_baryon=0.603,
                 lmax=3000, kmax=50.0, k_extrap=None, nl_model='mead2020'):
        if baryonic_feedback not in self._CAMB_MODEL:
            raise ValueError(
                f"baryonic_feedback must be one of {list(self._CAMB_MODEL)}, "
                f"got '{baryonic_feedback}'"
            )
        self.baryonic_feedback = baryonic_feedback
        self.logT_AGN   = logT_AGN
        self.A_baryon   = A_baryon
        self.eta_baryon = eta_baryon
        super().__init__(cosmo_params=cosmo_params, lmax=lmax, kmax=kmax,
                         k_extrap=k_extrap, nl_model=nl_model)
        print(f"Building HMcode ({baryonic_feedback}) feedback grid...")
        self._setup_feedback_grid()
        print(f"HMcodeCAMB ({baryonic_feedback}) ready.")

    def _setup_feedback_grid(self):
        camb_name = self._CAMB_MODEL[self.baryonic_feedback]
        pars = self._make_cosmo_pars()
        pars.set_matter_power(redshifts=list(self._z_nl), kmax=self.kmax)
        pars.NonLinear = camb_model.NonLinear_both
        pars.NonLinearModel.set_params(camb_name)
        pars.NonLinearModel.HMCode_A_baryon   = self.A_baryon
        pars.NonLinearModel.HMCode_eta_baryon = self.eta_baryon
        if self.baryonic_feedback == 'mead2020':
            pars.NonLinearModel.HMCode_logT_AGN = self.logT_AGN
        res = camb.get_results(pars)
        _, _, pk_fb = res.get_nonlinear_matter_power_spectrum(
            hubble_units=False, k_hunit=False)

        lk = np.log(self._k_grid)
        self._spline_pk_fb = RectBivariateSpline(
            self._z_nl, lk, np.log(pk_fb), kx=3, ky=3
        )

    def _eval_pk_fb(self, k, z):
        lk = np.log(np.clip(k, self._k_grid[0], self._k_grid[-1]))
        zc = np.clip(z, self._z_nl[0], self._z_nl[-1])
        return np.exp(self._spline_pk_fb(zc, lk, grid=True))

    def S(self, k, z_array):
        """S(k,z) = P^NL_feedback / P^NL_dmo. Returns 1 for z beyond grid."""
        k_arr = np.atleast_1d(k)
        z_arr = np.atleast_1d(z_array)

        z_valid = np.clip(z_arr, self._z_nl[0], self._z_nl[-1])
        k_min, k_max = self._k_grid[0], self._k_grid[-1]
        policy = self.k_extrap

        def _ratio_at(k_pts):
            lk = np.log(np.clip(k_pts, k_min, k_max))
            return np.clip(
                np.exp(self._spline_pk_fb(z_valid, lk, grid=True)) /
                np.exp(self._spline_pk_nl(z_valid, lk, grid=True)),
                1e-10, None)

        if policy == 'extrapolate':
            k_in = (k_arr >= k_min) & (k_arr <= k_max)
            k_hi = k_arr > k_max
            k_lo = k_arr < k_min
            Sk = np.ones((len(z_arr), len(k_arr)))
            if k_in.any():
                Sk[:, k_in] = _ratio_at(k_arr[k_in])
            if k_hi.any():
                pts = np.array([k_max * 0.97, k_max])
                S_e = _ratio_at(pts)
                alpha = (np.log(S_e[:, 1]) - np.log(S_e[:, 0])) / (
                    np.log(pts[1]) - np.log(pts[0]))
                S_bnd = S_e[:, 1]
                dlk   = np.log(k_arr[k_hi]) - np.log(k_max)
                extrap = S_bnd[:, np.newaxis] * np.exp(
                    alpha[:, np.newaxis] * dlk[np.newaxis, :])
                Sk[:, k_hi] = np.where(
                    (alpha <= 0)[:, np.newaxis], 1.0, np.clip(extrap, 0.0, None))
            if k_lo.any():
                pts = np.array([k_min, k_min * 1.03])
                S_e = _ratio_at(pts)
                alpha = (np.log(S_e[:, 1]) - np.log(S_e[:, 0])) / (
                    np.log(pts[1]) - np.log(pts[0]))
                S_bnd = S_e[:, 0]
                dlk   = np.log(k_arr[k_lo]) - np.log(k_min)
                Sk[:, k_lo] = np.clip(
                    S_bnd[:, np.newaxis] * np.exp(
                        alpha[:, np.newaxis] * dlk[np.newaxis, :]),
                    0.0, None)
        elif policy in ('fixed', 'same'):
            pk_fb  = self._eval_pk_fb(k_arr, z_valid)
            pk_dmo = self._eval_pk_nl(k_arr, z_valid)
            Sk = pk_fb / pk_dmo
        else:  # None or 1 — S=1 outside [k_min, k_max]
            k_c = np.clip(k_arr, k_min, k_max)
            pk_fb  = self._eval_pk_fb(k_c, z_valid)
            pk_dmo = self._eval_pk_nl(k_c, z_valid)
            Sk = pk_fb / pk_dmo
            k_out = (k_arr < k_min) | (k_arr > k_max)
            if k_out.any():
                Sk[:, k_out] = 1.0

        outside_z = z_arr > self._z_nl[-1]
        if outside_z.any():
            Sk[outside_z, :] = 1.0
        return Sk


class HydroSimCAMB(BaryonicCAMB):
    """
    Baryonic feedback from hydrodynamical simulation data via BCemu.

    S(k, z) is loaded from BCemu.HydroSimDataSk. k in BCemu data is in
    h/Mpc and is converted to Mpc^-1 internally using H0 from cosmo_params.
    S = 1 outside the simulation redshift range.

    Parameters
    ----------
    sim_name : str
        Simulation name. Use BCemu.HydroSimDataSk.available() to list options.
        E.g. 'BAHAMAS', 'TNG100', 'EAGLE', 'HorizonAGN', 'FLAMINGO'.
    cosmo_params : dict, optional
    lmax : int
    kmax : float
    """

    def __init__(self, sim_name, cosmo_params=None, lmax=3000, kmax=50.0,
                 k_extrap=None, nl_model='mead2020'):
        self.sim_name = sim_name
        super().__init__(cosmo_params=cosmo_params, lmax=lmax, kmax=kmax,
                         k_extrap=k_extrap, nl_model=nl_model)
        print(f"  Loading BCemu data for {sim_name}...", end=' ', flush=True)
        self._setup_hydrosim_boost()
        print("done.")
        print(f"HydroSimCAMB ({sim_name}) ready.")

    def _setup_hydrosim_boost(self):
        from .datasets import HydroSimDataSk
        h = self.cosmo_params['H0'] / 100.0

        data = HydroSimDataSk(self.sim_name).load()
        k_hMpc = np.asarray(data['k'], dtype=float)
        z_sim  = np.asarray(data['z'], dtype=float)
        Sk     = np.asarray(data['Sk'], dtype=float)   # (n_z, n_k)

        k_Mpc = k_hMpc * h
        k_ok  = (k_Mpc > 0) & np.isfinite(k_Mpc)
        k_Mpc = k_Mpc[k_ok]
        Sk    = Sk[:, k_ok]
        sort_k = np.argsort(k_Mpc)
        k_Mpc  = k_Mpc[sort_k]
        Sk     = Sk[:, sort_k]

        z_ok  = (z_sim >= 0.0) & np.isfinite(z_sim)
        z_sim = z_sim[z_ok]
        Sk    = Sk[z_ok]
        z_sim, uidx = np.unique(z_sim, return_index=True)
        Sk = Sk[uidx]

        Sk = np.clip(Sk, 0.0, 3.0)
        kx = min(3, len(z_sim) - 1)
        self._spline_S_sim = RectBivariateSpline(
            z_sim, np.log(k_Mpc), Sk, kx=kx, ky=3
        )
        self._k_sim = k_Mpc
        self._z_sim = z_sim

    def S(self, k, z_array):
        """S(k,z) interpolated from simulation data. Returns 1 outside z range."""
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        z_arr = np.atleast_1d(np.asarray(z_array, dtype=float))

        Sk = np.ones((len(z_arr), len(k_arr)))
        in_range = (z_arr >= self._z_sim[0]) & (z_arr <= self._z_sim[-1])
        if in_range.any():
            Sk[in_range] = self._apply_k_extrap(
                self._spline_S_sim, z_arr[in_range],
                k_arr, self._k_sim[0], self._k_sim[-1]
            )
        return Sk


class BCemuCAMB(BaryonicCAMB):
    """
    Baryonic feedback from BCemu emulators.

    Parameters
    ----------
    bcm_params : dict
        Baryonic model parameters passed to the emulator.
        BCemu2025 keys: Theta_co, log10Mc, mu, delta, eta, deta, Nstar, fb.
        BCemu2021 keys: log10Mc, mu, thej, gamma, delta, eta, deta.
    baryonic_feedback : {'BCemu2025', 'BCemu2021'}
        Emulator version to use.
    q2 : float
        Secondary cosmological parameter for BCemu2025 (default 0.70).
        Not used for BCemu2021.
    cosmo_params : dict, optional
    lmax : int
    kmax : float
    """

    def __init__(self, bcm_params, baryonic_feedback='BCemu2025', q2=0.70,
                 cosmo_params=None, lmax=3000, kmax=50.0, k_extrap=None,
                 nl_model='mead2020'):
        _valid = ('BCemu2025', 'BCemu2021')
        if baryonic_feedback not in _valid:
            raise ValueError(
                f"baryonic_feedback must be one of {_valid}, "
                f"got '{baryonic_feedback}'"
            )
        self.bcm_params = bcm_params
        self.baryonic_feedback = baryonic_feedback
        self.q2 = q2
        super().__init__(cosmo_params=cosmo_params, lmax=lmax, kmax=kmax,
                         k_extrap=k_extrap, nl_model=nl_model)
        print(f"Building BCemu ({baryonic_feedback}) suppression grid...")
        self._setup_emu_boost()
        print(f"BCemuCAMB ({baryonic_feedback}) ready.")

    def _setup_emu_boost(self):
        h = self.cosmo_params['H0'] / 100.0

        if self.baryonic_feedback == 'BCemu2025':
            # Cache emulator so model weights are not reloaded on every update_bcm_params call
            if not hasattr(self, '_emu_instance'):
                from .BaryonEffectsEmulator import BCemu2025 as _BCemu2025
                self._emu_instance = _BCemu2025(backend='numpy')
            emu = self._emu_instance

            z_min_emu = float(emu.z_grid[0])
            z_max_emu = float(emu.z_grid[-1])
            mask = (self._z_nl >= z_min_emu) & (self._z_nl <= z_max_emu)
            z_valid = self._z_nl[mask]

            # get_boost returns (k [h/Mpc], S(k))
            k_hMpc, S0 = emu.get_boost(self.bcm_params, float(z_valid[0]), self.q2)
            k_hMpc = np.asarray(k_hMpc)
            k_mpc  = k_hMpc * h

            Sk_grid = np.ones((len(z_valid), len(k_mpc)))
            Sk_grid[0] = np.asarray(S0)
            for i, z in enumerate(z_valid[1:], 1):
                _, Sk = emu.get_boost(self.bcm_params, float(z), self.q2)
                Sk_grid[i] = np.asarray(Sk)

        else:  # BCemu2021
            if not hasattr(self, '_emu_instance'):
                from .BaryonEffectsEmulator import BCM_7param
                self._emu_instance = BCM_7param()
            emu = self._emu_instance

            # BCemu2021 is trained at z in {0, 0.5, 1, 1.5, 2}; k in h/Mpc
            z_max_emu = 2.0
            mask = self._z_nl <= z_max_emu
            z_valid = self._z_nl[mask]

            k_hMpc  = np.geomspace(0.1, 35.0, 200)
            k_mpc   = k_hMpc * h
            Sk_grid = np.ones((len(z_valid), len(k_mpc)))
            for i, z in enumerate(z_valid):
                try:
                    Sk = emu.get_boost(float(z), self.bcm_params, k_hMpc)
                    Sk_grid[i] = np.asarray(Sk)
                except Exception:
                    pass  # keep S=1 if emulator fails at this z

        Sk_grid = np.clip(Sk_grid, 0.0, 3.0)
        kx = min(3, len(z_valid) - 1)
        self._spline_S_emu = RectBivariateSpline(
            z_valid, np.log(k_mpc), Sk_grid, kx=kx, ky=3
        )
        self._k_emu = k_mpc
        self._z_emu = z_valid

    def update_bcm_params(self, bcm_params):
        """
        Update BCM parameters and rebuild S(k,z) without re-running CAMB.

        Equivalent to creating a new instance but ~10–50× faster because the
        CAMB background and matter power grids are reused.  The emulator
        weights are also cached across calls.
        """
        self.bcm_params = bcm_params
        self._setup_emu_boost()

    def S(self, k, z_array):
        """S(k,z) from BCemu emulator. Returns 1 outside emulator z range."""
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        z_arr = np.atleast_1d(np.asarray(z_array, dtype=float))

        Sk = np.ones((len(z_arr), len(k_arr)))
        in_range = (z_arr >= self._z_emu[0]) & (z_arr <= self._z_emu[-1])
        if in_range.any():
            Sk[in_range] = self._apply_k_extrap(
                self._spline_S_emu, z_arr[in_range],
                k_arr, self._k_emu[0], self._k_emu[-1]
            )
        return Sk
