"""
Tests for BCemu.spectra module.

Requires camb:  pip install BCemu[spectra]
Tests are skipped gracefully when camb is not installed.
"""

import numpy as np
import pytest

camb = pytest.importorskip("camb", reason="camb not installed; skipping spectra tests")

import BCemu
from BCemu.spectra import BaryonicCAMB, HMcodeCAMB, HydroSimCAMB, BCemuCAMB


# ---------------------------------------------------------------------------
# Shared fixture — one BaryonicCAMB instance for the whole session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def base_model():
    return BaryonicCAMB(lmax=500, kmax=10.0)


@pytest.fixture(scope="module")
def k_arr():
    return np.geomspace(0.05, 5.0, 30)


@pytest.fixture(scope="module")
def z_arr():
    return np.array([0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# BaryonicCAMB — base class
# ---------------------------------------------------------------------------
class TestBaryonicCAMB:
    def test_S_equals_one(self, base_model, k_arr, z_arr):
        S = base_model.S(k_arr, z_arr)
        assert S.shape == (len(z_arr), len(k_arr))
        np.testing.assert_allclose(S, 1.0)

    def test_matter_spectrum_shapes(self, base_model, k_arr, z_arr):
        pk = base_model.matter_spectrum(k_arr, z_arr)
        for key in ('k', 'z', 'P_L_dmo', 'P_NL_dmo', 'P_dmb'):
            assert key in pk
        assert pk['P_L_dmo'].shape  == (len(z_arr), len(k_arr))
        assert pk['P_NL_dmo'].shape == (len(z_arr), len(k_arr))
        assert pk['P_dmb'].shape    == (len(z_arr), len(k_arr))

    def test_P_dmb_equals_P_NL_dmo_for_base(self, base_model, k_arr, z_arr):
        pk = base_model.matter_spectrum(k_arr, z_arr)
        np.testing.assert_allclose(pk['P_dmb'], pk['P_NL_dmo'])

    def test_matter_spectra_positive(self, base_model, k_arr, z_arr):
        pk = base_model.matter_spectrum(k_arr, z_arr)
        assert np.all(pk['P_L_dmo']  > 0)
        assert np.all(pk['P_NL_dmo'] > 0)

    def test_cmb_spectrum_keys(self, base_model):
        cmb = base_model.cmb_spectrum(lmax=100)
        expected = {'ell', 'C_kappakappa', 'C_TT', 'C_EE', 'C_BB', 'C_TE',
                    'C_TT_unlensed', 'C_EE_unlensed', 'C_TE_unlensed'}
        assert expected.issubset(cmb.keys())

    def test_cmb_spectrum_shapes(self, base_model):
        lmax = 100
        cmb = base_model.cmb_spectrum(lmax=lmax)
        for key in ('C_TT', 'C_EE', 'C_BB', 'C_TE', 'C_kappakappa'):
            assert cmb[key].shape == (lmax + 1,)

    def test_cmb_lmax_validation(self, base_model):
        with pytest.raises(ValueError):
            base_model.cmb_spectrum(lmax=base_model.lmax + 1)

    def test_no_camb_raises(self, monkeypatch):
        import BCemu.spectra as sp
        monkeypatch.setattr(sp, 'HAS_CAMB', False)
        with pytest.raises(ImportError, match="camb is required"):
            BaryonicCAMB.__init__(object.__new__(BaryonicCAMB))


# ---------------------------------------------------------------------------
# HMcodeCAMB
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hmcode20():
    return HMcodeCAMB(baryonic_feedback='mead2020', logT_AGN=7.8,
                      lmax=500, kmax=10.0)


@pytest.fixture(scope="module")
def hmcode15():
    return HMcodeCAMB(baryonic_feedback='mead2015', lmax=500, kmax=10.0)


class TestHMcodeCAMB:
    def test_invalid_feedback_raises(self):
        with pytest.raises(ValueError):
            HMcodeCAMB(baryonic_feedback='mead1999')

    def test_S_shape(self, hmcode20, k_arr, z_arr):
        S = hmcode20.S(k_arr, z_arr)
        assert S.shape == (len(z_arr), len(k_arr))

    def test_S_suppressed_at_small_scales_mead2020(self, hmcode20, z_arr):
        # Feedback suppresses power at small scales (large k)
        k_small = np.array([0.1])
        k_large = np.array([5.0])
        S_small = hmcode20.S(k_small, z_arr[:1])
        S_large = hmcode20.S(k_large, z_arr[:1])
        assert float(S_large) < float(S_small)

    def test_S_suppressed_at_small_scales_mead2015(self, hmcode15, z_arr):
        k_small = np.array([0.1])
        k_large = np.array([5.0])
        S_small = hmcode15.S(k_small, z_arr[:1])
        S_large = hmcode15.S(k_large, z_arr[:1])
        assert float(S_large) < float(S_small)

    def test_S_high_z_is_one(self, hmcode20, k_arr):
        # Beyond the NL grid (z > ~5) suppression should be 1
        S = hmcode20.S(k_arr, np.array([10.0]))
        np.testing.assert_allclose(S, 1.0)

    def test_P_dmb_less_than_P_NL_dmo_at_small_scales(self, hmcode20, z_arr):
        k_large = np.geomspace(1.0, 10.0, 10)
        pk = hmcode20.matter_spectrum(k_large, z_arr[:1])
        assert np.all(pk['P_dmb'] <= pk['P_NL_dmo'] * 1.01)  # small tolerance


# ---------------------------------------------------------------------------
# HydroSimCAMB
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hydro_model():
    available = BCemu.HydroSimDataSk.available()
    if not available:
        pytest.skip("No HydroSim data available")
    return HydroSimCAMB(available[0], lmax=500, kmax=10.0)


class TestHydroSimCAMB:
    def test_S_shape(self, hydro_model, k_arr, z_arr):
        S = hydro_model.S(k_arr, z_arr)
        assert S.shape == (len(z_arr), len(k_arr))

    def test_S_non_negative(self, hydro_model, k_arr, z_arr):
        S = hydro_model.S(k_arr, z_arr)
        assert np.all(S >= 0)

    def test_S_outside_z_range_is_one(self, hydro_model, k_arr):
        z_high = np.array([50.0])
        S = hydro_model.S(k_arr, z_high)
        np.testing.assert_allclose(S, 1.0)

    def test_matter_spectrum_shape(self, hydro_model, k_arr, z_arr):
        pk = hydro_model.matter_spectrum(k_arr, z_arr)
        assert pk['P_dmb'].shape == (len(z_arr), len(k_arr))


# ---------------------------------------------------------------------------
# BCemuCAMB
# ---------------------------------------------------------------------------
BCM_PARAMS_2025 = {
    'Theta_co': 0.3,
    'log10Mc':  13.1,
    'mu':       1.0,
    'delta':    6.0,
    'eta':      0.10,
    'deta':     0.22,
    'Nstar':    0.028,
    'fb':       0.0486 / 0.306,
}

BCM_PARAMS_2021 = {
    'log10Mc': 13.32,
    'mu':       0.93,
    'thej':     4.235,
    'gamma':    2.25,
    'delta':    6.40,
    'eta':      0.15,
    'deta':     0.14,
}


@pytest.fixture(scope="module")
def bcemu25():
    return BCemuCAMB(BCM_PARAMS_2025, baryonic_feedback='BCemu2025', q2=0.70,
                     lmax=500, kmax=10.0)


class TestBCemuCAMB:
    def test_invalid_feedback_raises(self):
        with pytest.raises(ValueError):
            BCemuCAMB(BCM_PARAMS_2025, baryonic_feedback='BCemu1999')

    def test_S_shape(self, bcemu25, k_arr, z_arr):
        S = bcemu25.S(k_arr, z_arr)
        assert S.shape == (len(z_arr), len(k_arr))

    def test_S_non_negative(self, bcemu25, k_arr, z_arr):
        S = bcemu25.S(k_arr, z_arr)
        assert np.all(S >= 0)

    def test_S_outside_z_range_is_one(self, bcemu25, k_arr):
        z_high = np.array([50.0])
        S = bcemu25.S(k_arr, z_high)
        np.testing.assert_allclose(S, 1.0)

    def test_matter_spectrum_shape(self, bcemu25, k_arr, z_arr):
        pk = bcemu25.matter_spectrum(k_arr, z_arr)
        assert pk['P_dmb'].shape == (len(z_arr), len(k_arr))

    def test_consistency_with_direct_emulator(self, bcemu25, k_arr):
        """S from BCemuCAMB should be close to direct BCemu2025 output at z=0."""
        from BCemu import BCemu2025
        emu = BCemu2025(backend='numpy')
        k_hMpc_emu, S_direct = emu.get_boost(BCM_PARAMS_2025, 0.0, 0.70)
        k_hMpc_emu = np.asarray(k_hMpc_emu)
        h = bcemu25.cosmo_params['H0'] / 100.0
        k_mpc_emu = k_hMpc_emu * h

        # Interpolate direct output onto k_arr
        k_clip = np.clip(k_arr, k_mpc_emu[0], k_mpc_emu[-1])
        S_ref = np.interp(k_clip, k_mpc_emu, np.asarray(S_direct))

        S_model = bcemu25.S(k_arr, np.array([0.0]))[0]
        k_ok = (k_arr >= k_mpc_emu[0]) & (k_arr <= k_mpc_emu[-1])
        np.testing.assert_allclose(S_model[k_ok], S_ref[k_ok], rtol=0.05)

    def test_bcemu2021(self, k_arr, z_arr):
        model = BCemuCAMB(BCM_PARAMS_2021, baryonic_feedback='BCemu2021',
                          lmax=500, kmax=10.0)
        S = model.S(k_arr, z_arr)
        assert S.shape == (len(z_arr), len(k_arr))
        assert np.all(S >= 0)
