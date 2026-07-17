"""
Tests for BCemu.cosmology module.

Cross-backend (numpy/jax/torch) numerical agreement, CPL -> LCDM reduction,
and gradient smoke tests. jax/torch tests are skipped gracefully when those
optional backends aren't installed.
"""
import numpy as np
import pytest

from BCemu import cosmology as cosmo
from BCemu import backend as bk

jax = pytest.importorskip("jax", reason="jax not installed; skipping jax backend tests")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

torch = pytest.importorskip("torch", reason="torch not installed; skipping torch backend tests")


def _t(x):
    """float64 torch tensor, without touching the global default dtype
    (other test modules rely on the default float32 for their model weights)."""
    return torch.tensor(x, dtype=torch.float64)


OMBH2, OMCH2, H0 = 0.02237, 0.1200, 68.0


@pytest.fixture(scope="module")
def Om():
    return cosmo.omega_m_total(OMBH2, OMCH2, H0)


# ---------------------------------------------------------------------------
# Backend validation
# ---------------------------------------------------------------------------
class TestBackendValidation:
    def test_invalid_backend_raises(self, Om):
        with pytest.raises(ValueError):
            cosmo.Ez(0.5, Om, backend='tensorflow')

    def test_jax_backend_raises_without_jax(self, Om, monkeypatch):
        monkeypatch.setattr(bk, 'JAX_AVAILABLE', False)
        with pytest.raises(ImportError, match="JAX is required"):
            cosmo.Ez(0.5, Om, backend='jax')

    def test_torch_backend_raises_without_torch(self, Om, monkeypatch):
        monkeypatch.setattr(bk, 'TORCH_AVAILABLE', False)
        with pytest.raises(ImportError, match="PyTorch is required"):
            cosmo.Ez(0.5, Om, backend='torch')


# ---------------------------------------------------------------------------
# CPL -> LCDM reduction
# ---------------------------------------------------------------------------
class TestCPLReduction:
    def test_dark_energy_density_is_one_at_defaults(self):
        z = np.array([0.0, 0.5, 1.0, 5.0])
        np.testing.assert_allclose(cosmo.dark_energy_density(z), 1.0)

    def test_Ez_defaults_equal_explicit_lcdm(self, Om):
        z = 0.7
        assert cosmo.Ez(z, Om) == cosmo.Ez(z, Om, w0=-1.0, wa=0.0)

    def test_comoving_distance_defaults_equal_explicit_lcdm(self, Om):
        z = 0.7
        d1 = cosmo.comoving_distance(z, H0, Om)
        d2 = cosmo.comoving_distance(z, H0, Om, w0=-1.0, wa=0.0)
        assert d1 == d2

    def test_cpl_deviates_from_lcdm_when_w0_changes(self, Om):
        z = 0.7
        d_lcdm = cosmo.comoving_distance(z, H0, Om)
        d_cpl = cosmo.comoving_distance(z, H0, Om, w0=-0.8, wa=0.3)
        assert not np.isclose(d_lcdm, d_cpl)


# ---------------------------------------------------------------------------
# Cross-backend numerical agreement
# ---------------------------------------------------------------------------
class TestCrossBackendAgreement:
    def test_Ez(self, Om):
        z = 0.5
        v_np = cosmo.Ez(z, Om, backend='numpy')
        v_jax = cosmo.Ez(z, jnp.asarray(Om), backend='jax')
        v_torch = cosmo.Ez(z, _t(Om), backend='torch')
        np.testing.assert_allclose(float(v_jax), v_np, rtol=1e-10)
        np.testing.assert_allclose(float(v_torch), v_np, rtol=1e-10)

    def test_comoving_distance(self, Om):
        z = 0.5
        v_np = cosmo.comoving_distance(z, H0, Om, backend='numpy')
        v_jax = cosmo.comoving_distance(z, jnp.asarray(H0), jnp.asarray(Om), backend='jax')
        v_torch = cosmo.comoving_distance(z, _t(H0), _t(Om), backend='torch')
        np.testing.assert_allclose(float(v_jax), v_np, rtol=1e-8)
        np.testing.assert_allclose(float(v_torch), v_np, rtol=1e-8)

    def test_luminosity_distance_and_distance_modulus(self, Om):
        z = 0.5
        dl_np = cosmo.luminosity_distance(z, H0, Om, backend='numpy')
        mu_np = cosmo.distance_modulus(z, H0, Om, backend='numpy')
        dl_jax = cosmo.luminosity_distance(z, jnp.asarray(H0), jnp.asarray(Om), backend='jax')
        mu_jax = cosmo.distance_modulus(z, jnp.asarray(H0), jnp.asarray(Om), backend='jax')
        np.testing.assert_allclose(float(dl_jax), dl_np, rtol=1e-8)
        np.testing.assert_allclose(float(mu_jax), mu_np, rtol=1e-8)

    @pytest.mark.parametrize("method", [
        'eh98', 'aubourg2015', 'aizpuru2021_narrow', 'aizpuru2021_broad', 'aizpuru2021_neutrino',
    ])
    def test_rdrag_fitting(self, method):
        rd_np = cosmo.rdrag_fitting(OMBH2, OMCH2, method=method, backend='numpy')
        rd_jax = cosmo.rdrag_fitting(jnp.asarray(OMBH2), jnp.asarray(OMCH2), method=method, backend='jax')
        rd_torch = cosmo.rdrag_fitting(_t(OMBH2), _t(OMCH2), method=method, backend='torch')
        np.testing.assert_allclose(float(rd_jax), rd_np, rtol=1e-8)
        np.testing.assert_allclose(float(rd_torch), rd_np, rtol=1e-8)

    def test_rdrag_invalid_method_raises(self):
        with pytest.raises(ValueError):
            cosmo.rdrag_fitting(OMBH2, OMCH2, method='not_a_method')

    def test_r_star_and_theta_star(self, Om):
        rs_np = cosmo.r_star(OMBH2, OMCH2, backend='numpy')
        rs_jax = cosmo.r_star(jnp.asarray(OMBH2), jnp.asarray(OMCH2), backend='jax')
        np.testing.assert_allclose(float(rs_jax), rs_np, rtol=1e-8)

        # sanity: known physical values, z* ~ 1090, theta* ~ 0.0104 (Planck-like)
        z_star = cosmo.photon_decoupling_redshift(OMBH2, OMCH2)
        assert 1085 < z_star < 1095
        theta = cosmo.theta_star(OMBH2, OMCH2, H0, Om, backend='numpy')
        assert 0.0100 < theta < 0.0108

    @pytest.mark.parametrize("method", ['aizpuru2021', 'hu_sugiyama1996'])
    def test_photon_decoupling_redshift_methods(self, method):
        z_np = cosmo.photon_decoupling_redshift(OMBH2, OMCH2, method=method)
        z_jax = cosmo.photon_decoupling_redshift(jnp.asarray(OMBH2), jnp.asarray(OMCH2), method=method)
        np.testing.assert_allclose(float(z_jax), z_np, rtol=1e-8)
        assert 1085 < z_np < 1095

    def test_photon_decoupling_redshift_methods_agree_closely(self):
        z_aizpuru = cosmo.photon_decoupling_redshift(OMBH2, OMCH2, method='aizpuru2021')
        z_hs = cosmo.photon_decoupling_redshift(OMBH2, OMCH2, method='hu_sugiyama1996')
        np.testing.assert_allclose(z_aizpuru, z_hs, rtol=5e-3)

    def test_photon_decoupling_redshift_invalid_method_raises(self):
        with pytest.raises(ValueError):
            cosmo.photon_decoupling_redshift(OMBH2, OMCH2, method='not_a_method')

    def test_r_star_zstar_method_forwarded(self, Om):
        rs_default = cosmo.r_star(OMBH2, OMCH2)
        rs_hs = cosmo.r_star(OMBH2, OMCH2, zstar_method='hu_sugiyama1996')
        assert rs_default != rs_hs   # different z_star -> different r_star

        theta_default = cosmo.theta_star(OMBH2, OMCH2, H0, Om)
        theta_hs = cosmo.theta_star(OMBH2, OMCH2, H0, Om, zstar_method='hu_sugiyama1996')
        assert theta_default != theta_hs

    def test_bao_kinematics(self, Om):
        z = 0.51
        DA_np, H_np, rd_np = cosmo.bao_kinematics(z, H0, OMBH2, OMCH2, backend='numpy')
        DA_jax, H_jax, rd_jax = cosmo.bao_kinematics(
            z, jnp.asarray(H0), jnp.asarray(OMBH2), jnp.asarray(OMCH2), backend='jax')
        np.testing.assert_allclose(float(DA_jax), DA_np, rtol=1e-8)
        np.testing.assert_allclose(float(H_jax), H_np, rtol=1e-8)
        np.testing.assert_allclose(float(rd_jax), rd_np, rtol=1e-8)

    @pytest.mark.parametrize("method", ['linder2005', 'ode_solve'])
    def test_growth_factor(self, Om, method):
        z = 0.5
        d_np = cosmo.growth_factor(z, Om, method=method, backend='numpy')
        d_jax = cosmo.growth_factor(z, jnp.asarray(Om), method=method, backend='jax')
        d_torch = cosmo.growth_factor(z, _t(Om), method=method, backend='torch')
        np.testing.assert_allclose(float(d_jax), d_np, rtol=1e-6)
        np.testing.assert_allclose(float(d_torch), d_np, rtol=1e-6)

    def test_growth_factor_normalized_at_z0(self, Om):
        assert cosmo.growth_factor(0.0, Om, method='linder2005') == pytest.approx(1.0, abs=1e-8)
        assert cosmo.growth_factor(0.0, Om, method='ode_solve') == pytest.approx(1.0, rel=1e-3)

    def test_growth_factor_linder2005_vs_ode_solve_agree(self, Om):
        d_fit = cosmo.growth_factor(0.5, Om, method='linder2005')
        d_ode = cosmo.growth_factor(0.5, Om, method='ode_solve')
        np.testing.assert_allclose(d_ode, d_fit, rtol=5e-3)

    def test_growth_factor_invalid_method_raises(self, Om):
        with pytest.raises(ValueError):
            cosmo.growth_factor(0.5, Om, method='not_a_method')

    @pytest.mark.parametrize("method", ['linder2005', 'ode_solve'])
    def test_growth_rate(self, Om, method):
        z = 0.5
        f_np = cosmo.growth_rate(z, Om, method=method, backend='numpy')
        f_jax = cosmo.growth_rate(z, jnp.asarray(Om), method=method, backend='jax')
        np.testing.assert_allclose(float(f_jax), f_np, rtol=1e-6)

    def test_sigma8_z(self, Om):
        s_np = cosmo.sigma8_z(0.8, 0.5, Om, backend='numpy')
        s_jax = cosmo.sigma8_z(0.8, 0.5, jnp.asarray(Om), backend='jax')
        np.testing.assert_allclose(float(s_jax), s_np, rtol=1e-6)

    def test_As_to_sigma8_bartlett2024(self):
        ln1e10As, ns = 3.044, 0.9649
        s_np = cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, backend='numpy')
        s_jax = cosmo.As_to_sigma8(
            jnp.asarray(OMBH2), jnp.asarray(OMCH2), jnp.asarray(H0), jnp.asarray(ns), jnp.asarray(ln1e10As),
            backend='jax')
        np.testing.assert_allclose(float(s_jax), s_np, rtol=1e-8)
        assert 0.5 < s_np < 1.2   # sanity range

    def test_As_to_sigma8_invalid_method_raises(self):
        with pytest.raises(ValueError):
            cosmo.As_to_sigma8(OMBH2, OMCH2, H0, 0.9649, 3.044, method='not_a_method')

    @pytest.mark.parametrize("method", ['syren_new', 'sui2025'])
    def test_As_to_sigma8_syren_new_cross_backend(self, method):
        ln1e10As, ns = 3.044, 0.9649
        mnu, w0, wa = 0.06, -1.05, 0.1
        s_np = cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, mnu=mnu, w0=w0, wa=wa,
                                   method=method, backend='numpy')
        s_jax = cosmo.As_to_sigma8(
            jnp.asarray(OMBH2), jnp.asarray(OMCH2), jnp.asarray(H0), jnp.asarray(ns), jnp.asarray(ln1e10As),
            mnu=mnu, w0=w0, wa=wa, method=method, backend='jax')
        s_torch = cosmo.As_to_sigma8(
            _t(OMBH2), _t(OMCH2), _t(H0), _t(ns), _t(ln1e10As),
            mnu=mnu, w0=w0, wa=wa, method=method, backend='torch')
        np.testing.assert_allclose(float(s_jax), s_np, rtol=1e-8)
        np.testing.assert_allclose(float(s_torch), s_np, rtol=1e-8)
        assert 0.5 < s_np < 1.2

    def test_As_to_sigma8_sui2025_aliases_syren_new(self):
        ln1e10As, ns = 3.044, 0.9649
        s_syren = cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, method='syren_new')
        s_sui = cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, method='sui2025')
        assert s_syren == s_sui

    def test_As_to_sigma8_syren_new_matches_reference_implementation(self):
        symbolic_pofk_linear_new = pytest.importorskip(
            "symbolic_pofk.linear_new",
            reason="symbolic_pofk not installed; skipping upstream cross-check "
                   "(pip install git+https://github.com/DeaglanBartlett/symbolic_pofk.git)")
        ln1e10As, ns = 3.044, 0.9649
        mnu, w0, wa = 0.06, -1.0, 0.0
        h = H0 / 100.0
        Om = cosmo.omega_m_total(OMBH2, OMCH2, H0, omega_nu_h2=mnu / 93.14)
        Ob = OMBH2 / h ** 2
        As_1e9 = np.exp(ln1e10As) / 10.0

        s_mine = cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, mnu=mnu, w0=w0, wa=wa, method='syren_new')
        s_ref = symbolic_pofk_linear_new.As_to_sigma8(As_1e9, Om, Ob, h, ns, mnu, w0, wa)
        np.testing.assert_allclose(s_mine, s_ref, rtol=1e-10)

    def test_sigma8_to_As_syren_new_roundtrip(self):
        ln1e10As, ns = 3.044, 0.9649
        mnu, w0, wa = 0.06, -1.05, 0.1
        s8 = cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, mnu=mnu, w0=w0, wa=wa, method='syren_new')
        ln1e10As_back = cosmo.sigma8_to_As(OMBH2, OMCH2, H0, ns, s8, mnu=mnu, w0=w0, wa=wa, method='syren_new')
        np.testing.assert_allclose(ln1e10As_back, ln1e10As, rtol=1e-8)

    def test_As_to_sigma8_syren_new_differs_from_bartlett2024_at_lcdm(self):
        # different fits -- shouldn't agree exactly, but should be in the same ballpark
        ln1e10As, ns = 3.044, 0.9649
        s_bartlett = cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, method='bartlett2024')
        s_syren = cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, method='syren_new')
        assert s_bartlett != s_syren
        np.testing.assert_allclose(s_syren, s_bartlett, rtol=3e-2)

    def test_sigma8_to_As_roundtrip_bartlett2024(self):
        ln1e10As, ns = 3.044, 0.9649
        s8 = cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, backend='numpy')
        ln1e10As_back = cosmo.sigma8_to_As(OMBH2, OMCH2, H0, ns, s8, backend='numpy')
        np.testing.assert_allclose(ln1e10As_back, ln1e10As, rtol=1e-10)

    def test_sigma8_to_As_cross_backend(self):
        ln1e10As, ns = 3.044, 0.9649
        s8 = cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, backend='numpy')
        v_np = cosmo.sigma8_to_As(OMBH2, OMCH2, H0, ns, s8, backend='numpy')
        v_jax = cosmo.sigma8_to_As(
            jnp.asarray(OMBH2), jnp.asarray(OMCH2), jnp.asarray(H0), jnp.asarray(ns), jnp.asarray(s8),
            backend='jax')
        v_torch = cosmo.sigma8_to_As(_t(OMBH2), _t(OMCH2), _t(H0), _t(ns), _t(s8), backend='torch')
        np.testing.assert_allclose(float(v_jax), v_np, rtol=1e-8)
        np.testing.assert_allclose(float(v_torch), v_np, rtol=1e-8)

    def test_sigma8_to_As_invalid_method_raises(self):
        with pytest.raises(ValueError):
            cosmo.sigma8_to_As(OMBH2, OMCH2, H0, 0.9649, 0.8, method='not_a_method')

    def test_As_to_sigma8_camb_warns_on_non_numpy_backend(self, capsys):
        try:
            cosmo.As_to_sigma8(OMBH2, OMCH2, H0, 0.9649, 3.044, method='camb', backend='jax')
        except ImportError:
            pass   # camb itself may not be installed; the warning must still fire first
        out = capsys.readouterr().out
        assert "backend='jax' is ignored" in out

    def test_As_to_sigma8_camb_silent_on_numpy_backend(self, capsys):
        try:
            cosmo.As_to_sigma8(OMBH2, OMCH2, H0, 0.9649, 3.044, method='camb', backend='numpy')
        except ImportError:
            pass
        out = capsys.readouterr().out
        assert "is ignored" not in out

    def test_sigma8_to_As_camb_warns_on_non_numpy_backend(self, capsys):
        try:
            cosmo.sigma8_to_As(OMBH2, OMCH2, H0, 0.9649, 0.8, method='camb', backend='torch')
        except ImportError:
            pass
        out = capsys.readouterr().out
        assert "backend='torch' is ignored" in out


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------
class TestGradients:
    def test_jax_grad_As_to_sigma8_syren_new(self):
        ln1e10As, ns = 3.044, 0.9649

        def f(w0_):
            return cosmo.As_to_sigma8(OMBH2, OMCH2, H0, ns, ln1e10As, w0=w0_, method='syren_new', backend='jax')

        g = jax.grad(f)(jnp.asarray(-1.05))
        assert np.isfinite(float(g)) and g != 0

    def test_torch_autograd_sigma8_to_As_syren_new(self):
        mnu_t = torch.tensor(0.06, dtype=torch.float64, requires_grad=True)
        cosmo.sigma8_to_As(OMBH2, OMCH2, H0, 0.9649, 0.81, mnu=mnu_t, method='syren_new',
                           backend='torch').backward()
        assert np.isfinite(mnu_t.grad.item()) and mnu_t.grad.item() != 0

    def test_jax_grad_comoving_distance(self, Om):
        def f(H0_, Om_):
            return cosmo.comoving_distance(0.5, H0_, Om_, backend='jax')
        gH0, gOm = jax.grad(f, argnums=(0, 1))(jnp.asarray(H0), jnp.asarray(Om))
        assert np.isfinite(float(gH0)) and gH0 != 0
        assert np.isfinite(float(gOm)) and gOm != 0

    def test_torch_autograd_comoving_distance(self, Om):
        H0_t = torch.tensor(H0, dtype=torch.float64, requires_grad=True)
        Om_t = torch.tensor(Om, dtype=torch.float64, requires_grad=True)
        cosmo.comoving_distance(0.5, H0_t, Om_t, backend='torch').backward()
        assert np.isfinite(H0_t.grad.item()) and H0_t.grad.item() != 0
        assert np.isfinite(Om_t.grad.item()) and Om_t.grad.item() != 0

    @pytest.mark.parametrize("method", ['linder2005', 'ode_solve'])
    def test_jax_grad_growth_factor(self, Om, method):
        def f(Om_):
            return cosmo.growth_factor(0.5, Om_, method=method, backend='jax')
        g = jax.grad(f)(jnp.asarray(Om))
        assert np.isfinite(float(g)) and g != 0

    @pytest.mark.parametrize("method", ['linder2005', 'ode_solve'])
    def test_torch_autograd_growth_factor(self, Om, method):
        Om_t = torch.tensor(Om, dtype=torch.float64, requires_grad=True)
        cosmo.growth_factor(0.5, Om_t, method=method, backend='torch').backward()
        assert np.isfinite(Om_t.grad.item()) and Om_t.grad.item() != 0

    def test_jax_grad_rdrag_fitting(self):
        def f(ombh2, omch2):
            return cosmo.rdrag_fitting(ombh2, omch2, method='aubourg2015', backend='jax')
        g_ombh2, g_omch2 = jax.grad(f, argnums=(0, 1))(jnp.asarray(OMBH2), jnp.asarray(OMCH2))
        assert np.isfinite(float(g_ombh2)) and np.isfinite(float(g_omch2))

    def test_jax_grad_theta_star(self):
        def f(H0_):
            Om_ = cosmo.omega_m_total(OMBH2, OMCH2, H0_)
            return cosmo.theta_star(OMBH2, OMCH2, H0_, Om_, backend='jax')
        g = jax.grad(f)(jnp.asarray(H0))
        assert np.isfinite(float(g)) and g != 0
