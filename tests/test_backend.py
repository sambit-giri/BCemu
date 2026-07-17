"""
Tests for BCemu.backend — the shared numpy/jax/torch dispatch layer used by
BCemu.cosmology (and available to any other module needing multi-backend
support). jax/torch tests are skipped gracefully when those optional
backends aren't installed.
"""
import numpy as np
import pytest

from BCemu import backend as bk

jax = pytest.importorskip("jax", reason="jax not installed; skipping jax backend tests")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

torch = pytest.importorskip("torch", reason="torch not installed; skipping torch backend tests")


def _t(x):
    return torch.tensor(x, dtype=torch.float64)


class TestCheckBackend:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            bk.check_backend('tensorflow')

    def test_valid_backends_pass(self):
        for backend in bk.SUPPORTED_BACKENDS:
            bk.check_backend(backend)   # no exception

    def test_jax_backend_raises_without_jax(self, monkeypatch):
        monkeypatch.setattr(bk, 'JAX_AVAILABLE', False)
        with pytest.raises(ImportError, match="JAX is required"):
            bk.check_backend('jax')

    def test_torch_backend_raises_without_torch(self, monkeypatch):
        monkeypatch.setattr(bk, 'TORCH_AVAILABLE', False)
        with pytest.raises(ImportError, match="PyTorch is required"):
            bk.check_backend('torch')


class TestOpsCrossBackendAgreement:
    def test_exp(self):
        x = 1.2345
        np.testing.assert_allclose(float(bk.exp(jnp.asarray(x), 'jax')), np.exp(x), rtol=1e-12)
        np.testing.assert_allclose(float(bk.exp(_t(x), 'torch')), np.exp(x), rtol=1e-12)

    def test_log(self):
        x = 3.21
        np.testing.assert_allclose(float(bk.log(jnp.asarray(x), 'jax')), np.log(x), rtol=1e-12)
        np.testing.assert_allclose(float(bk.log(_t(x), 'torch')), np.log(x), rtol=1e-12)

    def test_log10(self):
        x = 42.0
        np.testing.assert_allclose(float(bk.log10(jnp.asarray(x), 'jax')), np.log10(x), rtol=1e-12)
        np.testing.assert_allclose(float(bk.log10(_t(x), 'torch')), np.log10(x), rtol=1e-12)

    def test_sqrt(self):
        x = 7.0
        np.testing.assert_allclose(float(bk.sqrt(jnp.asarray(x), 'jax')), np.sqrt(x), rtol=1e-12)
        np.testing.assert_allclose(float(bk.sqrt(_t(x), 'torch')), np.sqrt(x), rtol=1e-12)

    def test_exp_accepts_plain_python_float(self):
        # torch.exp normally rejects a bare python float -- bk.exp must not.
        assert bk.exp(1.0, 'torch') == pytest.approx(np.e)
        assert float(bk.exp(1.0, 'jax')) == pytest.approx(np.e)

    def test_linspace(self):
        np_grid = bk.linspace(0.0, 1.0, 5, 'numpy')
        jax_grid = bk.linspace(0.0, 1.0, 5, 'jax')
        torch_grid = bk.linspace(0.0, 1.0, 5, 'torch')
        np.testing.assert_allclose(np.asarray(jax_grid), np_grid)
        np.testing.assert_allclose(torch_grid.numpy(), np_grid)

    def test_trapz(self):
        x = np.linspace(0.0, 2.0, 100)
        y = x ** 2
        v_np = bk.trapz(y, x, 'numpy')
        v_jax = bk.trapz(jnp.asarray(y), jnp.asarray(x), 'jax')
        v_torch = bk.trapz(_t(y), _t(x), 'torch')
        np.testing.assert_allclose(float(v_jax), v_np, rtol=1e-10)
        np.testing.assert_allclose(float(v_torch), v_np, rtol=1e-10)


class TestAsTorch:
    def test_wraps_plain_float_as_float64(self):
        t = bk.as_torch(3.14)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float64

    def test_preserves_existing_tensor_identity(self):
        """as_torch must not copy-and-detach an existing tensor (that would
        break an in-progress autograd graph)."""
        x = torch.tensor(2.0, requires_grad=True)
        y = x * 3.0
        wrapped = bk.as_torch(y)
        assert wrapped is y
        wrapped.backward()
        assert x.grad.item() == pytest.approx(3.0)
