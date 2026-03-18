import pytest
import numpy as np
import BCemu

bcmdict = {
    'Theta_co': 0.3,
    'log10Mc' : 13.1,
    'mu'      : 1.0,
    'delta'   : 6.0,
    'eta'     : 0.10,
    'deta'    : 0.22,
    'Nstar'   : 0.028,
    'fb'      : 0.0486 / 0.306,
}


@pytest.fixture(scope="module", params=["jax", "numpy", "torch"])
def emu(request):
    pytest.importorskip(
        {"jax": "jax", "numpy": "msgpack", "torch": "torch"}[request.param]
    )
    return BCemu.BCemu2025(backend=request.param)


@pytest.mark.bcemu2025
def test_get_boost_shape(emu):
    k, S = emu.get_boost(bcmdict, z=0.5)
    assert k.shape == S.shape
    assert S.shape[0] > 0


@pytest.mark.bcemu2025
def test_get_boost_range(emu):
    """Boost values should be close to 1 for fiducial parameters."""
    _, S = emu.get_boost(bcmdict, z=0.0)
    assert np.all(S > 0.5) and np.all(S <= 1.05)


@pytest.mark.bcemu2025
def test_get_boost_out_of_range(emu):
    with pytest.raises(ValueError):
        emu.get_boost(bcmdict, z=99.0)


@pytest.mark.bcemu2025
def test_get_boost_differentiable_jax():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    emu = BCemu.BCemu2025(backend="jax")
    params_jnp = jnp.array([bcmdict[k] for k in emu.param_names])
    S = emu.get_boost_differentiable(params_jnp, z=0.5)
    assert S.shape[0] > 0
    # Check that gradients can be computed
    J = jax.jacfwd(emu.get_boost_differentiable)(params_jnp, z=0.5)
    assert J.shape == (S.shape[0], params_jnp.shape[0])


@pytest.mark.bcemu2025
def test_get_boost_differentiable_torch():
    torch = pytest.importorskip("torch")
    emu = BCemu.BCemu2025(backend="torch")
    params_t = torch.tensor(
        [bcmdict[k] for k in emu.param_names], requires_grad=True
    )
    S = emu.get_boost_differentiable(params_t, z=0.5)
    assert S.shape[0] > 0
    # Check that autograd works
    S.sum().backward()
    assert params_t.grad is not None
