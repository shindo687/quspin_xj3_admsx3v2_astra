"""Second-order checks against analytic and independent central oracles."""

import numpy as np
import pytest

import quspin_ad
import chainrules as ad
from quspin.basis import coherent_state
from quspin.operators import commutator, anti_commutator
from quspin.tools.evolution import ED_state_vs_time
from quspin.tools.lanczos import lin_comb_Q_T
from quspin.tools.misc import KL_div, project_op


def mixed_difference(fun, values, u, v, h=1e-4):
    def at(s, r):
        return fun(**{k: x + s * u.get(k, 0) + r * v.get(k, 0) for k, x in values.items()})
    return (at(h, h) - at(h, -h) - at(-h, h) + at(-h, -h)) / (4 * h**2)


def test_kl_second_order_every_input_mixed_and_hvp():
    p, q = np.array([.2, .3, .5]), np.array([.3, .3, .4])
    u, v = np.array([.1, -.04, -.06]), np.array([-.02, .01, .01])
    for directions in ({"p1": u}, {"p2": v}, {"p1": u, "p2": v}):
        value, first, second = ad.nested_jvp(KL_div, p, q, tangents=directions)
        oracle = mixed_difference(lambda p1, p2: KL_div(p1, p2), {"p1": p, "p2": q}, directions, directions)
        assert value == KL_div(p, q)
        assert np.allclose(first, ad.jvp(KL_div, p, q, tangents=directions)[1])
        assert np.allclose(second, oracle, rtol=1e-5, atol=1e-7)
    _, _, mixed = ad.nested_jvp(KL_div, p, q, tangents={"p1": u}, nested_tangents={"p2": v})
    assert np.allclose(mixed, -np.sum(u * v / q))
    vec = {"p1": u, "p2": v}
    value, gradient, product = ad.value_grad_and_hvp(KL_div, p, q, wrt=("p1", "p2"), vector=vec)
    assert value == KL_div(p, q)
    assert np.allclose(product["p1"], u / p - v / q)
    assert np.allclose(product["p2"], -u / q + p * v / q**2)
    assert np.allclose(gradient["p1"], np.log(p / q) + 1)
    other = {"p1": 2 * v, "p2": -u}
    other_product = ad.hvp(KL_div, p, q, wrt=("p1", "p2"), vector=other)
    assert np.allclose(sum(np.vdot(other[k], product[k]) for k in vec), sum(np.vdot(vec[k], other_product[k]) for k in vec))


@pytest.mark.parametrize("a,u,v,dtype", [(0.8, .2, -.4, np.float64), (.8+.4j, .2-.7j, -.3+.1j, np.complex128)])
def test_coherent_second_order_oracle(a, u, v, dtype):
    value, first, mixed = ad.nested_jvp(coherent_state, a, 6, dtype=dtype, tangents={"a": u}, nested_tangents={"a": v})
    oracle = mixed_difference(lambda a: coherent_state(a, 6, dtype=dtype), {"a": a}, {"a": u}, {"a": v})
    assert value.dtype == dtype
    assert np.allclose(first, ad.jvp(coherent_state, a, 6, dtype=dtype, tangents={"a": u})[1])
    assert np.allclose(mixed, oracle, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("function,sign", [(commutator, -1), (anti_commutator, 1)])
def test_binary_analytic_zero_and_mixed(function, sign):
    rng = np.random.default_rng(5)
    a, b, u, v = [rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3)) for _ in range(4)]
    for key, direction in (("H1", u), ("H2", v)):
        _, _, second = ad.nested_jvp(function, a, b, tangents={key: direction})
        assert np.array_equal(second, np.zeros_like(second))
    _, _, mixed = ad.nested_jvp(function, a, b, tangents={"H1": u}, nested_tangents={"H2": v})
    assert np.allclose(mixed, u @ v + sign * v @ u)
    assert np.linalg.norm(mixed) > 0


def test_lincomb_analytic_zero_and_mixed():
    rng = np.random.default_rng(11)
    c, u = rng.normal(size=(2, 3))
    q, v = rng.normal(size=(2, 3, 4))
    for key, direction in (("coeff", u), ("Q_T", v)):
        _, _, second = ad.nested_jvp(lin_comb_Q_T, c, q, tangents={key: direction})
        assert np.array_equal(second, np.zeros_like(second))
    _, _, mixed = ad.nested_jvp(lin_comb_Q_T, c, q, tangents={"coeff": u}, nested_tangents={"Q_T": v})
    assert np.allclose(mixed, u @ v)


@pytest.mark.parametrize("down", [True, False])
def test_projection_every_input_and_mixed(down):
    rng = np.random.default_rng(13)
    obs = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
    shape = (3, 2) if down else (2, 3)
    proj = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    uo = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
    up = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    fun = lambda Obs, proj: project_op(Obs, proj)["Proj_Obs"]
    for directions in ({"Obs": uo}, {"proj": up}, {"Obs": uo, "proj": up}):
        _, _, mixed = ad.nested_jvp(project_op, obs, proj, tangents=directions)
        oracle = mixed_difference(fun, {"Obs": obs, "proj": proj}, directions, directions)
        assert set(mixed) == {"Proj_Obs"}
        assert np.allclose(mixed["Proj_Obs"], oracle, rtol=1e-5, atol=2e-6)
    _, _, mixed = ad.nested_jvp(project_op, obs, proj, tangents={"Obs": uo}, nested_tangents={"proj": up})
    expected = up.conj().T @ uo @ proj + proj.conj().T @ uo @ up if down else up @ uo @ proj.conj().T + proj @ uo @ up.conj().T
    assert np.allclose(mixed["Proj_Obs"], expected)


def test_ed_every_input_mixed_and_complex_duality():
    rng = np.random.default_rng(21)
    n, nt = 3, 4
    psi = rng.normal(size=n) + 1j*rng.normal(size=n)
    E = np.array([.2, .7, 1.1])
    V = np.linalg.qr(rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n)))[0]
    times = np.linspace(.1, .7, nt)
    directions = {"psi": rng.normal(size=n) + 1j*rng.normal(size=n), "E": rng.normal(size=n), "times": rng.normal(size=nt)}
    values = {"psi": psi, "E": E, "times": times}
    fun = lambda psi, E, times: ED_state_vs_time(psi, E, V, times)
    for direction in [*({k: v} for k, v in directions.items()), directions]:
        value, first, second = ad.nested_jvp(ED_state_vs_time, psi, E, V, times, tangents=direction)
        oracle = mixed_difference(fun, values, direction, direction)
        assert value.shape == first.shape == second.shape == (n, nt)
        assert np.allclose(second, oracle, rtol=1e-5, atol=2e-6)
    u, v = {"psi": directions["psi"]}, {"E": directions["E"], "times": directions["times"]}
    _, _, uv = ad.nested_jvp(ED_state_vs_time, psi, E, V, times, tangents=u, nested_tangents=v)
    _, _, vu = ad.nested_jvp(ED_state_vs_time, psi, E, V, times, tangents=v, nested_tangents=u)
    assert np.allclose(uv, vu)
    cotangent = rng.normal(size=(n, nt)) + 1j*rng.normal(size=(n, nt))
    scalar_oracle = mixed_difference(lambda **kw: np.real(np.vdot(cotangent, fun(**kw))), values, u, v)
    assert np.allclose(np.real(np.vdot(cotangent, uv)), scalar_oracle, rtol=1e-5, atol=2e-6)


def test_loschmidt_rate_second_directional_fixture():
    # Fixed spectrum/basis, smooth nonzero return amplitudes on this grid.
    psi = np.sqrt(np.array([.3, .7])).astype(complex)
    E, V, times = np.array([-.4, .7]), np.eye(2, dtype=complex), np.linspace(0, 1.2, 9)
    direction = np.array([.2, -.1])
    value, first, second = ad.nested_jvp(ED_state_vs_time, psi, E, V, times, tangents={"E": direction})
    amp, d_amp, dd_amp = psi.conj() @ value, psi.conj() @ first, psi.conj() @ second
    probability = np.abs(amp)**2
    d_probability = 2*np.real(amp.conj()*d_amp)
    dd_probability = 2*np.real(amp.conj()*dd_amp + d_amp.conj()*d_amp)
    result = np.mean((d_probability/probability)**2 - dd_probability/probability)
    # Independent two-level exact probability, no AD trajectory rule involved.
    def rate(E):
        exact_probability = .3**2 + .7**2 + 2*.3*.7*np.cos((E[1]-E[0])*times)
        return -np.mean(np.log(exact_probability))
    oracle = mixed_difference(rate, {"E": E}, {"E": direction}, {"E": direction})
    assert np.isfinite(result) and abs(result) > 1e-5
    assert np.allclose(result, oracle, rtol=1e-5, atol=1e-7)


def test_composed_complex_objective_hvp_duality():
    psi = np.array([1 + .2j, .3 - .4j])
    E = np.array([.2, .7])
    V = np.array([[1, .2j], [.1, .9j]])
    times = np.linspace(.1, .5, 4)
    weight = np.array([1., 2.])[:, None]
    objective = lambda E: np.real(np.vdot(
        quspin_ad.ED_state_vs_time(psi, E, V, times),
        weight * quspin_ad.ED_state_vs_time(psi, E, V, times),
    ))
    u, v = np.array([.1, -.2]), np.array([-.3, .4])
    value, first, mixed = ad.nested_jvp(objective, E, tangents={"E": u}, nested_tangents={"E": v})
    reverse = ad.nested_jvp(objective, E, tangents={"E": v}, nested_tangents={"E": u})[2]
    assert np.isfinite(value) and np.isfinite(first) and np.isfinite(mixed)
    assert np.allclose(mixed, reverse, rtol=1e-9, atol=1e-10)
    _, gradient, product = ad.value_grad_and_hvp(objective, E, wrt="E", vector=u)
    assert np.allclose(np.dot(v, product["E"]), mixed)
    assert np.allclose(np.dot(gradient["E"], u), first)


def test_second_order_boundaries_and_shapes():
    with pytest.raises(ad.NonDifferentiablePoint, match="a=0"):
        ad.nested_jvp(coherent_state, 0., 4, tangents={"a": 1.})
    with pytest.raises(ad.UnsupportedWrt):
        ad.nested_jvp(coherent_state, .8, 4, tangents={"n": 1.})
    with pytest.raises((ValueError, TypeError)):
        ad.nested_jvp(commutator, np.eye(2), np.eye(2), tangents={"H1": np.ones(3)})
    with pytest.raises(ad.NonDifferentiablePoint, match="out=None"):
        ad.nested_jvp(lin_comb_Q_T, np.ones(2), np.eye(2), out=np.empty(2), tangents={"coeff": np.ones(2)})
    with pytest.raises(ad.NonDifferentiablePoint, match="iterate=False"):
        ad.nested_jvp(ED_state_vs_time, np.ones(2), np.ones(2), np.eye(2), np.ones(2), iterate=True, tangents={"E": np.ones(2)})
    with pytest.raises(ad.UnsupportedWrt):
        ad.nested_jvp(ED_state_vs_time, np.ones(2), np.ones(2), np.eye(2), np.ones(2), tangents={"V": np.eye(2)})
    with pytest.raises((ValueError, TypeError)):
        ad.nested_jvp(KL_div, np.ones(2), np.ones(2), tangents={"p1": np.ones(3)})
