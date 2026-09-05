# quspin-ad

`quspin-ad` is a separately installable sidecar for QuSpin 1.0.1.  It registers
analytic first-order rules with the small `chainrules` protocol (or its bundled
compatible fallback when ChainRules is unavailable) while leaving
the upstream QuSpin callable and source tree unchanged.

Install it in a clean environment with:

```bash
python -m pip install .
```

Load the rules explicitly (registration never monkey-patches QuSpin):

```python
import quspin_ad  # registers the rules
import chainrules as ad
from quspin.tools.misc import KL_div

value, tangent = ad.jvp(
    KL_div, p1, p2,
    tangents={"p1": dp1, "p2": ad.ZERO},
)
value, pullback = ad.vjp(KL_div, p1, p2, wrt=("p1", "p2"))
gradients = pullback(1.0)
```

Second-order composition is available through the bundled sidecar interface.
`nested_jvp` returns `(value, J(u), D[J(u)]·v)` and uses the same analytic
primitive rules, so omitting `nested_tangents` computes a second directional
derivative.  `value_grad_and_hvp` returns `(value, gradient, H @ vector)` for
real scalar objectives; `hvp` returns only the last mapping.

```python
import quspin_ad as qad

loss = lambda p1, p2: qad.KL_div(p1, p2) ** 2
value, tangent, mixed = qad.nested_jvp(
    loss, p1, p2,
    tangents={"p1": dp1}, nested_tangents={"p2": dp2},
)
value, gradient, product = qad.value_grad_and_hvp(
    loss, p1, p2, wrt=("p1", "p2"),
    vector={"p1": dp1, "p2": dp2},
)
```

Composition is fixed-shape: use the `quspin_ad` callable inside a composed
objective, preserve the active input names, and avoid `out=` mutation,
branching on differentiable values, or conversions to `numpy.asarray` inside
the objective.  Primitive rules still reject unsupported inputs and the
documented non-differentiable boundaries.

Supported rules and their mathematical domains are specified in [SPEC.md](SPEC.md).
The package currently covers the continuous, array-valued APIs `KL_div`,
`coherent_state`, `commutator`, `anti_commutator`, `ED_state_vs_time`,
`lin_comb_Q_T`, and `project_op` (dense ndarray domain).  Discrete basis
construction, eigensolvers, entropy routines,
I/O, sparse/operator object methods, and non-array workflows are explicitly
reported as deferred or not suitable for AD rather than approximated by finite
differences.

The `upstream/` directory is a byte-for-byte snapshot of the official QuSpin
repository used for API inventory and tests; it is not imported by the wheel.
