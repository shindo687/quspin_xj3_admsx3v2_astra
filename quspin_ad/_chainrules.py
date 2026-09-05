"""Small bundled ChainRules-compatible fallback for offline sidecar installs.

The official ``chainrules`` package is used whenever it is installed.  This
module implements the same narrow v0.1 protocol so that a wheel remains
usable in an isolated environment where that dependency is not mirrored.
It intentionally contains no numerical differentiation fallback.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Iterable, Mapping
from numbers import Real


class _Zero:
    __slots__ = ()

    def __repr__(self) -> str:
        return "ZERO"


ZERO = _Zero()


def _name(function: Callable[..., object]) -> str:
    return getattr(function, "__qualname__", repr(function))


class RuleNotFound(LookupError):
    def __init__(self, function: Callable[..., object], mode: str) -> None:
        super().__init__(f"No {mode.upper()} rule is registered for {_name(function)}")


class UnsupportedWrt(ValueError):
    def __init__(
        self,
        function: Callable[..., object],
        requested: Iterable[str],
        *,
        supported: Iterable[str] | None = None,
    ) -> None:
        self.function = function
        self.requested = tuple(sorted(requested))
        self.supported = None if supported is None else tuple(sorted(supported))
        message = (
            f"{_name(function)} does not support differentiation with respect to "
            f"{self.requested!r}"
        )
        if self.supported is not None:
            message += f"; supported inputs are {self.supported!r}"
        super().__init__(message)


class NonDifferentiablePoint(RuntimeError):
    pass


class RuleRegistry:
    def __init__(self) -> None:
        self._jvp: dict[int, tuple[Callable[..., object], Callable[..., object]]] = {}
        self._vjp: dict[int, tuple[Callable[..., object], Callable[..., object]]] = {}
        self._jvp2: dict[int, tuple[Callable[..., object], Callable[..., object]]] = {}

    def _register(self, table, function):
        key = id(function)

        def decorator(rule):
            if key in table:
                raise RuntimeError(
                    f"A rule is already registered for {_name(function)}"
                )
            table[key] = (function, rule)
            return rule

        return decorator

    def jvp_for(self, function):
        return self._register(self._jvp, function)

    def vjp_for(self, function):
        return self._register(self._vjp, function)

    def jvp2_for(self, function):
        """Register a mixed second directional-JVP rule.

        A rule returns ``(value, J(u), D[J(u)]·v)``.  Keeping this separate
        from the first-order registry makes second-order support explicit and
        prevents accidental numerical differentiation of a primal.
        """
        return self._register(self._jvp2, function)

    def _get(self, table, function, mode):
        entry = table.get(id(function))
        if entry is None or entry[0] is not function:
            raise RuleNotFound(function, mode)
        return entry[1]

    def get_jvp(self, function):
        return self._get(self._jvp, function, "JVP")

    def get_vjp(self, function):
        return self._get(self._vjp, function, "VJP")

    def get_jvp2(self, function):
        return self._get(self._jvp2, function, "second-order JVP")


rules = RuleRegistry()


def _signature_bind(function, args, kwargs):
    signature = inspect.signature(function)
    signature.bind(*args, **kwargs).apply_defaults()
    return signature


def _names(names, signature, label):
    names = (names,) if isinstance(names, str) else tuple(names)
    if not names:
        raise ValueError("wrt must contain at least one parameter name")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("every name must be a string parameter name")
    if len(set(names)) != len(names):
        raise ValueError("wrt must contain unique parameter names")
    unknown = set(names) - set(signature.parameters)
    if unknown:
        raise TypeError(f"Unknown {label} parameter names: {sorted(unknown)!r}")
    return names


def jvp(function, /, *args, tangents, **kwargs):
    if not isinstance(tangents, Mapping):
        raise TypeError("tangents must be a mapping from parameter names to values")
    signature = _signature_bind(function, args, kwargs)
    _names(tuple(tangents), signature, "tangent")
    if not tangents or all(value is ZERO for value in tangents.values()):
        return function(*args, **kwargs), ZERO
    result = rules.get_jvp(function)(dict(tangents), *args, **kwargs)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("A JVP rule must return a two-tuple")
    return result


def vjp(function, /, *args, wrt, **kwargs):
    signature = _signature_bind(function, args, kwargs)
    names = _names(wrt, signature, "wrt")
    result = rules.get_vjp(function)(names, *args, **kwargs)
    if not isinstance(result, tuple) or len(result) != 2 or not callable(result[1]):
        raise TypeError("A VJP rule must return (value, pullback)")
    value, raw = result

    def pullback(cotangent):
        if cotangent is ZERO:
            return dict.fromkeys(names, ZERO)
        output = raw(cotangent)
        if not isinstance(output, Mapping) or set(output) != set(names):
            raise TypeError("Pullback keys must exactly match wrt")
        return {name: output[name] for name in names}

    return value, pullback


def grad(function, /, *args, wrt, **kwargs):
    value, pullback = vjp(function, *args, wrt=wrt, **kwargs)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("grad requires a single real scalar output")
    return pullback(1.0)


def value_and_grad(function, /, *args, wrt, **kwargs):
    value, pullback = vjp(function, *args, wrt=wrt, **kwargs)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("value_and_grad requires a single real scalar output")
    return value, pullback(1.0)


def nested_jvp(
    function: Callable[..., object],
    /,
    *args,
    tangents: Mapping[str, object],
    nested_tangents: Mapping[str, object] | None = None,
    **kwargs,
):
    """Evaluate a registered JVP and its directional derivative.

    ``nested_tangents`` is the direction in which the first JVP is
    differentiated.  Omitting it computes a second directional derivative.
    The return value is ``(primal, first_tangent, mixed_tangent)``.
    """
    if not isinstance(tangents, Mapping):
        raise TypeError("tangents must be a mapping from parameter names to values")
    if nested_tangents is None:
        nested_tangents = tangents
    if not isinstance(nested_tangents, Mapping):
        raise TypeError("nested_tangents must be a mapping from parameter names to values")
    signature = _signature_bind(function, args, kwargs)
    _names(tuple(tangents), signature, "tangent")
    _names(tuple(nested_tangents), signature, "nested tangent")
    try:
        rule = rules.get_jvp2(function)
        result = rule(dict(tangents), dict(nested_tangents), *args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("A second-order JVP rule must return a three-tuple")
        return result
    except RuleNotFound:
        # Compositions of sidecar wrappers are traced with exact primitive
        # rules and ordinary analytic array operations; no finite differences.
        from ._second_order import evaluate
        value, first, _, mixed = evaluate(function, args, kwargs, tangents, nested_tangents)
        return value, first, mixed


def value_grad_and_hvp(function: Callable[..., object], /, *args, wrt, vector, **kwargs):
    """Return ``(value, gradient, Hessian @ vector)`` for scalar rules.

    The computation uses the registered analytic mixed JVP rule.  ``vector``
    is a mapping when multiple inputs are active, or a single array/scalar for
    one active input.  Complex inputs use the package's real-linear convention.
    """
    signature = _signature_bind(function, args, kwargs)
    names = _names(wrt, signature, "wrt")
    if isinstance(vector, Mapping):
        vectors = dict(vector)
    elif len(names) == 1:
        vectors = {names[0]: vector}
    else:
        raise TypeError("vector must map every active input when wrt has multiple names")
    if set(vectors) != set(names):
        raise ValueError("vector keys must exactly match wrt")
    try:
        value, gradient = value_and_grad(function, *args, wrt=names, **kwargs)
    except (RuleNotFound, TypeError):
        from ._second_order import evaluate
        import numpy as np
        signature_bound = signature.bind(*args, **kwargs)
        signature_bound.apply_defaults()
        gradient = {}
        value = function(*args, **kwargs)
        for name in names:
            current = signature_bound.arguments[name]
            arr = np.asarray(current)
            shape = arr.shape
            complex_input = np.iscomplexobj(arr)
            grad_arr = np.zeros(shape, dtype=np.result_type(arr, np.complex128 if complex_input else np.float64))
            for index in ([()] if shape == () else np.ndindex(shape)):
                for imaginary in ((False, True) if complex_input else (False,)):
                    basis = np.zeros(shape, dtype=np.result_type(arr, np.complex128 if imaginary else np.float64))
                    basis[index] = 1j if imaginary else 1
                    _, first, _, _ = evaluate(function, args, kwargs, {name: basis}, {})
                    component = float(np.real(first))
                    if shape == ():
                        grad_arr = grad_arr + (1j if imaginary else 1) * component
                    elif imaginary:
                        grad_arr[index] += 1j * component
                    else:
                        grad_arr[index] += component
            gradient[name] = grad_arr.item() if shape == () else grad_arr
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("value_grad_and_hvp requires a single real scalar output")

    hvp = {}
    for name in names:
        current = signature.bind(*args, **kwargs).arguments[name]
        array = current if hasattr(current, "shape") else current
        complex_input = hasattr(array, "dtype") and getattr(array.dtype, "kind", "") == "c"
        shape = getattr(array, "shape", ())
        import numpy as np
        result = np.zeros(shape, dtype=np.result_type(array, np.float64))
        indices = [()] if shape == () else list(np.ndindex(shape))
        for index in indices:
            for imaginary in ((False, True) if complex_input else (False,)):
                basis = np.zeros(shape, dtype=np.result_type(array, np.complex128 if imaginary else np.float64))
                basis[index] = 1j if imaginary else 1.0
                first = {key: ZERO for key in names}
                first[name] = basis.item() if shape == () else basis
                _, _, mixed = nested_jvp(
                    function, *args, tangents=first, nested_tangents=vectors, **kwargs
                )
                component = float(np.real(mixed))
                if shape == ():
                    if imaginary:
                        result = result + 1j * component
                    else:
                        result = result + component
                elif imaginary:
                    result[index] += 1j * component
                else:
                    result[index] += component
        hvp[name] = result.item() if shape == () else result
    return value, gradient, hvp


def hvp(function: Callable[..., object], /, *args, wrt, vector, **kwargs):
    """Return only the Hessian-vector mapping from :func:`value_grad_and_hvp`."""
    return value_grad_and_hvp(function, *args, wrt=wrt, vector=vector, **kwargs)[2]


__version__ = "0.1.0"
sys.modules.setdefault("chainrules", sys.modules[__name__])
