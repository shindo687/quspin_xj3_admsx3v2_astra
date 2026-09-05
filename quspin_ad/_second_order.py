"""Fixed-shape, two-direction forward composition (no numerical differences).

The four coefficients represent x, D_u x, D_v x, and D_v D_u x.
QuSpin primitives are evaluated through their registered analytic rules;
only ordinary smooth array arithmetic is implemented here.
"""

from functools import wraps
import inspect

import numpy as np


class Jet:
    __array_priority__ = 1000

    def __init__(self, value, u=0, v=0, uv=0):
        self.value = np.asarray(value)
        self.u = np.broadcast_to(u, self.value.shape)
        self.v = np.broadcast_to(v, self.value.shape)
        self.uv = np.broadcast_to(uv, self.value.shape)

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    @property
    def dtype(self):
        return self.value.dtype

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Converting a differentiable value to ndarray discards derivatives; use supported array operations")

    def __bool__(self):
        raise TypeError("Branching on differentiable values is outside fixed-shape composition")

    def __getitem__(self, key):
        return Jet(self.value[key], self.u[key], self.v[key], self.uv[key])

    def __len__(self):
        return len(self.value)

    def _linear(self, operation):
        return Jet(*(operation(x) for x in (self.value, self.u, self.v, self.uv)))

    @property
    def real(self):
        return self._linear(np.real)

    @property
    def imag(self):
        return self._linear(np.imag)

    @property
    def T(self):
        return self.transpose()

    def conjugate(self):
        return self._linear(np.conj)

    conj = conjugate

    def transpose(self, *axes):
        return self._linear(lambda x: x.transpose(*axes))

    def reshape(self, *shape, **kwargs):
        return self._linear(lambda x: x.reshape(*shape, **kwargs))

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        if out is not None:
            raise TypeError("out mutation is unsupported during differentiation")
        return self._linear(lambda x: np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims))

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        if out is not None:
            raise TypeError("out mutation is unsupported during differentiation")
        return self._linear(lambda x: np.mean(x, axis=axis, dtype=dtype, keepdims=keepdims))

    def __add__(self, other):
        other = as_jet(other)
        return Jet(self.value + other.value, self.u + other.u, self.v + other.v, self.uv + other.uv)

    __radd__ = __add__

    def __neg__(self):
        return self._linear(np.negative)

    def __sub__(self, other):
        return self + (-as_jet(other))

    def __rsub__(self, other):
        return as_jet(other) + (-self)

    def _bilinear(self, other, operation):
        other = as_jet(other)
        return Jet(
            operation(self.value, other.value),
            operation(self.u, other.value) + operation(self.value, other.u),
            operation(self.v, other.value) + operation(self.value, other.v),
            operation(self.uv, other.value) + operation(self.u, other.v)
            + operation(self.v, other.u) + operation(self.value, other.uv),
        )

    def __mul__(self, other):
        return self._bilinear(other, np.multiply)

    __rmul__ = __mul__

    def __matmul__(self, other):
        return self._bilinear(other, np.matmul)

    def __rmatmul__(self, other):
        return as_jet(other)._bilinear(self, np.matmul)

    def _unary(self, value, first, second):
        return Jet(value, first * self.u, first * self.v, first * self.uv + second * self.u * self.v)

    def __pow__(self, exponent):
        if isinstance(exponent, Jet):
            return np.exp(exponent * np.log(self))
        if np.ndim(exponent) != 0:
            return np.exp(as_jet(exponent) * np.log(self))
        if exponent == 0:
            return Jet(np.ones_like(self.value))
        if exponent == 1:
            return self
        return self._unary(self.value**exponent, exponent * self.value**(exponent-1), exponent*(exponent-1)*self.value**(exponent-2))

    def __rpow__(self, base):
        return np.exp(self * np.log(base))

    def __truediv__(self, other):
        return self * (as_jet(other)**-1)

    def __rtruediv__(self, other):
        return as_jet(other) * self**-1

    def __abs__(self):
        if np.any(self.value == 0):
            raise ValueError("absolute value is not differentiable at zero; use (x.conj()*x).real for squared magnitude")
        return ((self.conj() * self).real)**.5

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            raise TypeError(f"Unsupported differentiation operation: {ufunc.__name__}.{method}")
        binary = {
            np.add: lambda a, b: a + b, np.subtract: lambda a, b: a - b,
            np.multiply: lambda a, b: a * b, np.divide: lambda a, b: a / b,
            np.matmul: lambda a, b: a @ b,
        }
        if ufunc in binary:
            return binary[ufunc](*(as_jet(x) for x in inputs))
        if ufunc is np.power:
            return as_jet(inputs[0]) ** inputs[1]
        x = as_jet(inputs[0])
        if ufunc is np.negative: return -x
        if ufunc is np.positive: return x
        if ufunc is np.conjugate: return x.conj()
        if ufunc is np.absolute: return abs(x)
        if ufunc is np.square: return x**2
        if ufunc is np.sqrt: return x**.5
        if ufunc is np.reciprocal: return x**-1
        if ufunc is np.exp:
            value = np.exp(x.value)
            return x._unary(value, value, value)
        if ufunc is np.log:
            return x._unary(np.log(x.value), 1/x.value, -1/x.value**2)
        if ufunc is np.sin:
            return x._unary(np.sin(x.value), np.cos(x.value), -np.sin(x.value))
        if ufunc is np.cos:
            return x._unary(np.cos(x.value), -np.sin(x.value), -np.cos(x.value))
        raise TypeError(f"No analytic composition rule for numpy.{ufunc.__name__}")

    def __array_function__(self, func, types, args, kwargs):
        if func is np.real: return as_jet(args[0]).real
        if func is np.imag: return as_jet(args[0]).imag
        if func is np.sum: return as_jet(args[0]).sum(*args[1:], **kwargs)
        if func is np.mean: return as_jet(args[0]).mean(*args[1:], **kwargs)
        if func is np.reshape: return as_jet(args[0]).reshape(*args[1:], **kwargs)
        if func is np.transpose:
            axes = args[1] if len(args) > 1 else kwargs.get("axes")
            return as_jet(args[0]).transpose() if axes is None else as_jet(args[0]).transpose(axes)
        if func is np.vdot:
            if kwargs: raise TypeError("vdot keyword arguments are unsupported")
            return (as_jet(args[0]).conj().reshape(-1) * as_jet(args[1]).reshape(-1)).sum()
        if func is np.dot:
            if kwargs: raise TypeError("dot keyword arguments are unsupported")
            return as_jet(args[0])._bilinear(args[1], np.dot)
        raise TypeError(f"No analytic composition rule for numpy.{func.__name__}")


def as_jet(value):
    return value if isinstance(value, Jet) else Jet(value)


def unpack(result):
    """Extract coefficients while retaining array/dictionary/tuple structure."""
    if isinstance(result, Jet):
        return tuple(x.item() if x.ndim == 0 else x for x in (result.value, result.u, result.v, result.uv))
    if isinstance(result, dict):
        parts = {key: unpack(value) for key, value in result.items()}
        return tuple({key: value[i] for key, value in parts.items()} for i in range(4))
    if isinstance(result, (tuple, list)):
        parts = [unpack(value) for value in result]
        return tuple(type(result)(value[i] for value in parts) for i in range(4))
    value = np.asarray(result)
    return result, np.zeros_like(value), np.zeros_like(value), np.zeros_like(value)


def evaluate(function, args, kwargs, u, v):
    from .rules import ad
    signature = inspect.signature(function)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    for name in set(u) | set(v):
        value = np.asarray(bound.arguments[name])
        directions = []
        for mapping in (u, v):
            direction = mapping.get(name, ad.ZERO)
            if direction is ad.ZERO:
                direction = np.zeros_like(value)
            direction = np.asarray(direction)
            if direction.shape != value.shape:
                raise ValueError(f"{name} direction shape {direction.shape} does not match {value.shape}")
            if not np.iscomplexobj(value) and np.iscomplexobj(direction):
                raise TypeError(f"{name} is real and requires a real direction")
            directions.append(direction)
        bound.arguments[name] = Jet(value, *directions)
    return unpack(function(*bound.args, **bound.kwargs))


def traceable(function):
    """Dispatch tracked wrapper calls through the exact primitive rules."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if not any(isinstance(x, Jet) for x in (*args, *kwargs.values())):
            return function(*args, **kwargs)
        from . import _chainrules
        from .rules import ad
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        tracked = {name: x for name, x in bound.arguments.items() if isinstance(x, Jet)}
        for name, x in tracked.items():
            bound.arguments[name] = x.value.item() if x.value.ndim == 0 else x.value
        u, v, uv = ({name: getattr(x, field) for name, x in tracked.items()} for field in ("u", "v", "uv"))
        # Mixed propagation includes curvature from the primitive and the
        # first-order pushforward of curvature arriving from earlier nodes.
        value, du, mixed = _chainrules.rules.get_jvp2(wrapped)(u, v, *bound.args, **bound.kwargs)
        _, dv = ad.rules.get_jvp(wrapped)(v, *bound.args, **bound.kwargs)
        _, arriving = ad.rules.get_jvp(wrapped)(uv, *bound.args, **bound.kwargs)
        def pack(value, du, dv, mixed, arriving):
            if isinstance(value, dict):
                return {k: pack(value[k], du[k], dv[k], mixed[k], arriving[k]) for k in value}
            return Jet(value, du, dv, mixed + arriving)
        return pack(value, du, dv, mixed, arriving)
    return wrapped
