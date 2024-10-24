from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, central_difference, backpropagate
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        """Automatically called after the Scalar is initialized, setting unique IDs and
        ensuring the scalar data is a float.
        """
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        """String representation of the Scalar."""
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Multiply two Scalars."""
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Divide two Scalars."""
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Handle reverse division (b / self)."""
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        """Boolean representation of the scalar value."""
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Handle reverse addition (b + self)."""
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Handle reverse multiplication (b * self)."""
        return self * b

    def __hash__(self) -> int:
        """Make Scalar hashable so it can be used in sets and as dictionary keys."""
        return hash(self.unique_id)  # Use the unique_id as the hash value

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the Scalar is a constant (no history).

        Returns
        -------
        bool
            True if the scalar is constant and does not require gradients.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent Scalars that were used to compute this Scalar."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Performs the chain rule to propagate the derivative to each input.

        Args:
        ----
        d_output: The derivative of the output with respect to some downstream quantity.

        Returns:
        -------
        Iterable of (variable, derivative) pairs for each input variable.

        """
        h = self.history
        assert h is not None, "History is required to apply the chain rule."
        assert h.last_fn is not None, "A function is required to apply the chain rule."
        assert h.ctx is not None, "Context is required to apply the chain rule."

        # Compute local derivatives using the backward function
        local_derivatives = h.last_fn.backward(h.ctx, d_output)  # type: ignore

        return list(zip(h.inputs, local_derivatives))

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this project"""
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Addition of two Scalars or a Scalar and a number."""
        return Add.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtraction of two Scalars or a Scalar and a number."""
        return Add.apply(self, -b)

    def __neg__(self) -> Scalar:
        """Negation of a Scalar."""
        return Neg.apply(self)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Less than comparison between two Scalars."""
        return LT.apply(self, b)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Equality comparison between two Scalars."""
        return EQ.apply(self, b)

    def log(self) -> Scalar:
        """Natural logarithm of the Scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Exponential of the Scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Sigmoid function of the Scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """ReLU function of the Scalar."""
        return ReLU.apply(self)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Args:
    ----
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
