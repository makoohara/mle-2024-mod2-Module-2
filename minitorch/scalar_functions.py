from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Wrapper method for backward pass.

        Args:
            ctx (Context): Context object storing intermediate values.
            d_out (float): Derivative of the output with respect to some downstream value.

        Returns:
            Tuple[float, ...]: Derivatives with respect to inputs.

        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Wrapper method for forward pass.

        Args:
            ctx (Context): Context object.
            *inps (float): Input values for the function.

        Returns:
            float: Output of the forward pass.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the function to input Scalar values.

        Args:
            *vals (ScalarLike): Input values, which can be Scalars or floats.

        Returns:
            Scalar: Result of applying the function to inputs.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(float(v.data))
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(float(v))

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition.

        Args:
            ctx (Context): Context object.
            a (float): First input.
            b (float): Second input.

        Returns:
            float: Sum.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition.

        Args:
            ctx (Context): Context object.
            d_output (float): Derivative of the output.

        Returns:
            Tuple[float, ...]: Derivatives with respect to inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for log function.

        Args:
            ctx (Context): Context object.
            a (float): Input value.

        Returns:
            float: log(a).

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for log function.

        Args:
            ctx (Context): Context object.
            d_output (float): Derivative of the output.

        Returns:
            Tuple[float]: Derivative with respect to the input.

        """
        (a,) = ctx.saved_values
        return (operators.log_back(a, d_output),)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication.

        Args:
            ctx (Context): Context object.
            a (float): First input.
            b (float): Second input.

        Returns:
            float: product.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication.

        Args:
            ctx (Context): Context object.
            d_output (float): Derivative of the output.

        Returns:
            Tuple[float, float]: Derivatives with respect to inputs.

        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse.

        Args:
            ctx (Context): Context object.
            a (float): Input value.

        Returns:
            float: Inverse of the input.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for inverse.

        Args:
            ctx (Context): Context object.
            d_output (float): Derivative of the output.

        Returns:
            Tuple[float]: Derivative with respect to the input.

        """
        (a,) = ctx.saved_values
        return (operators.inv_back(a, d_output),)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation.

        Args:
            ctx (Context): Context object.
            a (float): Input value.

        Returns:
            float: Negation of the input.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for negation.

        Args:
            ctx (Context): Context object.
            d_output (float): Derivative of the output.

        Returns:
            Tuple[float]: Derivative with respect to the input.

        """
        return (-d_output,)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation.

        Args:
            ctx (Context): Context object.
            a (float): Input value.

        Returns:
            float: Sigmoid of the input.

        """
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for sigmoid.

        Args:
            ctx (Context): Context object.
            d_output (float): Derivative of the output.

        Returns:
            Tuple[float]: Derivative with respect to the input.

        """
        (a,) = ctx.saved_values
        sigmoid_a = operators.sigmoid(a)
        return (d_output * sigmoid_a * (1 - sigmoid_a),)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU.

        Args:
            ctx (Context): Context object.
            a (float): Input value.

        Returns:
            float: ReLU of the input.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for ReLU.

        Args:
            ctx (Context): Context object.
            d_output (float): Derivative of the output.

        Returns:
            Tuple[float]: Derivative with respect to the input.

        """
        (a,) = ctx.saved_values
        return (operators.relu_back(a, d_output),)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential.

        Args:
            ctx (Context): Context object.
            a (float): Input value.

        Returns:
            float: Exponential of the input.

        """
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for exponential.

        Args:
            ctx (Context): Context object.
            d_output (float): Derivative of the output.

        Returns:
            Tuple[float]: Derivative with respect to the input.

        """
        (result,) = ctx.saved_values
        return (d_output * result,)


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than comparison.

        Args:
            ctx (Context): Context object.
            a (float): First input.
            b (float): Second input.

        Returns:
            float: Result of comparison.

        """
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than comparison.

        Args:
            ctx (Context): Context object.
            d_output (float): Derivative of the output.

        Returns:
            Tuple[float, float]: Zero derivatives, since the function is non-differentiable.

        """
        return 0.0, 0.0  # No gradient for boolean comparisons


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality comparison.

        Args:
            ctx (Context): Context object.
            a (float): First input.
            b (float): Second input.

        Returns:
            float: Result of comparison.

        """
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality comparison.

        Args:
            ctx (Context): Context object.
            d_output (float): Derivative of the output.

        Returns:
            Tuple[float, float]: Zero derivatives, since the function is non-differentiable.

        """
        return 0.0, 0.0  # No gradient for boolean comparisons
