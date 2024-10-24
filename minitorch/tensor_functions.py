"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple, Optional

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        # Too strict of a test
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Any) -> Tensor:
        """Call the forward function and track history."""
        from .tensor import Tensor  # Local import to avoid circular import

        raw_vals = []
        need_grad = False

        for v in vals:
            # Convert numbers to tensors if necessary
            if isinstance(v, (int, float)):
                v = tensor(v)
            if isinstance(v, Tensor) and v.requires_grad():
                need_grad = True
            # Store the raw value for forward without history
            raw_vals.append(v.detach() if isinstance(v, Tensor) else v)

        ctx = Context()

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, Tensor), f"Expected return type Tensor, got {type(c)}"

        if need_grad:
            # If any input needs gradient, track the operation
            back = minitorch.History(last_fn=cls, ctx=ctx, inputs=vals)
        else:
            back = None

        # Create a new tensor with the correct history
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the negative of a tensor element-wise."""
        ctx.save_for_backward(t1)
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward path"""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the element-wise inverse of a tensor."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the inverse operation."""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise addition of two tensors."""
        # Ensure t2 is properly handled as a tensor if needed
        t2 = t1._ensure_tensor(t2)
        ctx.save_for_backward(t1, t2)
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the addition operation."""
        t1, t2 = ctx.saved_values
        # Handle broadcasting if t1 and t2 have different shapes
        grad_t1 = grad_output
        grad_t2 = grad_output

        if t1.shape != grad_t1.shape:
            grad_t1 = grad_t1.expand(t1)
        if t2.shape != grad_t2.shape:
            grad_t2 = grad_t2.expand(t2)
        # if t1.shape != t2.shape:
        #     grad_t1 = grad_t1.expand(t1)
        #     grad_t2 = grad_t2.expand(t2)

        return grad_t1, grad_t2


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all are true."""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            shape = a.shape
            return a.f.mul_reduce(
                a.contiguous().view(int(operators.prod(list(shape)))), 0
            )


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise multiplication of two tensors."""
        t2 = t1._ensure_tensor(t2)
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)  # Use tensor_zip with multiplication function

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the multiplication operation"""
        t1, t2 = ctx.saved_values
        # Gradient with respect to t2 is grad_output * t1
        return grad_output.f.mul_zip(grad_output, t2), grad_output.f.mul_zip(
            grad_output, t1
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Apply the sigmoid function element-wise to the input tensor."""
        ctx.save_for_backward(t.f.sigmoid_map(t))
        return t.f.sigmoid_map(t)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass"""
        (sig,) = ctx.saved_values
        one = ones(sig.shape)
        one_minus_sig = sig.f.add_zip(sig.f.neg_map(sig), one)
        return grad_output.f.mul_zip(sig, one_minus_sig)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """ReLU forward"""
        ctx.save_for_backward(t)
        return t.f.relu_map(t)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the ReLU function."""
        (t,) = ctx.saved_values
        # zero_grad = zeros(t.shape, backend=t.backend)
        return grad_output.f.relu_back_zip(
            t, grad_output
        )  # if (t > tensor(0, backend=t.backend)).item() else zero_grad


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Apply the natural logarithm function element-wise to the input tensor."""
        ctx.save_for_backward(t)
        return t.f.log_map(t)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the logarithm function."""
        (t,) = ctx.saved_values
        # Gradient is 1 / t
        return grad_output.f.inv_map(t)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Apply the exponential function element-wise to the input tensor."""
        ctx.save_for_backward(t)
        return t.f.exp_map(t)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tensor:
        """Compute the gradient of the exponential function."""
        (t,) = ctx.saved_values
        # Gradient is exp(t)
        return grad_output.f.exp_map(t)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor, dim: Optional[Tensor]) -> Tensor:
        """Compute the sum of elements in a tensor along a specified dimension.

        Args:
            ctx: Context for backward computation.
            t: Input tensor.
            dim: Dimension along which to sum the elements.

        Returns:
            A tensor with summed elements along the specified dimension.

        """
        int_dim = int(dim.item())  # type: ignore
        ctx.save_for_backward(t.shape, int_dim)
        return t.f.add_reduce(t, int_dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Any]:
        """Compute the gradient of the sum operation."""
        original_shape, int_dim = ctx.saved_values
        shape = grad_output.shape
        expanded_grad = grad_output.view(*shape[:int_dim], 1, *shape[int_dim:])
        # grad_input = grad_output_reshaped.f.mul_zip(grad_output_reshaped, ones(original_shape))
        grad_input = expanded_grad.expand(*original_shape)
        return grad_input, grad_output.zeros()
        # original_shape, int_dim = ctx.saved_values

        # # Expand grad_output to the shape of the original tensor
        # grad_output = grad_output.view(*grad_output.shape[:int_dim], 1, *grad_output.shape[int_dim:])
        # grad_output = grad_output.expand(*original_shape)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Foward pass for LT"""
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for EQ (no gradient flow)."""
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for EQ (no gradient flow)."""
        shape1, shape2 = ctx.saved_values
        return zeros(shape1), zeros(shape2)

class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Close comparison"""
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor, *order: int) -> Tensor:
        """Permute the dimensions of a tensor based on a specified order."""
        from .tensor import Tensor

        order = tuple(int(o.item()) if isinstance(o, Tensor) else int(o) for o in order)
        ctx.save_for_backward(t.shape, order)
        return t._new(t._tensor.permute(*order))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Tensor, ...]:
        """Compute the gradient of the permute operation."""
        order, orig_shape = ctx.saved_values
        # Ensure `order` is a tuple of integers, not a Tensor
        inverse_order = sorted(range(len(orig_shape)), key=lambda x: order[x])
        grad = grad_output._tensor.permute(*inverse_order)
        return minitorch.Tensor.make(
            grad._storage,
            grad.shape,
            backend=grad_output.backend,
        ), zeros(grad.shape)


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshape a tensor to a new shape without changing its data."""
        # Ensure 'shape' is a list/tuple of integers, not a Tensor

        shape2 = [int(shape[i]) for i in range(shape.size)]
        ctx.save_for_backward(a.shape)  # Save the original shape for the backward pass
        assert a._tensor.is_contiguous(), "Must be contiguous to view"

        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Return the gradient with respect to the original tensor in the correct shape."""
        (original_shape,) = ctx.saved_values  # Retrieve the original shape
        # Reshape the grad_output back to the original shape
        # zero_grad = zeros(original_shape, backend=t.backend)
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage,
                original_shape,
                backend=grad_output.backend,
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))  # type: ignore
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (grad_output @ transpose(t2), transpose(t1) @ grad_output)


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(list(shape))), shape, backend=backend
    )


def ones(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [1.0] * int(operators.prod(list(shape))), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(list(shape))))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    if isinstance(ls, (int, float)):
        return _tensor([ls], (1,), backend=backend, requires_grad=requires_grad)

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Gradient calculation through central difference"""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
