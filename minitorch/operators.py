"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable
from typing import List, TypeVar

T = TypeVar("T")
U = TypeVar("U")
#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back


def mul(x: float, y: float) -> float:
    """Multiplies two numbers: x * y."""
    return x * y


def id(x: float) -> float:
    """Identity function: returns x."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers: x + y."""
    return x + y


def neg(x: float) -> float:
    """Negates a number: -x."""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if x is less than y: x < y."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if x is equal to y: x == y."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum of x and y."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if the absolute difference between x and y is within tolerance."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Sigmoid function: 1 / (1 + exp(-x)) if x >= 0, else exp(x) / (1 + exp(x)).

    Returns a value between 0 and 1.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """ReLU function: returns max(0, x)."""
    return max(0.0, x)


EPS = 1e-6


def log(x: float) -> float:
    """Natural logarithm of x. Raises ValueError if x <= 0."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Exponential function: exp(x)."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Backward pass for log: returns d / x. Raises ValueError if x <= 0."""
    return d / (x + EPS)


def inv(x: float) -> float:
    """Inverse function: 1 / x. Raises ValueError if x is 0."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Backward pass for inv: returns -d / (x^2). Raises ValueError if x is 0."""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Backward pass for ReLU: returns d if x > 0, else 0."""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# Core Functions
def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies the function `f` to each element in the list `lst` and returns a new list.

    Args:
    ----
    f : Callable[[T], U]
        The function to apply to each element of the list.
    lst : List[T]
        The input list of elements.

    Returns:
    -------
    List[U]
        A list with the results of applying `f` to each element of `lst`.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        return [f(x) for x in ls]

    return _map


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Applies the function `f` to pairs of elements from `lst1` and `lst2`, returning a new list.

    Args:
    ----
    f : Callable[[T, T], U]
        The function to apply to pairs of elements from both lists.
    lst1 : List[T]
        The first list of elements.
    lst2 : List[T]
        The second list of elements.

    Returns:
    -------
    List[U]
        A list with the results of applying `f` to pairs of elements from `lst1` and `lst2`.

    Raises:
    ------
    ValueError: If the two lists have different lengths.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [f(x, y) for x, y in zip(ls1, ls2)]

    return _zipWith


def reduce(
    f: Callable[[float, float], float], init: float
) -> Callable[[Iterable[float]], float]:
    """Reduces the list `lst` to a single value by applying the function `f` cumulatively,
    starting with the initial value `init`.

    Args:
    ----
    f : Callable[[T, T], T]
        The function to apply cumulatively to the elements of the list.
    init : T
        The initial value to start the reduction.
    lst : List[T]
        The input list of elements.

    Returns:
    -------
    T
        The final reduced value after applying `f` to all elements of `lst`.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = init
        for x in ls:
            val = f(val, x)
        return val

    return _reduce


# Derived Functions
def negList(lst: Iterable[float]) -> Iterable[float]:
    """Returns a new list with each element negated.

    Args:
    ----
    lst : List[float]
        The input list of floating-point numbers.

    Returns:
    -------
    List[float]
        A list where each element is the negation of the corresponding element in `lst`.

    """
    return map(neg)(lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Returns a new list by adding corresponding elements from `lst1` and `lst2`.

    Args:
    ----
    lst1 : List[float]
        The first list of floating-point numbers.
    lst2 : List[float]
        The second list of floating-point numbers.

    Returns:
    -------
    List[float]
        A list where each element is the sum of corresponding elements from `lst1` and `lst2`.

    Raises:
    ------
    ValueError: If the two lists have different lengths.

    """
    return zipWith(add)(lst1, lst2)


def sum(lst: List[float]) -> float:
    """Sums all elements in the list."""
    return reduce(add, 0.0)(lst)


def prod(lst: List[float]) -> float:
    """Computes the product of all elements in the list."""
    return reduce(mul, 1.0)(lst)
