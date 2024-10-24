from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage

    """
    position = 0
    for i, stride in zip(index, strides):
        # Handle negative indices (Python-style indexing)
        if i < 0:
            i += stride  # Adjust negative index based on stride size
        position += i * stride
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    for i in reversed(range(len(shape))):  # Work backwards from the last dimension
        out_index[i] = ordinal % shape[i]
        ordinal //= shape[i]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None

    """
    # Align both shapes by padding smaller shape with leading 1s
    offset = len(big_shape) - len(shape)
    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = 0  # Broadcasted dimensions always index at 0
        else:
            out_index[i] = big_index[i + offset]


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast

    """
    # Start from the rightmost dimension and broadcast each pair
    result = []
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)

    # Iterate from the last dimension towards the first
    for i in range(1, max_len + 1):
        dim1 = shape1[-i] if i <= len1 else 1
        dim2 = shape2[-i] if i <= len2 else 1

        if dim1 == 1 or dim2 == 1 or dim1 == dim2:
            result.append(max(dim1, dim2))
        else:
            raise IndexingError(f"Cannot broadcast dimensions {dim1} and {dim2}")

    return tuple(reversed(result))


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a shape"""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))  # type: ignore
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert to cuda"""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    # @staticmethod
    # def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
    #     return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Calculate the flat index for a given multi-dimensional index.

        Args:
            index (Union[int, UserIndex]): The multi-dimensional index or a single integer index.

        Returns:
            int: The flat index in the storage.

        Raises:
            IndexingError: If the index is not valid for the tensor shape.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:
            aindex = array(index)

        shape = self.shape

        # Special case: handle 0-dimensional (scalar) tensors
        if len(shape) == 0:
            if (len(aindex) == 1 and aindex[0] == 0) or len(aindex) == 0:
                return 0
            else:
                raise IndexingError(f"Index {aindex} is not valid for a scalar tensor.")

        # Check for dimension mismatch
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")

        # Check for out-of-bounds index values
        for dim_idx, (idx, dim_size) in enumerate(zip(aindex, self.shape)):
            if idx < 0 or idx >= dim_size:
                raise IndexingError(
                    f"Index {idx} is out of bounds for dimension {dim_idx} with size {dim_size}."
                )

        # Compute the flat index using the strides
        return sum(aindex[i] * self.strides[i] for i in range(len(aindex)))

    def indices(self) -> Iterable[UserIndex]:
        """Yields all possible indices for the tensor shape."""
        lshape: Shape = array(self.shape)
        out_index: Index = array([0] * len(self.shape))  # Initialize to all zeroes

        for ordinal in range(self.size):  # Iterate over all possible indices
            to_index(ordinal, lshape, out_index)
            yield tuple(out_index)  # Yield the tuple of the current index

    def sample(self) -> UserIndex:
        """Get a random valid index"""
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Retrieve the value at the specified index."""
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set the value at the specified index."""
        self._storage[self.index(key)] = val

    def fill(self, value: float) -> None:
        """Fill the storage of the tensor with the specified value."""
        for idx in range(self.size):
            # Directly set the value in the storage using the flat index
            self._storage[idx] = value

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        """Convert to string"""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
