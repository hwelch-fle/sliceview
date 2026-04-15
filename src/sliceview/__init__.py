"""
sliceview — Zero-copy slice views for Python sequences.

A sliceview presents a live window into an existing sequence:
reads and writes reflect the underlying sequence, view-to-view
slicing composes in O(1), and no data is copied unless explicitly
requested.

Basic usage::

    from sliceview import sliceview

    big = list(range(1_000_000))

    # O(1) window — no copy
    window = sliceview(big)[1000:2000]

    # Compose slices — still O(1)
    every_other = sliceview(big)[::2][500:1000]

    # In-place windowed update
    sv = sliceview(big)
    sv[0:10] = range(10, 20)

    # Sliding window without creating new objects
    view = sliceview(big, 0, 100)
    for _ in range(10):
        view.advance(100)
"""

from __future__ import annotations

from collections.abc import (
    Iterable, 
    Sequence, 
    Iterator, 
    MutableSequence,
)

from typing import (
    Any, 
    Protocol, 
    SupportsIndex, 
    TypeGuard, 
    overload, 
    runtime_checkable,
)


__all__ = ["sliceview"]
__version__ = "0.1.0"


@runtime_checkable
class SliceViewable(Protocol):
    """sliceview requires Sequence interface"""
    def __len__(self): ...
    def __getattr__(self, name: str): ...


def range_to_slice(r: range) -> slice:
    """Convert a range to a slice"""
    start, stop, step = r.start, r.stop, r.step

    # range(1, 0, 1) -> slice(1, None, 1) 
    _inverted = stop < start and step > 0
    # range(0, 1, -1) -> slice(0, None, -1)
    _reversed = stop > start and step < 0
    # range(n, -1, -1) -> slice(n, None, -1)
    _from_end = step < 0 and stop == -1
    if _inverted or _reversed or _from_end:
        stop = None
    
    return slice(start, stop, step)


def slice_to_range(s: slice, length: int) -> range:
    """Convert a slice to a range for a sequence of length `length`"""
    return range(*s.indices(length))


def clamp_range(r: range, length: int) -> range:
    """Create a new range clamped to `length`"""
    stop = length if r.step > 0 else -1
    return range(r.start, stop, r.step)


def guard_type[T](obj: object, typ: type[T]) -> TypeGuard[T]:
    """isinstance but with generic preservation (type checking only!)"""
    typ = typ.mro()[0]
    return isinstance(obj, typ)


class sliceview[T](Sequence[T]):
    """A zero-copy, composable slice view over any :class:`collections.abc.Sequence`.

    Parameters
    ----------
    base:
        The underlying sequence.  Any object that implements
        ``__len__`` and ``__getitem__`` with integer indices is accepted.
    start, stop, step:
        Slice parameters (same semantics as the built-in :class:`slice`).
        *start* may alternatively be a :class:`slice` object, in which case
        *stop* and *step* must be omitted.

    Examples
    --------
    >>> sv = sliceview([0, 1, 2, 3, 4, 5])
    >>> list(sv[1:4])
    [1, 2, 3]
    >>> list(sv[::2])
    [0, 2, 4]
    >>> sv2 = sv[1:][::2]   # composed — O(1), no copy
    >>> list(sv2)
    [1, 3, 5]
    """

    __slots__ = ("_base", "_range", "_unbound")

    @overload
    def __init__(self, base: Sequence[T]) -> None: ...
    @overload
    def __init__(self, base: Sequence[T], start: slice) -> None: ...
    @overload
    def __init__(self, base: Sequence[T], start: int, stop: int) -> None: ...
    @overload
    def __init__(self, base: Sequence[T], start: int, stop: int, step: int) -> None: ...
    @overload
    def __init__(self, base: Sequence[T], start: int, stop: None, step: int) -> None: ...

    def __init__(self, base: Sequence[T], 
                 start : int | slice | None = None, 
                 stop: int | None = None, 
                 step: int | None = None,
        ) -> None:
        if isinstance(base, SliceViewable):
            raise TypeError(
                f"sliceview requires a sequence with __len__ and __getitem__, "
                f"got {type(base).__name__!r}"
            )
        self._base = base
        
        if isinstance(start, slice):
            if (stop, step) != (None, None):
                raise ValueError(
                    'sliceview initialized with slice must not have stop/step arguments'
                )
            sl: slice = start
        else:
            sl = slice(start, stop, step)
        
        self._unbound = sl.stop is None
        self._range = slice_to_range(sl, len(base))

    @classmethod
    def from_range(cls, base: Sequence[T], r: range) -> sliceview[T]:
        """Internal constructor: build a view directly from a range object."""
        sv = cls.__new__(cls)
        sv._base = base
        sv._range = r
        sv._unbound = r.stop < 0
        return sv

    @property
    def base(self) -> Sequence[T] | MutableSequence[T]:
        """The base sequence of the sliceview"""
        return self._base

    @property
    def range(self) -> range:
        """Return a concrete `range` clamped to the current base length."""
        return (
            clamp_range(self._range, len(self.base)) 
            if self._unbound 
            else self._range
        )

    @property
    def slice(self) -> slice:
        """Return a slice object representing the current view"""
        return range_to_slice(self.range)

    def __len__(self) -> int:
        return len(self.range)

    @overload
    def __getitem__(self, index: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> sliceview[T]: ...

    def __getitem__(self, index: object) -> T | sliceview[T]:
        if not isinstance(index, (slice, SupportsIndex)):
            raise TypeError(
                f'{type(self).__name__} indices must be integers or slices, '
                f'not {type(index).__name__}'
            )

        if isinstance(index, slice):
            return sliceview[T].from_range(self.base, self.range[index])
        
        else:
            index = index.__index__()
            if index < 0:
                index += len(self.range)
            if not (0 <= index < len(self.range)):
                raise IndexError("sliceview index out of range")
            return self.base[self.range[index]]

    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    
    def __setitem__(self, index: object, value: T | Iterable[T]) -> None:
        if not isinstance(self._base, MutableSequence):
            raise TypeError("underlying sequence is not mutable")

        if not isinstance(index, (slice, SupportsIndex)):
            raise TypeError(
                f'{type(self).__name__} indices must be integers or slices, '
                f'not {type(index).__name__}'
            )

        if isinstance(index, slice):
            if not guard_type(value, Iterable[T]):
                raise TypeError('can only assign an iterable')
            self._base[range_to_slice(self.range[index])] = value
            return
        
        index = index.__index__()
        length = len(self.range)
        if index < 0:
            index += length
        if index not in range(length):
            raise IndexError("sliceview index out of range")
        self._base[self.range[index]] = value # type: ignore

    def __iter__(self) -> Iterator[T]:
        return iter(self.base[self.slice])

    def __contains__(self, item: object) -> bool:
        return any(item == el for el in self)

    def __reversed__(self) -> Iterator[T]:
        return (self.base[i] for i in reversed(self.range))

    def __eq__(self, other: Sequence[T] | object) -> bool:
        if guard_type(other, Sequence[T]) and len(self) == len(other):
            return all(a == b for a, b in zip(self, other))
        return False

    __hash__ = None  # type: ignore[assignment]  # unhashable by design

    def __repr__(self) -> str:
        _window = self.slice
        _window_repr = ':'.join(map(str, _window.indices(len(self.base))))
        _base_repr = repr(self.base[_window])
        return f"sliceview(<{_base_repr}>)[{_window_repr}]"

    def _setslice(self, sl: slice, values: Iterable[Any]) -> None:
        if not isinstance(self._base, MutableSequence):
            raise TypeError(f'base of sliceview {type(self._base)} is not mutable')
        
