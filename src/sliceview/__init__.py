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
    Self, 
    SupportsIndex, 
    TypeGuard, 
    overload, 
)


__all__ = ["sliceview"]
__version__ = "0.1.0"


def range_to_slice(r: range) -> slice:
    """Convert a range to a slice"""
    start, stop, step = r.start, r.stop, r.step

    # range (n, n, s) -> slice(0, 0, 1) [empty]
    if start == stop:
        return slice(0,0,1)
    
    # range(n, n-1, s) -> slice(n, None, s)
    _inverted = stop < start and step > 0
    # range(n, n+1, -s) -> slice(n, None, -s)
    _reversed = stop > start and step < 0
    # range(n, -1, -s) -> slice(n, None, -s)
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

    def __init__(self, base: Sequence[T] | object, 
                 start : object = None, 
                 stop: object = None, 
                 step: object = None,
        ) -> None:
        if not guard_type(base, Sequence[T]):
            raise TypeError(
                f"sliceview requires a sequence with __len__ and __getitem__, "
                f"got {type(base).__name__!r}"
            )
        self._base = base
        
        if guard_type(start, slice):
            if (stop, step) != (None, None):
                raise ValueError('sliceview initialized with slice must not have stop/step arguments')
            sl: slice = start
        else:
            sl = slice(start, stop, step)
        
        self._unbound = sl.stop is None
        self._range = slice_to_range(sl, len(base))

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
            return type(self)(self.base, range_to_slice(self.range[index]))
        
        else:
            index = int(index)
            if index < 0:
                index += len(self.range)
            if not (0 <= index < len(self.range)):
                raise IndexError("sliceview index out of range")
            return self.base[self.range[index]]

    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    
    def __setitem__(self, index: object, value: Any) -> None:
        if not guard_type(self._base, MutableSequence[T]):
            raise TypeError(f"underlying sequence of type '{type(self.base)}' has no '__setitem__'")
        
        match index:
            case slice():
                if not guard_type(value, Iterable[T]):
                    raise TypeError('can only assign an iterable')
                self._base[range_to_slice(self.range[index])] = value
            
            case SupportsIndex():
                w_size = len(self.range)
                index = int(index) + (w_size if int(index) < 0 else 0)
                if index not in range(w_size):
                    raise IndexError("sliceview index out of range")
                self._base[self.range[index]] = value
            
            case _:
                raise TypeError(
                    f'{type(self).__name__} indices must be integers or slices, '
                    f'not {type(index).__name__}'
                )

    def __iter__(self) -> Iterator[T]:
        yield from (self.base[i] for i in self.range)

    def __bool__(self) -> bool:
        for _ in self:
            return True
        else:
            return False

    def __contains__(self, item: object) -> bool:
        return any(item == el for el in self)

    def __reversed__(self) -> Iterator[T]:
        return (self.base[i] for i in reversed(self.range))

    def __eq__(self, other: Sequence[T] | object) -> bool:
        if guard_type(other, Sequence[T]) and len(self) == len(other):
            return all(a == b for a, b in zip(self, other))
        return False

    def __repr__(self) -> str:
        _window = self.slice.indices(len(self.base))
        _window_repr = ':'.join(map(str, _window))
        return f"sliceview[{_window_repr}]({repr(self.base)})"

    def __str__(self) -> str:
        _window = self.slice.indices(len(self.base))
        _window_repr = ':'.join(map(str, _window))
        return f"sliceview[{_window_repr}](>{list(self)}<)"
    
    def advance(self, n: int) -> Self:
        """Shift the view's window forward by *n* index positions in-place.

        Args:
            n: Positions to advance (negative to retreat).
            
        Returns:
            *self* so calls can be chained.

        Example:
            >>> data = list(range(10))
            >>> sv = sliceview(data, 0, 3)
            >>> list(sv)
            [0, 1, 2]
            >>> sv.advance(3)
            sliceview(...)[3:6:1]
            >>> list(sv)
            [3, 4, 5]
        """
        b_len = len(self.base)
        cr = self.range
        new_start = max(0, min(cr.start + n, b_len))
        delta = new_start - cr.start
        new_stop = max(0, min(cr.stop + delta, b_len))
        self._range = range(new_start, new_stop, cr.step)
        return self
    
    def retreat(self, n: int) -> Self:
        """Mirror of `advance` calls advance with negative `n`"""
        return self.advance(-n)
