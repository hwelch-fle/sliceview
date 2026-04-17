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
    overload, 
)


__all__ = ["sliceview"]
__version__ = "0.1.0"


# perf: ~200ns O(1)
def range_to_slice(r: range) -> slice:
    """Convert a range to a slice"""
    start, stop, step = r.start, r.stop, r.step

    if start == stop:
        return slice(0,0,1)
    
    no_stop = (
        # Unbound range
        step < 0 and stop == -1
        # Inverted ranges
        or stop < start and step > 0
        or stop > start and step < 0
    )
    return slice(start, None if no_stop else stop, step)


class sliceview[T](Sequence[T]):
    """A zero-copy, composable slice view over any :class:`Sequence`.

    Params:
        base: The underlying :class:`Sequence`
        start: Start index or a :class:`slice` object (optional)
        stop: Stop index (if *start* is a `slice`, this must be omitted) (optional)
        step: Step amount (if *start* is a `slice`, this must be omitted) (optional)

    Examples:
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
    def __init__(
        self, base: Sequence[T], start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> None: ...

    # perf: ~500ns O(1)
    def __init__(
        self, base: Sequence[T], start : Any = None, stop: Any = None, step: Any = None
    ) -> None:
        if not isinstance(base, Sequence):
            raise TypeError(
                f"sliceview requires a sequence with __len__ and __getitem__, "
                f"got {type(base).__name__!r}"
            )
        self._base: Sequence[T] = base
        
        if isinstance(start, slice):
            if stop or step:
                raise ValueError('sliceview initialized with slice must not have stop/step arguments')
            sl: slice = start
        else:
            sl = slice(start, stop, step)
        
        self._unbound = sl.stop is None
        self._range = range(*sl.indices(len(base)))

    # perf: ~40ns O(1)
    @property
    def base(self) -> Sequence[T] | MutableSequence[T]:
        """The base sequence of the sliceview"""
        return self._base

    # perf: ~50ns if bounded, ~125ns if unbounded O(1)
    @property
    def range(self) -> range:
        """Return a concrete `range` clamped to the current base length."""
        r = self._range
        return (
            range(r.start, len(self._base) if r.step > 0 else -1, r.step)
            if self._unbound 
            else r
        )

    # perf: ~150ns if bounded, ~320ns if unbounded O(1)
    @property
    def slice(self) -> slice:
        """Return a slice object representing the current view"""
        return range_to_slice(self.range if self._unbound else self._range)

    # perf: ~80ns if bounded, ~200ns if unbounded O(1)
    def __len__(self) -> int:
        return len(self.range if self._unbound else self._range)

    @overload
    def __getitem__(self, index: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> sliceview[T]: ...

    # perf: Sequence.__getitem__+ K
    def __getitem__(self, index: object) -> T | sliceview[T]:
        if not isinstance(index, (slice, SupportsIndex)):
            raise TypeError(
                f'{type(self).__name__} indices must be integers or slices, '
                f'not {type(index).__name__}'
            )
        
        r = self.range if self._unbound else self._range
        
        if isinstance(index, slice):
            return type(self)(self._base, range_to_slice(r[index]))
        
        else:
            index = int(index)
            if index < 0:
                index += len(r)
            if not (0 <= index < len(r)):
                raise IndexError("sliceview index out of range")
            return self._base[r[index]]

    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    
    # perf: Sequence.__setitem__ + K
    def __setitem__(self, index: object, value: Any) -> None:
        if not isinstance(self._base, MutableSequence):
            raise TypeError(f"underlying sequence of type '{type(self._base)}' has no '__setitem__'")
        
        if not isinstance(index, (slice, SupportsIndex)):
            raise TypeError(
                f'{type(self).__name__} indices must be integers or slices, '
                f'not {type(index).__name__}'
            )
        
        r = self.range if self._unbound else self._range
        
        if isinstance(index, slice):
            if r.step != 1:
                value = list(value)
                if len(r[index]) == len(value):
                    for i, v in zip(r[index], value):
                        self._base[i] = v
                    return
            self._base[range_to_slice(r[index])] = value
        else:
            index = index.__index__()
            if index < 0: 
                index += len(r)
            if not 0 <= index < len(r):
                raise IndexError("sliceview index out of range")
            self._base[r[index]] = value

    # perf: Sequence.__getitem__ * n
    def __iter__(self) -> Iterator[T]:
        r = self.range if self._unbound else self._range
        for i in r:
            yield self._base[i]

    # perf: O(n)*(item.__eq__) worst case
    def __contains__(self, item: object) -> bool:
        return any(item == el for el in self)

    # perf: same as __iter__
    def __reversed__(self) -> Iterator[T]:
        r = self.range if self._unbound else self._range
        for i in reversed(r):
            yield self._base[i]

    # perf: O(n)*(len(self)*item.__eq__) worst case
    def __eq__(self, other: Sequence[T] | object) -> bool:
        if isinstance(other, Sequence) and len(self) == len(other):
            return all(a == b for a, b in zip(self, other))
        return False

    # perf: ~1000ns O(1)
    def __repr__(self) -> str:
        _window = self.slice.indices(len(self._base))
        _window_repr = ':'.join(map(str, _window))
        return f"sliceview[{_window_repr}]({object.__repr__(self._base)})"

    # perf: same as __iter__
    def __str__(self) -> str:
        _window = self.slice.indices(len(self._base))
        _window_repr = ':'.join(map(str, _window))
        return f"sliceview[{_window_repr}](>{list(self)}<)"
    
    # perf: ~250ns O(1)
    def advance(self, n: int = 1) -> Self:
        """Shift the view's window forward by *n* index positions in-place.

        Args:
            n: Positions to advance (negative to retreat). (default: 1)
            
        Returns:
            *self* so calls can be chained.

        Example:
            >>> sv = sliceview(range(10), 0, 3)
            >>> print(sv)
            sliceview[0:3:1](>[0, 1, 2]<)
            >>> sv.advance(3)
            sliceview[3:6:1](>[3, 4, 5]<)
        """
        b_len = len(self._base)
        r = self.range if self._unbound else self._range
        
        new_start = max(0, min(r.start + n, b_len))
        delta = new_start - r.start
        new_stop = max(0, min(r.stop + delta, b_len))
        self._range = range(new_start, new_stop, r.step)
        return self

    # perf: same as advance
    def advance_window(self, n: int = 1) -> Self:
        """Shift the view's window forward by *n* window widths in-place
        
        Args:
            n: The number of window widths to advance (negative to retreat) (default: 1)
        
        Returns:
            *self* so calls can be chained.
        
        Example:
            >>> sv = sliceview(range(100), 0, 5)
            >>> print(sv)
            sliceview[0:5:1](>[0, 1, 2, 3, 4]<)
            >>> print(sv.advance_window(3))
            sliceview[15:20:1](>[15, 16, 17, 18, 19]<)
        """
        return self.advance(len(self)*n)