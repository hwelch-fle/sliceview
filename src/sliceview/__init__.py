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

from collections.abc import Iterable, Sequence, Iterator, MutableSequence
from typing import Any, Protocol, SupportsIndex, overload, runtime_checkable


__all__ = ["sliceview"]
__version__ = "0.1.0"


@runtime_checkable
class SliceViewable(Protocol):
    """sliceview requires Sequence interface"""
    def __len__(self): ...
    def __getattr__(self, name: str): ...


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

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @overload
    def __init__(self, base: Sequence[T], start: slice, /) -> None: ...
    @overload
    def __init__(self, base: Sequence[T], start: int, stop: int, step: int) -> None: ...
    @overload
    def __init__(self, base: Sequence[T], start: int, stop: None, step: int) -> None: ...

    def __init__(
        self,
        base: Sequence[T],
        start: int | slice | None = None,
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
                raise ValueError('sliceview initialized with slice must not have stop/step arguments')
            sl: slice = start
        else:
            sl = slice(start, stop, step)
        
        self._unbound = sl.stop is None
        b_len = len(base)
        s, e, st = sl.indices(b_len)
        self._range = range(s, e, st)

    @classmethod
    def _from_range(cls, base: Sequence[T], r: range) -> sliceview[T]:
        """Internal constructor: build a view directly from a range object."""
        sv = cls.__new__(cls)
        sv._base = base
        sv._range = r
        sv._unbound = False
        return sv

    # ------------------------------------------------------------------
    # Core helper
    # ------------------------------------------------------------------

    @property
    def range(self) -> range:
        """Return a concrete ``range`` clamped to the current base length."""
        b_len = len(self._base)

        # Unbound views grow and shrink with the base sequence
        if self._unbound:
            if self._range.stop != b_len:
                self._range = range(self._range.start, b_len, self._range.step)
        return self._range

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.range)

    @overload
    def __getitem__(self, index: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> sliceview[T]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> T | sliceview[T]:
        if isinstance(index, slice):
            # Compose slices using Python's range slicing — O(1), exact.
            sub = self.range[index]
            return sliceview[T]._from_range(self._base, sub)
        
        index = index.__index__()
        cr = self.range
        length = len(cr)
        if index < 0:
            index += length
        if not (0 <= index < length):
            raise IndexError("sliceview index out of range")
        return self._base[cr[index]]

    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    
    def __setitem__(self, index: slice | SupportsIndex, value: T | Iterable[T]) -> None:
        if not isinstance(self._base, MutableSequence):
            raise TypeError("underlying sequence is not mutable")

        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise TypeError('can only assign an iterable')
            self._setslice(index, value)
            return
        
        index = index.__index__()
        cr = self.range
        length = len(cr)
        if index < 0:
            index += length
        if not (0 <= index < length):
            raise IndexError("sliceview index out of range")
        self._base[cr[index]] = value

    def __iter__(self) -> Iterator[T]:
        base = self._base
        _cur_range = self.range
        return iter(base[_cur_range.start:_cur_range.stop:_cur_range.step])

    def __contains__(self, item: object) -> bool:
        return any(item == el for el in self)

    def __reversed__(self) -> Iterator[T]:
        base = self._base
        _cur_range = self.range
        return (base[i] for i in reversed(_cur_range))

    # ------------------------------------------------------------------
    # Equality and hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence):
            return len(self) == len(other) and all(a == b for a, b in zip(self, other))
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]  # unhashable by design

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cr = self.range
        return f"sliceview(<{self._base[cr.start:cr.stop:cr.step]!r}>)[{cr.start}:{cr.stop}:{cr.step}]"

    # ------------------------------------------------------------------
    # Advance — sliding-window helper
    # ------------------------------------------------------------------

    def advance(self, n: int) -> sliceview[T]:
        """Shift the view's window forward by *n* index positions in-place.

        Returns *self* so calls can be chained.  Useful for sliding windows:

        >>> data = list(range(10))
        >>> sv = sliceview(data, 0, 3)
        >>> list(sv)
        [0, 1, 2]
        >>> sv.advance(3)                    # doctest: +ELLIPSIS
        sliceview(...)[3:6:1]
        >>> list(sv)
        [3, 4, 5]

        Parameters
        ----------
        n:
            Positions to advance (negative to retreat).
        """
        b_len = len(self._base)
        cr = self.range
        new_start = max(0, min(cr.start + n, b_len))
        delta = new_start - cr.start
        new_stop = max(0, min(cr.stop + delta, b_len))
        self._range = range(new_start, new_stop, cr.step)
        return self

    # ------------------------------------------------------------------
    # tolist / copy
    # ------------------------------------------------------------------

    def tolist(self) -> list[T]:
        """Return a new list containing the elements covered by this view."""
        return list(self)

    def copy(self) -> list[T]:
        """Alias for :meth:`tolist`."""
        return self.tolist()

    # ------------------------------------------------------------------
    # base property
    # ------------------------------------------------------------------

    @property
    def base(self) -> Sequence[T]:
        """The underlying sequence this view points into."""
        return self._base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setslice(self, sl: slice, values: Iterable[Any]) -> None:
        target_range = self.range[sl]
        values = list(values)
        if not isinstance(self._base, MutableSequence):
            raise TypeError(f'Base type of sliceview must be Mutable')
        if abs(target_range.step) != 1:
            # Extended slice assignment must match length exactly.
            if len(values) != len(target_range):
                raise ValueError(
                    f"attempt to assign sequence of size {len(values)} "
                    f"to extended slice of size {len(target_range)}"
                )
            for i, v in zip(target_range, values):
                self._base[i] = v
        else:
            # Step ±1: delegate to the base (allows resizing if base supports it).
            self._base[
                slice(target_range.start, target_range.stop, target_range.step)
            ] = values
