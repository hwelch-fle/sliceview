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

from collections.abc import Sequence, Iterator
from typing import overload, Union, Optional


__all__ = ["sliceview"]
__version__ = "0.1.0"


class _OpenRange:
    """Sentinel for a range whose stop is open (None), so it grows with the base."""

    __slots__ = ("start", "step")

    def __init__(self, start: int, step: int) -> None:
        self.start = start
        self.step = step

    def resolve(self, b_len: int) -> range:
        stop = b_len if self.step > 0 else -1
        return range(self.start, stop, self.step)


class sliceview(Sequence):
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

    __slots__ = ("_base", "_range")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        base: Sequence,
        start: Union[int, slice, None] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        if not (hasattr(base, "__len__") and hasattr(base, "__getitem__")):
            raise TypeError(
                f"sliceview requires a sequence with __len__ and __getitem__, "
                f"got {type(base).__name__!r}"
            )
        self._base = base

        if isinstance(start, slice) and stop is None and step is None:
            sl: slice = start
        else:
            sl = slice(start, stop, step)

        b_len = len(base)
        s, e, st = sl.indices(b_len)

        # If the original stop was None, store an open sentinel so that the
        # view grows when elements are appended to the base.
        if sl.stop is None:
            self._range = _OpenRange(s, st)
        else:
            self._range = range(s, e, st)

    @classmethod
    def _from_range(cls, base: Sequence, r) -> "sliceview":
        """Internal constructor: build a view directly from a range object."""
        sv = cls.__new__(cls)
        sv._base = base
        sv._range = r
        return sv

    # ------------------------------------------------------------------
    # Core helper
    # ------------------------------------------------------------------

    def _current_range(self) -> range:
        """Return a concrete ``range`` clamped to the current base length."""
        r = self._range
        if isinstance(r, _OpenRange):
            return r.resolve(len(self._base))
        return r

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._current_range())

    @overload
    def __getitem__(self, index: int) -> object: ...
    @overload
    def __getitem__(self, index: slice) -> "sliceview": ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            # Compose slices using Python's range slicing — O(1), exact.
            sub = self._current_range()[index]
            return sliceview._from_range(self._base, sub)

        cr = self._current_range()
        length = len(cr)
        if index < 0:
            index += length
        if not (0 <= index < length):
            raise IndexError("sliceview index out of range")
        return self._base[cr[index]]

    def __setitem__(self, index, value) -> None:
        if not hasattr(self._base, "__setitem__"):
            raise TypeError("underlying sequence is not mutable")

        if isinstance(index, slice):
            self._setslice(index, value)
            return

        cr = self._current_range()
        length = len(cr)
        if index < 0:
            index += length
        if not (0 <= index < length):
            raise IndexError("sliceview index out of range")
        self._base[cr[index]] = value

    def __iter__(self) -> Iterator:
        base = self._base
        for i in self._current_range():
            yield base[i]

    def __contains__(self, item) -> bool:
        return any(item == el for el in self)

    def __reversed__(self) -> Iterator:
        base = self._base
        return (base[i] for i in reversed(self._current_range()))

    # ------------------------------------------------------------------
    # Equality and hashing
    # ------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        if isinstance(other, Sequence):
            return len(self) == len(other) and all(a == b for a, b in zip(self, other))
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]  # unhashable by design

    # ------------------------------------------------------------------
    # Repr & Str
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        _window = slice(
            self._range.start, self._range.stop, self._range.step
        ).indices(len(self._base))
        _window_repr = ':'.join(map(str, _window))
        return f"sliceview[{_window_repr}]({object.__repr__(self._base)})"

    def __str__(self) -> str:
        _window = slice(
            self._range.start, self._range.stop, self._range.step
        ).indices(len(self._base))
        _window_repr = ':'.join(map(str, _window))
        return f"sliceview[{_window_repr}](>{list(self)}<)"

    # ------------------------------------------------------------------
    # Advance — sliding-window helper
    # ------------------------------------------------------------------

    def advance(self, n: int) -> "sliceview":
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
        cr = self._current_range()
        new_start = max(0, min(cr.start + n, b_len))
        delta = new_start - cr.start
        new_stop = max(0, min(cr.stop + delta, b_len))
        self._range = range(new_start, new_stop, cr.step)
        return self

    # ------------------------------------------------------------------
    # tolist / copy
    # ------------------------------------------------------------------

    def tolist(self) -> list:
        """Return a new list containing the elements covered by this view."""
        return list(self)

    def copy(self) -> list:
        """Alias for :meth:`tolist`."""
        return self.tolist()

    # ------------------------------------------------------------------
    # base property
    # ------------------------------------------------------------------

    @property
    def base(self):
        """The underlying sequence this view points into."""
        return self._base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setslice(self, sl: slice, values) -> None:
        target_range = self._current_range()[sl]
        values = list(values)

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
