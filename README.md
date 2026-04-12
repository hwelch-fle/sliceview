# sliceview

**Zero-copy, composable slice views for Python sequences.**

`sliceview` lets you work with windows into lists, tuples, strings, or any
`Sequence` without copying the underlying data.  It is the proof-of-concept
library accompanying the Python discussion on
[Slice Views for Python Sequences](https://discuss.python.org/t/slice-views-for-python-sequences/103531).

```python
from sliceview import sliceview

data = list(range(1_000_000))

# O(1) — no copy
window = sliceview(data)[50_000:60_000]

# Compose slices — still O(1), still no copy
every_third = sliceview(data)[::3][100:200]

# In-place windowed update
sv = sliceview(data)
sv[0:5] = [10, 20, 30, 40, 50]

# Sliding window without creating new objects
view = sliceview(data, 0, 1000)
for _ in range(10):
    process(view)
    view.advance(1000)
```

## Why?

In Python today, `seq[a:b]` copies the data.  For large pipelines — text
processing, audio, genomics, sorting algorithms — those copies dominate
both time and memory.  `memoryview` solves this for *bytes-like* objects;
`sliceview` targets *generic* sequences of Python objects.

Inspired by Go slices, NumPy views, and `memoryview`.

## Installation

```bash
pip install sliceview
```

## Features

| Feature | Details |
|---|---|
| **Zero-copy slicing** | `sv[a:b:c]` returns a new `sliceview` in O(1) |
| **Composable** | `sv[2:][::3][5:10]` chains correctly with no intermediate copies |
| **Live view** | Mutations to the base are immediately visible through the view |
| **Write-through** | `sv[i] = x` and `sv[a:b] = iterable` forward to the base |
| **Sliding window** | `sv.advance(n)` shifts the window in-place — no new object |
| **Any sequence** | Works with `list`, `tuple`, `str`, `array`, or your own type |

## API

### `sliceview(base, start=None, stop=None, step=None)`

Create a view over `base`.  `start` may be a `slice` object.

```python
sv = sliceview(my_list)           # full view
sv = sliceview(my_list, 10, 20)   # [10:20]
sv = sliceview(my_list, slice(10, 20, 2))  # [10:20:2]
```

### Indexing and slicing

```python
sv[i]        # element access — maps to base[start + i*step]
sv[a:b:c]    # returns a new sliceview (O(1))
sv[i] = x    # write-through to the base
sv[a:b] = it # slice assignment (delegates to base)
```

### `sv.advance(n) -> self`

Shift the view's window forward by `n` index positions (negative to retreat).
Returns `self` for chaining.  Useful for sliding-window algorithms:

```python
view = sliceview(samples, 0, window_size)
while True:
    result = process(view)
    if view.advance(window_size)._start >= len(samples):
        break
```

### `sv.tolist()` / `sv.copy()`

Materialise the view as a new `list` (explicit copy).

### `sv.base`

The underlying sequence the view points into.

## Semantics

- **`len(sv)`** reflects the *current* base length, so appending to the base
  is immediately visible.
- **`sv[:]`** returns a new `sliceview` pointing at the same base — O(1).
- **Hashing**: `sliceview` is intentionally unhashable.
- **Equality**: compares element-wise to any `Sequence`.
- **Immutable bases**: `sv[i] = x` raises `TypeError` if the base does not
  support `__setitem__`.

## Design notes and open questions

This library implements the core proposal from the
[Python discussion](https://discuss.python.org/t/slice-views-for-python-sequences/103531).
Deliberately left out to keep the scope focused:

- `view()` builtin shortcut (consensus in the thread was it's unnecessary)
- `__sliceview__` dunder (motivation unclear until adoption data exists)
- Multidimensional views (NumPy is the right tool for that)
- ABC / Protocol additions to `collections.abc`

Feedback welcome — please open an issue or join the discussion thread.

## License

MIT
