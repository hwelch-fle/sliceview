from collections.abc import Sequence

from hypothesis import (
    given, 
    strategies as st,
)

from sliceview import sliceview

def correct_answer[T](seq: Sequence[T], sl: slice) -> list[T]:
    return list(seq[sl])

@given(lst=st.lists(st.integers(), min_size=100, max_size=100), sl=st.slices(100))
def test_fuzzy(lst: list[int], sl: slice):
    sv = sliceview(lst, sl)
    assert list(sv) == correct_answer(lst, sl), (sv.range, sv.slice, sl)

@given(lst=st.lists(st.integers(), min_size=100, max_size=100), sl=st.slices(100))
def test_slice_assignment(lst: list[int], sl: slice):
    _lst = lst[sl].copy()
    sv = sliceview(lst, sl)
    _lst[0:2] = [1,1]
    sv[0:2] = [1,1]
    assert sv == _lst
