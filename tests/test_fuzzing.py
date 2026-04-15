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

@given(lst=st.lists(st.integers(), min_size=5, max_size=5), sl=st.slices(5))
def test_slice_assignment(lst: list[int], sl: slice):
    _lst = lst[sl]
    if len(lst) < 2:
        return
    sv = sliceview(lst, sl)
    
    assert list(sv) == _lst, ('initial', str(sv), str(_lst), sv.slice)
    assert list(sv[0:2]) == _lst[0:2], ('sub-slice', str(sv), str(_lst), sv.slice)
    assert len(_lst) == len(sv), ('length', str(sv), str(_lst), sv.slice)
