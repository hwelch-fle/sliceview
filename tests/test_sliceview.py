"""Tests for sliceview."""

import pytest
from sliceview import sliceview


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic(self):
        sv = sliceview([1, 2, 3])
        assert list(sv) == [1, 2, 3]

    def test_with_start_stop(self):
        sv = sliceview([0, 1, 2, 3, 4], 1, 4)
        assert list(sv) == [1, 2, 3]

    def test_with_step(self):
        sv = sliceview([0, 1, 2, 3, 4], 0, 5, 2)
        assert list(sv) == [0, 2, 4]

    def test_with_slice_object(self):
        sv = sliceview([0, 1, 2, 3, 4], slice(1, 4))
        assert list(sv) == [1, 2, 3]

    def test_negative_step(self):
        sv = sliceview([0, 1, 2, 3, 4], 4, None, -1)
        assert list(sv) == [4, 3, 2, 1, 0]

    def test_invalid_base(self):
        with pytest.raises(TypeError):
            sliceview(42)

    def test_string_base(self):
        sv = sliceview("hello")[1:4]
        assert list(sv) == ["e", "l", "l"]

    def test_tuple_base(self):
        sv = sliceview((10, 20, 30, 40))[::2]
        assert list(sv) == [10, 30]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_positive_index(self):
        sv = sliceview([10, 20, 30])[:]
        assert sv[0] == 10
        assert sv[2] == 30

    def test_negative_index(self):
        sv = sliceview([10, 20, 30])
        assert sv[-1] == 30
        assert sv[-3] == 10

    def test_out_of_range(self):
        sv = sliceview([1, 2, 3])
        with pytest.raises(IndexError):
            _ = sv[10]

    def test_index_into_strided_view(self):
        sv = sliceview(list(range(10)))[::3]
        assert sv[0] == 0
        assert sv[1] == 3
        assert sv[2] == 6

    # def test_none_start(self):
        # lst = [1, 2, 3, 4, 5]
        # slc = slice(None, -len(lst), 1)
        # sv = sliceview(lst, slc)
        # sv.slice = slc
        # assert list(sv) == lst[slc], (sv.range, sv.slice)


# ---------------------------------------------------------------------------
# Slicing (composition)
# ---------------------------------------------------------------------------

class TestSlicing:

    def test_slice_returns_sliceview(self):
        sv = sliceview([1, 2, 3, 4, 5])
        assert isinstance(sv[1:3], sliceview)

    def test_composed_slice(self):
        data = list(range(20))
        sv = sliceview(data)[2:][::3][1:4]
        assert list(sv) == [5, 8, 11]

    def test_no_copy_on_slice(self):
        data = [1, 2, 3, 4, 5]
        sv = sliceview(data)
        sv2 = sv[1:4]
        assert sv2.base is data

    def test_full_slice_is_same_base(self):
        data = [1, 2, 3]
        sv = sliceview(data)
        assert sv[:].base is data

    def test_negative_step_composition(self):
        data = list(range(10))
        sv = sliceview(data)[::-1][::2]
        assert list(sv) == [9, 7, 5, 3, 1]

    def test_empty_slice(self):
        sv = sliceview([1, 2, 3])[5:10]
        assert list(sv) == []
        assert len(sv) == 0


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------

class TestLen:
    def test_full(self):
        assert len(sliceview([1, 2, 3])) == 3

    def test_partial(self):
        assert len(sliceview([1, 2, 3, 4, 5])[1:4]) == 3

    def test_step(self):
        assert len(sliceview(list(range(10)))[::3]) == 4

    def test_empty(self):
        assert len(sliceview([])) == 0

    def test_reflects_base_mutation(self):
        data = [1, 2, 3, 4, 5]
        sv = sliceview(data)
        assert len(sv) == 5
        data.append(6)
        assert len(sv) == 6


# ---------------------------------------------------------------------------
# Mutation (write-through)
# ---------------------------------------------------------------------------

class TestMutation:
    def test_setitem_int(self):
        data = [1, 2, 3, 4, 5]
        sv = sliceview(data)[1:4]
        sv[0] = 99
        assert data == [1, 99, 3, 4, 5]

    def test_setitem_slice(self):
        data = list(range(5))
        sv = sliceview(data)
        sv[1:4] = [10, 20, 30]
        assert data == [0, 10, 20, 30, 4]

    def test_setitem_strided(self):
        data = list(range(10))
        sv = sliceview(data)[::2]
        sv[0] = 99
        assert data[0] == 99

    def test_setitem_extended_slice_wrong_size(self):
        data = list(range(10))
        sv = sliceview(data)[::2]
        with pytest.raises(ValueError):
            sv[0:3] = [1, 2]  # 3 slots, 2 values

    def test_immutable_base_raises(self):
        sv = sliceview((1, 2, 3))
        with pytest.raises(TypeError):
            sv[0] = 99


# ---------------------------------------------------------------------------
# Advance
# ---------------------------------------------------------------------------

# class TestAdvance:
#     def test_advance_basic(self):
#         data = list(range(10))
#         sv = sliceview(data, 0, 3)
#         assert list(sv) == [0, 1, 2]
#         sv.advance(3)
#         assert list(sv) == [3, 4, 5]
# 
#     def test_advance_returns_self(self):
#         sv = sliceview(list(range(10)), 0, 5)
#         assert sv.advance(5) is sv
# 
#     def test_advance_past_end_clamps(self):
#         data = list(range(5))
#         sv = sliceview(data, 0, 3)
#         sv.advance(100)
#         assert list(sv) == []
# 
#     def test_advance_negative(self):
#         data = list(range(10))
#         sv = sliceview(data, 5, 8)
#         sv.advance(-3)
#         assert list(sv) == [2, 3, 4]
# 
#     def test_sliding_window(self):
#         data = list(range(12))
#         sv = sliceview(data, 0, 4)
#         result = []
#         for _ in range(3):
#             result.append(list(sv))
#             sv.advance(4)
#         assert result == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

class TestIteration:
    def test_iter(self):
        sv = sliceview([10, 20, 30, 40])[1:3]
        assert list(sv) == [20, 30]

    def test_contains(self):
        sv = sliceview([1, 2, 3, 4, 5])[1:4]
        assert 2 in sv
        assert 5 not in sv

    def test_reversed(self):
        sv = sliceview([1, 2, 3, 4, 5])
        assert list(reversed(sv)) == [5, 4, 3, 2, 1]


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------

class TestEquality:
    def test_equal_to_list(self):
        sv = sliceview([1, 2, 3])
        assert sv == [1, 2, 3]

    def test_equal_to_sliceview(self):
        data = [1, 2, 3, 4]
        sv1 = sliceview(data)[1:3]
        sv2 = sliceview(data)[1:3]
        assert sv1 == sv2

    def test_not_equal(self):
        sv = sliceview([1, 2, 3])
        assert sv != [1, 2, 4]

    def test_unhashable(self):
        sv = sliceview([1, 2, 3])
        with pytest.raises(TypeError):
            hash(sv)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_slice(self):
        sv = sliceview([1, 2, 3, 4])[1:3]
        r = repr(sv)
        assert "sliceview" in r
        assert "1:3" in r

    def test_repr_full(self):
        sv = sliceview([1, 2, 3])
        r = repr(sv)
        assert "sliceview" in r


# ---------------------------------------------------------------------------
# tolist / copy
# ---------------------------------------------------------------------------

# class TestCopy:
#     def test_tolist(self):
#         data = [1, 2, 3, 4, 5]
#         sv = sliceview(data)[1:4]
#         result = sv.tolist()
#         assert result == [2, 3, 4]
#         assert isinstance(result, list)
#         result[0] = 99
#         assert data[1] == 2  # original unchanged
# 
#     def test_copy(self):
#         sv = sliceview([1, 2, 3])
#         assert sv.copy() == [1, 2, 3]


# ---------------------------------------------------------------------------
# base property
# ---------------------------------------------------------------------------

class TestBase:
    def test_base_is_original(self):
        data = [1, 2, 3]
        sv = sliceview(data)
        assert sv.base is data

    def test_composed_base_is_original(self):
        data = [1, 2, 3, 4, 5]
        sv = sliceview(data)[1:][::2]
        assert sv.base is data


# ---------------------------------------------------------------------------
# Live mutation reflection
# ---------------------------------------------------------------------------

class TestLiveMutation:
    def test_write_through(self):
        data = [1, 2, 3, 4, 5]
        sv = sliceview(data)
        data[2] = 99
        assert sv[2] == 99

    def test_append_reflected_in_len(self):
        data = [1, 2, 3]
        sv = sliceview(data)
        data.append(4)
        assert len(sv) == 4
