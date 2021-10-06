import pytest

from recsys.recsys import recommend_simple


def test_simple():
    assert len(recommend_simple(["hello"], 5)) == 5


def test_error_neg_limit():
    with pytest.raises(Exception):
        recommend_simple(["hello"], -1)
