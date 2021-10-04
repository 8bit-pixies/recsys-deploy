from recsys.api import recommend


def test_simple():
    assert len(recommend([], 5)) == 5
