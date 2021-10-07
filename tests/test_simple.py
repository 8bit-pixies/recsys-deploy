import string

import numpy as np
import pytest

from recsys.recsys import recommend_lang


def test_simple():
    assert len(recommend_lang(["hello"], 5)) == 5


def test_error_neg_limit():
    with pytest.raises(Exception):
        recommend_lang(["hello"], -1)


def test_not_english():
    output = recommend_lang(["肇庆混凝土", "美灼物资"], 5)
    # at least 2 of the tags have no ascii_string characters
    output = [len([y for y in list(x["tag"]) if y in string.ascii_letters]) for x in output]
    assert np.sum(np.array(output) == 0) >= 2
