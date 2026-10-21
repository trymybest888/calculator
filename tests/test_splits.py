import pandas as pd

from src.splits import SPLIT_NAMES, split_frame, split_mask


def _hourly_frame():
    index = pd.date_range("2015-12-31 20:00", "2017-01-01 03:00", freq="h")
    return pd.DataFrame({"x": range(len(index))}, index=index)


def test_splits_are_chronological_and_disjoint():
    parts = split_frame(_hourly_frame())
    assert parts["train"].index.max() < parts["validation"].index.min()
    assert parts["validation"].index.max() < parts["test"].index.min()
    total = sum(len(parts[name]) for name in SPLIT_NAMES)
    assert total == len(_hourly_frame())


def test_boundaries():
    index = _hourly_frame().index
    assert split_mask(index, "train")[pd.Timestamp("2015-12-31 23:00")]
    assert split_mask(index, "validation")[pd.Timestamp("2016-01-01 00:00")]
    assert split_mask(index, "test")[pd.Timestamp("2017-01-01 00:00")]
