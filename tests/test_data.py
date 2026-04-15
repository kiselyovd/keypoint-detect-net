"""Data layer smoke tests."""
from __future__ import annotations

import pandas as pd

from vehicle_keypoints.data import load_dataset


def test_load_dataset(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)
    df = load_dataset(p)
    assert len(df) == 2
