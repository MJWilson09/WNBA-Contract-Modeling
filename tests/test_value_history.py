from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.wnba_salary import value_history


class ValueHistoryTest(unittest.TestCase):
    def test_upsert_is_idempotent(self) -> None:
        values = pd.DataFrame({
            "athlete_id": [1], "athlete_display_name": ["Test Player"],
            "team_id": [2], "team_games": [10], "g": [9], "mp": [200],
            "mpg": [22.2], "availability": [.9], "proj_minutes": [880],
            "rating": [1.5], "rating_se": [2.0], "war": [2.4],
            "value": [800_000], "market_value": [800_000],
            "salary": [500_000], "surplus": [300_000],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.sqlite"
            value_history.upsert(values, "2026-06-01", path)
            values.loc[0, "value"] = 810_000
            value_history.upsert(values, "2026-06-01", path)
            loaded = value_history.load(path=path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.iloc[0]["value"], 810_000)


if __name__ == "__main__":
    unittest.main()
