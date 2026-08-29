from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import update
from src.wnba_salary import data


class Response:
    def __init__(self, *, content: bytes = b"", text: str = "", status: int = 200):
        self.content = content
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    frame.to_parquet(out, index=False)
    return out.getvalue()


class RolloverReadinessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.game = "2027001"
        # A full 15-team, 50-game schedule. Only the first game is completed;
        # the guard must not mistake a one-game schedule stub for readiness.
        appearances = []
        for team in range(15):
            for n in range(50):
                opponent = (team + n + 1) % 15
                appearances.append((f"s{team}-{n}", team, opponent))
        schedule = pd.DataFrame(appearances, columns=["game_id", "home_id", "away_id"])
        schedule["home_abbreviation"] = "TEAM"
        schedule["away_abbreviation"] = "TEAM"
        schedule["season_type"] = 2
        schedule["status_type_completed"] = False
        schedule.loc[0, ["game_id", "status_type_completed"]] = [self.game, True]
        self.frames = {
            "schedule": schedule,
            "team_box": pd.DataFrame({
                "game_id": [self.game, self.game], "season_type": [2, 2],
            }),
            "player_box": pd.DataFrame({
                "game_id": [self.game] * 10, "season_type": [2] * 10,
            }),
            "pbp": pd.DataFrame({
                "game_id": [self.game] * 100, "season_type": [2] * 100,
            }),
            "player_core": pd.DataFrame({"athlete_id": range(10)}),
        }
        rosters = pd.DataFrame({
            "game_id": [self.game] * 10,
            "team_id": [1] * 5 + [2] * 5,
            "athlete_id": range(10),
            "starter": [True] * 10,
        })
        self.roster_content = parquet_bytes(rosters)
        rows = "".join(
            f'<tr><td sorttable_customkey="Last{i}, First{i}">P</td>'
            f'<td>${270000 + i}</td><td>Contract</td></tr>' for i in range(10)
        )
        self.salary_html = f"<table><tbody>{rows}</tbody></table>"
        self.bbref_html = '<table>' + '<th data-stat="player">P</th>' * 10 + '</table>'

    def remote(self, dataset: str, target: int) -> pd.DataFrame:
        self.assertEqual(target, 2027)
        return self.frames[dataset]

    def get(self, url: str, **kwargs) -> Response:
        if "game_rosters" in url:
            return Response(content=self.roster_content)
        if "herhoopstats" in url:
            return Response(text=self.salary_html)
        if "basketball-reference" in url:
            return Response(text=self.bbref_html)
        raise AssertionError(f"unexpected URL {url}")

    def test_first_completed_game_allows_rollover(self) -> None:
        with patch.object(update, "_remote_dataset", side_effect=self.remote), \
             patch("requests.get", side_effect=self.get):
            ready, reasons = update.rollover_readiness(2027)
        self.assertTrue(ready)
        self.assertEqual(reasons, [])

    def test_incomplete_schedule_blocks_rollover(self) -> None:
        self.frames["schedule"]["status_type_completed"] = False
        with patch.object(update, "_remote_dataset", side_effect=self.remote):
            ready, reasons = update.rollover_readiness(2027)
        self.assertFalse(ready)
        self.assertIn("schedule: no completed regular-season game", reasons)


class ExhibitionFilterTest(unittest.TestCase):
    def test_all_star_game_is_removed_by_event_team_identity(self) -> None:
        games = pd.DataFrame({
            "game_id": [1, 2],
            "home_abbreviation": ["NY", "COOP"],
            "away_abbreviation": ["LV", "SPO"],
        })
        filtered = data.exclude_exhibitions(games, 2027)
        self.assertEqual(filtered["game_id"].tolist(), [1])

    def test_game_rows_are_removed_by_exhibition_id(self) -> None:
        rows = pd.DataFrame({"game_id": [1, 1, 2, 2], "points": [1, 2, 3, 4]})
        with patch.object(data, "exhibition_game_ids", return_value={"2"}):
            filtered = data.exclude_exhibitions(rows, 2027)
        self.assertEqual(filtered["game_id"].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
