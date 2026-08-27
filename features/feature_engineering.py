"""
feature_engineering.py

Build leakage-safe multi-season pre-match features from the cleaned
Premier League matches table.

The seasons are processed as one chronological timeline.

Example:
    2024/25 completed matches
            ->
    2025/26 completed matches
            ->
    2026/27 completed matches
            ->
    2026/27 future fixtures

Completed matches update historical team statistics only AFTER their
own features have been created.

Future fixtures receive features but NEVER update historical statistics.

Writes:
    data/processed/features_2024_2026.csv
    data/processed/features_2024_2025.csv
    data/processed/features_2026.csv
    data/processed/features_2026_completed.csv

Usage:
    uv run python -m features.feature_engineering --start-season 2024 --end-season 2026
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

ROLLING_WINDOWS = (5, 10)
H2H_WINDOW = 5
CONGESTION_DAYS = 14

DEFAULT_GOALS_PER_GAME = 1.35
DEFAULT_REST_DAYS = 7.0


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_settings() -> tuple[str, str]:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    missing = []

    if not supabase_url:
        missing.append("SUPABASE_URL")

    if not supabase_key:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    return supabase_url, supabase_key


def mean_or_default(
    values: list[float],
    default: float = 0.0,
) -> float:
    if not values:
        return default

    return float(sum(values) / len(values))


class FeatureEngineer:
    """
    Build fixture-level features strictly from information available
    before each fixture's kickoff.

    Anti-leakage rule:
    1. Build features for a match.
    2. If the match is completed, update historical statistics.
    3. If the match is future/unplayed, do not update history.
    """

    def __init__(self) -> None:
        supabase_url, supabase_key = get_settings()

        self.supabase: Client = create_client(
            supabase_url,
            supabase_key,
        )

    def fetch_matches(
    self,
    start_season: int,
    end_season: int,
    ) -> list[dict[str, Any]]:
        page_size = 1000
        start = 0
        rows: list[dict[str, Any]] = []

        while True:
            result = (
                self.supabase
                .table("matches")
                .select("*")
                .eq("competition_code", "PL")
                .gte("season_start_year", start_season)
                .lte("season_start_year", end_season)
                .order("kickoff_at")
                .range(
                    start,
                    start + page_size - 1,
                )
                .execute()
            )

            batch = result.data or []

            rows.extend(batch)

            logger.info(
                "Fetched %s matches so far.",
                len(rows),
            )

            if len(batch) < page_size:
                break

            start += page_size

        logger.info(
            "Fetched %s Premier League fixtures across seasons %s to %s.",
            len(rows),
            start_season,
            end_season,
        )

        return rows

    def fetch_team_names(self) -> dict[str, str]:
        result = (
            self.supabase
            .table("teams")
            .select("*")
            .execute()
        )

        rows = result.data or []

        if not rows:
            raise RuntimeError(
                "The teams table is empty."
            )

        possible_name_columns = (
            "canonical_name",
            "name",
            "team_name",
            "short_name",
        )

        name_column = None

        for column in possible_name_columns:
            if column in rows[0]:
                name_column = column
                break

        if name_column is None:
            raise RuntimeError(
                "Could not detect a team-name column in teams. "
                f"Available columns: {list(rows[0].keys())}"
            )

        mapping: dict[str, str] = {}

        for row in rows:
            if row.get("id") and row.get(name_column):
                mapping[str(row["id"])] = str(
                    row[name_column]
                )

        logger.info(
            "Loaded %s team names using '%s'.",
            len(mapping),
            name_column,
        )

        return mapping

    @staticmethod
    def is_finished(
        match: dict[str, Any],
    ) -> bool:
        return (
            match.get("home_goals") is not None
            and match.get("away_goals") is not None
        )

    @staticmethod
    def points_for(
        goals_for: int,
        goals_against: int,
    ) -> int:
        if goals_for > goals_against:
            return 3

        if goals_for == goals_against:
            return 1

        return 0

    @staticmethod
    def rolling_average(
        history: deque[dict[str, Any]],
        key: str,
        window: int,
        default: float = 0.0,
    ) -> float:
        values = [
            float(item.get(key, 0.0) or 0.0)
            for item in list(history)[-window:]
        ]

        return mean_or_default(
            values,
            default,
        )

    @staticmethod
    def rolling_sum(
        history: deque[dict[str, Any]],
        key: str,
        window: int,
    ) -> float:
        return float(
            sum(
                float(item.get(key, 0.0) or 0.0)
                for item in list(history)[-window:]
            )
        )

    @staticmethod
    def count_matches_in_last_days(
        history: deque[dict[str, Any]],
        kickoff: pd.Timestamp,
        days: int,
    ) -> int:
        count = 0

        for item in history:
            previous_kickoff = item["kickoff"]

            delta_days = (
                kickoff - previous_kickoff
            ).total_seconds() / 86400

            if 0 < delta_days <= days:
                count += 1

        return count

    @staticmethod
    def days_since_last_match(
        history: deque[dict[str, Any]],
        kickoff: pd.Timestamp,
        default: float = DEFAULT_REST_DAYS,
    ) -> float:
        if not history:
            return default

        previous_kickoff = history[-1]["kickoff"]

        days = (
            kickoff - previous_kickoff
        ).total_seconds() / 86400

        return max(
            0.0,
            float(days),
        )

    @staticmethod
    def build_league_table(
        team_histories: dict[
            str,
            deque[dict[str, Any]],
        ],
        season_start_year: int,
    ) -> list[dict[str, Any]]:
        """
        Build the current-season league table.

        Cross-season history is used for form features, but league
        position and season points reset naturally at the start of
        every new season.
        """
        table = []

        for team_id, history in team_histories.items():
            current_season_records = [
                record
                for record in history
                if record.get("season_start_year")
                == season_start_year
            ]

            if not current_season_records:
                continue

            matches_played = len(
                current_season_records
            )

            points = sum(
                int(record.get("points", 0))
                for record in current_season_records
            )

            goals_for = sum(
                float(record.get("goals_for", 0))
                for record in current_season_records
            )

            goals_against = sum(
                float(record.get("goals_against", 0))
                for record in current_season_records
            )

            goal_difference = (
                goals_for - goals_against
            )

            points_per_game = (
                points / matches_played
                if matches_played > 0
                else 0.0
            )

            table.append(
                {
                    "team_id": team_id,
                    "matches_played": matches_played,
                    "points": points,
                    "goal_difference": goal_difference,
                    "goals_for": goals_for,
                    "points_per_game": points_per_game,
                }
            )

        table.sort(
            key=lambda row: (
                -row["points"],
                -row["goal_difference"],
                -row["goals_for"],
                row["team_id"],
            )
        )

        for position, row in enumerate(
            table,
            start=1,
        ):
            row["position"] = position

        return table

    @staticmethod
    def get_table_row(
        table: list[dict[str, Any]],
        team_id: str,
    ) -> dict[str, Any] | None:
        for row in table:
            if row["team_id"] == team_id:
                return row

        return None

    @staticmethod
    def h2h_key(
        home_team_id: str,
        away_team_id: str,
    ) -> tuple[str, str]:
        return tuple(
            sorted(
                (
                    home_team_id,
                    away_team_id,
                )
            )
        )

    def h2h_features(
        self,
        h2h_history: dict[
            tuple[str, str],
            deque[dict[str, Any]],
        ],
        home_team_id: str,
        away_team_id: str,
    ) -> dict[str, float]:
        key = self.h2h_key(
            home_team_id,
            away_team_id,
        )

        meetings = list(
            h2h_history[key]
        )[-H2H_WINDOW:]

        if not meetings:
            return {
                "h2h_matches": 0.0,
                "h2h_home_team_win_rate": 0.0,
                "h2h_draw_rate": 0.0,
                "h2h_avg_total_goals": 0.0,
            }

        home_wins = 0
        draws = 0
        total_goals = 0.0

        for meeting in meetings:
            total_goals += (
                meeting["home_goals"]
                + meeting["away_goals"]
            )

            if meeting["winner_id"] is None:
                draws += 1

            elif meeting["winner_id"] == home_team_id:
                home_wins += 1

        count = len(meetings)

        return {
            "h2h_matches": float(count),
            "h2h_home_team_win_rate": (
                home_wins / count
            ),
            "h2h_draw_rate": (
                draws / count
            ),
            "h2h_avg_total_goals": (
                total_goals / count
            ),
        }

    @staticmethod
    def league_goal_averages(
        team_histories: dict[
            str,
            deque[dict[str, Any]],
        ],
    ) -> tuple[float, float]:
        all_records = []

        for history in team_histories.values():
            all_records.extend(history)

        if not all_records:
            return (
                DEFAULT_GOALS_PER_GAME,
                DEFAULT_GOALS_PER_GAME,
            )

        goals_for = mean_or_default(
            [
                float(
                    record.get(
                        "goals_for",
                        0,
                    ) or 0
                )
                for record in all_records
            ],
            DEFAULT_GOALS_PER_GAME,
        )

        goals_against = mean_or_default(
            [
                float(
                    record.get(
                        "goals_against",
                        0,
                    ) or 0
                )
                for record in all_records
            ],
            DEFAULT_GOALS_PER_GAME,
        )

        return (
            goals_for,
            goals_against,
        )

    @staticmethod
    def history_before_current_season(
        history: deque[dict[str, Any]],
        season_start_year: int,
    ) -> int:
        return sum(
            1
            for record in history
            if record.get("season_start_year")
            < season_start_year
        )

    def build_features(
        self,
        matches: list[dict[str, Any]],
        team_names: dict[str, str],
    ) -> pd.DataFrame:
        matches = sorted(
            matches,
            key=lambda row: (
                row.get("kickoff_at") or ""
            ),
        )

        team_histories: dict[
            str,
            deque[dict[str, Any]],
        ] = defaultdict(
            lambda: deque(maxlen=500)
        )

        home_histories: dict[
            str,
            deque[dict[str, Any]],
        ] = defaultdict(
            lambda: deque(maxlen=250)
        )

        away_histories: dict[
            str,
            deque[dict[str, Any]],
        ] = defaultdict(
            lambda: deque(maxlen=250)
        )

        h2h_history: dict[
            tuple[str, str],
            deque[dict[str, Any]],
        ] = defaultdict(
            lambda: deque(maxlen=100)
        )

        feature_rows: list[
            dict[str, Any]
        ] = []

        for index, match in enumerate(
            matches,
            start=1,
        ):
            match_id = match.get("id")

            home_team_id = str(
                match.get("home_team_id")
            )

            away_team_id = str(
                match.get("away_team_id")
            )

            season_start_year = match.get(
                "season_start_year"
            )

            if (
                not match_id
                or home_team_id == "None"
                or away_team_id == "None"
                or season_start_year is None
            ):
                logger.warning(
                    "Skipping malformed match row %s.",
                    match_id,
                )
                continue

            season_start_year = int(
                season_start_year
            )

            kickoff = pd.to_datetime(
                match.get("kickoff_at"),
                utc=True,
                errors="coerce",
            )

            if pd.isna(kickoff):
                logger.warning(
                    "Skipping match %s with invalid kickoff.",
                    match_id,
                )
                continue

            home_history = team_histories[
                home_team_id
            ]

            away_history = team_histories[
                away_team_id
            ]

            home_home_history = home_histories[
                home_team_id
            ]

            away_away_history = away_histories[
                away_team_id
            ]

            league_table = self.build_league_table(
                team_histories,
                season_start_year,
            )

            home_table = self.get_table_row(
                league_table,
                home_team_id,
            )

            away_table = self.get_table_row(
                league_table,
                away_team_id,
            )

            (
                league_avg_goals_for,
                league_avg_goals_against,
            ) = self.league_goal_averages(
                team_histories
            )

            home_previous_season_matches = (
                self.history_before_current_season(
                    home_history,
                    season_start_year,
                )
            )

            away_previous_season_matches = (
                self.history_before_current_season(
                    away_history,
                    season_start_year,
                )
            )

            feature_row: dict[str, Any] = {
                "match_id": str(match_id),
                "raw_match_id": (
                    str(match.get("raw_match_id"))
                    if match.get("raw_match_id")
                    else None
                ),
                "season_start_year": season_start_year,
                "matchday": match.get("matchday"),
                "gameweek": match.get("matchday"),
                "kickoff_at": kickoff.isoformat(),
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team": team_names.get(
                    home_team_id,
                    home_team_id,
                ),
                "away_team": team_names.get(
                    away_team_id,
                    away_team_id,
                ),
                "home_matches_played_before": len(
                    home_history
                ),
                "away_matches_played_before": len(
                    away_history
                ),
                "home_previous_season_matches": (
                    home_previous_season_matches
                ),
                "away_previous_season_matches": (
                    away_previous_season_matches
                ),
                "home_has_prior_pl_history": float(
                    home_previous_season_matches > 0
                ),
                "away_has_prior_pl_history": float(
                    away_previous_season_matches > 0
                ),
            }

            for window in ROLLING_WINDOWS:
                for prefix, history in (
                    ("home", home_history),
                    ("away", away_history),
                ):
                    feature_row[
                        f"{prefix}_points_last_{window}"
                    ] = self.rolling_sum(
                        history,
                        "points",
                        window,
                    )

                    feature_row[
                        f"{prefix}_points_per_game_last_{window}"
                    ] = self.rolling_average(
                        history,
                        "points",
                        window,
                    )

                    feature_row[
                        f"{prefix}_goals_for_avg_last_{window}"
                    ] = self.rolling_average(
                        history,
                        "goals_for",
                        window,
                    )

                    feature_row[
                        f"{prefix}_goals_against_avg_last_{window}"
                    ] = self.rolling_average(
                        history,
                        "goals_against",
                        window,
                    )

                    feature_row[
                        f"{prefix}_xg_for_avg_last_{window}"
                    ] = self.rolling_average(
                        history,
                        "xg_for",
                        window,
                    )

                    feature_row[
                        f"{prefix}_xg_against_avg_last_{window}"
                    ] = self.rolling_average(
                        history,
                        "xg_against",
                        window,
                    )

            feature_row.update(
                {
                    "home_goals_for_avg_home_last_5": (
                        self.rolling_average(
                            home_home_history,
                            "goals_for",
                            5,
                        )
                    ),
                    "home_goals_against_avg_home_last_5": (
                        self.rolling_average(
                            home_home_history,
                            "goals_against",
                            5,
                        )
                    ),
                    "home_xg_for_avg_home_last_5": (
                        self.rolling_average(
                            home_home_history,
                            "xg_for",
                            5,
                        )
                    ),
                    "home_xg_against_avg_home_last_5": (
                        self.rolling_average(
                            home_home_history,
                            "xg_against",
                            5,
                        )
                    ),
                    "away_goals_for_avg_away_last_5": (
                        self.rolling_average(
                            away_away_history,
                            "goals_for",
                            5,
                        )
                    ),
                    "away_goals_against_avg_away_last_5": (
                        self.rolling_average(
                            away_away_history,
                            "goals_against",
                            5,
                        )
                    ),
                    "away_xg_for_avg_away_last_5": (
                        self.rolling_average(
                            away_away_history,
                            "xg_for",
                            5,
                        )
                    ),
                    "away_xg_against_avg_away_last_5": (
                        self.rolling_average(
                            away_away_history,
                            "xg_against",
                            5,
                        )
                    ),
                }
            )

            feature_row.update(
                {
                    "home_rest_days": (
                        self.days_since_last_match(
                            home_history,
                            kickoff,
                        )
                    ),
                    "away_rest_days": (
                        self.days_since_last_match(
                            away_history,
                            kickoff,
                        )
                    ),
                    "home_matches_last_14_days": (
                        self.count_matches_in_last_days(
                            home_history,
                            kickoff,
                            CONGESTION_DAYS,
                        )
                    ),
                    "away_matches_last_14_days": (
                        self.count_matches_in_last_days(
                            away_history,
                            kickoff,
                            CONGESTION_DAYS,
                        )
                    ),
                }
            )

            feature_row.update(
                {
                    "home_table_position": (
                        float(home_table["position"])
                        if home_table
                        else 0.0
                    ),
                    "away_table_position": (
                        float(away_table["position"])
                        if away_table
                        else 0.0
                    ),
                    "home_points_per_game": (
                        float(
                            home_table[
                                "points_per_game"
                            ]
                        )
                        if home_table
                        else 0.0
                    ),
                    "away_points_per_game": (
                        float(
                            away_table[
                                "points_per_game"
                            ]
                        )
                        if away_table
                        else 0.0
                    ),
                    "home_total_points_before": (
                        float(home_table["points"])
                        if home_table
                        else 0.0
                    ),
                    "away_total_points_before": (
                        float(away_table["points"])
                        if away_table
                        else 0.0
                    ),
                }
            )

            home_attack_avg = (
                self.rolling_average(
                    home_history,
                    "goals_for",
                    10,
                    DEFAULT_GOALS_PER_GAME,
                )
            )

            home_defence_avg = (
                self.rolling_average(
                    home_history,
                    "goals_against",
                    10,
                    DEFAULT_GOALS_PER_GAME,
                )
            )

            away_attack_avg = (
                self.rolling_average(
                    away_history,
                    "goals_for",
                    10,
                    DEFAULT_GOALS_PER_GAME,
                )
            )

            away_defence_avg = (
                self.rolling_average(
                    away_history,
                    "goals_against",
                    10,
                    DEFAULT_GOALS_PER_GAME,
                )
            )

            feature_row.update(
                {
                    "home_attack_strength": (
                        home_attack_avg
                        / league_avg_goals_for
                        if league_avg_goals_for > 0
                        else 1.0
                    ),
                    "home_defence_strength": (
                        home_defence_avg
                        / league_avg_goals_against
                        if league_avg_goals_against > 0
                        else 1.0
                    ),
                    "away_attack_strength": (
                        away_attack_avg
                        / league_avg_goals_for
                        if league_avg_goals_for > 0
                        else 1.0
                    ),
                    "away_defence_strength": (
                        away_defence_avg
                        / league_avg_goals_against
                        if league_avg_goals_against > 0
                        else 1.0
                    ),
                }
            )

            feature_row.update(
                self.h2h_features(
                    h2h_history,
                    home_team_id,
                    away_team_id,
                )
            )

            feature_row["home_goals"] = match.get(
                "home_goals"
            )

            feature_row["away_goals"] = match.get(
                "away_goals"
            )

            feature_row["home_xg"] = match.get(
                "home_xg"
            )

            feature_row["away_xg"] = match.get(
                "away_xg"
            )

            feature_row["is_completed"] = float(
                self.is_finished(match)
            )

            if self.is_finished(match):
                home_goals = int(
                    match["home_goals"]
                )

                away_goals = int(
                    match["away_goals"]
                )

                if home_goals > away_goals:
                    feature_row["outcome"] = "H"
                    winner_id = home_team_id

                elif home_goals < away_goals:
                    feature_row["outcome"] = "A"
                    winner_id = away_team_id

                else:
                    feature_row["outcome"] = "D"
                    winner_id = None

                home_points = self.points_for(
                    home_goals,
                    away_goals,
                )

                away_points = self.points_for(
                    away_goals,
                    home_goals,
                )

                home_xg_value = match.get("home_xg")
                away_xg_value = match.get("away_xg")

                home_xg = (
                    float(home_xg_value)
                    if home_xg_value is not None
                    else DEFAULT_GOALS_PER_GAME
                )

                away_xg = (
                    float(away_xg_value)
                    if away_xg_value is not None
                    else DEFAULT_GOALS_PER_GAME
                )

                home_record = {
                    "kickoff": kickoff,
                    "season_start_year": (
                        season_start_year
                    ),
                    "points": home_points,
                    "goals_for": home_goals,
                    "goals_against": away_goals,
                    "xg_for": home_xg,
                    "xg_against": away_xg,
                }

                away_record = {
                    "kickoff": kickoff,
                    "season_start_year": (
                        season_start_year
                    ),
                    "points": away_points,
                    "goals_for": away_goals,
                    "goals_against": home_goals,
                    "xg_for": away_xg,
                    "xg_against": home_xg,
                }

                team_histories[
                    home_team_id
                ].append(home_record)

                team_histories[
                    away_team_id
                ].append(away_record)

                home_histories[
                    home_team_id
                ].append(home_record)

                away_histories[
                    away_team_id
                ].append(away_record)

                h2h_history[
                    self.h2h_key(
                        home_team_id,
                        away_team_id,
                    )
                ].append(
                    {
                        "home_team_id": (
                            home_team_id
                        ),
                        "away_team_id": (
                            away_team_id
                        ),
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "winner_id": winner_id,
                    }
                )

            else:
                feature_row["outcome"] = None

            feature_rows.append(
                feature_row
            )

            if index % 50 == 0:
                logger.info(
                    "Built features for %s/%s matches.",
                    index,
                    len(matches),
                )

        return pd.DataFrame(
            feature_rows
        )

    def run(
        self,
        start_season: int,
        end_season: int,
    ) -> tuple[Path, Path, Path, Path]:
        matches = self.fetch_matches(
            start_season,
            end_season,
        )

        team_names = self.fetch_team_names()

        if not matches:
            raise RuntimeError(
                "No matches found for the requested seasons."
            )

        logger.info(
            "Building multi-season features chronologically "
            "with leakage protection."
        )

        features = self.build_features(
            matches,
            team_names,
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        combined_path = (
            OUTPUT_DIR
            / (
                f"features_{start_season}_"
                f"{end_season}.csv"
            )
        )

        completed_path = (
            OUTPUT_DIR
            / (
                f"features_{start_season}_"
                f"{end_season - 1}.csv"
            )
        )

        current_season_path = (
            OUTPUT_DIR
            / f"features_{end_season}.csv"
        )

        current_completed_path = (
            OUTPUT_DIR
            / (
                f"features_{end_season}_"
                "completed.csv"
            )
        )

        features.to_csv(
            combined_path,
            index=False,
        )

        completed_history = features[
            (
                features["is_completed"] == 1.0
            )
            & (
                features["season_start_year"]
                < end_season
            )
        ].copy()

        completed_history.to_csv(
            completed_path,
            index=False,
        )

        current_season = features[
            features["season_start_year"]
            == end_season
        ].copy()

        current_season.to_csv(
            current_season_path,
            index=False,
        )

        current_completed = current_season[
            current_season["is_completed"]
            == 1.0
        ].copy()

        current_completed.to_csv(
            current_completed_path,
            index=False,
        )

        logger.info(
            "Feature engineering complete."
        )

        logger.info(
            "Total rows written: %s",
            len(features),
        )

        logger.info(
            "Feature columns: %s",
            len(features.columns),
        )

        logger.info(
            "Completed historical rows: %s",
            len(completed_history),
        )

        logger.info(
            "Current-season rows: %s",
            len(current_season),
        )

        logger.info(
            "Current completed rows: %s",
            len(current_completed),
        )

        logger.info(
            "Combined output: %s",
            combined_path,
        )

        logger.info(
            "Current-season output: %s",
            current_season_path,
        )

        return (
            combined_path,
            completed_path,
            current_season_path,
            current_completed_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe multi-season Premier League "
            "pre-match features."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help=(
            "Shortcut for building one target season. "
            "For example, --season 2025 builds features_2025.csv "
            "using the previous season as history."
        ),
    )

    parser.add_argument(
        "--start-season",
        type=int,
        default=None,
        help=(
            "First season start year. "
            "Use together with --end-season."
        ),
    )

    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help=(
            "Last season start year. "
            "Use together with --start-season."
        ),
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    # Convenience mode:
    # --season 2025
    #
    # This builds a chronological timeline from 2024/25 through 2025/26
    # and writes data/processed/features_2025.csv.
    if args.season is not None:
        if (
            args.start_season is not None
            or args.end_season is not None
        ):
            raise ValueError(
                "Use either --season OR "
                "--start-season/--end-season, not both."
            )

        start_season = args.season - 1
        end_season = args.season

    else:
        if (
            args.start_season is None
            and args.end_season is None
        ):
            start_season = 2024
            end_season = 2026

        elif (
            args.start_season is None
            or args.end_season is None
        ):
            raise ValueError(
                "--start-season and --end-season must be "
                "provided together."
            )

        else:
            start_season = args.start_season
            end_season = args.end_season

    if end_season < start_season:
        raise ValueError(
            "--end-season must be greater than or equal to "
            "--start-season."
        )

    logger.info(
        "Starting multi-season feature engineering "
        "for seasons %s/%s through %s/%s",
        start_season,
        str(start_season + 1)[-2:],
        end_season,
        str(end_season + 1)[-2:],
    )

    engineer = FeatureEngineer()

    engineer.run(
        start_season,
        end_season,
    )


if __name__ == "__main__":
    main()