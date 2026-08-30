"""
promoted_team_normalizer.py

Build promoted-team-aware Dixon-Coles training data.

Rules:

- Established Premier League teams keep only genuine Premier League history.

- Ipswich keeps genuine Premier League history plus Championship history as a
  lower-weight prior.

- Hull and Coventry use Championship history as a lower-weight prior plus any
  genuine current Premier League matches.

- Championship scorelines remain real integer scorelines.

- Championship -> PL normalization is applied through row weights, NOT by
  rounding individual scorelines into synthetic results.

- Every prior row keeps its real kickoff timestamp so historical backtests can
  safely filter out future information.

- max_repeats is accepted for compatibility with poisson_dixon_coles.py.
  Since Championship priors are now handled through training_weight rather
  than physically duplicating rows, max_repeats does not duplicate data.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client


load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Teams that need Championship priors because they have very few PL matches
# ONLY include teams with < 10 PL matches in the training data
# Teams with 30+ PL matches should rely on their PL data alone
PROMOTED_TEAMS = {
    "Coventry City FC",  # 1 PL match - needs Championship priors
    "Hull City AFC",     # 1 PL match - needs Championship priors
    # "Ipswich Town FC" removed - has 39 PL matches (sufficient data!)
}

DEFAULT_PL_FEATURES_PATH = DATA_DIR / "features_2024_2026.csv"
DEFAULT_OUTPUT_PATH = DATA_DIR / "dixon_coles_training_2024_2026.csv"

DEFAULT_HISTORY_START_SEASON = 2023
DEFAULT_HISTORY_END_SEASON = 2025

DEFAULT_SEASON_WEIGHTS = {
    2023: 0.35,
    2024: 0.60,
    2025: 1.00,
}

DEFAULT_PRIOR_MATCH_WEIGHT = 0.40

# Compatibility with poisson_dixon_coles.py.
# Priors are weighted directly, so this does NOT duplicate rows.
DEFAULT_MAX_REPEATS = 1


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "must be present in .env"
        )

    return create_client(url, key)


def get_season_weights(
    start_season: int,
    end_season: int,
) -> dict[int, float]:
    if (
        start_season == DEFAULT_HISTORY_START_SEASON
        and end_season == DEFAULT_HISTORY_END_SEASON
    ):
        return {
            season: weight
            for season, weight in DEFAULT_SEASON_WEIGHTS.items()
            if start_season <= season <= end_season
        }

    seasons = list(range(start_season, end_season + 1))

    if len(seasons) == 1:
        return {
            seasons[0]: 1.0,
        }

    weights = np.linspace(
        0.35,
        1.0,
        len(seasons),
    )

    return {
        season: float(weight)
        for season, weight in zip(seasons, weights)
    }


def outcome_from_score(
    home_goals: int,
    away_goals: int,
) -> str:
    if home_goals > away_goals:
        return "Home Win"

    if home_goals < away_goals:
        return "Away Win"

    return "Draw"


def load_pl_matches(
    features_path: Path,
) -> pd.DataFrame:
    logger.info(
        "Loading Premier League data from %s",
        features_path,
    )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Premier League feature file not found: {features_path}"
        )

    df = pd.read_csv(features_path)

    required_columns = {
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required PL columns: "
            + ", ".join(sorted(missing_columns))
        )

    date_column = None

    for candidate in [
        "kickoff_at",
        "match_date",
        "date",
    ]:
        if candidate in df.columns:
            date_column = candidate
            break

    if date_column is None:
        raise ValueError(
            "Premier League features must contain one of: "
            "kickoff_at, match_date, date"
        )

    completed = df.copy()

    completed["home_goals"] = pd.to_numeric(
        completed["home_goals"],
        errors="coerce",
    )

    completed["away_goals"] = pd.to_numeric(
        completed["away_goals"],
        errors="coerce",
    )

    completed["kickoff_at"] = pd.to_datetime(
        completed[date_column],
        errors="coerce",
        utc=True,
    )

    completed = completed.dropna(
        subset=[
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "kickoff_at",
        ]
    ).copy()

    completed["home_team"] = completed["home_team"].astype(str)
    completed["away_team"] = completed["away_team"].astype(str)

    completed["home_goals"] = completed["home_goals"].astype(int)
    completed["away_goals"] = completed["away_goals"].astype(int)

    if "season_start_year" not in completed.columns:
        completed["season_start_year"] = np.nan

    logger.info(
        "Loaded %s completed Premier League matches.",
        len(completed),
    )

    return completed


def load_championship_matches(
    start_season: int,
    end_season: int,
) -> pd.DataFrame:
    supabase = get_supabase()

    logger.info(
        "Loading Championship history from seasons %s through %s.",
        start_season,
        end_season,
    )

    response = (
        supabase
        .table("raw_matches")
        .select(
            "source_match_id,"
            "competition_code,"
            "season_start_year,"
            "matchday,"
            "utc_date,"
            "status,"
            "home_team_name,"
            "away_team_name,"
            "home_score,"
            "away_score"
        )
        .eq(
            "competition_code",
            "ELC",
        )
        .eq(
            "status",
            "FINISHED",
        )
        .gte(
            "season_start_year",
            start_season,
        )
        .lte(
            "season_start_year",
            end_season,
        )
        .execute()
    )

    df = pd.DataFrame(response.data)

    if df.empty:
        raise RuntimeError(
            "No Championship matches were found in raw_matches "
            "for the requested seasons."
        )

    required_columns = {
        "source_match_id",
        "season_start_year",
        "utc_date",
        "home_team_name",
        "away_team_name",
        "home_score",
        "away_score",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Championship data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = df[
        df["home_team_name"].isin(PROMOTED_TEAMS)
        | df["away_team_name"].isin(PROMOTED_TEAMS)
    ].copy()

    df["home_score"] = pd.to_numeric(
        df["home_score"],
        errors="coerce",
    )

    df["away_score"] = pd.to_numeric(
        df["away_score"],
        errors="coerce",
    )

    df["utc_date"] = pd.to_datetime(
        df["utc_date"],
        errors="coerce",
        utc=True,
    )

    df["season_start_year"] = pd.to_numeric(
        df["season_start_year"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "source_match_id",
            "season_start_year",
            "utc_date",
            "home_team_name",
            "away_team_name",
            "home_score",
            "away_score",
        ]
    ).copy()

    if df.empty:
        raise RuntimeError(
            "No valid Championship matches were found for "
            "Ipswich, Coventry or Hull."
        )

    df["home_team_name"] = df["home_team_name"].astype(str)
    df["away_team_name"] = df["away_team_name"].astype(str)

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["season_start_year"] = (
        df["season_start_year"].astype(int)
    )

    df = df.sort_values(
        "utc_date",
        kind="stable",
    ).reset_index(drop=True)

    logger.info(
        "Loaded %s Championship matches involving promoted teams.",
        len(df),
    )

    return df


def calculate_league_environment(
    matches: pd.DataFrame,
    home_goal_column: str,
    away_goal_column: str,
) -> dict[str, float]:
    if matches.empty:
        raise ValueError(
            "Cannot calculate a scoring environment from zero matches."
        )

    match_count = len(matches)

    home_goals_per_match = float(
        matches[home_goal_column].sum() / match_count
    )

    away_goals_per_match = float(
        matches[away_goal_column].sum() / match_count
    )

    goals_per_team_match = float(
        (
            matches[home_goal_column].sum()
            + matches[away_goal_column].sum()
        )
        / (2 * match_count)
    )

    return {
        "home_goals_per_match": home_goals_per_match,
        "away_goals_per_match": away_goals_per_match,
        "goals_per_team_match": goals_per_team_match,
    }


def calculate_normalization_factors(
    pl_environment: dict[str, float],
    championship_environment: dict[str, float],
) -> dict[str, float]:
    championship_home = (
        championship_environment["home_goals_per_match"]
    )

    championship_away = (
        championship_environment["away_goals_per_match"]
    )

    if championship_home <= 0 or championship_away <= 0:
        raise ValueError(
            "Championship scoring averages must be greater than zero."
        )

    return {
        "home_goal_factor": float(
            pl_environment["home_goals_per_match"]
            / championship_home
        ),
        "away_goal_factor": float(
            pl_environment["away_goals_per_match"]
            / championship_away
        ),
    }


def calculate_environment_weight(
    normalization_factors: dict[str, float],
) -> float:
    """
    Convert Championship -> PL scoring-environment adjustment into
    a conservative row-weight adjustment.

    Real scorelines remain untouched.
    """

    average_factor = float(
        (
            normalization_factors["home_goal_factor"]
            + normalization_factors["away_goal_factor"]
        )
        / 2.0
    )

    return float(
        np.clip(
            average_factor,
            0.75,
            1.25,
        )
    )


def build_promoted_team_summary(
    championship_matches: pd.DataFrame,
    season_weights: dict[int, float],
    normalization_factors: dict[str, float],
    prior_match_weight: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    environment_weight = calculate_environment_weight(
        normalization_factors
    )

    for team in sorted(PROMOTED_TEAMS):
        team_matches = championship_matches[
            (
                championship_matches["home_team_name"]
                == team
            )
            | (
                championship_matches["away_team_name"]
                == team
            )
        ].copy()

        if team_matches.empty:
            raise ValueError(
                f"No Championship history found for {team}."
            )

        weighted_goals_for = 0.0
        weighted_goals_against = 0.0
        total_weight = 0.0

        for row in team_matches.itertuples(index=False):
            season_weight = season_weights.get(
                int(row.season_start_year),
                0.25,
            )

            effective_weight = (
                season_weight
                * prior_match_weight
                * environment_weight
            )

            if row.home_team_name == team:
                goals_for = float(row.home_score)
                goals_against = float(row.away_score)
            else:
                goals_for = float(row.away_score)
                goals_against = float(row.home_score)

            weighted_goals_for += (
                goals_for * effective_weight
            )

            weighted_goals_against += (
                goals_against * effective_weight
            )

            total_weight += effective_weight

        rows.append(
            {
                "team": team,
                "championship_matches": len(team_matches),
                "weighted_championship_gf_per_match": (
                    weighted_goals_for / total_weight
                    if total_weight > 0
                    else np.nan
                ),
                "weighted_championship_ga_per_match": (
                    weighted_goals_against / total_weight
                    if total_weight > 0
                    else np.nan
                ),
                "environment_weight": environment_weight,
            }
        )

    return pd.DataFrame(rows)


def create_normalized_prior_matches(
    championship_matches: pd.DataFrame,
    season_weights: dict[int, float],
    normalization_factors: dict[str, float],
    prior_match_weight: float,
) -> pd.DataFrame:
    """
    Create Championship prior rows.

    IMPORTANT:

    - Real Championship scorelines are preserved.
    - No synthetic integer scoreline rounding.
    - The Dixon-Coles trainer uses training_weight directly.
    - kickoff_at remains the real Championship match date so backtests can
      filter every prior using the historical prediction timestamp.
    """

    environment_weight = calculate_environment_weight(
        normalization_factors
    )

    rows: list[dict[str, object]] = []

    for row in championship_matches.itertuples(index=False):
        season_weight = season_weights.get(
            int(row.season_start_year),
            0.25,
        )

        effective_weight = float(
            season_weight
            * prior_match_weight
            * environment_weight
        )

        home_goals = int(row.home_score)
        away_goals = int(row.away_score)

        rows.append(
            {
                "home_team": str(row.home_team_name),
                "away_team": str(row.away_team_name),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "season_start_year": int(
                    row.season_start_year
                ),
                "kickoff_at": pd.to_datetime(
                    row.utc_date,
                    utc=True,
                ),
                "source": "normalized_championship_prior",
                "source_match_id": str(
                    row.source_match_id
                ),
                "outcome": outcome_from_score(
                    home_goals,
                    away_goals,
                ),
                "training_weight": effective_weight,
            }
        )

    prior_matches = pd.DataFrame(rows)

    if prior_matches.empty:
        raise RuntimeError(
            "No Championship prior matches were created."
        )

    return prior_matches


def prepare_pl_training_rows(
    pl_matches: pd.DataFrame,
) -> pd.DataFrame:
    rows = pl_matches[
        [
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "season_start_year",
            "kickoff_at",
        ]
    ].copy()

    if "raw_match_id" in pl_matches.columns:
        rows["source_match_id"] = (
            pl_matches["raw_match_id"]
            .astype(str)
        )

    elif "fixture_id" in pl_matches.columns:
        rows["source_match_id"] = (
            pl_matches["fixture_id"]
            .astype(str)
        )

    else:
        rows["source_match_id"] = (
            np.arange(len(rows))
            .astype(str)
        )

    rows["source"] = "real_premier_league"
    rows["training_weight"] = 1.0

    rows["outcome"] = [
        outcome_from_score(
            int(home_goals),
            int(away_goals),
        )
        for home_goals, away_goals in zip(
            rows["home_goals"],
            rows["away_goals"],
        )
    ]

    return rows


def run(
    pl_features_path: Path,
    output_path: Path,
    history_start_season: int,
    history_end_season: int,
    prior_match_weight: float,
    max_repeats: int = DEFAULT_MAX_REPEATS,
) -> Path:
    """
    Build the combined training file.

    max_repeats is accepted for backward compatibility with
    poisson_dixon_coles.py. It intentionally does not duplicate prior rows,
    because training_weight already represents their contribution.
    """

    if max_repeats < 1:
        raise ValueError(
            "max_repeats must be at least 1."
        )

    if max_repeats != 1:
        logger.info(
            "max_repeats=%s requested, but prior rows are not duplicated. "
            "Using training_weight directly.",
            max_repeats,
        )

    pl_matches = load_pl_matches(
        pl_features_path
    )

    championship_matches = (
        load_championship_matches(
            start_season=history_start_season,
            end_season=history_end_season,
        )
    )

    season_weights = get_season_weights(
        start_season=history_start_season,
        end_season=history_end_season,
    )

    pl_environment = (
        calculate_league_environment(
            matches=pl_matches,
            home_goal_column="home_goals",
            away_goal_column="away_goals",
        )
    )

    championship_environment = (
        calculate_league_environment(
            matches=championship_matches,
            home_goal_column="home_score",
            away_goal_column="away_score",
        )
    )

    normalization_factors = (
        calculate_normalization_factors(
            pl_environment=pl_environment,
            championship_environment=championship_environment,
        )
    )

    summary = build_promoted_team_summary(
        championship_matches=championship_matches,
        season_weights=season_weights,
        normalization_factors=normalization_factors,
        prior_match_weight=prior_match_weight,
    )

    championship_prior_rows = (
        create_normalized_prior_matches(
            championship_matches=championship_matches,
            season_weights=season_weights,
            normalization_factors=normalization_factors,
            prior_match_weight=prior_match_weight,
        )
    )

    pl_training_rows = prepare_pl_training_rows(
        pl_matches
    )

    combined = pd.concat(
        [
            pl_training_rows,
            championship_prior_rows,
        ],
        ignore_index=True,
    )

    combined["kickoff_at"] = pd.to_datetime(
        combined["kickoff_at"],
        errors="coerce",
        utc=True,
    )

    combined = combined.dropna(
        subset=["kickoff_at"]
    )

    combined = combined.sort_values(
        "kickoff_at",
        kind="stable",
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    environment_weight = calculate_environment_weight(
        normalization_factors
    )

    print()
    print("=" * 78)
    print("PROMOTED-TEAM PRIOR SUMMARY")
    print("=" * 78)
    print()

    print("PL scoring environment:")
    print(
        "  Home goals/match: "
        f"{pl_environment['home_goals_per_match']:.3f}"
    )
    print(
        "  Away goals/match: "
        f"{pl_environment['away_goals_per_match']:.3f}"
    )

    print()
    print("Championship scoring environment:")
    print(
        "  Home goals/match: "
        f"{championship_environment['home_goals_per_match']:.3f}"
    )
    print(
        "  Away goals/match: "
        f"{championship_environment['away_goals_per_match']:.3f}"
    )

    print()
    print("Championship -> PL adjustment:")
    print(
        "  Home factor: "
        f"{normalization_factors['home_goal_factor']:.3f}"
    )
    print(
        "  Away factor: "
        f"{normalization_factors['away_goal_factor']:.3f}"
    )
    print(
        "  Applied environment weight: "
        f"{environment_weight:.3f}"
    )

    print()
    print(
        summary.to_string(
            index=False,
            formatters={
                "weighted_championship_gf_per_match": (
                    lambda value: f"{value:.3f}"
                ),
                "weighted_championship_ga_per_match": (
                    lambda value: f"{value:.3f}"
                ),
                "environment_weight": (
                    lambda value: f"{value:.3f}"
                ),
            },
        )
    )

    print()
    print("=" * 78)
    print("COMBINED DIXON-COLES TRAINING DATA")
    print("=" * 78)

    print(
        "Real completed Premier League matches: "
        f"{len(pl_training_rows)}"
    )

    print(
        "Championship prior rows: "
        f"{len(championship_prior_rows)}"
    )

    print(
        "Total rows: "
        f"{len(combined)}"
    )

    print(
        f"Output: {output_path}"
    )

    print()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build promoted-team-aware Dixon-Coles "
            "training data."
        )
    )

    parser.add_argument(
        "--pl-features",
        default=str(DEFAULT_PL_FEATURES_PATH),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
    )

    parser.add_argument(
        "--history-start-season",
        type=int,
        default=DEFAULT_HISTORY_START_SEASON,
    )

    parser.add_argument(
        "--history-end-season",
        type=int,
        default=DEFAULT_HISTORY_END_SEASON,
    )

    parser.add_argument(
        "--prior-match-weight",
        type=float,
        default=DEFAULT_PRIOR_MATCH_WEIGHT,
    )

    parser.add_argument(
        "--max-repeats",
        type=int,
        default=DEFAULT_MAX_REPEATS,
        help=(
            "Compatibility argument. Prior rows are weighted directly "
            "and are not duplicated."
        ),
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    if args.history_end_season < args.history_start_season:
        raise ValueError(
            "--history-end-season cannot be earlier than "
            "--history-start-season."
        )

    if args.prior_match_weight <= 0:
        raise ValueError(
            "--prior-match-weight must be greater than zero."
        )

    if args.max_repeats < 1:
        raise ValueError(
            "--max-repeats must be at least 1."
        )

    run(
        pl_features_path=Path(
            args.pl_features
        ),
        output_path=Path(
            args.output
        ),
        history_start_season=(
            args.history_start_season
        ),
        history_end_season=(
            args.history_end_season
        ),
        prior_match_weight=(
            args.prior_match_weight
        ),
        max_repeats=(
            args.max_repeats
        ),
    )


if __name__ == "__main__":
    main()