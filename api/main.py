"""
main.py

Standalone Premier League prediction runner. No API, no server, no localhost.

Each run:
  1. Loads the current season's feature file.
  2. Detects the current gameweek (earliest gameweek with an incomplete fixture).
  3. If the *previous* gameweek is fully completed in Supabase but the local
     feature file still shows it as incomplete, refreshes real results by
     re-running ingest + feature engineering for this season. This makes sure
     the model trains on real final scores, never on its own predictions.
  4. Retrains the Dixon-Coles model on the refreshed data.
  5. Predicts every fixture in the current gameweek and prints the results
     to the terminal.

Usage:
    uv run python main.py --season 2025
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from features.feature_engineering import main as run_feature_engineering
from ingest.football_data_client import FootballDataClient
from models.poisson_dixon_coles import (
    FEATURES_DIR,
    DixonColesModel,
    train_model,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_feature_path(season: int) -> "pd.io.common.Path | __import__('pathlib').Path":
    from pathlib import Path

    return FEATURES_DIR / f"features_{season}.csv"


def load_features(season: int) -> pd.DataFrame:
    path = get_feature_path(season)

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}. "
            "Run feature engineering at least once before using main.py."
        )

    frame = pd.read_csv(path)

    frame["gameweek"] = pd.to_numeric(frame["gameweek"], errors="coerce")
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")

    return frame


def detect_current_gameweek(features: pd.DataFrame) -> int:
    incomplete = features[
        features["home_goals"].isna() | features["away_goals"].isna()
    ].dropna(subset=["gameweek"])

    if incomplete.empty:
        # Nothing left to predict this season; fall back to the final gameweek.
        return int(features["gameweek"].max())

    return int(incomplete["gameweek"].min())


def previous_gameweek_needs_refresh(features: pd.DataFrame, current_gameweek: int) -> bool:
    """
    True when the gameweek right before the current one still has any fixture
    locally marked incomplete. That means real results exist by now but
    haven't been pulled in yet.
    """

    previous_gameweek = current_gameweek - 1

    if previous_gameweek < 1:
        return False

    previous_fixtures = features[features["gameweek"] == previous_gameweek]

    if previous_fixtures.empty:
        return False

    return bool(
        previous_fixtures["home_goals"].isna().any()
        or previous_fixtures["away_goals"].isna().any()
    )


def refresh_real_results(season: int) -> None:
    """
    Pull the latest real match results into Supabase, then rebuild the local
    feature file from that data. This is the only way completed-match rows
    change from predicted/blank to real scores.
    """

    logger.info("Refreshing real results for season %s from football-data.org.", season)

    client = FootballDataClient()
    processed = client.ingest_matches(season=season)
    logger.info("Ingest complete. %s matches processed.", processed)

    logger.info("Rebuilding feature file for season %s.", season)

    import sys

    original_argv = sys.argv
    sys.argv = ["feature_engineering.py", "--season", str(season)]
    try:
        run_feature_engineering()
    finally:
        sys.argv = original_argv


def retrain(season: int, model_name: str) -> DixonColesModel:
    logger.info("Retraining model '%s' on season %s.", model_name, season)
    model_path = train_model(start_season=season - 1, end_season=season, model_name=model_name)
    return DixonColesModel.load(model_path)


def print_gameweek_predictions(
    model: DixonColesModel,
    fixtures: pd.DataFrame,
    season: int,
    gameweek: int,
) -> None:
    print()
    print("=" * 100)
    print(f"Premier League {season}/{str(season + 1)[-2:]} — Gameweek {gameweek}")
    print("=" * 100)
    print()

    rows = []

    for _, fixture in fixtures.iterrows():
        home_team = str(fixture["home_team"])
        away_team = str(fixture["away_team"])

        prediction = model.predict(home_team, away_team)

        rows.append(
            {
                "home": home_team,
                "away": away_team,
                "home_xg": prediction.expected_home_goals,
                "away_xg": prediction.expected_away_goals,
                "home_win": prediction.home_win_probability,
                "draw": prediction.draw_probability,
                "away_win": prediction.away_win_probability,
                "score": f"{prediction.most_likely_home_goals}-{prediction.most_likely_away_goals}",
            }
        )

    results_df = pd.DataFrame(rows)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "home_win": lambda v: f"{v * 100:.1f}%",
                "draw": lambda v: f"{v * 100:.1f}%",
                "away_win": lambda v: f"{v * 100:.1f}%",
            },
        )
    )
    print()


def run(season: int, model_name: str) -> None:
    features = load_features(season)
    current_gameweek = detect_current_gameweek(features)

    logger.info("Detected current gameweek: %s", current_gameweek)

    if previous_gameweek_needs_refresh(features, current_gameweek):
        logger.info(
            "Gameweek %s appears complete upstream but is stale locally. Refreshing.",
            current_gameweek - 1,
        )
        refresh_real_results(season)
        features = load_features(season)
        current_gameweek = detect_current_gameweek(features)
        logger.info("Gameweek after refresh: %s", current_gameweek)
    else:
        logger.info("No refresh needed; local data for the previous gameweek is current.")

    model = retrain(season, model_name)

    gameweek_fixtures = features[features["gameweek"] == current_gameweek].copy()

    if gameweek_fixtures.empty:
        raise ValueError(f"No fixtures found for gameweek {current_gameweek}.")

    print_gameweek_predictions(model, gameweek_fixtures, season, current_gameweek)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Premier League prediction runner (no server)."
    )

    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season starting year, e.g. 2025 for the 2025/26 season.",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model artifact name. Defaults to the season year.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    model_name = args.model_name or str(args.season)

    run(season=args.season, model_name=model_name)


if __name__ == "__main__":
    main()