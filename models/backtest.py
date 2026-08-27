"""
backtest.py

Leakage-safe walk-forward backtesting for the Dixon-Coles model.

For every historical prediction date T:

- Premier League training matches: kickoff_at < T
- Championship promoted-team priors: kickoff_at < T
- The current test match is never included in training

This prevents future information from leaking into
historical predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from models.poisson_dixon_coles import DixonColesModel


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

DEFAULT_INITIAL_TRAIN_SIZE = 150
DEFAULT_TEST_BLOCK_SIZE = 20

OUTCOME_TO_CLASS = {
    "Away Win": 0,
    "Draw": 1,
    "Home Win": 2,
}

CLASS_TO_OUTCOME = {
    0: "Away Win",
    1: "Draw",
    2: "Home Win",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def load_features(season: int) -> pd.DataFrame:
    """
    Load completed Premier League matches for the season being backtested.
    """

    path = FEATURES_DIR / f"features_{season}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    logger.info(
        "Loading backtest features from %s",
        path,
    )

    df = pd.read_csv(path)

    required_columns = {
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
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
            "Could not find a match date column. "
            "Expected one of: kickoff_at, match_date, date"
        )

    df["_backtest_date"] = pd.to_datetime(
        df[date_column],
        errors="coerce",
        utc=True,
    )

    df["home_goals"] = pd.to_numeric(
        df["home_goals"],
        errors="coerce",
    )

    df["away_goals"] = pd.to_numeric(
        df["away_goals"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "_backtest_date",
            "home_goals",
            "away_goals",
            "home_team",
            "away_team",
        ]
    ).copy()

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    df = df.sort_values(
        "_backtest_date",
        kind="stable",
    ).reset_index(drop=True)

    df["_target"] = np.select(
        [
            df["home_goals"] > df["away_goals"],
            df["home_goals"] == df["away_goals"],
            df["home_goals"] < df["away_goals"],
        ],
        [
            OUTCOME_TO_CLASS["Home Win"],
            OUTCOME_TO_CLASS["Draw"],
            OUTCOME_TO_CLASS["Away Win"],
        ],
    ).astype(int)

    logger.info(
        "Loaded %s completed matches for backtesting.",
        len(df),
    )

    return df


def load_promoted_aware_training_data() -> pd.DataFrame:
    """
    Load the combined historical training dataset containing:

    - Premier League historical matches
    - Championship promoted-team priors
    """

    path = (
        FEATURES_DIR
        / "dixon_coles_training_2024_2026.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Promoted-aware training data not found. "
            "Run promoted_team_normalizer first: "
            f"{path}"
        )

    logger.info(
        "Loading promoted-aware historical data from %s",
        path,
    )

    df = pd.read_csv(path)

    required_columns = {
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "kickoff_at",
        "source",
        "training_weight",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Promoted-aware training data is missing: "
            + ", ".join(sorted(missing))
        )

    df["kickoff_at"] = pd.to_datetime(
        df["kickoff_at"],
        errors="coerce",
        utc=True,
    )

    df["home_goals"] = pd.to_numeric(
        df["home_goals"],
        errors="coerce",
    )

    df["away_goals"] = pd.to_numeric(
        df["away_goals"],
        errors="coerce",
    )

    df["training_weight"] = pd.to_numeric(
        df["training_weight"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "kickoff_at",
            "training_weight",
        ]
    ).copy()

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["training_weight"] = df["training_weight"].astype(float)

    df = df[
        df["training_weight"] > 0
    ].copy()

    df = df.sort_values(
        "kickoff_at",
        kind="stable",
    ).reset_index(drop=True)

    logger.info(
        "Loaded %s promoted-aware historical rows.",
        len(df),
    )

    return df


def build_dixon_coles_training_window(
    historical_training_data: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Strict leakage protection.

    Every training row must have:

        kickoff_at < cutoff_date

    This applies to both Premier League matches and
    Championship promoted-team prior matches.
    """

    cutoff_date = pd.to_datetime(
        cutoff_date,
        utc=True,
    )

    train_df = historical_training_data[
        historical_training_data["kickoff_at"]
        < cutoff_date
    ].copy()

    if train_df.empty:
        raise ValueError(
            "No Dixon-Coles training data exists before "
            f"{cutoff_date}."
        )

    return train_df.sort_values(
        "kickoff_at",
        kind="stable",
    ).reset_index(drop=True)


def fit_dixon_coles(
    train_df: pd.DataFrame,
) -> DixonColesModel:
    """
    Train a fresh Dixon-Coles model.
    """

    model = DixonColesModel()

    training_columns = [
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    training_data = train_df[
        training_columns
    ].copy()

    model.fit(training_data)

    return model


def dixon_coles_probabilities(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
) -> np.ndarray:
    """
    Return probabilities in this fixed class order:

    0 = Away Win
    1 = Draw
    2 = Home Win
    """

    try:
        prediction = model.predict(
            home_team=home_team,
            away_team=away_team,
        )

        probabilities = np.array(
            [
                prediction.away_win_probability,
                prediction.draw_probability,
                prediction.home_win_probability,
            ],
            dtype=float,
        )

        total = probabilities.sum()

        if total <= 0:
            return np.array(
                [1 / 3, 1 / 3, 1 / 3],
                dtype=float,
            )

        return probabilities / total

    except ValueError as error:
        logger.warning(
            "Dixon-Coles could not predict %s vs %s: %s",
            home_team,
            away_team,
            error,
        )

        return np.array(
            [1 / 3, 1 / 3, 1 / 3],
            dtype=float,
        )


def calculate_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Multiclass Brier score.
    """

    one_hot = np.eye(
        3,
        dtype=float,
    )[y_true]

    return float(
        np.mean(
            np.sum(
                (probabilities - one_hot) ** 2,
                axis=1,
            )
        )
    )


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """
    Calculate Accuracy, multiclass Log Loss and Brier Score.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        1e-15,
        1.0,
    )

    probabilities = probabilities / probabilities.sum(
        axis=1,
        keepdims=True,
    )

    predicted_classes = np.argmax(
        probabilities,
        axis=1,
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predicted_classes,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1, 2],
            )
        ),
        "brier_score": calculate_brier_score(
            y_true,
            probabilities,
        ),
    }


def run_walk_forward_backtest(
    df: pd.DataFrame,
    historical_training_data: pd.DataFrame,
    initial_train_size: int,
    test_block_size: int,
) -> pd.DataFrame:
    """
    Run leakage-safe walk-forward backtesting.

    For every test match, Dixon-Coles is trained only on
    matches strictly before that fixture's kickoff time.

    The initial_train_size controls how many completed matches
    at the beginning of the backtest season are skipped before
    evaluation starts.
    """

    if initial_train_size < 1:
        raise ValueError(
            "initial_train_size must be at least 1."
        )

    if initial_train_size >= len(df):
        raise ValueError(
            "initial_train_size must be smaller than "
            "the total number of matches."
        )

    if test_block_size < 1:
        raise ValueError(
            "test_block_size must be at least 1."
        )

    records: list[dict] = []

    block_number = 1
    test_start = initial_train_size

    while test_start < len(df):
        test_end = min(
            test_start + test_block_size,
            len(df),
        )

        test_df = df.iloc[
            test_start:test_end
        ].copy()

        logger.info(
            "Walk-forward block %s | Test matches: %s",
            block_number,
            len(test_df),
        )

        for local_index, (_, row) in enumerate(
            test_df.iterrows()
        ):
            cutoff_date = row["_backtest_date"]

            dc_train_df = (
                build_dixon_coles_training_window(
                    historical_training_data=
                    historical_training_data,
                    cutoff_date=cutoff_date,
                )
            )

            logger.info(
                "Block %s | Fixture %s/%s | "
                "Training Dixon-Coles on %s rows",
                block_number,
                local_index + 1,
                len(test_df),
                len(dc_train_df),
            )

            model = fit_dixon_coles(
                dc_train_df
            )

            probabilities = (
                dixon_coles_probabilities(
                    model=model,
                    home_team=str(
                        row["home_team"]
                    ),
                    away_team=str(
                        row["away_team"]
                    ),
                )
            )

            predicted_class = int(
                np.argmax(probabilities)
            )

            actual_class = int(
                row["_target"]
            )

            records.append(
                {
                    "block": block_number,
                    "fixture_row": (
                        test_start
                        + local_index
                    ),
                    "kickoff_at": (
                        row["_backtest_date"]
                    ),
                    "home_team": (
                        row["home_team"]
                    ),
                    "away_team": (
                        row["away_team"]
                    ),
                    "home_goals": int(
                        row["home_goals"]
                    ),
                    "away_goals": int(
                        row["away_goals"]
                    ),
                    "actual_class": actual_class,
                    "actual_outcome": (
                        CLASS_TO_OUTCOME[
                            actual_class
                        ]
                    ),
                    "away_win_probability": float(
                        probabilities[0]
                    ),
                    "draw_probability": float(
                        probabilities[1]
                    ),
                    "home_win_probability": float(
                        probabilities[2]
                    ),
                    "prediction": (
                        CLASS_TO_OUTCOME[
                            predicted_class
                        ]
                    ),
                    "prediction_correct": bool(
                        predicted_class
                        == actual_class
                    ),
                    "training_rows": len(
                        dc_train_df
                    ),
                }
            )

        logger.info(
            "Completed block %s.",
            block_number,
        )

        test_start = test_end
        block_number += 1

    if not records:
        raise ValueError(
            "No backtest predictions were generated."
        )

    return pd.DataFrame(records)


def summarize_backtest(
    results: pd.DataFrame,
) -> dict[str, float]:
    """
    Summarize the Dixon-Coles backtest.
    """

    y_true = results[
        "actual_class"
    ].to_numpy(
        dtype=int
    )

    probabilities = results[
        [
            "away_win_probability",
            "draw_probability",
            "home_win_probability",
        ]
    ].to_numpy(
        dtype=float
    )

    return calculate_metrics(
        y_true,
        probabilities,
    )


def print_summary(
    summary: dict[str, float],
    total_predictions: int,
) -> None:
    """
    Print backtest results.
    """

    print()
    print("=" * 72)
    print("DIXON-COLES WALK-FORWARD BACKTEST RESULTS")
    print("=" * 72)
    print()

    print(
        f"Total unseen predictions: "
        f"{total_predictions}"
    )

    print()

    print(
        f"Accuracy:    "
        f"{summary['accuracy'] * 100:.2f}%"
    )

    print(
        f"Log Loss:    "
        f"{summary['log_loss']:.4f}"
    )

    print(
        f"Brier Score: "
        f"{summary['brier_score']:.4f}"
    )

    print()
    print("=" * 72)
    print()


def save_results(
    results: pd.DataFrame,
    summary: dict[str, float],
    season: int,
) -> tuple[Path, Path]:
    """
    Save detailed predictions and summary metrics.
    """

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        ARTIFACTS_DIR
        / f"backtest_predictions_{season}.csv"
    )

    summary_path = (
        ARTIFACTS_DIR
        / f"backtest_summary_{season}.json"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    logger.info(
        "Detailed predictions saved to %s",
        results_path,
    )

    logger.info(
        "Summary saved to %s",
        summary_path,
    )

    return (
        results_path,
        summary_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe Dixon-Coles walk-forward "
            "backtest for the Premier League predictor."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help=(
            "Season starting year, e.g. "
            "2025 for 2025/26."
        ),
    )

    parser.add_argument(
        "--initial-train-size",
        type=int,
        default=DEFAULT_INITIAL_TRAIN_SIZE,
        help=(
            "Number of early season matches to skip "
            "before starting backtest evaluation."
        ),
    )

    parser.add_argument(
        "--test-block-size",
        type=int,
        default=DEFAULT_TEST_BLOCK_SIZE,
        help=(
            "Number of fixtures processed per "
            "walk-forward block."
        ),
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    logger.info(
        "Starting leakage-safe Dixon-Coles "
        "walk-forward backtest for %s/%s",
        args.season,
        str(args.season + 1)[-2:],
    )

    df = load_features(
        args.season
    )

    historical_training_data = (
        load_promoted_aware_training_data()
    )

    results = run_walk_forward_backtest(
        df=df,
        historical_training_data=
        historical_training_data,
        initial_train_size=
        args.initial_train_size,
        test_block_size=
        args.test_block_size,
    )

    summary = summarize_backtest(
        results
    )

    save_results(
        results=results,
        summary=summary,
        season=args.season,
    )

    print_summary(
        summary=summary,
        total_predictions=len(results),
    )


if __name__ == "__main__":
    main()