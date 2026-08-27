from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from models.poisson_dixon_coles import DixonColesModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dixon_coles_training_2024_2026.csv"
)

ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
)


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def load_training_data() -> pd.DataFrame:
    """
    Load the combined promoted-aware training dataset.

    This dataset contains:
    1. Real Premier League matches
    2. Normalized Championship matches for promoted teams

    The original poisson_dixon_coles.py is not modified.
    """

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            "Training file not found:\n"
            f"{TRAINING_DATA_PATH}"
        )

    logger.info(
        "Loading promoted-aware training data from %s",
        TRAINING_DATA_PATH,
    )

    df = pd.read_csv(TRAINING_DATA_PATH)

    required_columns = [
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    df = df.dropna(
        subset=required_columns,
    ).copy()

    df["home_goals"] = (
        pd.to_numeric(
            df["home_goals"],
            errors="raise",
        ).astype(int)
    )

    df["away_goals"] = (
        pd.to_numeric(
            df["away_goals"],
            errors="raise",
        ).astype(int)
    )

    logger.info(
        "Loaded %s training matches.",
        len(df),
    )

    return df


def print_training_summary(
    df: pd.DataFrame,
) -> None:
    print()
    print("=" * 72)
    print("PROMOTED-AWARE DIXON-COLES TRAINING DATA")
    print("=" * 72)
    print()

    print(
        f"Total matches: {len(df)}"
    )

    if "source" in df.columns:
        print()
        print("MATCHES BY SOURCE")
        print("-" * 72)

        source_counts = (
            df["source"]
            .fillna("unknown")
            .value_counts()
        )

        for source, count in source_counts.items():
            print(
                f"{source}: {count}"
            )

    promoted_teams = [
        "Coventry City FC",
        "Hull City AFC",
        "Ipswich Town FC",
    ]

    print()
    print("PROMOTED TEAM MATCHES IN TRAINING DATA")
    print("-" * 72)

    for team in promoted_teams:
        home_matches = (
            df["home_team"] == team
        ).sum()

        away_matches = (
            df["away_team"] == team
        ).sum()

        print(
            f"{team}: "
            f"{home_matches + away_matches} appearances "
            f"({home_matches} home, "
            f"{away_matches} away)"
        )


def train_model(
    model_name: str,
) -> None:
    """
    Train the original Dixon-Coles model using the promoted-aware
    combined dataset.

    This does NOT modify poisson_dixon_coles.py.
    """

    df = load_training_data()

    print_training_summary(df)

    teams = sorted(
        set(df["home_team"])
        | set(df["away_team"])
    )

    logger.info(
        "Training Dixon-Coles model on %s matches and %s teams.",
        len(df),
        len(teams),
    )

    model = DixonColesModel()

    model.fit(
        matches=df,
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        ARTIFACTS_DIR
        / f"dixon_coles_{model_name}.json"
    )

    model.save(
        model_path,
    )

    logger.info(
        "Model saved to %s",
        model_path,
    )

    strengths = []

    for team in model.teams:
        team_index = (
            model.team_to_index[team]
        )

        strengths.append(
            {
                "team": team,
                "attack": float(
                    model.attack[team_index]
                ),
                "defence": float(
                    model.defence[team_index]
                ),
            }
        )

    strengths_df = pd.DataFrame(
        strengths,
    )

    strengths_path = (
        ARTIFACTS_DIR
        / f"team_strengths_{model_name}.csv"
    )

    strengths_df.to_csv(
        strengths_path,
        index=False,
    )

    logger.info(
        "Team strengths saved to %s",
        strengths_path,
    )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print()

    print(
        f"Total matches used: {len(df)}"
    )

    print(
        f"Total teams: {len(model.teams)}"
    )

    print()

    print(
        "PROMOTED TEAM MODEL STRENGTHS"
    )
    print("-" * 72)

    promoted_teams = [
        "Coventry City FC",
        "Hull City AFC",
        "Ipswich Town FC",
    ]

    for team in promoted_teams:
        if team not in model.team_to_index:
            print(
                f"{team}: NOT FOUND"
            )
            continue

        team_index = (
            model.team_to_index[team]
        )

        print(
            f"{team} | "
            f"Attack={model.attack[team_index]:.3f} | "
            f"Defence={model.defence[team_index]:.3f}"
        )


def predict_match(
    model_name: str,
    home_team: str,
    away_team: str,
) -> None:
    """
    Load the promoted-aware Dixon-Coles model and predict one fixture.
    """

    model_path = (
        ARTIFACTS_DIR
        / f"dixon_coles_{model_name}.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found:\n"
            f"{model_path}"
        )

    model = DixonColesModel.load(
        model_path,
    )

    logger.info(
        "Model loaded from %s",
        model_path,
    )

    if home_team not in model.team_to_index:
        raise ValueError(
            f"Unknown home team: {home_team}"
        )

    if away_team not in model.team_to_index:
        raise ValueError(
            f"Unknown away team: {away_team}"
        )

    prediction = model.predict(
        home_team,
        away_team,
    )

    matrix = model.scoreline_matrix(
        home_team,
        away_team,
    )

    print()
    print("=" * 60)
    print(
        f"{prediction.home_team} vs "
        f"{prediction.away_team}"
    )
    print("=" * 60)
    print()

    print(
        "Expected goals: "
        f"{prediction.expected_home_goals:.3f} - "
        f"{prediction.expected_away_goals:.3f}"
    )

    print()

    print(
        "Home win: "
        f"{prediction.home_win_probability * 100:.2f}%"
    )

    print(
        "Draw:     "
        f"{prediction.draw_probability * 100:.2f}%"
    )

    print(
        "Away win: "
        f"{prediction.away_win_probability * 100:.2f}%"
    )

    print()

    print(
        "Most likely score: "
        f"{prediction.most_likely_home_goals}-"
        f"{prediction.most_likely_away_goals} "
        f"("
        f"{prediction.most_likely_score_probability * 100:.2f}%"
        f")"
    )

    print()

    print(
        "Scoreline probability matrix "
        "(rows=home goals, columns=away goals):"
    )

    print()

    matrix_df = pd.DataFrame(
        matrix * 100,
        index=[
            f"H{i}"
            for i in range(
                matrix.shape[0]
            )
        ],
        columns=[
            f"A{i}"
            for i in range(
                matrix.shape[1]
            )
        ],
    )

    print(
        matrix_df.round(2).to_string()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promoted-team-aware Dixon-Coles model."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help=(
            "Train using Premier League data plus "
            "normalized Championship history."
        ),
    )

    train_parser.add_argument(
        "--model-name",
        required=True,
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict one fixture.",
    )

    predict_parser.add_argument(
        "--model-name",
        required=True,
    )

    predict_parser.add_argument(
        "--home",
        required=True,
    )

    predict_parser.add_argument(
        "--away",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    if args.command == "train":
        train_model(
            model_name=args.model_name,
        )

    elif args.command == "predict":
        predict_match(
            model_name=args.model_name,
            home_team=args.home,
            away_team=args.away,
        )


if __name__ == "__main__":
    main()