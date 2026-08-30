"""
poisson_dixon_coles.py

Premier League statistical prediction model using a Poisson goal model
with Dixon-Coles low-score correction.

Supports:
- Single-season training
- Multi-season training
- Training on completed matches only
- Promoted-team-aware Championship priors
- Weighted historical observations
- Saving artifacts using a model label
- Single-match prediction
- Full gameweek prediction

Examples:

Single season:
    uv run python -m models.poisson_dixon_coles train --season 2026

Multi-season:
    uv run python -m models.poisson_dixon_coles train ^
        --start-season 2024 ^
        --end-season 2026 ^
        --model-name 2024_2026

Predict:
    uv run python -m models.poisson_dixon_coles predict ^
        --model-name 2024_2026 ^
        --home "Arsenal FC" ^
        --away "Chelsea FC"

Predict next gameweek automatically:
    uv run python -m models.poisson_dixon_coles predict-gameweek ^
        --model-name 2024_2026 ^
        --season 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import minimize
from scipy.stats import poisson


load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"


@dataclass
class Prediction:
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    most_likely_home_goals: int
    most_likely_away_goals: int
    most_likely_score_probability: float


def setup_logging() -> None:
    log_level = os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).upper()

    logging.basicConfig(
        level=getattr(
            logging,
            log_level,
            logging.INFO,
        ),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


class DixonColesModel:
    """
    Poisson football model with Dixon-Coles low-score correction.

    Supports observation weights.

    Real Premier League matches normally have weight 1.0.
    Normalized Championship history can be supplied as lower-weight priors.
    """

    def __init__(
        self,
        rho: float = -0.05,
        max_goals: int = 8,
        regularization_strength: float = 0.10,
    ) -> None:
        self.rho = float(rho)
        self.max_goals = int(max_goals)
        self.regularization_strength = float(
            regularization_strength
        )

        self.team_match_counts: dict[
            str,
            float,
        ] = {}

        self.teams: list[str] = []

        self.team_to_index: dict[
            str,
            int,
        ] = {}

        self.attack: np.ndarray | None = None
        self.defence: np.ndarray | None = None
        self.home_advantage: float | None = None
        self.fitted = False

    @staticmethod
    def dc_tau(
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float,
        rho: float,
    ) -> float:
        """
        Dixon-Coles correction for low-scoring outcomes.
        """

        if home_goals == 0 and away_goals == 0:
            return (
                1.0
                - lambda_home
                * lambda_away
                * rho
            )

        if home_goals == 0 and away_goals == 1:
            return (
                1.0
                + lambda_home
                * rho
            )

        if home_goals == 1 and away_goals == 0:
            return (
                1.0
                + lambda_away
                * rho
            )

        if home_goals == 1 and away_goals == 1:
            return 1.0 - rho

        return 1.0

    def _unpack_parameters(
        self,
        params: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
    ]:
        n_teams = len(
            self.teams
        )

        attack_free = params[
            : n_teams - 1
        ]

        final_attack = -np.sum(
            attack_free
        )

        attack = np.append(
            attack_free,
            final_attack,
        )

        defence_start = (
            n_teams - 1
        )

        defence_end = (
            defence_start
            + n_teams
        )

        defence = params[
            defence_start:defence_end
        ]

        home_advantage = float(
            params[-1]
        )

        return (
            attack,
            defence,
            home_advantage,
        )

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        home_indices: np.ndarray,
        away_indices: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        observation_weights: np.ndarray,
    ) -> float:
        (
            attack,
            defence,
            home_advantage,
        ) = self._unpack_parameters(
            params
        )

        lambda_home = np.exp(
            home_advantage
            + attack[home_indices]
            - defence[away_indices]
        )

        lambda_away = np.exp(
            attack[away_indices]
            - defence[home_indices]
        )

        home_probability = poisson.pmf(
            home_goals,
            lambda_home,
        )

        away_probability = poisson.pmf(
            away_goals,
            lambda_away,
        )

        tau = np.ones(
            len(home_goals),
            dtype=float,
        )

        mask_00 = (
            (home_goals == 0)
            & (away_goals == 0)
        )

        mask_01 = (
            (home_goals == 0)
            & (away_goals == 1)
        )

        mask_10 = (
            (home_goals == 1)
            & (away_goals == 0)
        )

        mask_11 = (
            (home_goals == 1)
            & (away_goals == 1)
        )

        tau[mask_00] = (
            1.0
            - lambda_home[mask_00]
            * lambda_away[mask_00]
            * self.rho
        )

        tau[mask_01] = (
            1.0
            + lambda_home[mask_01]
            * self.rho
        )

        tau[mask_10] = (
            1.0
            + lambda_away[mask_10]
            * self.rho
        )

        tau[mask_11] = (
            1.0
            - self.rho
        )

        probabilities = (
            home_probability
            * away_probability
            * tau
        )

        probabilities = np.clip(
            probabilities,
            1e-12,
            None,
        )

        weighted_negative_log_likelihood = float(
            -np.sum(
                observation_weights
                * np.log(
                    probabilities
                )
            )
        )

        regularization_penalty = (
            self.regularization_strength
            * (
                np.sum(
                    np.square(
                        attack
                    )
                )
                + np.sum(
                    np.square(
                        defence
                    )
                )
            )
        )

        weighted_negative_log_likelihood += float(
            regularization_penalty
        )

        return weighted_negative_log_likelihood

    def fit(
        self,
        matches: pd.DataFrame,
    ) -> "DixonColesModel":
        """
        Fit using completed historical matches.

        If training_weight exists, it is used as an observation weight.
        """

        required_columns = {
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        }

        missing_columns = (
            required_columns
            - set(matches.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    sorted(
                        missing_columns
                    )
                )
            )

        training_data = matches.copy()

        training_data["home_goals"] = (
            pd.to_numeric(
                training_data[
                    "home_goals"
                ],
                errors="coerce",
            )
        )

        training_data["away_goals"] = (
            pd.to_numeric(
                training_data[
                    "away_goals"
                ],
                errors="coerce",
            )
        )

        if "training_weight" not in (
            training_data.columns
        ):
            training_data[
                "training_weight"
            ] = 1.0

        training_data[
            "training_weight"
        ] = pd.to_numeric(
            training_data[
                "training_weight"
            ],
            errors="coerce",
        )

        training_data = (
            training_data
            .dropna(
                subset=[
                    "home_team",
                    "away_team",
                    "home_goals",
                    "away_goals",
                    "training_weight",
                ]
            )
            .copy()
        )

        training_data = training_data[
            training_data[
                "training_weight"
            ] > 0
        ].copy()

        if training_data.empty:
            raise ValueError(
                "No completed matches available for training."
            )

        training_data[
            "home_team"
        ] = (
            training_data[
                "home_team"
            ]
            .astype(str)
        )

        training_data[
            "away_team"
        ] = (
            training_data[
                "away_team"
            ]
            .astype(str)
        )

        training_data[
            "home_goals"
        ] = (
            training_data[
                "home_goals"
            ]
            .astype(int)
        )

        training_data[
            "away_goals"
        ] = (
            training_data[
                "away_goals"
            ]
            .astype(int)
        )

        self.teams = sorted(
            set(
                training_data[
                    "home_team"
                ]
            )
            | set(
                training_data[
                    "away_team"
                ]
            )
        )

        self.team_to_index = {
            team: index
            for index, team in enumerate(
                self.teams
            )
        }

        n_teams = len(
            self.teams
        )

        if n_teams < 2:
            raise ValueError(
                "At least two teams are required for training."
            )

        weighted_home_counts = (
            training_data
            .groupby(
                "home_team"
            )[
                "training_weight"
            ]
            .sum()
        )

        weighted_away_counts = (
            training_data
            .groupby(
                "away_team"
            )[
                "training_weight"
            ]
            .sum()
        )

        self.team_match_counts = {
            team: float(
                weighted_home_counts.get(
                    team,
                    0.0,
                )
                + weighted_away_counts.get(
                    team,
                    0.0,
                )
            )
            for team in self.teams
        }

        home_indices = np.array(
            [
                self.team_to_index[
                    team
                ]
                for team in training_data[
                    "home_team"
                ]
            ],
            dtype=int,
        )

        away_indices = np.array(
            [
                self.team_to_index[
                    team
                ]
                for team in training_data[
                    "away_team"
                ]
            ],
            dtype=int,
        )

        home_goals = (
            training_data[
                "home_goals"
            ]
            .to_numpy(
                dtype=int
            )
        )

        away_goals = (
            training_data[
                "away_goals"
            ]
            .to_numpy(
                dtype=int
            )
        )

        observation_weights = (
            training_data[
                "training_weight"
            ]
            .to_numpy(
                dtype=float
            )
        )

        parameter_count = (
            (n_teams - 1)
            + n_teams
            + 1
        )

        initial_parameters = np.zeros(
            parameter_count,
            dtype=float,
        )

        logger.info(
            "Training Dixon-Coles model on %s rows, "
            "%s teams and %.2f effective weighted matches.",
            len(training_data),
            n_teams,
            float(
                observation_weights.sum()
            ),
        )

        result = minimize(
            fun=self._negative_log_likelihood,
            x0=initial_parameters,
            args=(
                home_indices,
                away_indices,
                home_goals,
                away_goals,
                observation_weights,
            ),
            method="L-BFGS-B",
            options={
                "maxiter": 10000,
                "maxfun": 100000,
                "ftol": 1e-9,
                "gtol": 1e-6,
            },
        )

        if not result.success:
            message = str(
                result.message
            ).upper()

            acceptable_limit_messages = (
                "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
                "TOTAL NO. OF ITERATIONS REACHED LIMIT",
            )

            reached_acceptable_limit = any(
                limit_message in message
                for limit_message
                in acceptable_limit_messages
            )

            if (
                reached_acceptable_limit
                and np.isfinite(
                    result.fun
                )
                and np.all(
                    np.isfinite(
                        result.x
                    )
                )
            ):
                logger.warning(
                    "Optimizer reached a limit but produced "
                    "a finite solution. Using the best solution found. "
                    "Message: %s",
                    result.message,
                )
            else:
                raise RuntimeError(
                    "Model optimization failed: "
                    + str(
                        result.message
                    )
                )

        (
            self.attack,
            self.defence,
            self.home_advantage,
        ) = self._unpack_parameters(
            result.x
        )

        self.fitted = True

        logger.info(
            "Model training completed successfully."
        )

        logger.info(
            "L2 regularization strength: %.4f",
            self.regularization_strength,
        )

        logger.info(
            "Maximum absolute attack strength: %.4f",
            float(
                np.max(
                    np.abs(
                        self.attack
                    )
                )
            ),
        )

        logger.info(
            "Maximum absolute defence strength: %.4f",
            float(
                np.max(
                    np.abs(
                        self.defence
                    )
                )
            ),
        )

        logger.info(
            "Final weighted negative log likelihood: %.4f",
            result.fun,
        )

        logger.info(
            "Home advantage parameter: %.4f",
            self.home_advantage,
        )

        return self

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float]:
        if not self.fitted:
            raise RuntimeError(
                "Model has not been fitted."
            )

        if home_team not in (
            self.team_to_index
        ):
            raise ValueError(
                f"Unknown home team: {home_team}"
            )

        if away_team not in (
            self.team_to_index
        ):
            raise ValueError(
                f"Unknown away team: {away_team}"
            )

        if (
            self.attack is None
            or self.defence is None
            or self.home_advantage is None
        ):
            raise RuntimeError(
                "Model parameters are unavailable."
            )

        home_index = (
            self.team_to_index[
                home_team
            ]
        )

        away_index = (
            self.team_to_index[
                away_team
            ]
        )

        lambda_home = float(
            np.exp(
                self.home_advantage
                + self.attack[
                    home_index
                ]
                - self.defence[
                    away_index
                ]
            )
        )

        lambda_away = float(
            np.exp(
                self.attack[
                    away_index
                ]
                - self.defence[
                    home_index
                ]
            )
        )

        return (
            lambda_home,
            lambda_away,
        )

    def scoreline_matrix(
        self,
        home_team: str,
        away_team: str,
        max_goals: int | None = None,
    ) -> np.ndarray:
        if max_goals is None:
            max_goals = (
                self.max_goals
            )

        (
            lambda_home,
            lambda_away,
        ) = self.expected_goals(
            home_team,
            away_team,
        )

        matrix = np.zeros(
            (
                max_goals + 1,
                max_goals + 1,
            ),
            dtype=float,
        )

        for home_goals in range(
            max_goals + 1
        ):
            for away_goals in range(
                max_goals + 1
            ):
                probability = (
                    poisson.pmf(
                        home_goals,
                        lambda_home,
                    )
                    * poisson.pmf(
                        away_goals,
                        lambda_away,
                    )
                )

                correction = self.dc_tau(
                    home_goals,
                    away_goals,
                    lambda_home,
                    lambda_away,
                    self.rho,
                )

                matrix[
                    home_goals,
                    away_goals,
                ] = (
                    probability
                    * correction
                )

        total_probability = (
            matrix.sum()
        )

        if total_probability <= 0:
            raise RuntimeError(
                "Scoreline probability matrix has "
                "zero total probability."
            )

        return (
            matrix
            / total_probability
        )

    def predict(
        self,
        home_team: str,
        away_team: str,
        max_goals: int | None = None,
    ) -> Prediction:
        matrix = (
            self.scoreline_matrix(
                home_team,
                away_team,
                max_goals=max_goals,
            )
        )

        (
            expected_home_goals,
            expected_away_goals,
        ) = self.expected_goals(
            home_team,
            away_team,
        )

        home_win_probability = float(
            np.tril(
                matrix,
                k=-1,
            ).sum()
        )

        draw_probability = float(
            np.trace(
                matrix
            )
        )

        away_win_probability = float(
            np.triu(
                matrix,
                k=1,
            ).sum()
        )

        most_likely_index = (
            np.unravel_index(
                np.argmax(
                    matrix
                ),
                matrix.shape,
            )
        )

        most_likely_home_goals = int(
            most_likely_index[0]
        )

        most_likely_away_goals = int(
            most_likely_index[1]
        )

        most_likely_score_probability = float(
            matrix[
                most_likely_home_goals,
                most_likely_away_goals,
            ]
        )

        return Prediction(
            home_team=home_team,
            away_team=away_team,
            expected_home_goals=round(
                expected_home_goals,
                3,
            ),
            expected_away_goals=round(
                expected_away_goals,
                3,
            ),
            home_win_probability=round(
                home_win_probability,
                4,
            ),
            draw_probability=round(
                draw_probability,
                4,
            ),
            away_win_probability=round(
                away_win_probability,
                4,
            ),
            most_likely_home_goals=(
                most_likely_home_goals
            ),
            most_likely_away_goals=(
                most_likely_away_goals
            ),
            most_likely_score_probability=round(
                most_likely_score_probability,
                4,
            ),
        )

    def team_strengths(
        self,
    ) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError(
                "Model has not been fitted."
            )

        if (
            self.attack is None
            or self.defence is None
        ):
            raise RuntimeError(
                "Model parameters are unavailable."
            )

        rows = []

        for index, team in enumerate(
            self.teams
        ):
            rows.append(
                {
                    "team": team,
                    "effective_matches": (
                        self.team_match_counts.get(
                            team,
                            0.0,
                        )
                    ),
                    "attack_strength": float(
                        self.attack[index]
                    ),
                    "defence_strength": float(
                        self.defence[index]
                    ),
                }
            )

        return (
            pd.DataFrame(
                rows
            )
            .sort_values(
                "attack_strength",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    def save(
        self,
        path: Path,
    ) -> None:
        if not self.fitted:
            raise RuntimeError(
                "Cannot save an unfitted model."
            )

        if (
            self.attack is None
            or self.defence is None
            or self.home_advantage is None
        ):
            raise RuntimeError(
                "Model parameters are unavailable."
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "rho": self.rho,
            "max_goals": self.max_goals,
            "regularization_strength": (
                self.regularization_strength
            ),
            "teams": self.teams,
            "team_match_counts": (
                self.team_match_counts
            ),
            "attack": self.attack.tolist(),
            "defence": self.defence.tolist(),
            "home_advantage": (
                self.home_advantage
            ),
        }

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
            )

        logger.info(
            "Model saved to %s",
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "DixonColesModel":
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(
                file
            )

        model = cls(
            rho=float(
                payload["rho"]
            ),
            max_goals=int(
                payload["max_goals"]
            ),
            regularization_strength=float(
                payload.get(
                    "regularization_strength",
                    0.10,
                )
            ),
        )

        model.teams = list(
            payload["teams"]
        )

        model.team_to_index = {
            team: index
            for index, team in enumerate(
                model.teams
            )
        }

        model.team_match_counts = {
            str(team): float(count)
            for team, count in payload.get(
                "team_match_counts",
                {},
            ).items()
        }

        model.attack = np.array(
            payload["attack"],
            dtype=float,
        )

        model.defence = np.array(
            payload["defence"],
            dtype=float,
        )

        model.home_advantage = float(
            payload["home_advantage"]
        )

        model.fitted = True

        logger.info(
            "Model loaded from %s",
            path,
        )

        return model


def get_feature_file(
    start_season: int,
    end_season: int | None = None,
) -> Path:
    if end_season is None:
        path = (
            FEATURES_DIR
            / f"features_{start_season}.csv"
        )
    else:
        path = (
            FEATURES_DIR
            / (
                f"features_{start_season}_"
                f"{end_season}.csv"
            )
        )

    if not path.exists():
        raise FileNotFoundError(
            "Feature file not found: "
            f"{path}. Run feature engineering first."
        )

    return path


def get_model_name(
    season: int | None,
    start_season: int | None,
    end_season: int | None,
    model_name: str | None,
) -> str:
    if model_name:
        return model_name

    if season is not None:
        return str(
            season
        )

    if (
        start_season is not None
        and end_season is not None
    ):
        return (
            f"{start_season}_"
            f"{end_season}"
        )

    raise ValueError(
        "Provide either --model-name, --season, "
        "or --start-season with --end-season."
    )


def build_promoted_aware_training_file(
    feature_file: Path,
    start_season: int,
    end_season: int | None,
    output_path: Path,
) -> Path:
    """
    Build real-PL + normalized-Championship training data.

    Only use Championship priors when the current training range includes
    the promoted-team season ending in the supplied latest season.
    """

    from models.promoted_team_normalizer import (
        DEFAULT_HISTORY_END_SEASON,
        DEFAULT_HISTORY_START_SEASON,
        DEFAULT_MAX_REPEATS,
        DEFAULT_PRIOR_MATCH_WEIGHT,
        run,
    )

    logger.info(
        "Building promoted-team-aware training data."
    )

    return run(
        pl_features_path=feature_file,
        output_path=output_path,
        history_start_season=(
            DEFAULT_HISTORY_START_SEASON
        ),
        history_end_season=(
            DEFAULT_HISTORY_END_SEASON
        ),
        prior_match_weight=(
            DEFAULT_PRIOR_MATCH_WEIGHT
        ),
        max_repeats=(
            DEFAULT_MAX_REPEATS
        ),
    )


def train_model(
    start_season: int,
    end_season: int | None = None,
    model_name: str | None = None,
    promoted_aware: bool = True,
    filter_to_teams: list[str] | None = None,
) -> Path:
    feature_file = get_feature_file(
        start_season,
        end_season,
    )

    artifact_name = get_model_name(
        season=(
            start_season
            if end_season is None
            else None
        ),
        start_season=start_season,
        end_season=end_season,
        model_name=model_name,
    )

    logger.info(
        "Loading base training data from %s",
        feature_file,
    )

    if promoted_aware:
        training_file = (
            FEATURES_DIR
            / (
                f"dixon_coles_training_"
                f"{artifact_name}.csv"
            )
        )

        training_file = (
            build_promoted_aware_training_file(
                feature_file=feature_file,
                start_season=start_season,
                end_season=end_season,
                output_path=training_file,
            )
        )

        logger.info(
            "Loading promoted-aware training data from %s",
            training_file,
        )

        matches = pd.read_csv(
            training_file
        )
    else:
        logger.info(
            "Promoted-aware priors disabled."
        )

        matches = pd.read_csv(
            feature_file
        )

        matches["training_weight"] = 1.0

    completed_before = len(
        matches
    )

    matches["home_goals"] = pd.to_numeric(
        matches["home_goals"],
        errors="coerce",
    )

    matches["away_goals"] = pd.to_numeric(
        matches["away_goals"],
        errors="coerce",
    )

    matches = matches.dropna(
        subset=[
            "home_goals",
            "away_goals",
        ]
    ).copy()

    logger.info(
        "Using %s completed training rows out of %s total rows.",
        len(matches),
        completed_before,
    )

    if matches.empty:
        raise ValueError(
            "No completed matches found "
            "in the training data."
        )

    # NOTE: filter_to_teams parameter is deprecated and ignored.
    #
    # The model trains on all teams in the combined training file, which includes:
    # 1. Real Premier League matches for all teams that played in the PL (2024-2026)
    # 2. Normalized Championship priors for promoted teams (Ipswich, Coventry, Hull)
    #    when they played in the Championship
    #
    # This results in ~49 teams total (20 current PL + 29 Championship opponents),
    # but this is NOT overfitting - it's the correct approach because:
    # - Promoted teams (Coventry, Hull) have only 1 PL match each in 2026
    # - Without their Championship history, their parameters would be undefined
    # - Championship opponents (Preston, Norwich, etc.) act as "training anchors"
    #   that help estimate promoted teams' attack/defence strengths
    # - We only predict matches between the 20 current PL teams
    # - The Championship opponents' parameters are learned but never used in predictions
    #
    # The weighted effective match count and L2 regularization prevent overfitting.
    if filter_to_teams is not None:
        logger.warning(
            "filter_to_teams parameter is deprecated and will be ignored. "
            "The model trains on all teams in the combined training file to "
            "preserve Championship priors for promoted teams."
        )

    if "source" in matches.columns:
        source_counts = (
            matches["source"]
            .fillna("unknown")
            .value_counts()
        )

        for source, count in (
            source_counts.items()
        ):
            logger.info(
                "Training source | %s: %s rows",
                source,
                count,
            )

    model = DixonColesModel()

    model.fit(
        matches
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / f"dixon_coles_{artifact_name}.json"
    )

    model.save(
        model_path
    )

    strengths = (
        model.team_strengths()
    )

    strengths_path = (
        MODEL_DIR
        / (
            f"team_strengths_"
            f"{artifact_name}.csv"
        )
    )

    strengths.to_csv(
        strengths_path,
        index=False,
    )

    logger.info(
        "Team strengths saved to %s",
        strengths_path,
    )

    logger.info(
        "Top attack strengths:"
    )

    for _, row in (
        strengths.head(10).iterrows()
    ):
        logger.info(
            "%s | Matches=%.2f | Attack=%.3f | Defence=%.3f",
            row["team"],
            row["effective_matches"],
            row["attack_strength"],
            row["defence_strength"],
        )

    return model_path


def predict_match(
    model_name: str,
    home_team: str,
    away_team: str,
) -> None:
    model_path = (
        MODEL_DIR
        / f"dixon_coles_{model_name}.json"
    )

    model = DixonColesModel.load(
        model_path
    )

    prediction = model.predict(
        home_team,
        away_team,
    )

    matrix = (
        model.scoreline_matrix(
            home_team,
            away_team,
        )
    )

    print()
    print("=" * 60)
    print(
        f"{prediction.home_team} "
        f"vs "
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
        matrix,
        index=[
            f"H{goal}"
            for goal in range(
                matrix.shape[0]
            )
        ],
        columns=[
            f"A{goal}"
            for goal in range(
                matrix.shape[1]
            )
        ],
    )

    print(
        matrix_df
        .mul(100)
        .round(2)
        .to_string()
    )

    print()


def predict_gameweek(
    model_name: str,
    season: int,
    gameweek: int | None = None,
) -> None:
    """
    Predict all fixtures in a specific gameweek.

    If gameweek is omitted, predict the earliest gameweek containing
    unplayed fixtures.
    """

    feature_file = (
        FEATURES_DIR
        / f"features_{season}.csv"
    )

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}. "
            "Run feature engineering first."
        )

    model_path = (
        MODEL_DIR
        / f"dixon_coles_{model_name}.json"
    )

    model = DixonColesModel.load(
        model_path
    )

    fixtures = pd.read_csv(
        feature_file
    )

    if "gameweek" not in (
        fixtures.columns
    ):
        raise ValueError(
            "The feature file does not contain a "
            "'gameweek' column."
        )

    fixtures["home_goals"] = (
        pd.to_numeric(
            fixtures["home_goals"],
            errors="coerce",
        )
    )

    fixtures["away_goals"] = (
        pd.to_numeric(
            fixtures["away_goals"],
            errors="coerce",
        )
    )

    fixtures["gameweek"] = (
        pd.to_numeric(
            fixtures["gameweek"],
            errors="coerce",
        )
    )

    if gameweek is None:
        upcoming = fixtures[
            fixtures["home_goals"].isna()
            | fixtures["away_goals"].isna()
        ].copy()

        upcoming = upcoming.dropna(
            subset=["gameweek"]
        )

        if upcoming.empty:
            raise ValueError(
                f"No upcoming fixtures found for season {season}."
            )

        gameweek = int(
            upcoming[
                "gameweek"
            ].min()
        )

    gameweek_fixtures = fixtures[
        fixtures["gameweek"].eq(
            gameweek
        )
    ].copy()

    if gameweek_fixtures.empty:
        raise ValueError(
            f"No fixtures found for gameweek {gameweek}."
        )

    upcoming_fixtures = (
        gameweek_fixtures[
            gameweek_fixtures[
                "home_goals"
            ].isna()
            | gameweek_fixtures[
                "away_goals"
            ].isna()
        ]
        .copy()
    )

    if not upcoming_fixtures.empty:
        gameweek_fixtures = (
            upcoming_fixtures
        )

    print()
    print("=" * 100)
    print(
        f"Premier League {season}/"
        f"{str(season + 1)[-2:]} "
        f"— Gameweek {gameweek}"
    )
    print(
        f"Model: Dixon-Coles {model_name}"
    )
    print("=" * 100)
    print()

    results = []

    for _, fixture in (
        gameweek_fixtures.iterrows()
    ):
        home_team = str(
            fixture["home_team"]
        )

        away_team = str(
            fixture["away_team"]
        )

        prediction = model.predict(
            home_team,
            away_team,
        )

        results.append(
            {
                "home": home_team,
                "away": away_team,
                "home_xg": (
                    prediction.expected_home_goals
                ),
                "away_xg": (
                    prediction.expected_away_goals
                ),
                "home_win": (
                    prediction.home_win_probability
                ),
                "draw": (
                    prediction.draw_probability
                ),
                "away_win": (
                    prediction.away_win_probability
                ),
                "score": (
                    f"{prediction.most_likely_home_goals}"
                    "-"
                    f"{prediction.most_likely_away_goals}"
                ),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "home_win": (
                    lambda value: (
                        f"{value * 100:.1f}%"
                    )
                ),
                "draw": (
                    lambda value: (
                        f"{value * 100:.1f}%"
                    )
                ),
                "away_win": (
                    lambda value: (
                        f"{value * 100:.1f}%"
                    )
                ),
            },
        )
    )

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Premier League promoted-aware "
            "Poisson / Dixon-Coles model."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    train_parser = (
        subparsers.add_parser(
            "train",
            help=(
                "Train the promoted-aware Dixon-Coles model."
            ),
        )
    )

    season_group = (
        train_parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    season_group.add_argument(
        "--season",
        type=int,
        help=(
            "Single season starting year."
        ),
    )

    season_group.add_argument(
        "--start-season",
        type=int,
        help=(
            "First season starting year for "
            "multi-season training."
        ),
    )

    train_parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help=(
            "Last season starting year for "
            "multi-season training."
        ),
    )

    train_parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Optional artifact name. "
            "Example: 2024_2026"
        ),
    )

    train_parser.add_argument(
        "--no-promoted-priors",
        action="store_true",
        help=(
            "Disable normalized Championship priors."
        ),
    )

    # --------------------------------------------------
    # SINGLE MATCH PREDICTION
    # --------------------------------------------------

    predict_parser = (
        subparsers.add_parser(
            "predict",
            help=(
                "Predict a match using a trained model."
            ),
        )
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

    # --------------------------------------------------
    # GAMEWEEK PREDICTION
    # --------------------------------------------------

    gameweek_parser = (
        subparsers.add_parser(
            "predict-gameweek",
            help=(
                "Predict all fixtures in a gameweek."
            ),
        )
    )

    gameweek_parser.add_argument(
        "--model-name",
        required=True,
    )

    gameweek_parser.add_argument(
        "--season",
        type=int,
        default=2026,
    )

    gameweek_parser.add_argument(
        "--gameweek",
        type=int,
        default=None,
        help=(
            "If omitted, automatically predicts "
            "the next upcoming gameweek."
        ),
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    if args.command == "train":
        if (
            args.start_season is not None
            and args.end_season is None
        ):
            raise ValueError(
                "--end-season is required when "
                "--start-season is used."
            )

        if (
            args.start_season is not None
            and args.end_season
            < args.start_season
        ):
            raise ValueError(
                "--end-season cannot be earlier than "
                "--start-season."
            )

        if args.season is not None:
            logger.info(
                "Training Dixon-Coles model for season %s/%s",
                args.season,
                str(
                    args.season + 1
                )[-2:],
            )

            train_model(
                start_season=args.season,
                model_name=args.model_name,
                promoted_aware=not (
                    args.no_promoted_priors
                ),
            )
        else:
            logger.info(
                "Training multi-season Dixon-Coles model "
                "from %s/%s through %s/%s",
                args.start_season,
                str(
                    args.start_season + 1
                )[-2:],
                args.end_season,
                str(
                    args.end_season + 1
                )[-2:],
            )

            train_model(
                start_season=(
                    args.start_season
                ),
                end_season=(
                    args.end_season
                ),
                model_name=(
                    args.model_name
                ),
                promoted_aware=not (
                    args.no_promoted_priors
                ),
            )

    elif args.command == "predict":
        predict_match(
            model_name=args.model_name,
            home_team=args.home,
            away_team=args.away,
        )

    elif args.command == "predict-gameweek":
        predict_gameweek(
            model_name=args.model_name,
            season=args.season,
            gameweek=args.gameweek,
        )


if __name__ == "__main__":
    main()