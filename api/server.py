"""
api/server.py

FastAPI wrapper around the existing Dixon-Coles prediction pipeline.

Serves JSON for the frontend. Does not change any existing files —
reuses the same feature CSVs, model training, and Supabase teams data
that main.py and models/poisson_dixon_coles.py already use.

Run locally:
    uv run uvicorn api.server:app --reload --port 8000

Endpoints:
    GET /health
    GET /fixtures/current-gameweek?season=2026
    GET /predictions/{home_team}/{away_team}?season=2026
    GET /accuracy?season=2025
"""

from __future__ import annotations

import math
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from models.poisson_dixon_coles import (
    FEATURES_DIR,
    MODEL_DIR,
    DixonColesModel,
    train_model,
)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="Premier League Predictor API")

# Allow the local Next.js dev server (and any origin during local testing)
# to call this API. Tighten this once you deploy for real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Supabase client (optional — only used for team crest/name lookup)
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_supabase_client():
    """
    Lazily create a Supabase client. Returns None if env vars are missing,
    so the API still works (without crest URLs) even if Supabase isn't
    configured locally.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

    if not url or not key:
        logger.warning(
            "SUPABASE_URL / SUPABASE_KEY not set — team crest lookup disabled."
        )
        return None

    from supabase import create_client

    return create_client(url, key)


@lru_cache(maxsize=1)
def get_team_lookup() -> dict[str, dict]:
    """
    Returns {team_name: {"name": ..., "crest_url": ..., "short_name": ...}}

    Falls back to an empty dict if Supabase isn't reachable — the API
    still returns predictions, just without crest URLs.
    """
    client = get_supabase_client()

    if client is None:
        return {}

    try:
        result = client.table("teams").select(
            "name, short_name, tla, crest_url"
        ).execute()
    except Exception:
        logger.exception("Failed to fetch teams from Supabase.")
        return {}

    return {row["name"]: row for row in (result.data or [])}


def team_payload(team_name: str) -> dict:
    lookup = get_team_lookup()
    row = lookup.get(team_name)

    return {
        "name": team_name,
        "short_name": (row or {}).get("short_name"),
        "crest_url": (row or {}).get("crest_url"),
    }


# --------------------------------------------------------------------------
# Model loading / training (mirrors main.py logic, but cached in-process)
# --------------------------------------------------------------------------

_model_cache: dict[str, DixonColesModel] = {}


def load_features(season: int) -> pd.DataFrame:
    path = FEATURES_DIR / f"features_{season}.csv"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Feature file not found: {path.name}. "
            "Run feature engineering for this season first.",
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
        return int(features["gameweek"].max())

    return int(incomplete["gameweek"].min())


DEFAULT_MODEL_NAME = "2024_2026"  # matches the multi-season artifact already
                                    # trained via the CLI: predict-gameweek
                                    # --model-name 2024_2026


def get_model(season: int, model_name: str = DEFAULT_MODEL_NAME) -> DixonColesModel:
    """
    Loads the multi-season model artifact (e.g. 2024_2026) that was already
    trained via the CLI, the same one `predict-gameweek --model-name
    2024_2026` uses. This is important: a single season (e.g. just 2026) has
    only ~10 completed matches at the start of a season, which is far too
    little data to fit 20 teams' attack/defence parameters — that produced
    extreme/nonsensical probabilities. Multi-season data fixes that.

    Cached in memory for the life of the server process. Restart the server
    (or call retrain via the CLI) to pick up newly completed matches.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    artifact_path = MODEL_DIR / f"dixon_coles_{model_name}.json"

    if artifact_path.exists():
        logger.info("Loading existing multi-season model artifact: %s", artifact_path)
        model = DixonColesModel.load(artifact_path)
    else:
        logger.info(
            "No artifact found for '%s' — training multi-season model now "
            "(seasons %s-%s)...",
            model_name,
            season - 2,
            season,
        )
        model_path = train_model(
            start_season=season - 2, end_season=season, model_name=model_name
        )
        model = DixonColesModel.load(model_path)

    _model_cache[model_name] = model
    return model


# --------------------------------------------------------------------------
# Response shaping
# --------------------------------------------------------------------------

def prediction_to_payload(model: DixonColesModel, home_team: str, away_team: str) -> dict:
    prediction = model.predict(home_team, away_team)

    # BUGFIX: Use Dixon-Coles corrected scoreline matrix instead of independent Poisson.
    # The model.scoreline_matrix() method applies the Dixon-Coles correction factor (τ)
    # to low-scoring outcomes (0-0, 0-1, 1-0, 1-1), which is the whole point of using
    # Dixon-Coles over plain Poisson. The previous code used independent Poisson
    # (p_home × p_away), which ignored this correction.
    max_grid_goals = 4
    scoreline_probabilities: dict[str, float] = {}

    # Get the Dixon-Coles corrected probability matrix
    matrix = model.scoreline_matrix(home_team, away_team, max_goals=max_grid_goals)

    for h in range(max_grid_goals + 1):
        for a in range(max_grid_goals + 1):
            scoreline_probabilities[f"{h}-{a}"] = round(float(matrix[h, a]), 4)

    return {
        "home_team": team_payload(home_team),
        "away_team": team_payload(away_team),
        "expected_home_goals": round(prediction.expected_home_goals, 3),
        "expected_away_goals": round(prediction.expected_away_goals, 3),
        "home_win_probability": round(prediction.home_win_probability, 4),
        "draw_probability": round(prediction.draw_probability, 4),
        "away_win_probability": round(prediction.away_win_probability, 4),
        "most_likely_home_goals": prediction.most_likely_home_goals,
        "most_likely_away_goals": prediction.most_likely_away_goals,
        "most_likely_score_probability": round(
            prediction.most_likely_score_probability, 4
        ),
        "scoreline_probabilities": scoreline_probabilities,
        "explanation_summary": build_explanation_summary(
            home_team, away_team, prediction
        ),
    }


def build_explanation_summary(home_team: str, away_team: str, prediction) -> str:
    """
    Simple human-readable summary from the Poisson model's own numbers.
    Not SHAP — this project doesn't have an XGBoost/SHAP layer yet — just a
    plain-language read of the expected goals and win probabilities.
    """
    favourite = home_team if prediction.home_win_probability > prediction.away_win_probability else away_team
    fav_prob = max(prediction.home_win_probability, prediction.away_win_probability)

    return (
        f"{favourite} are favoured with a {fav_prob * 100:.0f}% win probability. "
        f"Model expects {prediction.expected_home_goals:.1f} goals for {home_team} "
        f"and {prediction.expected_away_goals:.1f} for {away_team}, with "
        f"{prediction.most_likely_home_goals}-{prediction.most_likely_away_goals} "
        f"as the single most likely scoreline "
        f"({prediction.most_likely_score_probability * 100:.0f}% chance)."
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/admin/refresh-model")
def refresh_model(model_name: str = Query(default=DEFAULT_MODEL_NAME)):
    """
    Call this after running predict_weekend.ps1 (or any manual retrain) so
    the API picks up the freshly-written dixon_coles_{model_name}.json
    artifact from disk without needing a server restart.

    Also clears the leave-one-out backtest caches and the combined-features
    cache, since those are derived from the same underlying data and would
    otherwise keep serving stale results for /history endpoints too.
    """
    artifact_path = MODEL_DIR / f"dixon_coles_{model_name}.json"

    if not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No artifact found at {artifact_path.name}. "
            "Run the training step first (predict_weekend.ps1 steps 1-4).",
        )

    _model_cache.pop(model_name, None)
    _backtest_model_cache.clear()
    load_combined_features.cache_clear()

    model = get_model(0, model_name=model_name)  # season unused when artifact exists

    return {
        "status": "refreshed",
        "model_name": model_name,
        "teams_loaded": len(model.teams),
    }


@app.post("/admin/refresh-history")
def refresh_history():
    """
    Force recompute and persist accuracy history and all GW history files.
    Called by the weekly retrain workflow after model retraining.
    """
    # Clear disk caches so they get rebuilt
    for f in MODEL_DIR.glob("accuracy_history.json"):
        f.unlink(missing_ok=True)
    for f in MODEL_DIR.glob("history_gw*.json"):
        f.unlink(missing_ok=True)

    # Trigger background recompute
    _warm_history_caches()

    return {"status": "history refresh triggered (background)"}


@app.get("/fixtures/current-gameweek")
def current_gameweek_fixtures(season: int = Query(default=2026)):
    features = load_features(season)
    gameweek = detect_current_gameweek(features)

    fixtures = features[features["gameweek"] == gameweek].copy()

    if fixtures.empty:
        raise HTTPException(
            status_code=404, detail=f"No fixtures found for gameweek {gameweek}."
        )

    model = get_model(season)

    results = []
    for _, fixture in fixtures.iterrows():
        home_team = str(fixture["home_team"])
        away_team = str(fixture["away_team"])

        payload = prediction_to_payload(model, home_team, away_team)
        payload["match_id"] = f"{season}-{gameweek}-{home_team}-{away_team}".replace(" ", "_")
        payload["gameweek"] = gameweek
        payload["season"] = season

        results.append(payload)

    return {
        "season": season,
        "gameweek": gameweek,
        "fixtures": results,
    }


@app.get("/predictions/{home_team}/{away_team}")
def single_prediction(home_team: str, away_team: str, season: int = Query(default=2026)):
    home_team = unquote(home_team)
    away_team = unquote(away_team)

    model = get_model(season)

    try:
        return prediction_to_payload(model, home_team, away_team)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team name: {exc}. Team names must match exactly "
            "as they appear in the feature data (e.g. 'Arsenal FC').",
        )


@app.get("/accuracy")
def accuracy(season: int = Query(default=2026)):
    """Live overall accuracy across all completed fixtures."""
    frame = load_combined_features()
    completed = frame[
        (frame["season_start_year"] == season) &
        frame["home_goals"].notna() &
        frame["away_goals"].notna()
    ]
    if completed.empty:
        return {
            "season": season,
            "model_accuracy": None,
            "model_log_loss": None,
            "model_brier_score": None,
            "bookmaker_accuracy": None,
            "matches_evaluated": 0,
        }

    model = get_model(season)
    hits = 0
    total = 0
    log_loss_sum = 0.0
    for _, row in completed.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        try:
            pred = model.predict(home_team, away_team)
        except Exception:
            continue
        actual_home = int(row["home_goals"])
        actual_away = int(row["away_goals"])
        pred_outcome = outcome_from_probs(pred.home_win_probability, pred.draw_probability, pred.away_win_probability)
        actual_outcome = outcome_from_goals(actual_home, actual_away)
        if pred_outcome == actual_outcome:
            hits += 1
        total += 1
        p = max(pred.home_win_probability if pred_outcome == "HOME" else
                pred.draw_probability if pred_outcome == "DRAW" else
                pred.away_win_probability, 1e-12)
        log_loss_sum += -math.log(p)

    return {
        "season": season,
        "model_accuracy": round(hits / total, 4) if total else None,
        "model_log_loss": round(log_loss_sum / total, 4) if total else None,
        "model_brier_score": None,
        "bookmaker_accuracy": None,
        "matches_evaluated": total,
    }


# --------------------------------------------------------------------------
# Past-gameweek history: real leave-one-gameweek-out backtest.
#
# For each COMPLETED gameweek of the current season, the model is retrained
# on every match EXCEPT that gameweek's own results (using the full
# 2024-2026 multi-season file as the base), then used to predict that
# gameweek's fixtures. This avoids hindsight bias — the model never sees
# the result it's being asked to predict.
#
# Gameweek 1 of a new season is intentionally excluded: with almost no
# season-specific signal yet, a leave-one-out backtest there is barely
# different from the full model, so it's not a meaningful check. History
# starts appearing once gameweek 2 is complete.
#
# Backtest models are cached separately per gameweek (they're different
# fits) and are NOT saved to disk / do not touch dixon_coles_2024_2026.json.
# --------------------------------------------------------------------------

COMBINED_FEATURES_FILE = "features_2024_2026.csv"
CURRENT_SEASON_START_YEAR = 2026
MIN_GAMEWEEK_FOR_HISTORY = 2  # skip GW1 — not enough prior signal to be a fair test

# Persisted history artifacts
ACCURACY_HISTORY_FILE = "accuracy_history.json"
HISTORY_GW_FILE_PATTERN = "history_gw{}.json"


def _history_cache_path(filename: str) -> Path:
    return MODEL_DIR / filename


def _load_accuracy_history_from_disk() -> list[dict] | None:
    """Load precomputed accuracy history from JSON file."""
    path = _history_cache_path(ACCURACY_HISTORY_FILE)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("gameweeks", [])
    except Exception as e:
        logger.warning("Failed to load accuracy history cache: %s", e)
        return None


def _save_accuracy_history_to_disk(gameweeks: list[dict]) -> None:
    """Persist computed accuracy history to JSON file."""
    path = _history_cache_path(ACCURACY_HISTORY_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"season": CURRENT_SEASON_START_YEAR, "gameweeks": gameweeks}, f, indent=2)
        logger.info("Saved accuracy history cache to %s (%d GWs)", path, len(gameweeks))
    except Exception as e:
        logger.error("Failed to save accuracy history cache: %s", e)


def _load_gw_history_from_disk(gameweek: int) -> dict | None:
    """Load precomputed single GW history from JSON file."""
    path = _history_cache_path(HISTORY_GW_FILE_PATTERN.format(gameweek))
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load GW%d history cache: %s", gameweek, e)
        return None


def _save_gw_history_to_disk(gameweek: int, data: dict) -> None:
    """Persist computed single GW history to JSON file."""
    path = _history_cache_path(HISTORY_GW_FILE_PATTERN.format(gameweek))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved GW%d history cache to %s", gameweek, path)
    except Exception as e:
        logger.error("Failed to save GW%d history cache: %s", gameweek, e)


def _warm_history_caches() -> None:
    """Pre-compute and persist history for all completed gameweeks on startup.
    Runs in background; doesn't block API readiness."""
    import threading

    def _do_warm():
        try:
            frame = load_combined_features()
            available = completed_gameweeks(frame, CURRENT_SEASON_START_YEAR)
            if not available:
                return

            # Accuracy history
            cached_acc = _load_accuracy_history_from_disk()
            if cached_acc is None:
                history = []
                for gameweek in available:
                    model = get_leave_one_gameweek_out_model(gameweek)
                    rows = frame[
                        (frame["season_start_year"] == CURRENT_SEASON_START_YEAR) & (frame["gameweek"] == gameweek)
                    ]
                    outcome_hits = 0
                    total = 0
                    for _, row in rows.iterrows():
                        home_team = str(row["home_team"])
                        away_team = str(row["away_team"])
                        actual_home_goals = int(row["home_goals"])
                        actual_away_goals = int(row["away_goals"])
                        prediction = model.predict(home_team, away_team)
                        predicted_outcome = outcome_from_probs(
                            prediction.home_win_probability,
                            prediction.draw_probability,
                            prediction.away_win_probability,
                        )
                        actual_outcome = outcome_from_goals(actual_home_goals, actual_away_goals)
                        if predicted_outcome == actual_outcome:
                            outcome_hits += 1
                        total += 1
                    accuracy_pct = round((outcome_hits / total * 100), 1) if total > 0 else 0
                    history.append({
                        "gameweek": gameweek,
                        "accuracy": accuracy_pct,
                        "correct": outcome_hits,
                        "total": total,
                        "predicted_fixtures": total,
                        "results_in": total,
                    })
                _save_accuracy_history_to_disk(history)
            else:
                # Only compute any newly completed GWs
                existing_gws = {p["gameweek"] for p in cached_acc}
                missing = [gw for gw in available if gw not in existing_gws]
                if missing:
                    history = list(cached_acc)
                    for gameweek in missing:
                        model = get_leave_one_gameweek_out_model(gameweek)
                        rows = frame[
                            (frame["season_start_year"] == CURRENT_SEASON_START_YEAR) & (frame["gameweek"] == gameweek)
                        ]
                        outcome_hits = 0
                        total = 0
                        for _, row in rows.iterrows():
                            home_team = str(row["home_team"])
                            away_team = str(row["away_team"])
                            actual_home_goals = int(row["home_goals"])
                            actual_away_goals = int(row["away_goals"])
                            prediction = model.predict(home_team, away_team)
                            predicted_outcome = outcome_from_probs(
                                prediction.home_win_probability,
                                prediction.draw_probability,
                                prediction.away_win_probability,
                            )
                            actual_outcome = outcome_from_goals(actual_home_goals, actual_away_goals)
                            if predicted_outcome == actual_outcome:
                                outcome_hits += 1
                            total += 1
                        accuracy_pct = round((outcome_hits / total * 100), 1) if total > 0 else 0
                        history.append({
                            "gameweek": gameweek,
                            "accuracy": accuracy_pct,
                            "correct": outcome_hits,
                            "total": total,
                            "predicted_fixtures": total,
                            "results_in": total,
                        })
                    history.sort(key=lambda x: x["gameweek"])
                    _save_accuracy_history_to_disk(history)

            # Individual GW detail files
            for gameweek in available:
                cached = _load_gw_history_from_disk(gameweek)
                if cached is None:
                    # compute
                    model = get_leave_one_gameweek_out_model(gameweek)
                    rows = frame[
                        (frame["season_start_year"] == CURRENT_SEASON_START_YEAR) & (frame["gameweek"] == gameweek)
                    ]
                    results = []
                    for _, row in rows.iterrows():
                        home_team = str(row["home_team"])
                        away_team = str(row["away_team"])
                        actual_home_goals = int(row["home_goals"])
                        actual_away_goals = int(row["away_goals"])
                        prediction = model.predict(home_team, away_team)
                        predicted_outcome = outcome_from_probs(
                            prediction.home_win_probability,
                            prediction.draw_probability,
                            prediction.away_win_probability,
                        )
                        actual_outcome = outcome_from_goals(actual_home_goals, actual_away_goals)
                        outcome_correct = predicted_outcome == actual_outcome
                        scoreline_correct = (
                            prediction.most_likely_home_goals == actual_home_goals
                            and prediction.most_likely_away_goals == actual_away_goals
                        )
                        results.append({
                            "match_id": f"{CURRENT_SEASON_START_YEAR}-{gameweek}-{home_team}-{away_team}".replace(" ", "_"),
                            "home_team": team_payload(home_team),
                            "away_team": team_payload(away_team),
                            "actual_home_goals": actual_home_goals,
                            "actual_away_goals": actual_away_goals,
                            "predicted_home_goals": prediction.most_likely_home_goals,
                            "predicted_away_goals": prediction.most_likely_away_goals,
                            "home_win_probability": round(prediction.home_win_probability, 4),
                            "draw_probability": round(prediction.draw_probability, 4),
                            "away_win_probability": round(prediction.away_win_probability, 4),
                            "predicted_outcome": predicted_outcome,
                            "actual_outcome": actual_outcome,
                            "outcome_correct": outcome_correct,
                            "scoreline_correct": scoreline_correct,
                        })
                    outcome_hits = sum(1 for r in results if r["outcome_correct"])
                    scoreline_hits = sum(1 for r in results if r["scoreline_correct"])
                    _save_gw_history_to_disk(gameweek, {
                        "season": CURRENT_SEASON_START_YEAR,
                        "gameweek": gameweek,
                        "matches": results,
                        "summary": {
                            "total_matches": len(results),
                            "outcome_correct": outcome_hits,
                            "scoreline_correct": scoreline_hits,
                            "outcome_accuracy": round(outcome_hits / len(results), 4) if results else None,
                            "scoreline_accuracy": round(scoreline_hits / len(results), 4) if results else None,
                        },
                    })
        except Exception as e:
            logger.warning("History cache warm-up failed: %s", e)

    # Fire and forget - don't block startup
    t = threading.Thread(target=_do_warm, daemon=True)
    t.start()


@lru_cache(maxsize=1)
def load_combined_features() -> pd.DataFrame:
    path = FEATURES_DIR / COMBINED_FEATURES_FILE

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Combined feature file not found: {path.name}.",
        )

    frame = pd.read_csv(path)
    frame["gameweek"] = pd.to_numeric(frame["gameweek"], errors="coerce")
    frame["season_start_year"] = pd.to_numeric(
        frame["season_start_year"], errors="coerce"
    )
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")

    return frame


_backtest_model_cache: dict[int, DixonColesModel] = {}


def get_leave_one_gameweek_out_model(gameweek: int) -> DixonColesModel:
    """
    Trains (or returns a cached) model on all combined-feature rows EXCEPT
    the given current-season gameweek. Never writes to disk.
    """
    if gameweek in _backtest_model_cache:
        return _backtest_model_cache[gameweek]

    frame = load_combined_features()

    holdout_mask = (
        (frame["season_start_year"] == CURRENT_SEASON_START_YEAR)
        & (frame["gameweek"] == gameweek)
    )

    training_rows = frame[~holdout_mask].dropna(
        subset=["home_team", "away_team", "home_goals", "away_goals"]
    ).copy()

    if training_rows.empty:
        raise HTTPException(
            status_code=500,
            detail=f"No training data left after excluding gameweek {gameweek}.",
        )

    if "training_weight" not in training_rows.columns:
        training_rows["training_weight"] = 1.0

    model = DixonColesModel()
    model.fit(training_rows)

    _backtest_model_cache[gameweek] = model
    return model


def completed_gameweeks(frame: pd.DataFrame, season: int) -> list[int]:
    """
    Gameweeks (>= MIN_GAMEWEEK_FOR_HISTORY) in the given season where every
    fixture has a real final score.
    """
    season_rows = frame[frame["season_start_year"] == season]

    complete = []
    for gw, group in season_rows.groupby("gameweek"):
        if gw < MIN_GAMEWEEK_FOR_HISTORY:
            continue
        if group["home_goals"].notna().all() and group["away_goals"].notna().all():
            complete.append(int(gw))

    return sorted(complete)


def outcome_from_goals(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "HOME"
    if away_goals > home_goals:
        return "AWAY"
    return "DRAW"


def outcome_from_probs(home_p: float, draw_p: float, away_p: float) -> str:
    best = max(home_p, draw_p, away_p)
    if best == home_p:
        return "HOME"
    if best == away_p:
        return "AWAY"
    return "DRAW"


@app.on_event("startup")
def _on_startup():
    """Pre-compute history caches in background so History tab loads instantly."""
    _warm_history_caches()


@app.get("/history/gameweeks")
def history_available_gameweeks(season: int = Query(default=CURRENT_SEASON_START_YEAR)):
    """
    Lists which gameweeks currently have a fair (leave-one-out) backtest
    available. Empty list is expected/normal early in a season.
    """
    frame = load_combined_features()
    return {"season": season, "gameweeks": completed_gameweeks(frame, season)}


@app.get("/history/gameweek/{gameweek}")
def history_gameweek(gameweek: int, season: int = Query(default=CURRENT_SEASON_START_YEAR)):
    # Read from persisted cache first (instant)
    cached = _load_gw_history_from_disk(gameweek)
    if cached is not None:
        return cached

    # Fallback: compute live (slower) and persist
    frame = load_combined_features()
    available = completed_gameweeks(frame, season)
    if gameweek not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Gameweek {gameweek} isn't complete yet, or is below the "
            f"minimum gameweek ({MIN_GAMEWEEK_FOR_HISTORY}) used for fair history.",
        )

    model = get_leave_one_gameweek_out_model(gameweek)
    rows = frame[
        (frame["season_start_year"] == season) & (frame["gameweek"] == gameweek)
    ]

    results = []
    for _, row in rows.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        actual_home_goals = int(row["home_goals"])
        actual_away_goals = int(row["away_goals"])

        prediction = model.predict(home_team, away_team)

        predicted_outcome = outcome_from_probs(
            prediction.home_win_probability,
            prediction.draw_probability,
            prediction.away_win_probability,
        )
        actual_outcome = outcome_from_goals(actual_home_goals, actual_away_goals)

        outcome_correct = predicted_outcome == actual_outcome
        scoreline_correct = (
            prediction.most_likely_home_goals == actual_home_goals
            and prediction.most_likely_away_goals == actual_away_goals
        )

        results.append({
            "match_id": f"{season}-{gameweek}-{home_team}-{away_team}".replace(" ", "_"),
            "home_team": team_payload(home_team),
            "away_team": team_payload(away_team),
            "actual_home_goals": actual_home_goals,
            "actual_away_goals": actual_away_goals,
            "predicted_home_goals": prediction.most_likely_home_goals,
            "predicted_away_goals": prediction.most_likely_away_goals,
            "home_win_probability": round(prediction.home_win_probability, 4),
            "draw_probability": round(prediction.draw_probability, 4),
            "away_win_probability": round(prediction.away_win_probability, 4),
            "predicted_outcome": predicted_outcome,
            "actual_outcome": actual_outcome,
            "outcome_correct": outcome_correct,
            "scoreline_correct": scoreline_correct,
        })

    outcome_hits = sum(1 for r in results if r["outcome_correct"])
    scoreline_hits = sum(1 for r in results if r["scoreline_correct"])
    result = {
        "season": season,
        "gameweek": gameweek,
        "matches": results,
        "summary": {
            "total_matches": len(results),
            "outcome_correct": outcome_hits,
            "scoreline_correct": scoreline_hits,
            "outcome_accuracy": round(outcome_hits / len(results), 4) if results else None,
            "scoreline_accuracy": round(scoreline_hits / len(results), 4) if results else None,
        },
    }
    _save_gw_history_to_disk(gameweek, result)
    return result


@app.get("/accuracy/history")
def accuracy_history(season: int = Query(default=CURRENT_SEASON_START_YEAR)):
    """
    Returns per-gameweek accuracy progression for the accuracy-over-time chart.
    Each gameweek uses a leave-one-out model (trained without that gameweek's
    results) to ensure no hindsight bias.

    Reads from persisted JSON file for instant response; computes any missing
    gameweeks in the background on first call.
    """
    cached = _load_accuracy_history_from_disk()
    if cached is not None and len(cached) > 0:
        return {"season": season, "gameweeks": cached}

    # Fallback: compute live (slower) and persist
    frame = load_combined_features()
    available = completed_gameweeks(frame, season)

    if not available:
        _save_accuracy_history_to_disk([])
        return {"season": season, "gameweeks": []}

    history = []
    for gameweek in available:
        model = get_leave_one_gameweek_out_model(gameweek)
        rows = frame[
            (frame["season_start_year"] == season) & (frame["gameweek"] == gameweek)
        ]

        outcome_hits = 0
        total = 0

        for _, row in rows.iterrows():
            home_team = str(row["home_team"])
            away_team = str(row["away_team"])
            actual_home_goals = int(row["home_goals"])
            actual_away_goals = int(row["away_goals"])

            prediction = model.predict(home_team, away_team)

            predicted_outcome = outcome_from_probs(
                prediction.home_win_probability,
                prediction.draw_probability,
                prediction.away_win_probability,
            )
            actual_outcome = outcome_from_goals(actual_home_goals, actual_away_goals)

            if predicted_outcome == actual_outcome:
                outcome_hits += 1
            total += 1

        accuracy_pct = round((outcome_hits / total * 100), 1) if total > 0 else 0
        history.append({
            "gameweek": gameweek,
            "accuracy": accuracy_pct,
            "correct": outcome_hits,
            "total": total,
            "predicted_fixtures": total,
            "results_in": total,
        })

    _save_accuracy_history_to_disk(history)
    return {"season": season, "gameweeks": history}