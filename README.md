# ⚽ Premier League Predictor

A machine learning and statistical modelling project that predicts Premier League football matches using a **Poisson / Dixon–Coles model**.

The system analyses historical match data to estimate team attacking and defensive strength, account for home advantage, handle newly promoted Championship teams through league-strength normalization, and generate probabilistic predictions for upcoming fixtures.

## What the model predicts

- 🏆 Home win, draw and away win probabilities
- ⚽ Expected goals for both teams
- 🎯 Most likely exact scoreline
- 📊 Probability distribution across possible scorelines
- 📅 Current gameweek predictions
- 📈 Historical gameweek backtesting and accuracy metrics

The prediction engine is exposed through a FastAPI API and presented through a web dashboard.

---

# 🧠 Model Methodology

## Poisson Goal Modelling

At the core of the project is a **Dixon–Coles adjusted Poisson model**, a statistical approach specifically designed for modelling football scores.

For a fixture:

> **Home Team vs Away Team**

the model estimates the expected number of goals scored by each team:

**λ_home** — expected goals for the home team

**λ_away** — expected goals for the away team

These expectations are derived from learned team-level parameters such as:

- ⚔️ Attacking strength
- 🛡️ Defensive strength
- 🏟️ Home advantage
- 📈 Historical match performance

The expected goals are then used to estimate the probability of possible scorelines.

For example:

| Home Goals | Away Goals |
|---|---|
| 0 | 0, 1, 2, 3, ... |
| 1 | 0, 1, 2, 3, ... |
| 2 | 0, 1, 2, 3, ... |
| 3 | 0, 1, 2, 3, ... |

This creates a probability distribution covering outcomes such as:

```text
0-0   1-0   2-0   3-0
0-1   1-1   2-1   3-1
0-2   1-2   2-2   3-2
0-3   1-3   2-3   3-3
```

---

## Dixon–Coles Adjustment

A standard independent Poisson model can struggle to accurately represent dependencies in low-scoring football matches.

The Dixon–Coles adjustment improves the modelling of outcomes such as:

- `0–0`
- `1–0`
- `0–1`
- `1–1`

This makes the model better suited to football score prediction than a basic independent Poisson approach alone.

---

# 🔄 Handling Promoted Championship Teams

One challenge in Premier League prediction is handling teams that have recently been promoted from the Championship.

A newly promoted team may have little or no recent Premier League data, meaning a model trained only on Premier League seasons would have insufficient historical information to estimate that team's strength reliably.

This project addresses that problem by incorporating the promoted team's **Championship season performance**.

The process is conceptually:

```text
Championship Match Data
        │
        ▼
Analyse Team Performance
        │
        ▼
Estimate Attacking & Defensive Strength
        │
        ▼
Normalise Championship Statistics
to the Premier League Level
        │
        ▼
Generate Comparable Premier League
Strength Estimates
        │
        ▼
Use in Premier League Predictions
```

Rather than treating promoted teams as completely unknown, the model analyses their Championship performance and **normalizes their statistical strength to account for the difference between the Championship and Premier League**.

This provides the model with a more informed starting point for promoted teams while recognising that raw Championship statistics cannot be directly treated as equivalent to Premier League statistics.

This is particularly important at the beginning of a new season, before enough Premier League matches have been played to establish reliable season-specific estimates.

---

# 📊 Understanding the Predictions

The model produces several different types of predictions.

## 1. Match Outcome Probabilities

The model calculates the total probability of:

- Home Win
- Draw
- Away Win

For example:

```text
Home Win   42%
Draw       29%
Away Win   29%
```

These probabilities are calculated by aggregating the probabilities of all relevant scorelines.

---

## 2. Most Likely Exact Scoreline

The system also identifies the **single most probable exact scoreline**.

For example:

```text
Most Likely Exact Score: 1–1
Probability: 12%
```

This can be different from the most likely overall match outcome.

For example:

```text
Most Likely Exact Score: 1–1

Most Likely Overall Outcome:
Home Win — 42%
```

This is **not a contradiction**.

`1–1` may be the most likely **individual scoreline**, but there are many different ways for the home team to win.

Conceptually:

```text
HOME WINS

1-0
2-0
2-1
3-0
3-1
3-2
4-0
...
```

All of those probabilities are combined to calculate the total probability of a home win.

Similarly:

```text
DRAWS

0-0
1-1
2-2
3-3
4-4
...
```

and:

```text
AWAY WINS

0-1
0-2
1-2
0-3
1-3
2-3
...
```

Therefore, a single scoreline such as `1–1` can be the most likely exact result, while the **combined probability of all home-winning scorelines** can still be greater than the combined probability of all draws.

---

## 3. Expected Goals

The model estimates the expected number of goals for each team.

For example:

```text
Home Team Expected Goals: 1.72
Away Team Expected Goals: 1.14
```

These values represent the model's underlying scoring expectations and are used to construct the probability distribution across possible scorelines.

---

# ✨ Features

## ⚽ Current Gameweek Predictions

The system identifies the current gameweek from the available fixture data and generates predictions for every fixture.

Each prediction includes:

- Home and away teams
- Team crests
- Expected goals
- Home win probability
- Draw probability
- Away win probability
- Most likely exact score
- Exact score probability
- Scoreline probability distribution
- Human-readable prediction summary

---

## 📈 Multi-Season Training

The project supports a multi-season model rather than relying only on the small amount of data available in the current season.

This is especially important early in a Premier League season.

Using multiple seasons gives the model a larger body of historical data for estimating team attack and defence parameters.

The deployed API loads a cached multi-season model artifact when available.

This avoids the instability that could occur if a full model were trained using only a very small number of completed matches from a new season.

---

## 🧪 Historical Gameweek Backtesting

The project includes a historical evaluation system for completed gameweeks.

For each completed gameweek, the system uses a **leave-one-gameweek-out backtesting approach**.

The target gameweek is excluded from the training data before the model is used to predict it.

```text
Historical Match Data
        │
        ▼
Select Completed Gameweek
        │
        ▼
Exclude That Gameweek
from Training Data
        │
        ▼
Train Dixon–Coles Model
        │
        ▼
Predict Excluded Fixtures
        │
        ▼
Compare with Actual Results
```

This helps reduce hindsight bias because the model does not train on the same results it is later evaluated against.

The system tracks:

- Outcome prediction accuracy
- Exact score prediction accuracy
- Gameweek-level performance
- Historical predictions versus actual results

---

# 🗂️ Data Pipeline

The project includes a pipeline for collecting and processing football data.

```text
Football Data Sources
        │
        ▼
Match & Team Ingestion
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ├──────────────────────┐
        ▼                      ▼
Premier League Data    Championship Data
        │                      │
        │              Promoted Team Analysis
        │                      │
        └──────────────┬───────┘
                       ▼
              League Strength Normalization
                       │
                       ▼
                Model Training
                       │
                       ▼
              Dixon–Coles Model
                       │
                       ▼
                 Predictions
```

The data pipeline supports the model by preparing historical match information and team-level data before training and prediction.

---

# 🏗️ System Architecture

```text
                        ┌─────────────────────┐
                        │ Football Data Source │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Data Ingestion    │
                        │  Cleaning & Features│
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         ┌───────────────────┐         ┌───────────────────┐
         │ Premier League    │         │ Championship Data │
         │ Historical Data   │         │ Promoted Teams    │
         └─────────┬─────────┘         └─────────┬─────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ League Strength         │
                     │ Normalization &         │
                     │ Feature Preparation     │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Dixon–Coles / Poisson   │
                     │ Prediction Model        │
                     │                         │
                     │ • Attack Strength       │
                     │ • Defence Strength      │
                     │ • Home Advantage        │
                     │ • Expected Goals        │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ FastAPI Prediction API  │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Web Dashboard           │
                     │ Predictions & History   │
                     └─────────────────────────┘
```

---

# 🛠️ Tech Stack

## Modelling & Data Science

- **Python 3.11+**
- **NumPy**
- **Pandas**
- **SciPy**
- **Poisson distribution modelling**
- **Dixon–Coles model**

## Data Pipeline

- **Supabase**
- **HTTPX**
- **Requests**
- **BeautifulSoup**
- **LXML**

## API

- **FastAPI**
- **Uvicorn**
- **Pydantic**

## Development & Deployment

- **uv** for Python dependency management
- **Pytest** for testing
- **Ruff** for linting
- **Next.js** for the frontend
- **Vercel** for frontend deployment
- **Render** for backend deployment

---

# 📁 Project Structure

```text
premier-league-predictor/
│
├── api/
│   └── server.py
│       FastAPI wrapper around the prediction pipeline
│
├── ingest/
│   └── Data ingestion and source integration
│
├── clean/
│   └── Data cleaning pipeline
│
├── data/
│   └── Project data
│
├── features/
│   └── Feature engineering and generated feature data
│
├── models/
│   └── poisson_dixon_coles.py
│       Dixon–Coles prediction model
│
├── tests/
│   └── Project tests
│
├── main.py
│   Main project workflow / CLI entry point
│
├── pyproject.toml
│   Project dependencies and tooling configuration
│
├── uv.lock
│   Locked dependency versions
│
└── README.md
```

---

# 🚀 Running Locally

## 1. Clone the Repository

```bash
git clone https://github.com/SahilK-10/premier-league-predictor.git
cd premier-league-predictor
```

## 2. Install Dependencies

This project uses `uv` for dependency management.

```bash
uv sync
```

---

## 3. Configure Environment Variables

Create a `.env` file in the project root.

Example:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

FOOTBALL_DATA_API_KEY=your_football_data_api_key
```

Depending on the local configuration, the project can also use:

```env
SUPABASE_KEY=your_supabase_key
```

> Never commit `.env` files, API keys, or service-role credentials to GitHub.

---

## 4. Run the API

Start the FastAPI server:

```bash
uv run uvicorn api.server:app --reload --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Check the API health endpoint:

```text
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

---

# 🔌 API Endpoints

## Health Check

```http
GET /health
```

Returns the API status.

---

## Current Gameweek Predictions

```http
GET /fixtures/current-gameweek?season=2026
```

Returns:

- Current gameweek
- Upcoming fixtures
- Team information
- Expected goals
- Home/draw/away probabilities
- Most likely scoreline
- Scoreline probability distribution

---

## Single Match Prediction

```http
GET /predictions/{home_team}/{away_team}?season=2026
```

Example:

```text
/predictions/Arsenal%20FC/Chelsea%20FC?season=2026
```

---

## Accuracy Metrics

```http
GET /accuracy?season=2025
```

Returns available evaluation metrics, including:

- Model accuracy
- Log loss
- Brier score

---

## Available Historical Gameweeks

```http
GET /history/gameweeks?season=2026
```

Returns completed gameweeks with available historical backtests.

---

## Historical Gameweek Backtest

```http
GET /history/gameweek/{gameweek}?season=2026
```

Returns:

- Actual scores
- Predicted scores
- Outcome correctness
- Exact score correctness
- Gameweek summary statistics

---

# 🌐 Deployment Architecture

The application is deployed as separate frontend and backend services.

```text
Users
  │
  ▼
Next.js Frontend
(Vercel)
  │
  ▼
FastAPI Prediction API
(Render)
  │
  ├──────────────► Supabase
  │                 Team & Supporting Data
  │
  └──────────────► Feature Data
                    & Model Artifacts
```

## Backend

The FastAPI prediction API is deployed on Render.

The service runs using:

```bash
uv run uvicorn api.server:app --host 0.0.0.0 --port $PORT
```

## Frontend

The prediction dashboard is deployed on Vercel and communicates with the deployed FastAPI backend.

Production credentials and API configuration are stored through environment variables rather than being committed to the repository.

---

# 🧪 Testing

The project uses:

- `pytest`
- `pytest-asyncio`

Run the test suite with:

```bash
uv run pytest
```

---

# 🧹 Code Quality

The project uses **Ruff** for linting.

Run:

```bash
uv run ruff check .
```

The configured linting rules include:

- `E` — Pycodestyle errors
- `F` — Pyflakes
- `I` — Import sorting
- `UP` — Pyupgrade
- `B` — Flake8 Bugbear

---

# ⚠️ Limitations

This project is a statistical prediction system and should not be interpreted as a guarantee of match results.

Football matches contain substantial uncertainty, including factors that may not be fully represented by the model, such as:

- Injuries
- Suspensions
- Starting lineups
- Tactical changes
- Short-term player form
- Fixture congestion
- Red cards
- In-match randomness

Similarly, while promoted-team normalization provides a more informed starting point than treating Championship teams as completely unknown, movement between leagues introduces uncertainty that cannot be captured perfectly through historical statistics alone.

A prediction probability represents the model's estimated likelihood based on the data and modelling assumptions used by the system.

---

# 🗺️ Future Improvements

Planned improvements include:

- [ ] Fully automate post-gameweek data updates
- [ ] Automatically ingest completed match results
- [ ] Automatically move completed predictions into historical results
- [ ] Automatically calculate outcome and exact-score accuracy
- [ ] Automatically fetch upcoming Premier League fixtures
- [ ] Automatically generate predictions for the next gameweek
- [ ] Scheduled model retraining after new results
- [ ] Additional feature engineering
- [ ] More detailed promoted-team adjustment methods
- [ ] Prediction calibration analysis
- [ ] Comparison against additional prediction models
- [ ] More detailed model explainability
- [ ] Expanded historical analytics

---

# 📚 Key Learning Areas

This project combines several areas of machine learning, statistics and data engineering:

- Statistical modelling
- Poisson processes
- Dixon–Coles football modelling
- Probability distributions
- Maximum likelihood estimation
- Team strength estimation
- League-strength normalization
- Promoted-team modelling
- Feature engineering
- Historical backtesting
- Model evaluation
- Data ingestion pipelines
- API development
- Production deployment

---

# 👤 Author

**Sahil Khade**

GitHub: [@SahilK-10](https://github.com/SahilK-10)

Repository: [Premier League Predictor](https://github.com/SahilK-10/premier-league-predictor)

---

## ⭐ If you found this project interesting

Consider giving the repository a star!

The long-term goal of this project is to build a progressively more automated Premier League prediction system while continuing to improve the underlying statistical modelling, promoted-team handling, data pipeline and evaluation framework.