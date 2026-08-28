# ⚽ Premier League Predictor

A machine learning-driven Premier League match prediction system built around a **Poisson / Dixon–Coles statistical model**.

The project estimates team attacking and defensive strengths from historical match data and uses them to generate:

- Match outcome probabilities
- Expected goals
- Most likely exact scorelines
- Scoreline probability distributions
- Current gameweek predictions
- Historical gameweek backtesting and accuracy metrics

The prediction engine is exposed through a FastAPI API and visualized through a web dashboard.

**Live Demo:** https://premier-league-predictor-iota.vercel.app

---

## 🧠 The Prediction Model

At the core of the project is a **Dixon–Coles adjusted Poisson model**, a statistical approach designed specifically for modelling football scores.

For a fixture:

> **Home Team vs Away Team**

the model estimates the expected number of goals scored by both teams:

\[
\lambda_{home}
\]

and

\[
\lambda_{away}
\]

These estimates are derived from learned team-level parameters including:

- ⚔️ Attacking strength
- 🛡️ Defensive strength
- 🏟️ Home advantage
- 📈 Historical match results

The expected goals are then used to estimate the probability of every possible scoreline.

For example:

```text
          Away Goals
          0      1      2      3
Home 0   0-0    0-1    0-2    0-3
Goals 1  1-0    1-1    1-2    1-3
      2  2-0    2-1    2-2    2-3
      3  3-0    3-1    3-2    3-3
```

The Dixon–Coles adjustment is applied to improve modelling of low-scoring football results, particularly:

- `0–0`
- `1–0`
- `0–1`
- `1–1`

These outcomes can exhibit dependencies that are not captured perfectly by a completely independent Poisson model.

---

## 📊 What the Model Predicts

For every fixture, the system produces three different types of predictions.

### 1. Match Outcome Probabilities

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

### 2. Most Likely Exact Score

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

Most Likely Outcome:
Home Win — 42%
```

This is not a contradiction.

`1–1` may be the most probable **individual scoreline**, while the combined probability of all possible home-winning scorelines may still be greater than the total probability of all draws.

Conceptually:

```text
HOME WINS
1-0 + 2-0 + 2-1 + 3-0 + 3-1 + ...

DRAWS
0-0 + 1-1 + 2-2 + 3-3 + ...

AWAY WINS
0-1 + 0-2 + 1-2 + 0-3 + 1-3 + ...
```

---

### 3. Expected Goals

The model estimates the expected number of goals for each team.

For example:

```text
Arsenal Expected Goals: 1.72
Chelsea Expected Goals: 1.14
```

These values represent the underlying scoring expectation used to generate the score probability distribution.

---

## ✨ Features

### ⚽ Current Gameweek Predictions

Automatically identifies the current gameweek from the available fixture data and generates predictions for every fixture.

Each prediction includes:

- Home and away teams
- Team crests
- Expected goals
- Home win probability
- Draw probability
- Away win probability
- Most likely exact score
- Exact score probability
- Scoreline probability grid
- Human-readable prediction summary

---

### 📊 Historical Gameweek Backtesting

The project includes a historical evaluation system for completed gameweeks.

For each completed gameweek, the model uses a **leave-one-gameweek-out backtesting approach**.

Instead of training the model on a gameweek and then evaluating it on the same matches, the target gameweek is excluded from the training data.

```text
Historical Data
      │
      ▼
Exclude Target Gameweek
      │
      ▼
Train Dixon–Coles Model
      │
      ▼
Predict Excluded Gameweek
      │
      ▼
Compare Predictions with Actual Results
```

This helps avoid hindsight bias and provides a fairer measure of predictive performance.

The system tracks:

- Outcome prediction accuracy
- Exact score prediction accuracy
- Gameweek-level performance
- Historical prediction results

---

### 📈 Multi-Season Model Training

The project supports a multi-season model artifact rather than relying only on the small amount of data available in the current season.

This is particularly important early in a Premier League season.

Using multiple seasons provides the model with substantially more match data for estimating team attack and defence parameters.

The deployed prediction API loads a cached multi-season model artifact when available.

---

### 🔄 Team and Match Data Pipeline

The project includes a data pipeline responsible for:

```text
Football Data Source
        ↓
Match & Team Ingestion
        ↓
Supabase
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Predictions
```

Team information stored in Supabase includes data such as:

- Team name
- Short name
- Team abbreviation
- Crest URL

---

## 🏗️ Project Architecture

```text
                         ┌──────────────────────┐
                         │   Football Data API  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Data Ingestion      │
                         │  & Feature Pipeline  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Supabase        │
                         │ Teams & Match Data   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────┐
                  │   Dixon–Coles Prediction     │
                  │          Model               │
                  │                              │
                  │ • Attack Strength            │
                  │ • Defence Strength           │
                  │ • Home Advantage             │
                  │ • Expected Goals             │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │        FastAPI API           │
                  │                              │
                  │ /fixtures/current-gameweek   │
                  │ /predictions/...             │
                  │ /history/...                 │
                  │ /accuracy                    │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │      Next.js Dashboard       │
                  │                              │
                  │ Predictions • History        │
                  │ Probabilities • Analytics    │
                  └──────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Modelling & Data Science

- **Python 3.11+**
- **NumPy**
- **Pandas**
- **SciPy**
- **Poisson distribution**
- **Dixon–Coles model**

### Data Pipeline

- **football-data.org**
- **Supabase**
- **Requests / HTTPX**
- **BeautifulSoup**
- **LXML**

### API

- **FastAPI**
- **Uvicorn**
- **Pydantic**

### Frontend & Deployment

- **Next.js**
- **Vercel**
- **Render**

---

## 📁 Project Structure

```text
premier-league-predictor/
│
├── api/
│   └── server.py                 # FastAPI prediction API
│
├── ingest/                       # Match and team data ingestion
│
├── clean/                        # Data cleaning pipeline
│
├── data/                         # Project data
│
├── features/                     # Feature engineering
│
├── models/
│   └── poisson_dixon_coles.py    # Dixon–Coles model implementation
│
├── tests/                        # Tests
│
├── main.py                       # Project CLI / workflow entry point
├── pyproject.toml                # Python dependencies and configuration
├── uv.lock                       # Locked dependency versions
└── README.md
```

---

# 🚀 Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/SahilK-10/premier-league-predictor.git
cd premier-league-predictor
```

## 2. Install Dependencies

This project uses `uv` for dependency management.

Install dependencies:

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

> Never commit `.env` files or secret API keys to GitHub.

---

## 4. Run the API

Start the FastAPI backend:

```bash
uv run uvicorn api.server:app --reload --port 8000
```

The API will then be available locally at:

```text
http://localhost:8000
```

Health check:

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

Checks whether the API is running.

---

## Current Gameweek Fixtures

```http
GET /fixtures/current-gameweek?season=2026
```

Returns:

- Current gameweek
- Fixtures
- Team information
- Expected goals
- Win/draw probabilities
- Most likely scorelines
- Scoreline probabilities

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

## Accuracy

```http
GET /accuracy?season=2025
```

Returns stored model evaluation metrics where available, including:

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

## Historical Gameweek Results

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

# 🌐 Deployment

The application is deployed as separate frontend and backend services.

```text
Users
  │
  ▼
Vercel
Next.js Frontend
  │
  ▼
Render
FastAPI + Prediction Model
  │
  ├──────────────► Supabase
  │                 Team Data
  │
  └──────────────► Model Artifacts
                    & Feature Data
```

### Frontend

Deployed using **Vercel**.

### Backend

The FastAPI prediction API is deployed using **Render**.

The backend starts using:

```bash
uv sync --frozen
```

and:

```bash
uv run uvicorn api.server:app --host 0.0.0.0 --port $PORT
```

### Environment Variables

Production credentials are stored as environment variables rather than being committed to the repository.

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
- `B` — Flake8 bugbear

---

# ⚠️ Important Limitations

This project is a statistical prediction system and should not be interpreted as a guarantee of match results.

Football matches contain significant uncertainty, including factors that may not be fully represented by the model, such as:

- Injuries
- Suspensions
- Lineups
- Tactical changes
- Player form
- Fixture congestion
- Red cards
- In-match randomness

A prediction probability represents the model's estimated likelihood based on the information and historical data used by the system.

---

# 🗺️ Future Improvements

Planned improvements include:

- [ ] Fully automate post-gameweek data updates
- [ ] Automatically move completed predictions into historical results
- [ ] Automatically calculate outcome and exact-score accuracy
- [ ] Automatically fetch upcoming Premier League fixtures
- [ ] Automatically generate predictions for the next gameweek
- [ ] Scheduled model retraining after new match results
- [ ] Improved feature engineering
- [ ] Additional team/player-level features
- [ ] Prediction calibration analysis
- [ ] Model comparison against additional statistical baselines
- [ ] More detailed explainability and analytics

---

# 📌 Key Learning Areas

This project combines several areas of machine learning and data engineering:

- Statistical modelling
- Poisson processes
- Dixon–Coles football modelling
- Maximum likelihood estimation
- Probability distributions
- Model evaluation
- Backtesting
- Feature engineering
- Data ingestion pipelines
- API development
- Production deployment

---

# 👤 Author

**Sahil Khade**

GitHub: [@SahilK-10](https://github.com/SahilK-10)

---

## ⭐ If you found this project interesting

Consider giving the repository a star!

The goal of this project is to build a progressively more automated Premier League prediction system while continuing to improve the underlying statistical modelling and evaluation pipeline.