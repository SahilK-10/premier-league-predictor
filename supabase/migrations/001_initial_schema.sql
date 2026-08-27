```sql
-- ============================================================
-- Premier League Predictor
-- Initial Supabase / PostgreSQL Schema
-- ============================================================

create extension if not exists pgcrypto;

-- ============================================================
-- ENUMS
-- ============================================================

do $$
begin
    create type match_status as enum (
        'SCHEDULED',
        'TIMED',
        'IN_PLAY',
        'PAUSED',
        'FINISHED',
        'POSTPONED',
        'SUSPENDED',
        'CANCELLED'
    );
exception
    when duplicate_object then null;
end $$;

do $$
begin
    create type prediction_outcome as enum (
        'HOME',
        'DRAW',
        'AWAY'
    );
exception
    when duplicate_object then null;
end $$;

-- ============================================================
-- UPDATED_AT HELPER
-- ============================================================

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = timezone('utc', now());
    return new;
end;
$$;

-- ============================================================
-- 1. TEAMS
-- Canonical team dimension used across all sources.
-- ============================================================

create table if not exists public.teams (
    id uuid primary key default gen_random_uuid(),

    name text not null,
    normalized_name text not null unique,

    football_data_id integer unique,
    understat_id integer unique,

    short_name text,
    tla text,
    crest_url text,

    country text not null default 'England',

    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now())
);

create index if not exists idx_teams_normalized_name
    on public.teams(normalized_name);

-- ============================================================
-- 2. TEAM NAME MAPPINGS
-- Maps inconsistent names from different providers to one team.
-- ============================================================

create table if not exists public.team_name_mappings (
    id uuid primary key default gen_random_uuid(),

    source text not null,
    source_team_name text not null,
    normalized_source_name text not null,

    team_id uuid not null references public.teams(id) on delete cascade,

    created_at timestamptz not null default timezone('utc', now()),

    unique(source, normalized_source_name)
);

create index if not exists idx_team_name_mappings_lookup
    on public.team_name_mappings(source, normalized_source_name);

-- ============================================================
-- 3. RAW MATCHES
-- Original football-data.org payload plus selected normalized fields.
-- ============================================================

create table if not exists public.raw_matches (
    id uuid primary key default gen_random_uuid(),

    source text not null default 'football-data.org',
    source_match_id bigint not null,

    competition_code text,
    season_start_year integer,
    matchday integer,

    utc_date timestamptz,
    status text,

    home_team_source_id integer,
    home_team_name text,
    away_team_source_id integer,
    away_team_name text,

    home_score integer,
    away_score integer,

    winner text,

    referee_name text,

    venue text,
    neutral_venue boolean not null default false,

    raw_payload jsonb not null,

    fetched_at timestamptz not null default timezone('utc', now()),
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),

    unique(source, source_match_id)
);

create index if not exists idx_raw_matches_season_matchday
    on public.raw_matches(season_start_year, matchday);

create index if not exists idx_raw_matches_utc_date
    on public.raw_matches(utc_date);

-- ============================================================
-- 4. RAW XG
-- Raw Understat or equivalent xG provider data.
-- ============================================================

create table if not exists public.raw_xg (
    id uuid primary key default gen_random_uuid(),

    source text not null default 'understat',
    source_match_id text not null,

    competition_code text default 'PL',
    season_start_year integer,

    match_date timestamptz,

    home_team_name text,
    away_team_name text,

    home_goals integer,
    away_goals integer,

    home_xg numeric(8,4),
    away_xg numeric(8,4),

    raw_payload jsonb not null,

    fetched_at timestamptz not null default timezone('utc', now()),
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),

    unique(source, source_match_id)
);

create index if not exists idx_raw_xg_season
    on public.raw_xg(season_start_year);

create index if not exists idx_raw_xg_match_date
    on public.raw_xg(match_date);

-- ============================================================
-- 5. RAW ODDS
-- Odds are stored only as an external benchmark.
-- They must never be used as model features.
-- ============================================================

create table if not exists public.raw_odds (
    id uuid primary key default gen_random_uuid(),

    source text not null,
    source_event_id text not null,

    bookmaker text,

    competition_code text default 'PL',
    season_start_year integer,

    match_date timestamptz,

    home_team_name text,
    away_team_name text,

    home_odds numeric(10,4),
    draw_odds numeric(10,4),
    away_odds numeric(10,4),

    implied_home_probability numeric(10,6),
    implied_draw_probability numeric(10,6),
    implied_away_probability numeric(10,6),

    raw_payload jsonb not null,

    fetched_at timestamptz not null default timezone('utc', now()),
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),

    unique(source, source_event_id, bookmaker)
);

create index if not exists idx_raw_odds_match
    on public.raw_odds(match_date, home_team_name, away_team_name);

-- ============================================================
-- 6. CLEAN MATCHES
-- Main fact table used by features and modeling.
-- ============================================================

create table if not exists public.matches (
    id uuid primary key default gen_random_uuid(),

    raw_match_id uuid unique references public.raw_matches(id) on delete set null,
    raw_xg_id uuid unique references public.raw_xg(id) on delete set null,

    source_match_id bigint,

    competition_code text not null default 'PL',
    season_start_year integer not null,
    matchday integer,

    kickoff_at timestamptz not null,
    status match_status not null default 'SCHEDULED',

    home_team_id uuid not null references public.teams(id),
    away_team_id uuid not null references public.teams(id),

    home_goals integer,
    away_goals integer,

    home_xg numeric(8,4),
    away_xg numeric(8,4),

    referee text,
    venue text,

    neutral_venue boolean not null default false,
    is_postponed boolean not null default false,

    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),

    check (home_team_id <> away_team_id),

    unique(
        competition_code,
        season_start_year,
        kickoff_at,
        home_team_id,
        away_team_id
    )
);

create index if not exists idx_matches_kickoff
    on public.matches(kickoff_at);

create index if not exists idx_matches_season_matchday
    on public.matches(season_start_year, matchday);

create index if not exists idx_matches_home_team
    on public.matches(home_team_id, kickoff_at);

create index if not exists idx_matches_away_team
    on public.matches(away_team_id, kickoff_at);

-- ============================================================
-- 7. FIXTURE FEATURES
-- One pre-kickoff feature snapshot per fixture.
-- ============================================================

create table if not exists public.fixture_features (
    id uuid primary key default gen_random_uuid(),

    match_id uuid not null references public.matches(id) on delete cascade,

    feature_version text not null default 'v1',
    computed_at timestamptz not null default timezone('utc', now()),

    home_form_points_5 numeric(10,4),
    away_form_points_5 numeric(10,4),

    home_form_points_10 numeric(10,4),
    away_form_points_10 numeric(10,4),

    home_goals_for_5 numeric(10,4),
    home_goals_against_5 numeric(10,4),

    away_goals_for_5 numeric(10,4),
    away_goals_against_5 numeric(10,4),

    home_xg_for_5 numeric(10,4),
    home_xg_against_5 numeric(10,4),

    away_xg_for_5 numeric(10,4),
    away_xg_against_5 numeric(10,4),

    home_attack_strength numeric(10,6),
    home_defense_strength numeric(10,6),

    away_attack_strength numeric(10,6),
    away_defense_strength numeric(10,6),

    home_table_position integer,
    away_table_position integer,

    home_points_per_game numeric(10,4),
    away_points_per_game numeric(10,4),

    home_rest_days integer,
    away_rest_days integer,

    home_matches_last_14_days integer,
    away_matches_last_14_days integer,

    h2h_home_wins integer,
    h2h_draws integer,
    h2h_away_wins integer,

    feature_payload jsonb not null default '{}'::jsonb,

    created_at timestamptz not null default timezone('utc', now()),

    unique(match_id, feature_version)
);

create index if not exists idx_fixture_features_match
    on public.fixture_features(match_id);

-- ============================================================
-- 8. MODEL RUNS
-- Tracks every training / backtest / production model version.
-- ============================================================

create table if not exists public.model_runs (
    id uuid primary key default gen_random_uuid(),

    model_name text not null,
    model_version text not null,

    model_type text not null,

    training_start_date date,
    training_end_date date,

    feature_version text,

    hyperparameters jsonb not null default '{}'::jsonb,
    metrics jsonb not null default '{}'::jsonb,

    artifact_path text,

    is_active boolean not null default false,

    trained_at timestamptz not null default timezone('utc', now()),
    created_at timestamptz not null default timezone('utc', now()),

    unique(model_name, model_version)
);

create index if not exists idx_model_runs_active
    on public.model_runs(model_name, is_active);

-- ============================================================
-- 9. PREDICTIONS
-- Stores ensemble and component predictions.
-- ============================================================

create table if not exists public.predictions (
    id uuid primary key default gen_random_uuid(),

    match_id uuid not null references public.matches(id) on delete cascade,
    model_run_id uuid not null references public.model_runs(id) on delete restrict,

    prediction_version text not null default 'v1',

    predicted_home_goals numeric(10,4),
    predicted_away_goals numeric(10,4),

    home_win_probability numeric(10,6) not null,
    draw_probability numeric(10,6) not null,
    away_win_probability numeric(10,6) not null,

    predicted_outcome prediction_outcome not null,

    most_likely_home_goals integer,
    most_likely_away_goals integer,
    most_likely_score_probability numeric(10,6),

    poisson_home_win_probability numeric(10,6),
    poisson_draw_probability numeric(10,6),
    poisson_away_win_probability numeric(10,6),

    xgboost_home_win_probability numeric(10,6),
    xgboost_draw_probability numeric(10,6),
    xgboost_away_win_probability numeric(10,6),

    scoreline_probabilities jsonb not null default '{}'::jsonb,

    shap_values jsonb,
    explanation_summary text,

    created_at timestamptz not null default timezone('utc', now()),

    unique(match_id, model_run_id, prediction_version),

    check (
        home_win_probability >= 0
        and draw_probability >= 0
        and away_win_probability >= 0
    )
);

create index if not exists idx_predictions_match
    on public.predictions(match_id);

create index if not exists idx_predictions_model_run
    on public.predictions(model_run_id);

-- ============================================================
-- 10. PREDICTION EVALUATIONS
-- Actual result vs model and bookmaker baseline.
-- ============================================================

create table if not exists public.prediction_evaluations (
    id uuid primary key default gen_random_uuid(),

    prediction_id uuid not null unique
        references public.predictions(id) on delete cascade,

    match_id uuid not null unique
        references public.matches(id) on delete cascade,

    actual_home_goals integer not null,
    actual_away_goals integer not null,
    actual_outcome prediction_outcome not null,

    prediction_correct boolean not null,

    model_log_loss numeric(12,8),
    model_brier_score numeric(12,8),

    bookmaker_source text,

    bookmaker_home_probability numeric(10,6),
    bookmaker_draw_probability numeric(10,6),
    bookmaker_away_probability numeric(10,6),

    bookmaker_prediction_correct boolean,
    bookmaker_log_loss numeric(12,8),
    bookmaker_brier_score numeric(12,8),

    evaluated_at timestamptz not null default timezone('utc', now()),
    created_at timestamptz not null default timezone('utc', now())
);

create index if not exists idx_prediction_evaluations_match
    on public.prediction_evaluations(match_id);

-- ============================================================
-- 11. ACCURACY SNAPSHOTS
-- Powers the frontend performance tracker over time.
-- ============================================================

create table if not exists public.accuracy_snapshots (
    id uuid primary key default gen_random_uuid(),

    snapshot_date date not null,

    model_run_id uuid references public.model_runs(id) on delete set null,

    matches_evaluated integer not null default 0,

    model_accuracy numeric(10,6),
    model_log_loss numeric(12,8),
    model_brier_score numeric(12,8),

    bookmaker_accuracy numeric(10,6),
    bookmaker_log_loss numeric(12,8),
    bookmaker_brier_score numeric(12,8),

    rolling_window integer,

    metadata jsonb not null default '{}'::jsonb,

    created_at timestamptz not null default timezone('utc', now()),

    unique(snapshot_date, model_run_id, rolling_window)
);

create index if not exists idx_accuracy_snapshots_date
    on public.accuracy_snapshots(snapshot_date);

-- ============================================================
-- UPDATED_AT TRIGGERS
-- ============================================================

drop trigger if exists set_teams_updated_at on public.teams;
create trigger set_teams_updated_at
before update on public.teams
for each row
execute function public.set_updated_at();

drop trigger if exists set_raw_matches_updated_at on public.raw_matches;
create trigger set_raw_matches_updated_at
before update on public.raw_matches
for each row
execute function public.set_updated_at();

drop trigger if exists set_raw_xg_updated_at on public.raw_xg;
create trigger set_raw_xg_updated_at
before update on public.raw_xg
for each row
execute function public.set_updated_at();

drop trigger if exists set_raw_odds_updated_at on public.raw_odds;
create trigger set_raw_odds_updated_at
before update on public.raw_odds
for each row
execute function public.set_updated_at();

drop trigger if exists set_matches_updated_at on public.matches;
create trigger set_matches_updated_at
before update on public.matches
for each row
execute function public.set_updated_at();

-- ============================================================
-- HELPFUL VIEW
-- Latest prediction for each fixture.
-- ============================================================

create or replace view public.latest_predictions as
select distinct on (p.match_id)
    p.*
from public.predictions p
join public.model_runs mr
    on mr.id = p.model_run_id
order by
    p.match_id,
    mr.trained_at desc,
    p.created_at desc;

-- ============================================================
-- NOTES
-- ============================================================
-- 1. raw_odds are intentionally isolated from fixture_features.
--    Bookmaker odds are a benchmark, NOT a model input.
--
-- 2. fixture_features stores only information available before kickoff.
--
-- 3. predictions preserve component probabilities so we can inspect
--    the Poisson, XGBoost, and final ensemble contributions.
--
-- 4. scoreline_probabilities is JSONB because the complete probability
--    matrix can vary in maximum goal range between model versions.
-- ============================================================
```
