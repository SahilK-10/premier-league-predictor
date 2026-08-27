
"""
football_data_client.py

Fetch Championship match data from football-data.org and store it
idempotently in Supabase.

Usage:
    uv run python -m ingest.football_data_client

Examples:
    uv run python -m ingest.football_data_client --season 2025
    uv run python -m ingest.football_data_client --date-from 2026-08-01 --date-to 2026-08-31
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
CHAMPIONSHIP_CODE = "ELC"

PROMOTED_TEAMS = {
    "Ipswich Town FC",
    "Coventry City FC",
    "Hull City AFC",
}

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure readable console logging."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize_team_name(name: str) -> str:
    """
    Create a stable normalized version of a team name.

    The same normalization strategy will later be reused by the cleaning layer
    and team-name mapping system.
    """
    normalized = name.lower().strip()
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def get_settings() -> tuple[str, str, str]:
    """Load and validate required environment variables."""
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    missing = []

    if not api_key:
        missing.append("FOOTBALL_DATA_API_KEY")
    if not supabase_url:
        missing.append("SUPABASE_URL")
    if not supabase_key:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")

    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variables: {missing_text}. "
            "Add them to your .env file."
        )

    return api_key, supabase_url, supabase_key


class ChampionshipDataClient:
    """Client for football-data.org Championship match ingestion."""

    def __init__(self) -> None:
        api_key, supabase_url, supabase_key = get_settings()

        self.api_key = api_key
        self.supabase: Client = create_client(supabase_url, supabase_key)

        self.headers = {
            "X-Auth-Token": self.api_key,
            "Accept": "application/json",
        }

    def fetch_matches(
        self,
        season: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch Championship matches.

        Args:
            season: Season starting year, for example 2025 for 2025/26.
            date_from: Optional YYYY-MM-DD lower date bound.
            date_to: Optional YYYY-MM-DD upper date bound.
        """
        url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{CHAMPIONSHIP_CODE}/matches"

        params: dict[str, Any] = {}

        if season is not None:
            params["season"] = season

        if date_from is not None:
            params["dateFrom"] = date_from

        if date_to is not None:
            params["dateTo"] = date_to

        logger.info("Fetching Championship matches with params=%s", params)

        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                url,
                headers=self.headers,
                params=params,
            )

        if response.status_code == 429:
            raise RuntimeError(
                "football-data.org rate limit reached. "
                "Wait before running the ingestion again."
            )

        if response.status_code == 403:
            raise RuntimeError(
                "football-data.org rejected the API request. "
                "Check your FOOTBALL_DATA_API_KEY and your plan permissions."
            )

        response.raise_for_status()

        payload = response.json()
        matches = payload.get("matches", [])

        matches = [
            match
            for match in matches
            if match.get("homeTeam", {}).get("name") in PROMOTED_TEAMS
            or match.get("awayTeam", {}).get("name") in PROMOTED_TEAMS
        ]

        logger.info(
            "Filtered Championship matches to promoted teams. Remaining=%s",
            len(matches),
        )

        logger.info("Fetched %s matches.", len(matches))

        return matches

    def upsert_team(self, team_data: dict[str, Any]) -> str:
        """
        Upsert one team into the canonical teams table.

        Returns:
            The UUID of the canonical team row.
        """
        source_id = team_data.get("id")
        name = team_data.get("name")

        if not source_id or not name:
            raise ValueError(
                f"Team data is missing required id or name: {team_data}"
            )

        normalized_name = normalize_team_name(name)

        row = {
            "name": name,
            "normalized_name": normalized_name,
            "football_data_id": source_id,
            "short_name": team_data.get("shortName"),
            "tla": team_data.get("tla"),
            "crest_url": team_data.get("crest"),
            "country": "England",
        }

        result = (
            self.supabase.table("teams")
            .upsert(
                row,
                on_conflict="football_data_id",
            )
            .execute()
        )

        if not result.data:
            raise RuntimeError(f"Failed to upsert team: {name}")

        team_id = result.data[0]["id"]

        self.upsert_team_mapping(
            source="football-data.org",
            source_team_name=name,
            team_id=team_id,
        )

        return team_id

    def upsert_team_mapping(
        self,
        source: str,
        source_team_name: str,
        team_id: str,
    ) -> None:
        """Store the provider-specific team name mapping."""
        row = {
            "source": source,
            "source_team_name": source_team_name,
            "normalized_source_name": normalize_team_name(source_team_name),
            "team_id": team_id,
        }

        (
            self.supabase.table("team_name_mappings")
            .upsert(
                row,
                on_conflict="source,normalized_source_name",
            )
            .execute()
        )

    @staticmethod
    def extract_season_start_year(match: dict[str, Any]) -> int | None:
        """Extract the starting year from the football-data season object."""
        season = match.get("season") or {}
        start_date = season.get("startDate")

        if not start_date:
            return None

        return datetime.fromisoformat(
            start_date.replace("Z", "+00:00")
        ).year

    @staticmethod
    def extract_score(match: dict[str, Any]) -> tuple[int | None, int | None]:
        """
        Extract full-time scores.

        Scheduled and postponed fixtures may not have scores yet.
        """
        score = match.get("score") or {}
        full_time = score.get("fullTime") or {}

        return (
            full_time.get("home"),
            full_time.get("away"),
        )

    @staticmethod
    def extract_referee(match: dict[str, Any]) -> str | None:
        """Extract the primary referee name when available."""
        referees = match.get("referees") or []

        if not referees:
            return None

        return referees[0].get("name")

    def upsert_match(self, match: dict[str, Any]) -> None:
        """Upsert one raw football-data.org match."""
        home_team = match.get("homeTeam") or {}
        away_team = match.get("awayTeam") or {}

        if not home_team.get("id") or not away_team.get("id"):
            raise ValueError(
                f"Match {match.get('id')} is missing team identifiers."
            )

        # Upsert canonical teams first.
        self.upsert_team(home_team)
        self.upsert_team(away_team)

        home_score, away_score = self.extract_score(match)

        utc_date = match.get("utcDate")
        season_start_year = self.extract_season_start_year(match)

        row = {
            "source": "football-data.org",
            "source_match_id": match["id"],
            "competition_code": (
                match.get("competition") or {}
            ).get("code", CHAMPIONSHIP_CODE),
            "season_start_year": season_start_year,
            "matchday": match.get("matchday"),
            "utc_date": utc_date,
            "status": match.get("status"),
            "home_team_source_id": home_team.get("id"),
            "home_team_name": home_team.get("name"),
            "away_team_source_id": away_team.get("id"),
            "away_team_name": away_team.get("name"),
            "home_score": home_score,
            "away_score": away_score,
            "winner": (match.get("score") or {}).get("winner"),
            "referee_name": self.extract_referee(match),
            "venue": None,
            "neutral_venue": False,
            "raw_payload": match,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        (
            self.supabase.table("raw_matches")
            .upsert(
                row,
                on_conflict="source,source_match_id",
            )
            .execute()
        )

    def ingest_matches(
        self,
        season: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> int:
        """
        Fetch and persist Premier League matches.

        Returns:
            Number of matches processed.
        """
        matches = self.fetch_matches(
            season=season,
            date_from=date_from,
            date_to=date_to,
        )

        successful = 0
        failed = 0

        for match in matches:
            match_id = match.get("id")

            try:
                self.upsert_match(match)
                successful += 1

                logger.info(
                    "Upserted match %s: %s vs %s",
                    match_id,
                    (match.get("homeTeam") or {}).get("name"),
                    (match.get("awayTeam") or {}).get("name"),
                )

            except Exception:
                failed += 1

                logger.exception(
                    "Failed to process match %s",
                    match_id,
                )

            # Be conservative with API/database request bursts.
            time.sleep(0.05)

        logger.info(
            "Ingestion complete. Successful=%s Failed=%s",
            successful,
            failed,
        )

        if failed:
            logger.warning(
                "%s matches failed. Check the logs above.",
                failed,
            )

        return successful


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest Championship matches from football-data.org."
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season starting year, e.g. 2025 for the 2025/26 season.",
    )

    parser.add_argument(
        "--date-from",
        dest="date_from",
        default=None,
        help="Start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--date-to",
        dest="date_to",
        default=None,
        help="End date in YYYY-MM-DD format.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = parse_args()

    client = ChampionshipDataClient()

    processed = client.ingest_matches(
        season=args.season,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    logger.info("Finished. Processed %s matches.", processed)


if __name__ == "__main__":
    main()

