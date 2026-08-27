"""
understat_scraper.py

Fetch Premier League match-level xG data from Understat and store it
idempotently in the raw_xg Supabase table.

Usage:
    uv run python -m ingest.understat_scraper --season 2025
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv()

UNDERSTAT_BASE_URL = "https://understat.com"
LEAGUE_CODE = "EPL"

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_settings() -> tuple[str, str]:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    missing: list[str] = []

    if not supabase_url:
        missing.append("SUPABASE_URL")

    if not supabase_key:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Add them to your .env file."
        )

    return supabase_url, supabase_key


class UnderstatScraper:
    """Fetch and store Premier League match-level xG data."""

    def __init__(self) -> None:
        supabase_url, supabase_key = get_settings()

        self.supabase: Client = create_client(
            supabase_url,
            supabase_key,
        )

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://understat.com/",
            "X-Requested-With": "XMLHttpRequest",
        }


    def fetch_league_data(self, season: int) -> dict[str, Any]:
        """
        Fetch Understat league data from its internal endpoint.

        The endpoint returns the match data directly as JSON.
        """
        url = (
            f"{UNDERSTAT_BASE_URL}/getLeagueData/"
            f"{LEAGUE_CODE}/{season}"
        )

        logger.info(
            "Fetching Understat Premier League data for season %s/%s",
            season,
            str(season + 1)[-2:],
        )

        with httpx.Client(
            headers=self.headers,
            timeout=30.0,
            follow_redirects=True,
        ) as client:
            response = client.get(url)

        response.raise_for_status()

        logger.info(
            "Understat endpoint responded successfully. HTTP %s",
            response.status_code,
        )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Understat did not return valid JSON. "
                f"Response starts with: {response.text[:300]}"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Unexpected Understat response format. "
                f"Expected an object, got {type(data).__name__}."
            )

        logger.info(
            "Understat response keys: %s",
            list(data.keys()),
        )

        return data


    @staticmethod
    def extract_matches(
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Extract match records from the Understat response.

        Different endpoint versions may expose the match collection
        under slightly different keys, so we check the expected options.
        """
        for key in ("dates", "matches", "results"):
            value = data.get(key)

            if isinstance(value, list):
                logger.info(
                    "Found %s match records under '%s'.",
                    len(value),
                    key,
                )
                return value

        raise RuntimeError(
            "Could not find match records in Understat response. "
            f"Available keys: {list(data.keys())}"
        )


    @staticmethod
    def get_nested_value(
        data: dict[str, Any],
        *keys: str,
        default: Any = None,
    ) -> Any:
        current: Any = data

        for key in keys:
            if not isinstance(current, dict):
                return default

            current = current.get(key)

        return default if current is None else current


    @staticmethod
    def to_int(value: Any) -> int | None:
        if value is None or value == "":
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None


    @staticmethod
    def to_float(value: Any) -> float | None:
        if value is None or value == "":
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None


    @staticmethod
    def parse_match_date(value: Any) -> str | None:
        """Convert Understat datetime into an ISO timestamp."""
        if not value:
            return None

        value = str(value).strip()

        formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d",
        )

        for date_format in formats:
            try:
                parsed = datetime.strptime(value, date_format)

                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)

                return parsed.isoformat()

            except ValueError:
                continue

        logger.warning(
            "Could not parse Understat match date: %s",
            value,
        )

        return None


    def normalize_match(
        self,
        match: dict[str, Any],
        season: int,
    ) -> dict[str, Any] | None:
        """Convert one Understat match into our raw_xg schema."""

        source_match_id = match.get("id")

        home_team = self.get_nested_value(
            match,
            "h",
            "title",
        )

        away_team = self.get_nested_value(
            match,
            "a",
            "title",
        )

        if not source_match_id or not home_team or not away_team:
            logger.warning(
                "Skipping malformed Understat match with id=%s",
                source_match_id,
            )
            return None

        return {
            "source": "understat",
            "source_match_id": str(source_match_id),
            "competition_code": "PL",
            "season_start_year": season,
            "match_date": self.parse_match_date(
                match.get("datetime")
            ),
            "home_team_name": home_team,
            "away_team_name": away_team,
            "home_goals": self.to_int(
                self.get_nested_value(
                    match,
                    "goals",
                    "h",
                )
            ),
            "away_goals": self.to_int(
                self.get_nested_value(
                    match,
                    "goals",
                    "a",
                )
            ),
            "home_xg": self.to_float(
                self.get_nested_value(
                    match,
                    "xG",
                    "h",
                )
            ),
            "away_xg": self.to_float(
                self.get_nested_value(
                    match,
                    "xG",
                    "a",
                )
            ),
            "raw_payload": match,
            "fetched_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }


    def upsert_match(
        self,
        row: dict[str, Any],
    ) -> None:
        """Idempotently insert or update one xG record."""

        (
            self.supabase
            .table("raw_xg")
            .upsert(
                row,
                on_conflict="source,source_match_id",
            )
            .execute()
        )


    def ingest_season(
        self,
        season: int,
    ) -> int:
        """Fetch, normalize, and store one Premier League season."""

        data = self.fetch_league_data(season)
        matches = self.extract_matches(data)

        successful = 0
        skipped = 0
        failed = 0

        for match in matches:
            row = self.normalize_match(
                match,
                season,
            )

            if row is None:
                skipped += 1
                continue

            try:
                self.upsert_match(row)
                successful += 1

                logger.info(
                    "Upserted Understat match %s: %s vs %s | "
                    "xG %.2f - %.2f",
                    row["source_match_id"],
                    row["home_team_name"],
                    row["away_team_name"],
                    row["home_xg"] or 0.0,
                    row["away_xg"] or 0.0,
                )

            except Exception:
                failed += 1

                logger.exception(
                    "Failed to process Understat match %s",
                    row["source_match_id"],
                )

            time.sleep(0.05)

        logger.info(
            "Understat ingestion complete. "
            "Successful=%s Skipped=%s Failed=%s",
            successful,
            skipped,
            failed,
        )

        return successful


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest Premier League match-level xG data "
            "from Understat."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help=(
            "Season starting year, e.g. "
            "2025 for the 2025/26 season."
        ),
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    scraper = UnderstatScraper()

    processed = scraper.ingest_season(
        args.season
    )

    logger.info(
        "Finished Understat ingestion. "
        "Processed %s matches.",
        processed,
    )


if __name__ == "__main__":
    main()