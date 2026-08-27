"""
clean_matches.py

Clean and reconcile football-data.org match data with Understat xG data.

Reads:
    raw_matches
    raw_xg
    teams

Writes:
    matches

Usage:
    uv run python -m clean.clean_matches --season 2025
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv()

logger = logging.getLogger(__name__)

DATE_TOLERANCE_DAYS = 2


TEAM_NAME_ALIASES = {
    "manchester united": "manchester united",
    "manchester united fc": "manchester united",
    "man united": "manchester united",

    "manchester city": "manchester city",
    "manchester city fc": "manchester city",
    "man city": "manchester city",

    "tottenham": "tottenham hotspur",
    "tottenham hotspur": "tottenham hotspur",
    "tottenham hotspur fc": "tottenham hotspur",
    "spurs": "tottenham hotspur",

    "newcastle": "newcastle united",
    "newcastle united": "newcastle united",
    "newcastle united fc": "newcastle united",

    "wolves": "wolverhampton wanderers",
    "wolverhampton": "wolverhampton wanderers",
    "wolverhampton wanderers": "wolverhampton wanderers",
    "wolverhampton wanderers fc": "wolverhampton wanderers",

    "brighton": "brighton and hove albion",
    "brighton and hove albion": "brighton and hove albion",
    "brighton hove albion": "brighton and hove albion",
    "brighton and hove albion fc": "brighton and hove albion",

    "west ham": "west ham united",
    "west ham united": "west ham united",
    "west ham united fc": "west ham united",

    "leeds": "leeds united",
    "leeds united": "leeds united",
    "leeds united fc": "leeds united",

    "leicester": "leicester city",
    "leicester city": "leicester city",
    "leicester city fc": "leicester city",

    "nottingham forest": "nottingham forest",
    "nottingham forest fc": "nottingham forest",
    "nottm forest": "nottingham forest",

    "bournemouth": "bournemouth",
    "afc bournemouth": "bournemouth",
    "bournemouth afc": "bournemouth",

    "crystal palace": "crystal palace",
    "crystal palace fc": "crystal palace",

    "aston villa": "aston villa",
    "aston villa fc": "aston villa",

    "everton": "everton",
    "everton fc": "everton",

    "arsenal": "arsenal",
    "arsenal fc": "arsenal",

    "chelsea": "chelsea",
    "chelsea fc": "chelsea",

    "liverpool": "liverpool",
    "liverpool fc": "liverpool",

    "brentford": "brentford",
    "brentford fc": "brentford",

    "burnley": "burnley",
    "burnley fc": "burnley",

    "sunderland": "sunderland",
    "sunderland afc": "sunderland",

    "fulham": "fulham",
    "fulham fc": "fulham",
    
    "ipswich": "ipswich town",
    "ipswich town": "ipswich town",
    "ipswich town fc": "ipswich town",
    
    "coventry": "coventry city",
    "coventry city": "coventry city",
    "coventry city fc": "coventry city",

    "hull": "hull city",
    "hull city": "hull city",
    "hull city afc": "hull city",
    "hull city fc": "hull city",
}


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
        )

    return supabase_url, supabase_key


def normalize_team_name(name: str | None) -> str:
    """
    Convert provider-specific team names into one canonical comparison key.
    """
    if not name:
        return ""

    normalized = str(name).lower().strip()
    normalized = normalized.replace("&", " and ")

    normalized = re.sub(
        r"[\.\,\-\_\'\"]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    if normalized in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[normalized]

    normalized = re.sub(
        r"\s+(fc|afc)$",
        "",
        normalized,
    ).strip()

    return TEAM_NAME_ALIASES.get(
        normalized,
        normalized,
    )


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(
            timezone.utc
        )

    except (TypeError, ValueError):
        return None


class MatchCleaner:
    def __init__(self) -> None:
        supabase_url, supabase_key = get_settings()

        self.supabase: Client = create_client(
            supabase_url,
            supabase_key,
        )

        self.team_index: dict[str, str] = {}


    def fetch_raw_matches(
        self,
        season: int,
    ) -> list[dict[str, Any]]:
        result = (
            self.supabase
            .table("raw_matches")
            .select("*")
            .eq("season_start_year", season)
            .eq("competition_code", "PL")
            .order("utc_date")
            .execute()
        )

        rows = result.data or []

        logger.info(
            "Fetched %s raw football-data.org matches.",
            len(rows),
        )

        return rows


    def fetch_raw_xg(
        self,
        season: int,
    ) -> list[dict[str, Any]]:
        result = (
            self.supabase
            .table("raw_xg")
            .select("*")
            .eq("season_start_year", season)
            .eq("competition_code", "PL")
            .order("match_date")
            .execute()
        )

        rows = result.data or []

        logger.info(
            "Fetched %s raw Understat xG records.",
            len(rows),
        )

        return rows


    def load_teams(self) -> None:
        """
        Load the teams table and automatically detect its name column.

        This avoids hardcoding a guessed teams-table schema.
        """
        result = (
            self.supabase
            .table("teams")
            .select("*")
            .execute()
        )

        rows = result.data or []

        if not rows:
            raise RuntimeError(
                "The teams table contains no rows. "
                "Cannot create matches because home_team_id and "
                "away_team_id are required."
            )

        possible_name_columns = (
            "canonical_name",
            "name",
            "team_name",
            "short_name",
        )

        detected_name_column: str | None = None

        for column in possible_name_columns:
            if column in rows[0]:
                detected_name_column = column
                break

        if detected_name_column is None:
            raise RuntimeError(
                "Could not determine the team-name column in the "
                "teams table. Available columns: "
                + ", ".join(rows[0].keys())
            )

        team_index: dict[str, str] = {}

        for team in rows:
            team_id = team.get("id")
            team_name = team.get(
                detected_name_column
            )

            if not team_id or not team_name:
                continue

            normalized_name = normalize_team_name(
                str(team_name)
            )

            team_index[normalized_name] = str(
                team_id
            )

        self.team_index = team_index

        logger.info(
            "Loaded %s teams using '%s' as the name column.",
            len(self.team_index),
            detected_name_column,
        )


    def resolve_team_id(
        self,
        team_name: str | None,
    ) -> str | None:
        normalized_name = normalize_team_name(
            team_name
        )

        return self.team_index.get(
            normalized_name
        )


    def build_xg_index(
        self,
        xg_matches: list[dict[str, Any]],
    ) -> dict[
        tuple[str, str],
        list[dict[str, Any]],
    ]:
        index: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = {}

        for match in xg_matches:
            home_team = normalize_team_name(
                match.get("home_team_name")
            )

            away_team = normalize_team_name(
                match.get("away_team_name")
            )

            if not home_team or not away_team:
                continue

            key = (
                home_team,
                away_team,
            )

            index.setdefault(
                key,
                [],
            ).append(match)

        return index


    def find_matching_xg(
        self,
        raw_match: dict[str, Any],
        xg_index: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ],
    ) -> dict[str, Any] | None:
        home_team = normalize_team_name(
            raw_match.get("home_team_name")
        )

        away_team = normalize_team_name(
            raw_match.get("away_team_name")
        )

        candidates = xg_index.get(
            (home_team, away_team),
            [],
        )

        if not candidates:
            return None

        raw_date = parse_datetime(
            raw_match.get("utc_date")
        )

        if raw_date is None:
            return None

        best_match: dict[str, Any] | None = None
        smallest_difference: float | None = None

        for candidate in candidates:
            candidate_date = parse_datetime(
                candidate.get("match_date")
            )

            if candidate_date is None:
                continue

            difference_seconds = abs(
                (
                    raw_date - candidate_date
                ).total_seconds()
            )

            difference_days = (
                difference_seconds / 86400
            )

            if difference_days > DATE_TOLERANCE_DAYS:
                continue

            if (
                smallest_difference is None
                or difference_seconds
                < smallest_difference
            ):
                smallest_difference = (
                    difference_seconds
                )

                best_match = candidate

        return best_match


    @staticmethod
    def to_int(
        value: Any,
    ) -> int | None:
        if value is None or value == "":
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None


    def build_clean_row(
        self,
        raw_match: dict[str, Any],
        xg_match: dict[str, Any] | None,
        home_team_id: str,
        away_team_id: str,
    ) -> dict[str, Any]:
        """
        Build a row using the EXACT columns visible in your matches table.
        """
        status = raw_match.get("status")

        is_postponed = (
            str(status).upper() == "POSTPONED"
        )

        source_match_id = self.to_int(
            raw_match.get("source_match_id")
        )

        return {
            "raw_match_id": raw_match["id"],

            "raw_xg_id": (
                xg_match.get("id")
                if xg_match
                else None
            ),

            "source_match_id": source_match_id,

            "competition_code": raw_match.get(
                "competition_code",
                "PL",
            ),

            "season_start_year": raw_match.get(
                "season_start_year"
            ),

            "matchday": raw_match.get(
                "matchday"
            ),

            "kickoff_at": raw_match.get(
                "utc_date"
            ),

            "status": status,

            "home_team_id": home_team_id,

            "away_team_id": away_team_id,

            "home_goals": raw_match.get(
                "home_score"
            ),

            "away_goals": raw_match.get(
                "away_score"
            ),

            "home_xg": (
                xg_match.get("home_xg")
                if xg_match
                else None
            ),

            "away_xg": (
                xg_match.get("away_xg")
                if xg_match
                else None
            ),

            "referee": raw_match.get(
                "referee_name"
            ),

            "venue": raw_match.get(
                "venue"
            ),

            "neutral_venue": bool(
                raw_match.get(
                    "neutral_venue",
                    False,
                )
            ),

            "is_postponed": is_postponed,
        }


    def upsert_clean_match(
        self,
        row: dict[str, Any],
    ) -> None:
        (
            self.supabase
            .table("matches")
            .upsert(
                row,
                on_conflict="raw_match_id",
            )
            .execute()
        )


    def clean_season(
        self,
        season: int,
    ) -> dict[str, int]:
        raw_matches = self.fetch_raw_matches(
            season
        )

        raw_xg = self.fetch_raw_xg(
            season
        )

        self.load_teams()

        xg_index = self.build_xg_index(
            raw_xg
        )

        total = 0
        matched_xg = 0
        unmatched_xg = 0
        unresolved_teams = 0
        successful = 0
        failed = 0

        for raw_match in raw_matches:
            total += 1

            try:
                home_team_name = raw_match.get(
                    "home_team_name"
                )

                away_team_name = raw_match.get(
                    "away_team_name"
                )

                home_team_id = self.resolve_team_id(
                    home_team_name
                )

                away_team_id = self.resolve_team_id(
                    away_team_name
                )

                if (
                    home_team_id is None
                    or away_team_id is None
                ):
                    unresolved_teams += 1

                    logger.error(
                        "Could not resolve team IDs for: %s vs %s",
                        home_team_name,
                        away_team_name,
                    )

                    continue

                xg_match = self.find_matching_xg(
                    raw_match,
                    xg_index,
                )

                if xg_match:
                    matched_xg += 1
                else:
                    unmatched_xg += 1

                    logger.warning(
                        "No Understat match found for: %s vs %s on %s",
                        home_team_name,
                        away_team_name,
                        raw_match.get("utc_date"),
                    )

                clean_row = self.build_clean_row(
                    raw_match,
                    xg_match,
                    home_team_id,
                    away_team_id,
                )

                self.upsert_clean_match(
                    clean_row
                )

                successful += 1

            except Exception:
                failed += 1

                logger.exception(
                    "Failed to clean match %s",
                    raw_match.get(
                        "source_match_id"
                    ),
                )

        return {
            "total": total,
            "successful": successful,
            "matched_xg": matched_xg,
            "unmatched_xg": unmatched_xg,
            "unresolved_teams": unresolved_teams,
            "failed": failed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean and reconcile Premier League match data."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help=(
            "Season starting year, for example 2025 "
            "for the 2025/26 season."
        ),
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    logger.info(
        "Starting cleaning pipeline for season %s/%s",
        args.season,
        str(args.season + 1)[-2:],
    )

    cleaner = MatchCleaner()

    stats = cleaner.clean_season(
        args.season
    )

    logger.info("Cleaning complete.")
    logger.info(
        "Total matches processed: %s",
        stats["total"],
    )
    logger.info(
        "Successfully written: %s",
        stats["successful"],
    )
    logger.info(
        "Matched with Understat xG: %s",
        stats["matched_xg"],
    )
    logger.info(
        "Without Understat xG: %s",
        stats["unmatched_xg"],
    )
    logger.info(
        "Unresolved teams: %s",
        stats["unresolved_teams"],
    )
    logger.info(
        "Failed records: %s",
        stats["failed"],
    )


if __name__ == "__main__":
    main()