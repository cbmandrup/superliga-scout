"""
scraper.py
----------
Scrapes player statistics from FBref and transfer data from Transfermarkt
via the soccerdata library.

Covers:
  - Danish Superliga (target league)
  - Polish Ekstraklasa, Czech First League, Croatian HNL,
    Romanian Liga 1, Bulgarian First League, Serbian SuperLiga

Stores raw CSVs in data/raw/ and loads into SQLite via pipeline.py.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import soccerdata as sd
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# League map
# FBref league IDs used by soccerdata's FBref wrapper.
# Key  = short name used inside this project
# Value = league name as expected by soccerdata / FBref
# ---------------------------------------------------------------------------
LEAGUES: dict[str, str] = {
    "superliga":  "Danish Superliga",
    "ekstraklasa": "Polish Ekstraklasa",
    "czech":       "Czech First League",
    "croatia":     "Croatian Football League",
    "romania":     "Romanian Liga I",
    "bulgaria":    "Bulgarian First Professional Football League",
    "serbia":      "Serbian SuperLiga",
}

# Stat categories available on FBref player pages
STAT_TYPES: list[str] = [
    "standard",
    "shooting",
    "passing",
    "passing_types",
    "goal_shot_creation",
    "defense",
    "possession",
    "playing_time",
    "misc",
]

SEASONS: list[str] = [str(y) for y in range(2018, 2026)]  # 2018-2025

RATE_LIMIT_SLEEP: float = 3.0   # seconds between requests (FBref ToS)


# ---------------------------------------------------------------------------
# FBref scraper
# ---------------------------------------------------------------------------

def _cache_path(league_key: str, season: str, stat: str) -> Path:
    return RAW_DIR / f"fbref_{league_key}_{season}_{stat}.csv"


def scrape_fbref_league(
    league_key: str,
    seasons: Optional[list[str]] = None,
    stat_types: Optional[list[str]] = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Scrape all requested stat tables for one league across multiple seasons.

    Returns a dict keyed by stat_type containing concatenated DataFrames
    (all seasons stacked).
    """
    league_name = LEAGUES[league_key]
    seasons = seasons or SEASONS
    stat_types = stat_types or STAT_TYPES

    logger.info(f"Starting FBref scrape: {league_key} ({league_name})")

    all_frames: dict[str, list[pd.DataFrame]] = {s: [] for s in stat_types}

    fbref = sd.FBref(leagues=league_name, seasons=seasons)

    for stat in stat_types:
        for season in seasons:
            cache_file = _cache_path(league_key, season, stat)

            if cache_file.exists() and not force:
                logger.debug(f"  Cache hit: {cache_file.name}")
                df = pd.read_csv(cache_file, low_memory=False)
                all_frames[stat].append(df)
                continue

            logger.info(f"  Fetching {stat} | {league_key} {season} …")
            try:
                # soccerdata returns a DataFrame with MultiIndex columns
                raw = getattr(fbref, f"read_player_{stat}")(
                    # some stat methods have slightly different names
                )
                # Filter to this season if the library returns multiple
                if "season" in raw.columns:
                    df = raw[raw["season"] == season].copy()
                else:
                    df = raw.copy()

                df["league_key"] = league_key
                df["season"] = season

                df.to_csv(cache_file, index=False)
                logger.success(f"  Saved {len(df)} rows → {cache_file.name}")
                all_frames[stat].append(df)

            except Exception as exc:
                logger.warning(f"  SKIP {stat}/{season}: {exc}")

            time.sleep(RATE_LIMIT_SLEEP)

    # Concatenate each stat type across seasons
    combined: dict[str, pd.DataFrame] = {}
    for stat, frames in all_frames.items():
        if frames:
            combined[stat] = pd.concat(frames, ignore_index=True)
            logger.info(
                f"  [{stat}] total rows after concat: {len(combined[stat])}"
            )

    return combined


def scrape_fbref_all_leagues(
    league_keys: Optional[list[str]] = None,
    seasons: Optional[list[str]] = None,
    force: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Scrape all leagues. Returns nested dict[league_key][stat_type]."""
    league_keys = league_keys or list(LEAGUES.keys())
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for lk in league_keys:
        results[lk] = scrape_fbref_league(lk, seasons=seasons, force=force)
    return results


# ---------------------------------------------------------------------------
# Transfermarkt scraper (via soccerdata)
# ---------------------------------------------------------------------------

def _tm_cache_path(league_key: str, data_type: str) -> Path:
    return RAW_DIR / f"tm_{league_key}_{data_type}.csv"


def scrape_transfermarkt_league(
    league_key: str,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Pull player valuations and team rosters from Transfermarkt.
    Returns dict with keys 'players' and 'transfers'.
    """
    league_name = LEAGUES[league_key]
    logger.info(f"Transfermarkt scrape: {league_key} ({league_name})")

    result: dict[str, pd.DataFrame] = {}

    # --- Player market values ---
    cache_players = _tm_cache_path(league_key, "players")
    if cache_players.exists() and not force:
        logger.debug(f"  Cache hit: {cache_players.name}")
        result["players"] = pd.read_csv(cache_players, low_memory=False)
    else:
        try:
            tm = sd.Transfermarkt(leagues=league_name, seasons=SEASONS)
            df_players = tm.read_player_market_values()
            df_players["league_key"] = league_key
            df_players.to_csv(cache_players, index=False)
            result["players"] = df_players
            logger.success(
                f"  players: {len(df_players)} rows → {cache_players.name}"
            )
            time.sleep(RATE_LIMIT_SLEEP)
        except Exception as exc:
            logger.warning(f"  SKIP players/{league_key}: {exc}")
            result["players"] = pd.DataFrame()

    # --- Transfer history ---
    cache_transfers = _tm_cache_path(league_key, "transfers")
    if cache_transfers.exists() and not force:
        logger.debug(f"  Cache hit: {cache_transfers.name}")
        result["transfers"] = pd.read_csv(cache_transfers, low_memory=False)
    else:
        try:
            tm = sd.Transfermarkt(leagues=league_name, seasons=SEASONS)
            df_transfers = tm.read_transfers()
            df_transfers["league_key"] = league_key
            df_transfers.to_csv(cache_transfers, index=False)
            result["transfers"] = df_transfers
            logger.success(
                f"  transfers: {len(df_transfers)} rows → {cache_transfers.name}"
            )
            time.sleep(RATE_LIMIT_SLEEP)
        except Exception as exc:
            logger.warning(f"  SKIP transfers/{league_key}: {exc}")
            result["transfers"] = pd.DataFrame()

    return result


# ---------------------------------------------------------------------------
# StatsBomb open data (GitHub)
# ---------------------------------------------------------------------------

STATSBOMB_BASE = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
)


def scrape_statsbomb_competitions() -> pd.DataFrame:
    """Download the StatsBomb open-data competition index."""
    import requests

    cache = RAW_DIR / "statsbomb_competitions.csv"
    if cache.exists():
        return pd.read_csv(cache)

    url = f"{STATSBOMB_BASE}/competitions.json"
    logger.info(f"Fetching StatsBomb competitions list …")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df.to_csv(cache, index=False)
    logger.success(f"StatsBomb competitions: {len(df)} rows")
    return df


def scrape_statsbomb_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Download match list for a StatsBomb competition/season pair."""
    import requests

    cache = RAW_DIR / f"statsbomb_matches_{competition_id}_{season_id}.csv"
    if cache.exists():
        return pd.read_csv(cache)

    url = f"{STATSBOMB_BASE}/matches/{competition_id}/{season_id}.json"
    logger.info(f"Fetching StatsBomb matches {competition_id}/{season_id} …")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.json_normalize(resp.json())
    df.to_csv(cache, index=False)
    logger.success(f"StatsBomb matches: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Quick test entry-point: one league, two seasons
# ---------------------------------------------------------------------------

def run_test_pull() -> None:
    """
    Minimal test: scrape Polish Ekstraklasa for two recent seasons
    and print a summary.  Run with:  python -m src.scraper
    """
    logger.info("=" * 60)
    logger.info("TEST PULL: Polish Ekstraklasa (2023, 2024)")
    logger.info("=" * 60)

    test_seasons = ["2023", "2024"]
    test_stats   = ["standard", "shooting", "passing"]

    data = scrape_fbref_league(
        "ekstraklasa",
        seasons=test_seasons,
        stat_types=test_stats,
    )

    for stat, df in data.items():
        if df.empty:
            logger.warning(f"  [{stat}] — empty DataFrame")
            continue

        logger.info(f"\n--- {stat} ---")
        logger.info(f"  Shape  : {df.shape}")
        logger.info(f"  Columns: {list(df.columns[:10])} …")
        logger.info(f"  Seasons: {sorted(df['season'].unique()) if 'season' in df.columns else 'N/A'}")

        # Print first 5 rows (player name + key cols)
        display_cols = [
            c for c in ["player", "team", "season", "league_key", "pos",
                         "age", "min", "gls", "ast", "xg", "xa"]
            if c in df.columns
        ]
        if display_cols:
            print(df[display_cols].head().to_string(index=False))

    # Also pull Transfermarkt data
    logger.info("\n--- Transfermarkt test ---")
    tm_data = scrape_transfermarkt_league("ekstraklasa")
    for key, df in tm_data.items():
        logger.info(f"  [{key}] shape: {df.shape}")
        if not df.empty:
            print(df.head(3).to_string(index=False))

    # StatsBomb competitions index
    logger.info("\n--- StatsBomb competitions ---")
    comps = scrape_statsbomb_competitions()
    logger.info(f"  {len(comps)} competitions available")
    print(comps[["competition_id", "competition_name", "country_name"]].head(10).to_string(index=False))

    logger.info("\nTest pull complete. Raw files saved to data/raw/")


if __name__ == "__main__":
    run_test_pull()
