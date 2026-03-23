"""
pipeline.py
-----------
Takes raw DataFrames from scraper.py and:

  1. Cleans & normalises all stats to per-90 minutes
  2. Builds a unified player_stats table
  3. Builds the historical_transfers table (Eastern EU → Superliga, 2015-2025)
     with pre-transfer stats AND post-transfer Superliga performance
  4. Writes everything to SQLite at data/processed/scouting.db

Schema
------
  player_stats        — one row per (player, league, season, stat_type_group)
  player_profiles     — one row per (player, league, season) with all stats merged
  historical_transfers — training data for the prediction model
  market_values       — Transfermarkt valuations
  transfers_raw       — raw Transfermarkt transfer events
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "scouting.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Eastern European leagues (source leagues for scouting)
EASTERN_EU_KEYS = [
    "ekstraklasa", "czech", "croatia", "romania", "bulgaria", "serbia"
]
TARGET_LEAGUE = "superliga"

# Columns that should be converted to per-90 minutes
# Any numeric col not in EXCLUDED_FROM_PER90 will be normalised if it
# represents a counting stat.
EXCLUDED_FROM_PER90 = {
    "age", "born", "min", "90s", "mp", "starts",
    "league_key", "season", "player", "team", "nation", "pos",
    "league_key", "matches",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_multiindex_cols(df: pd.DataFrame) -> pd.DataFrame:
    """soccerdata sometimes returns MultiIndex columns; flatten them."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col if c).strip("_")
                      for col in df.columns]
    return df


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _per_90(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise all counting stats to per-90-minute values.
    Requires a '90s' column (minutes played / 90) or 'min'.
    Skips players with fewer than 270 minutes (3 × 90) to avoid noise.
    """
    df = df.copy()

    # Build the minutes-played proxy column
    if "90s" not in df.columns:
        if "min" in df.columns:
            df["90s"] = pd.to_numeric(df["min"], errors="coerce") / 90
        else:
            logger.warning("No '90s' or 'min' column — skipping per-90 normalisation")
            return df

    df["90s"] = pd.to_numeric(df["90s"], errors="coerce")
    df = df[df["90s"] >= 3.0].copy()   # at least 270 mins

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cols_to_norm = [
        c for c in numeric_cols
        if c not in EXCLUDED_FROM_PER90 and c != "90s"
    ]

    for col in cols_to_norm:
        df[col] = df[col] / df["90s"]

    return df


def _standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, replace spaces/dashes with underscores."""
    df.columns = [
        re.sub(r"[^a-z0-9_]", "_", c.lower().strip())
        for c in df.columns
    ]
    return df


def _add_percentile_ranks(
    df: pd.DataFrame,
    group_cols: list[str] = ["league_key", "season", "pos"],
) -> pd.DataFrame:
    """
    Add percentile rank columns (suffix _pct) for every numeric stat,
    grouped by league + season + broad position.
    """
    df = df.copy()
    if not all(c in df.columns for c in group_cols):
        logger.warning(f"Skipping percentile ranks — missing grouping columns")
        return df

    # Broad position grouping
    if "pos" in df.columns:
        df["pos_broad"] = df["pos"].str.split(",").str[0].str.strip()
    else:
        df["pos_broad"] = "Outfield"

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    pct_cols = [c for c in numeric_cols if not c.endswith("_pct")]

    def _rank_group(grp: pd.DataFrame) -> pd.DataFrame:
        for col in pct_cols:
            grp[f"{col}_pct"] = grp[col].rank(pct=True) * 100
        return grp

    df = df.groupby(["league_key", "season", "pos_broad"], group_keys=False).apply(
        _rank_group
    )
    return df


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_stat_frames(
    raw_frames: dict[str, dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    raw_frames: dict[league_key][stat_type] → raw DataFrame

    Returns a single merged DataFrame with one row per
    (player, team, league_key, season) containing all stat types merged.
    """
    per_league_dfs: list[pd.DataFrame] = []

    for league_key, stat_dict in raw_frames.items():
        logger.info(f"Processing league: {league_key}")

        merged: Optional[pd.DataFrame] = None
        join_keys = ["player", "team", "season", "league_key"]

        for stat_type, df in stat_dict.items():
            if df is None or df.empty:
                logger.warning(f"  [{stat_type}] empty — skipping")
                continue

            df = _flatten_multiindex_cols(df)
            df = _standardise_column_names(df)
            df["league_key"] = league_key

            # Ensure join keys exist
            if "player" not in df.columns or "season" not in df.columns:
                logger.warning(f"  [{stat_type}] missing player/season cols — skipping")
                continue

            # Normalise to per-90
            df = _per_90(df)

            # Suffix non-key columns with the stat type to avoid collisions
            non_key = [c for c in df.columns if c not in join_keys]
            df = df.rename(columns={c: f"{stat_type}_{c}" for c in non_key})

            if merged is None:
                merged = df
            else:
                # outer join so we keep all players even if one stat is missing
                overlap = [c for c in join_keys if c in merged.columns and c in df.columns]
                merged = merged.merge(df, on=overlap, how="outer", suffixes=("", f"_{stat_type}"))

        if merged is not None and not merged.empty:
            per_league_dfs.append(merged)
            logger.info(f"  Merged shape: {merged.shape}")

    if not per_league_dfs:
        logger.error("No data to process!")
        return pd.DataFrame()

    result = pd.concat(per_league_dfs, ignore_index=True)
    logger.info(f"Total combined shape: {result.shape}")
    return result


def build_historical_transfer_table(
    player_profiles: pd.DataFrame,
    transfers_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the training dataset:
      - Find all transfers FROM Eastern EU leagues TO the Superliga
      - Attach the player's stats from the SEASON BEFORE transfer
      - Attach the player's stats in their FIRST FULL Superliga season

    Returns a DataFrame with pre-transfer features and post-transfer target.
    """
    logger.info("Building historical transfer table …")

    if transfers_raw.empty or player_profiles.empty:
        logger.warning("Insufficient data for historical transfer table")
        return pd.DataFrame()

    # Standardise transfers table
    tr = _standardise_column_names(transfers_raw.copy())
    pp = player_profiles.copy()

    # Filter transfers: Eastern EU → Superliga
    eastern_eu_set = set(EASTERN_EU_KEYS)

    # soccerdata Transfermarkt uses 'league_key' we added, plus 'transfer_to'
    # columns may vary; adapt to whatever columns exist
    from_col = next((c for c in tr.columns if "from" in c and "league" in c), None)
    to_col   = next((c for c in tr.columns if "to"   in c and "league" in c), None)

    if from_col is None or to_col is None:
        # Fall back: try to infer from the league_key column
        logger.warning("Could not find explicit from/to league columns in transfers; "
                       "will rely on league_key matching.")
        return pd.DataFrame()

    mask = (
        tr[from_col].isin(eastern_eu_set) &
        tr[to_col].str.lower().str.contains("superliga", na=False)
    )
    ee_to_sl = tr[mask].copy()
    logger.info(f"  Found {len(ee_to_sl)} Eastern EU → Superliga transfers")

    if ee_to_sl.empty:
        return pd.DataFrame()

    # For each transfer, get pre-transfer stats and post-transfer stats
    rows = []
    for _, t in ee_to_sl.iterrows():
        player  = t.get("player", t.get("player_name", None))
        season  = t.get("season", None)
        if player is None or season is None:
            continue

        # Season before transfer
        pre_season = str(int(str(season)[:4]) - 1) if season else None

        pre_stats = pp[
            (pp["player"] == player) &
            (pp["season"] == pre_season) &
            (pp["league_key"].isin(eastern_eu_set))
        ]
        post_stats = pp[
            (pp["player"] == player) &
            (pp["season"] == season) &
            (pp["league_key"] == TARGET_LEAGUE)
        ]

        if pre_stats.empty or post_stats.empty:
            continue

        row = pre_stats.iloc[0].to_dict()
        row["transfer_season"] = season
        # Target: some composite performance score in Superliga
        # For now: average of xg_pct and xa_pct if available
        post = post_stats.iloc[0]
        xg_col = next((c for c in post.index if "xg" in c and "_pct" in c), None)
        xa_col = next((c for c in post.index if "xa" in c and "_pct" in c), None)
        row["target_performance_pct"] = np.nanmean([
            post[xg_col] if xg_col else np.nan,
            post[xa_col] if xa_col else np.nan,
        ])
        rows.append(row)

    hist_df = pd.DataFrame(rows)
    logger.info(f"  Historical transfer table: {hist_df.shape}")
    return hist_df


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------

def write_to_sqlite(
    frames: dict[str, pd.DataFrame],
    db_path: Path = DB_PATH,
    if_exists: str = "replace",
) -> None:
    """Write a dict of table_name → DataFrame into SQLite."""
    logger.info(f"Writing {len(frames)} tables to {db_path}")
    with sqlite3.connect(db_path) as conn:
        for table_name, df in frames.items():
            if df is None or df.empty:
                logger.warning(f"  Skipping empty table: {table_name}")
                continue
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            logger.success(f"  {table_name}: {len(df)} rows written")


def load_table(table_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load a table from SQLite."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

def run_full_pipeline(
    raw_frames: dict[str, dict[str, pd.DataFrame]],
    tm_players: dict[str, pd.DataFrame],
    tm_transfers: dict[str, pd.DataFrame],
) -> None:
    """
    Orchestrate the full processing pipeline.

    Parameters
    ----------
    raw_frames   : dict[league_key][stat_type] from scraper.scrape_fbref_*
    tm_players   : dict[league_key] → player market values DataFrame
    tm_transfers : dict[league_key] → transfers DataFrame
    """

    # 1. Merge all stat frames into player profiles
    logger.info("Step 1: Merging stat frames …")
    player_profiles = process_stat_frames(raw_frames)

    # 2. Add percentile ranks
    logger.info("Step 2: Adding percentile ranks …")
    player_profiles = _add_percentile_ranks(player_profiles)

    # 3. Combine Transfermarkt data
    logger.info("Step 3: Combining Transfermarkt data …")
    all_players   = pd.concat(list(tm_players.values()),   ignore_index=True) if tm_players   else pd.DataFrame()
    all_transfers = pd.concat(list(tm_transfers.values()), ignore_index=True) if tm_transfers else pd.DataFrame()

    if not all_players.empty:
        all_players = _standardise_column_names(all_players)
    if not all_transfers.empty:
        all_transfers = _standardise_column_names(all_transfers)

    # 4. Build historical transfer table (training data)
    logger.info("Step 4: Building historical transfer table …")
    hist_transfers = build_historical_transfer_table(player_profiles, all_transfers)

    # 5. Write to SQLite
    logger.info("Step 5: Writing to SQLite …")
    tables = {
        "player_profiles":      player_profiles,
        "market_values":        all_players,
        "transfers_raw":        all_transfers,
        "historical_transfers": hist_transfers,
    }
    write_to_sqlite(tables)

    # Summary
    logger.info("\n=== Pipeline complete ===")
    for name, df in tables.items():
        rows = len(df) if df is not None and not df.empty else 0
        logger.info(f"  {name:30s}: {rows:,} rows")
