"""
Folketing Open Data API (ODA) client.

Base URL: https://oda.ft.dk/api/
Protocol: OData v4 (JSON responses)
No authentication required.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests
from loguru import logger

BASE_URL = "https://oda.ft.dk/api"

# Folketing member annual salary in DKK (grundvederlag 2024/2025)
FOLKETING_ANNUAL_SALARY_DKK = 662_400

_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})


def _get(endpoint: str, params: dict | None = None) -> dict:
    """Make a GET request to the ODA API and return the JSON response."""
    url = f"{BASE_URL}/{endpoint}"
    default_params = {"$format": "json"}
    if params:
        default_params.update(params)
    try:
        resp = _SESSION.get(url, params=default_params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"ODA API request failed: {url} — {e}")
        return {"value": []}


def _paginate(endpoint: str, params: dict | None = None, max_records: int = 5000) -> list[dict]:
    """Fetch all pages from a paginated ODA endpoint."""
    params = params or {}
    params["$top"] = 100
    params["$skip"] = 0
    records: list[dict] = []

    while len(records) < max_records:
        data = _get(endpoint, params.copy())
        batch = data.get("value", [])
        if not batch:
            break
        records.extend(batch)
        if len(batch) < params["$top"]:
            break
        params["$skip"] += params["$top"]
        time.sleep(0.1)  # polite rate limiting

    return records[:max_records]


@lru_cache(maxsize=1)
def get_periods() -> pd.DataFrame:
    """Return all parliamentary periods (Periode)."""
    records = _paginate("Periode", {"$orderby": "startdato desc"})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in ["startdato", "slutdato"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def get_meetings(periode_id: Optional[int] = None, max_records: int = 2000) -> pd.DataFrame:
    """Return meetings (Møde), optionally filtered by parliamentary period."""
    params: dict = {"$orderby": "dato desc"}
    if periode_id is not None:
        params["$filter"] = f"periode_id eq {periode_id}"
    records = _paginate("M%C3%B8de", params, max_records=max_records)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in ["dato", "starttidspunkt", "sluttidspunkt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def get_votes(periode_id: Optional[int] = None, max_records: int = 5000) -> pd.DataFrame:
    """Return votes (Afstemning), optionally filtered by parliamentary period.

    The ODA API is OData v3. Vote results are in the 'konklusion' field
    (string: "Vedtaget" / "Forkastet") and optionally a boolean 'vedtaget'.
    """
    params: dict = {"$orderby": "id desc"}
    if periode_id is not None:
        params["$filter"] = f"m%C3%B8de/periode_id eq {periode_id}"
        params["$expand"] = "m%C3%B8de"
    records = _paginate("Afstemning", params, max_records=max_records)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Flatten nested møde info if expanded
    if "møde" in df.columns:
        moed_df = df["møde"].apply(pd.Series).add_prefix("møde_")
        df = pd.concat([df.drop(columns=["møde"]), moed_df], axis=1)
    # Normalise result field — API may return 'konklusion' (string) or 'vedtaget' (bool)
    if "konklusion" in df.columns:
        df["vedtaget"] = df["konklusion"].str.strip().str.lower() == "vedtaget"
    elif "vedtaget" in df.columns:
        df["vedtaget"] = df["vedtaget"].astype(bool)
    return df


@lru_cache(maxsize=8)
def get_actors(typeid: int = 5) -> pd.DataFrame:
    """
    Return actors (Aktør).
    typeid=5 corresponds to Folketing members (Medlem af Folketinget).
    """
    params = {
        "$filter": f"typeid eq {typeid}",
        "$orderby": "navn",
    }
    records = _paginate("Akt%C3%B8r", params, max_records=300)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def get_individual_votes(periode_id: Optional[int] = None, max_records: int = 50000) -> pd.DataFrame:
    """Return individual politician votes (Stemme)."""
    params: dict = {}
    if periode_id is not None:
        params["$filter"] = f"afstemning/m%C3%B8de/periode_id eq {periode_id}"
        params["$expand"] = "afstemning($expand=m%C3%B8de)"
    records = _paginate("Stemme", params, max_records=max_records)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df


def get_vote_summary(votes_df: pd.DataFrame) -> dict:
    """Compute summary stats from a votes DataFrame."""
    if votes_df.empty:
        return {"total": 0, "vedtaget": 0, "forkastet": 0}
    total = len(votes_df)
    vedtaget = int(votes_df["vedtaget"].sum()) if "vedtaget" in votes_df.columns else 0
    return {
        "total": total,
        "vedtaget": vedtaget,
        "forkastet": total - vedtaget,
    }


def build_politician_scoreboard(
    individual_votes_df: pd.DataFrame,
    actors_df: pd.DataFrame,
    periode_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a scoreboard of politicians with activity metrics and løn per arbejdsdag.

    Returns a DataFrame with:
    - navn, parti, antal_stemmer, arbejdsdage, løn_per_dag
    """
    if individual_votes_df.empty or actors_df.empty:
        return pd.DataFrame()

    # Count votes per politician
    vote_counts = (
        individual_votes_df.groupby("aktør_id")
        .size()
        .reset_index(name="antal_stemmer")
    )

    # Merge with actor info
    actors_slim = actors_df[["id", "navn", "gruppenavnkort"]].rename(
        columns={"id": "aktør_id", "gruppenavnkort": "parti"}
    )
    scoreboard = vote_counts.merge(actors_slim, on="aktør_id", how="left")

    # Estimate workdays: each vote is on a meeting day; count unique dates
    # If we have møde date info via expand, use it; else estimate from vote count
    if "afstemning_møde_dato" in individual_votes_df.columns:
        date_col = "afstemning_møde_dato"
    elif "møde_dato" in individual_votes_df.columns:
        date_col = "møde_dato"
    else:
        date_col = None

    if date_col:
        work_days = (
            individual_votes_df.dropna(subset=[date_col])
            .groupby("aktør_id")[date_col]
            .apply(lambda x: x.dt.date.nunique() if hasattr(x, "dt") else x.nunique())
            .reset_index(name="arbejdsdage")
        )
        scoreboard = scoreboard.merge(work_days, on="aktør_id", how="left")
    else:
        # Rough estimate: assume ~8 votes per meeting day
        scoreboard["arbejdsdage"] = (scoreboard["antal_stemmer"] / 8).round(0).astype(int)

    scoreboard["arbejdsdage"] = scoreboard["arbejdsdage"].fillna(1).clip(lower=1).astype(int)

    # Calculate salary per workday
    # Assume current period spans roughly 1 year; adjust proportionally
    scoreboard["løn_per_dag"] = (
        FOLKETING_ANNUAL_SALARY_DKK / scoreboard["arbejdsdage"]
    ).round(0).astype(int)

    # Fill missing
    scoreboard["navn"] = scoreboard["navn"].fillna("Ukendt")
    scoreboard["parti"] = scoreboard["parti"].fillna("Ukendt")

    # Sort by votes descending
    scoreboard = scoreboard.sort_values("antal_stemmer", ascending=False).reset_index(drop=True)
    scoreboard.index += 1  # 1-based rank

    return scoreboard[["navn", "parti", "antal_stemmer", "arbejdsdage", "løn_per_dag"]]


def get_meetings_by_month(meetings_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate meeting counts by month."""
    if meetings_df.empty or "dato" not in meetings_df.columns:
        return pd.DataFrame()
    df = meetings_df.dropna(subset=["dato"]).copy()
    df["måned"] = df["dato"].dt.to_period("M")
    return df.groupby("måned").size().reset_index(name="antal_møder")


def get_votes_by_month(votes_df: pd.DataFrame, meetings_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vote counts by month using møde dates."""
    if votes_df.empty:
        return pd.DataFrame()
    if not meetings_df.empty and "møde_id" in votes_df.columns:
        date_map = meetings_df.set_index("id")["dato"].to_dict()
        votes_df = votes_df.copy()
        votes_df["dato"] = votes_df["møde_id"].map(date_map)
        votes_df = votes_df.dropna(subset=["dato"])
        votes_df["måned"] = pd.to_datetime(votes_df["dato"]).dt.to_period("M")
        return votes_df.groupby("måned").size().reset_index(name="antal_afstemninger")
    return pd.DataFrame()
