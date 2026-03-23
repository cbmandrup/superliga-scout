"""
features.py
-----------
Feature engineering for the scouting model.

Implements:
  1. Per-90 percentile metrics (computed in pipeline.py, consumed here)
  2. League adjustment coefficients (Eastern EU → Superliga)
  3. Adaptability score (performance variance across contexts)
     Based on Ribeiro et al. 2025 & Bonetti et al. 2025 (PNAS)
  4. Style fingerprint via K-means clustering
     Based on Frontiers 2025 cross-league role-stability study
  5. Similarity engine (cosine similarity, 3 nearest neighbours)
  6. Simplified TacticAI-style passing-network feature
     (inspired by DeepMind TacticAI 2024)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Key metrics used for feature construction
# ---------------------------------------------------------------------------

CORE_METRICS = [
    # Attacking / creation
    "standard_xg", "standard_xa", "standard_gls", "standard_ast",
    "goal_shot_creation_sca90",
    # Passing
    "passing_prog_p", "passing_xa",
    # Carrying / possession
    "possession_prg_c", "possession_touches",
    # Defensive
    "defense_tkl", "defense_press", "defense_int",
    # Playing time proxy
    "standard_90s",
]

# Metrics used for style clustering
STYLE_METRICS = [
    "standard_xg", "standard_xa",
    "goal_shot_creation_sca90",
    "passing_prog_p",
    "possession_prg_c",
    "defense_press", "defense_tkl",
]

N_STYLE_CLUSTERS = 6  # empirically chosen; tunable

STYLE_LABELS = {
    0: "Pressing Forward",
    1: "Wide Creator",
    2: "Deep-Lying Playmaker",
    3: "Box-to-Box Midfielder",
    4: "Defensive Anchor",
    5: "Advanced Playmaker",
}

# Historical success rate per cluster (Eastern EU → Superliga)
# Will be calibrated from historical_transfers; defaults shown here
CLUSTER_SUCCESS_DEFAULTS = {
    0: 0.62,  # Pressing Forward — decent adaptation
    1: 0.55,  # Wide Creator — perimeter role; less stable (Frontiers 2025)
    2: 0.71,  # Deep-Lying Playmaker — interior role; most stable
    3: 0.65,  # Box-to-Box
    4: 0.68,  # Defensive Anchor — interior/defensive; stable (Frontiers 2025)
    5: 0.58,  # Advanced Playmaker — high ceiling, high variance
}


# ---------------------------------------------------------------------------
# 1. League adjustment coefficients
# ---------------------------------------------------------------------------

def compute_league_coefficients(
    hist_transfers: pd.DataFrame,
    player_profiles: pd.DataFrame,
) -> dict[str, float]:
    """
    Calibrate a league-level adjustment factor for each Eastern EU league.

    Method (inspired by Malikov & Kim 2024):
      For each player who moved from league L → Superliga:
        ratio = superliga_performance_pct / eastern_eu_performance_pct (pre-transfer)
      league_coeff[L] = median(ratio) across all such players

    Coefficients < 1 mean the league is weaker than Superliga.
    """
    logger.info("Computing league coefficients …")

    if hist_transfers.empty:
        logger.warning("No historical transfer data — using default coefficients")
        return _default_league_coefficients()

    coefficients: dict[str, float] = {}

    for league_key in hist_transfers["league_key"].dropna().unique():
        subset = hist_transfers[hist_transfers["league_key"] == league_key].copy()

        # We need a pre-transfer performance score and the target
        pre_cols = [c for c in subset.columns
                    if "_pct" in c and "target" not in c and "90s" not in c]
        if not pre_cols or "target_performance_pct" not in subset.columns:
            coefficients[league_key] = 0.75  # fallback
            continue

        subset["pre_performance"] = subset[pre_cols].mean(axis=1, skipna=True)
        valid = subset.dropna(subset=["pre_performance", "target_performance_pct"])

        if len(valid) < 3:
            coefficients[league_key] = 0.75
            continue

        ratios = valid["target_performance_pct"] / valid["pre_performance"].replace(0, np.nan)
        coefficients[league_key] = float(ratios.median())

    logger.info("League coefficients:")
    for k, v in sorted(coefficients.items(), key=lambda x: -x[1]):
        logger.info(f"  {k:15s}: {v:.3f}")

    return coefficients


def _default_league_coefficients() -> dict[str, float]:
    """
    Fallback coefficients based on general football analyst consensus
    and UEFA coefficient rankings for these leagues.
    """
    return {
        "ekstraklasa": 0.82,   # Poland — closest in quality
        "czech":       0.80,
        "croatia":     0.78,
        "romania":     0.74,
        "serbia":      0.72,
        "bulgaria":    0.70,
        "superliga":   1.00,
    }


def apply_league_adjustment(
    df: pd.DataFrame,
    coefficients: dict[str, float],
) -> pd.DataFrame:
    """
    Add league-adjusted versions of key metrics to the DataFrame.
    Adjusted metric = raw_metric / league_coeff  (projects to Superliga level).
    """
    df = df.copy()
    if "league_key" not in df.columns:
        return df

    df["league_coeff"] = df["league_key"].map(coefficients).fillna(0.75)

    for metric in CORE_METRICS:
        if metric in df.columns:
            df[f"{metric}_adj"] = df[metric] / df["league_coeff"]

    return df


# ---------------------------------------------------------------------------
# 2. Adaptability score
# Based on Ribeiro et al. 2025 scoping review + Bonetti et al. 2025 PNAS
# ---------------------------------------------------------------------------

def compute_adaptability_score(
    player_profiles: pd.DataFrame,
    context_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Adaptability = inverse of performance variance across match contexts.

    FBref does not expose context-level splits directly, so we proxy via:
      - Home vs. away performance (if available)
      - Variance of monthly/period performance across seasons

    Here we use cross-season variance of core metrics as a proxy:
      players who perform consistently season-to-season = high adaptability

    Score: 0-10 (10 = most adaptable)
    Based on: Ribeiro et al. 2025 (contextual constraints),
              Bonetti et al. 2025 (cognitive flexibility → consistency)
    """
    logger.info("Computing adaptability scores …")

    df = player_profiles.copy()
    metric_cols = [c for c in CORE_METRICS if c in df.columns]

    if not metric_cols or "player" not in df.columns:
        df["adaptability_score"] = 5.0  # neutral default
        return df

    # Compute per-player cross-season CV (coefficient of variation) for each metric
    def _cv(series: pd.Series) -> float:
        mean = series.mean()
        std  = series.std()
        if mean == 0 or pd.isna(mean):
            return 1.0  # high uncertainty
        return std / abs(mean)

    player_cv = (
        df.groupby("player")[metric_cols]
        .agg(_cv)
        .mean(axis=1)       # average CV across metrics
        .rename("mean_cv")
    )

    # Normalise CV to 0-10 adaptability score (lower CV = higher score)
    cv_min = player_cv.quantile(0.05)
    cv_max = player_cv.quantile(0.95)
    player_cv_norm = (player_cv - cv_min) / (cv_max - cv_min + 1e-9)
    adaptability = (1 - player_cv_norm.clip(0, 1)) * 10

    df = df.merge(
        adaptability.reset_index().rename(columns={"mean_cv": "adaptability_score"}),
        on="player",
        how="left",
    )
    df["adaptability_score"] = df["adaptability_score"].fillna(5.0).clip(0, 10)

    return df


# ---------------------------------------------------------------------------
# 3. Style fingerprint (K-means clustering)
# Based on Frontiers 2025: interior/defensive roles transfer more stably
# ---------------------------------------------------------------------------

def build_style_clusters(
    df: pd.DataFrame,
    n_clusters: int = N_STYLE_CLUSTERS,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Cluster players into style archetypes using K-means on STYLE_METRICS.

    Returns
    -------
    df           : original df with added 'style_cluster' and 'style_label' columns
    kmeans       : fitted KMeans (for predicting new players)
    scaler       : fitted StandardScaler (must be used consistently)
    """
    logger.info(f"Building style clusters (k={n_clusters}) …")

    available = [c for c in STYLE_METRICS if c in df.columns]
    if len(available) < 3:
        logger.warning("Too few style metrics — using default cluster 0 for all")
        df["style_cluster"] = 0
        df["style_label"]   = STYLE_LABELS[0]
        return df, None, None

    feature_matrix = df[available].copy().fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df = df.copy()
    df["style_cluster"] = kmeans.fit_predict(X)
    df["style_label"]   = df["style_cluster"].map(STYLE_LABELS).fillna("Unknown")

    # Log cluster composition
    for c in range(n_clusters):
        n = (df["style_cluster"] == c).sum()
        label = STYLE_LABELS.get(c, "?")
        logger.info(f"  Cluster {c} [{label:25s}]: {n} player-seasons")

    return df, kmeans, scaler


def calibrate_cluster_success_rates(
    hist_transfers: pd.DataFrame,
    cluster_col: str = "style_cluster",
) -> dict[int, float]:
    """
    From historical transfer data, compute what fraction of players from
    each style cluster performed above the 50th percentile in Superliga.
    """
    if hist_transfers.empty or cluster_col not in hist_transfers.columns:
        return CLUSTER_SUCCESS_DEFAULTS

    if "target_performance_pct" not in hist_transfers.columns:
        return CLUSTER_SUCCESS_DEFAULTS

    rates: dict[int, float] = {}
    for cluster_id, grp in hist_transfers.groupby(cluster_col):
        above_median = (grp["target_performance_pct"] >= 50).mean()
        rates[int(cluster_id)] = float(above_median)

    # Fill missing clusters with defaults
    for k, v in CLUSTER_SUCCESS_DEFAULTS.items():
        rates.setdefault(k, v)

    return rates


# ---------------------------------------------------------------------------
# 4. Similarity engine (cosine similarity)
# ---------------------------------------------------------------------------

def build_similarity_index(
    hist_transfers: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare the reference set of players who already made the
    Eastern EU → Superliga transfer, for use as similarity targets.

    Returns the cleaned reference DataFrame and the feature columns used.
    """
    if hist_transfers.empty:
        return pd.DataFrame(), []

    if feature_cols is None:
        feature_cols = [c for c in hist_transfers.columns
                        if c.endswith("_adj") or c in CORE_METRICS]
        feature_cols = [c for c in feature_cols if c in hist_transfers.columns]

    ref = hist_transfers[["player", "league_key", "transfer_season",
                           "target_performance_pct", "style_cluster",
                           "adaptability_score"] + feature_cols].dropna(
        subset=feature_cols
    ).copy()

    return ref, feature_cols


def find_similar_players(
    candidate_row: pd.Series,
    reference_df: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Find the top_n most similar historical players to a candidate
    using cosine similarity on feature_cols.

    Returns a DataFrame with columns:
      player, league_key, transfer_season, target_performance_pct,
      similarity_score
    """
    if reference_df.empty or not feature_cols:
        return pd.DataFrame()

    cand_vec = candidate_row[feature_cols].fillna(0).values.astype(float)
    if np.linalg.norm(cand_vec) == 0:
        return pd.DataFrame()

    similarities = []
    for _, ref_row in reference_df.iterrows():
        ref_vec = ref_row[feature_cols].fillna(0).values.astype(float)
        if np.linalg.norm(ref_vec) == 0:
            continue
        sim = 1 - cosine(cand_vec, ref_vec)
        similarities.append({
            "player":                  ref_row.get("player"),
            "league_key":              ref_row.get("league_key"),
            "transfer_season":         ref_row.get("transfer_season"),
            "target_performance_pct":  ref_row.get("target_performance_pct"),
            "adaptability_score":      ref_row.get("adaptability_score"),
            "style_cluster":           ref_row.get("style_cluster"),
            "similarity_score":        sim,
        })

    if not similarities:
        return pd.DataFrame()

    sim_df = (
        pd.DataFrame(similarities)
        .sort_values("similarity_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return sim_df


# ---------------------------------------------------------------------------
# 5. Simplified passing-network feature (TacticAI-inspired, DeepMind 2024)
# ---------------------------------------------------------------------------

def compute_passing_network_centrality(
    events_df: pd.DataFrame,
    player_col: str = "player",
    recipient_col: str = "pass_recipient_name",
) -> pd.DataFrame:
    """
    Build a simplified passing-network centrality score per player.

    Inspired by TacticAI (Google DeepMind 2024): model player
    relationships as a graph where edges = passes.

    Uses PageRank-like weighting: players who receive passes from
    many different teammates score higher (network centrality).

    Parameters
    ----------
    events_df : StatsBomb event DataFrame with pass events
    player_col, recipient_col : column names

    Returns a DataFrame with columns [player, network_centrality]
    """
    if events_df.empty:
        return pd.DataFrame(columns=["player", "network_centrality"])

    # Filter to completed passes
    passes = events_df[
        (events_df.get("type_name", events_df.get("type", "")) == "Pass") &
        (events_df[recipient_col].notna())
    ].copy() if recipient_col in events_df.columns else pd.DataFrame()

    if passes.empty:
        return pd.DataFrame(columns=["player", "network_centrality"])

    # Build adjacency counts: from_player → to_player
    edge_df = passes.groupby([player_col, recipient_col]).size().reset_index(name="weight")

    # In-degree centrality: sum of weights directed INTO each player
    in_degree = edge_df.groupby(recipient_col)["weight"].sum().reset_index()
    in_degree.columns = ["player", "network_centrality"]

    # Normalise to 0-10
    max_c = in_degree["network_centrality"].max()
    if max_c > 0:
        in_degree["network_centrality"] = (
            in_degree["network_centrality"] / max_c * 10
        )

    return in_degree


# ---------------------------------------------------------------------------
# 6. Age curve factor
# Peak performance: 26-27 for most outfield positions (Dendir 2016 + updates)
# ---------------------------------------------------------------------------

def compute_age_curve_factor(age: float, position: str = "MF") -> float:
    """
    Return a multiplier (0.7–1.0) representing where a player sits
    on the age-performance curve.

    Peak: 26-27 for most positions.
    Attackers peak slightly earlier (~25), defenders later (~28).
    """
    pos = position.upper()
    if "GK" in pos:
        peak = 29
    elif "CB" in pos or "LB" in pos or "RB" in pos or "DF" in pos:
        peak = 27
    elif "FW" in pos or "ST" in pos or "CF" in pos:
        peak = 25
    else:
        peak = 26  # midfielders / default

    distance = abs(age - peak)
    # Gaussian-like decay; plateau ±2 years around peak
    if distance <= 2:
        factor = 1.0
    elif distance <= 5:
        factor = 1.0 - 0.04 * (distance - 2)
    else:
        factor = max(0.70, 1.0 - 0.04 * (distance - 2))

    return round(factor, 3)


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add age_curve_factor and years_to_peak columns."""
    df = df.copy()

    if "standard_age" in df.columns:
        age_col = "standard_age"
    elif "age" in df.columns:
        age_col = "age"
    else:
        df["age_curve_factor"] = 0.9
        df["years_to_peak"]    = 0
        return df

    pos_col = next((c for c in ["pos", "standard_pos", "position"] if c in df.columns), None)

    df["age_curve_factor"] = df.apply(
        lambda r: compute_age_curve_factor(
            float(r[age_col]) if pd.notna(r[age_col]) else 26,
            str(r[pos_col]) if pos_col and pd.notna(r[pos_col]) else "MF",
        ),
        axis=1,
    )

    # Generic peak ~26; position-specific handled inside the function
    df["years_to_peak"] = df[age_col].apply(
        lambda a: max(0, round(26 - float(a), 1)) if pd.notna(a) else 0
    )

    return df


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_all_features(
    player_profiles: pd.DataFrame,
    hist_transfers: pd.DataFrame,
    league_coefficients: Optional[dict[str, float]] = None,
) -> tuple[pd.DataFrame, KMeans, StandardScaler, dict[int, float]]:
    """
    Run the complete feature engineering pipeline.

    Returns
    -------
    df                  : enriched player profiles
    kmeans              : fitted style clusterer
    scaler              : fitted scaler
    cluster_success     : dict[cluster_id → success_rate]
    """
    coefficients = league_coefficients or _default_league_coefficients()

    # Step 1: league adjustment
    df = apply_league_adjustment(player_profiles, coefficients)

    # Step 2: age features
    df = add_age_features(df)

    # Step 3: adaptability
    df = compute_adaptability_score(df)

    # Step 4: style clusters
    df, kmeans, scaler = build_style_clusters(df)

    # Step 5: calibrate cluster success rates from historical transfers
    if not hist_transfers.empty and "style_cluster" not in hist_transfers.columns:
        hist_transfers = hist_transfers.merge(
            df[["player", "league_key", "season", "style_cluster"]].drop_duplicates(),
            on=["player", "league_key"],
            how="left",
        )
    cluster_success = calibrate_cluster_success_rates(hist_transfers)

    df["cluster_historical_success_rate"] = df["style_cluster"].map(cluster_success)

    logger.info("Feature engineering complete.")
    logger.info(f"  Final shape: {df.shape}")
    return df, kmeans, scaler, cluster_success
