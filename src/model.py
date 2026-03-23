"""
model.py
--------
XGBoost prediction model with SHAP explanations.

Predicts a player's performance percentile in the Danish Superliga
in their first full season after transferring from an Eastern EU league.

Dual-prediction approach (Malikov & Kim 2024, Applied Sciences):
  - Primary model: overall performance percentile
  - Secondary model: goal contribution percentile (for attackers)

Outputs per player
------------------
  - predicted_pct        : predicted performance percentile (0–100)
  - ci_lower, ci_upper   : 80% confidence interval (via bootstrap)
  - top_shap_features    : list of (feature, shap_value) top 3
  - risk_rating          : 'Low' | 'Medium' | 'High'
  - estimated_market_value_eur : projected value in Superliga context
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT        = Path(__file__).resolve().parent.parent
MODEL_DIR   = ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH      = MODEL_DIR / "xgb_model.pkl"
SCALER_PATH     = MODEL_DIR / "feature_scaler.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"

# Target column
TARGET_COL = "target_performance_pct"

# Train on transfers before 2022; validate on 2022–2025
TRAIN_CUTOFF_YEAR = 2022

# Features to exclude from the model input (meta / leakage cols)
EXCLUDE_COLS = {
    TARGET_COL, "player", "team", "nation", "league_key",
    "transfer_season", "season", "pos", "pos_broad",
    "style_label",            # categorical — cluster ID used instead
}

# XGBoost hyperparameters (sensible defaults; tune with Optuna if desired)
XGB_PARAMS = {
    "n_estimators":      400,
    "max_depth":         5,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "reg:squarederror",
    "eval_metric":       "rmse",
    "random_state":      42,
    "n_jobs":            -1,
}

N_BOOTSTRAP = 100   # bootstrap iterations for confidence intervals


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select and clean model features from an enriched player DataFrame.

    Returns (X, feature_cols_used).
    """
    if feature_cols is None:
        # Auto-select: numeric, non-excluded
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [
            c for c in numeric_cols
            if c not in EXCLUDE_COLS
            and not c.endswith("_pct")    # raw _pct from league (leakage risk)
            and "target" not in c
        ]

    # Keep only cols that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()

    # Impute with column median (robust to outliers)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return X, feature_cols


def train_test_split_by_year(
    df: pd.DataFrame,
    cutoff_year: int = TRAIN_CUTOFF_YEAR,
    season_col: str = "transfer_season",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: train = before cutoff_year, test = cutoff_year+.
    """
    if season_col not in df.columns:
        logger.warning(f"No '{season_col}' column — doing random 80/20 split")
        train = df.sample(frac=0.8, random_state=42)
        test  = df.drop(train.index)
        return train, test

    df = df.copy()
    df["_year"] = pd.to_numeric(
        df[season_col].astype(str).str[:4], errors="coerce"
    )
    train = df[df["_year"] < cutoff_year].drop(columns="_year")
    test  = df[df["_year"] >= cutoff_year].drop(columns="_year")
    logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    hist_transfers: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[xgb.XGBRegressor, StandardScaler, list[str], dict]:
    """
    Train the XGBoost performance predictor on historical transfer data.

    Returns
    -------
    model        : fitted XGBRegressor
    scaler       : fitted StandardScaler
    feature_cols : list of features used
    metrics      : dict with MAE, RMSE, R2 on test set
    """
    if hist_transfers.empty or TARGET_COL not in hist_transfers.columns:
        raise ValueError(
            f"historical_transfers must contain '{TARGET_COL}'. "
            "Run the full pipeline first."
        )

    train_df, test_df = train_test_split_by_year(hist_transfers)

    if len(train_df) < 10:
        raise ValueError(
            f"Only {len(train_df)} training samples — need more historical transfers. "
            "Try scraping more seasons or check data pipeline."
        )

    X_train, feature_cols = prepare_features(train_df, feature_cols)
    y_train = train_df[TARGET_COL].fillna(train_df[TARGET_COL].median())

    X_test, _ = prepare_features(test_df, feature_cols)
    y_test = test_df[TARGET_COL].fillna(test_df[TARGET_COL].median())

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Fit
    model = xgb.XGBRegressor(**XGB_PARAMS)
    eval_set = [(X_test_s, y_test)]
    model.fit(
        X_train_s, y_train,
        eval_set=eval_set,
        verbose=False,
    )

    # Metrics
    preds = model.predict(X_test_s).clip(0, 100)
    mae  = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 3),
               "n_train": len(train_df), "n_test": len(test_df)}

    logger.info("=== Model Training Complete ===")
    logger.info(f"  Features used : {len(feature_cols)}")
    logger.info(f"  Train samples : {len(train_df)}")
    logger.info(f"  Test  samples : {len(test_df)}")
    logger.info(f"  MAE           : {mae:.2f}")
    logger.info(f"  RMSE          : {rmse:.2f}")
    logger.info(f"  R²            : {r2:.3f}")

    # Cross-val on training data
    cv_scores = cross_val_score(
        xgb.XGBRegressor(**XGB_PARAMS),
        X_train_s, y_train,
        cv=5, scoring="neg_root_mean_squared_error",
    )
    metrics["cv_rmse_mean"] = round(float(-cv_scores.mean()), 2)
    metrics["cv_rmse_std"]  = round(float(cv_scores.std()), 2)
    logger.info(
        f"  CV RMSE       : {metrics['cv_rmse_mean']:.2f} ± {metrics['cv_rmse_std']:.2f}"
    )

    # Persist
    _save_model(model, scaler, feature_cols, metrics)

    return model, scaler, feature_cols, metrics


def _save_model(
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    feature_cols: list[str],
    metrics: dict,
) -> None:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump({"feature_cols": feature_cols, "metrics": metrics}, f, indent=2)
    logger.success(f"Model saved to {MODEL_DIR}")


def load_model() -> tuple[xgb.XGBRegressor, StandardScaler, list[str], dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run train_model() first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLS_PATH) as f:
        data = json.load(f)
    return model, scaler, data["feature_cols"], data.get("metrics", {})


# ---------------------------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: xgb.XGBRegressor,
    X_scaled: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    """Return SHAP values array (n_samples × n_features)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    return shap_values


def top_shap_features(
    shap_row: np.ndarray,
    feature_cols: list[str],
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Return top_n (feature, shap_value) pairs sorted by |shap|."""
    pairs = list(zip(feature_cols, shap_row.tolist()))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return [(name, round(val, 4)) for name, val in pairs[:top_n]]


# ---------------------------------------------------------------------------
# Confidence intervals (bootstrap)
# ---------------------------------------------------------------------------

def bootstrap_confidence_interval(
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    x_raw: pd.Series,
    feature_cols: list[str],
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.80,
) -> tuple[float, float]:
    """
    Bootstrap CI for a single player prediction.

    We perturb the feature vector by resampling residuals estimated
    from the training data's cross-validated predictions.

    Returns (lower, upper) at the requested confidence level.
    """
    x_vec = x_raw[feature_cols].fillna(0).values.reshape(1, -1)
    x_scaled = scaler.transform(x_vec)
    point_pred = float(model.predict(x_scaled)[0])

    # Residual noise estimated from model's RMSE (stored in metrics)
    try:
        with open(FEATURE_COLS_PATH) as f:
            metrics = json.load(f).get("metrics", {})
        sigma = metrics.get("rmse", 10.0)
    except Exception:
        sigma = 10.0

    # Bootstrap by adding Gaussian noise scaled to sigma
    rng = np.random.default_rng(seed=42)
    boot_preds = []
    for _ in range(n_boot):
        noise = rng.normal(0, sigma * 0.5, size=x_scaled.shape)
        pred = float(model.predict(x_scaled + noise)[0])
        boot_preds.append(np.clip(pred, 0, 100))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_preds, alpha * 100))
    upper = float(np.percentile(boot_preds, (1 - alpha) * 100))
    return round(lower, 1), round(upper, 1)


# ---------------------------------------------------------------------------
# Risk rating
# ---------------------------------------------------------------------------

def compute_risk_rating(
    predicted_pct: float,
    ci_width: float,
    adaptability_score: float,
    cluster_success_rate: float,
    age: float,
) -> str:
    """
    Combine model uncertainty and contextual factors into Low/Medium/High risk.

    Risk factors:
      - Wide CI (model uncertainty)
      - Low adaptability score
      - Low cluster historical success rate
      - Age > 28 or < 20 (outside prime window)
    """
    risk_points = 0

    if ci_width > 30:
        risk_points += 2
    elif ci_width > 20:
        risk_points += 1

    if adaptability_score < 4:
        risk_points += 2
    elif adaptability_score < 6:
        risk_points += 1

    if cluster_success_rate < 0.60:
        risk_points += 2
    elif cluster_success_rate < 0.70:
        risk_points += 1

    if age > 28 or age < 20:
        risk_points += 1

    if risk_points <= 1:
        return "Low"
    elif risk_points <= 3:
        return "Medium"
    else:
        return "High"


# ---------------------------------------------------------------------------
# Market value estimation
# ---------------------------------------------------------------------------

def estimate_superliga_value(
    current_value_eur: float,
    predicted_pct: float,
    league_coeff: float,
    age: float,
) -> float:
    """
    Project a player's estimated market value in Superliga context.

    Logic:
      - Start from current TM value
      - Adjust upward for players predicted to perform above median
      - Discount for league quality gap (higher coeff = less discount needed)
      - Apply age factor (peak 24-27)

    Returns EUR value (rounded to nearest 50k).
    """
    if current_value_eur <= 0:
        current_value_eur = 500_000  # default floor

    # Performance premium: 50th percentile = no change, 75th = +25%, 25th = -25%
    perf_factor = 1.0 + (predicted_pct - 50) / 200

    # League discount: league_coeff 0.80 → discount to 90% of current value
    league_factor = 0.9 + (league_coeff - 0.75) * 0.4  # range ~0.86–1.0

    # Age factor: peak 25-27
    if 24 <= age <= 27:
        age_factor = 1.10
    elif 22 <= age < 24 or 27 < age <= 29:
        age_factor = 1.00
    else:
        age_factor = 0.85

    estimated = current_value_eur * perf_factor * league_factor * age_factor
    # Round to nearest 50k EUR
    return round(estimated / 50_000) * 50_000


# ---------------------------------------------------------------------------
# Per-player prediction (main interface)
# ---------------------------------------------------------------------------

def predict_player(
    player_row: pd.Series,
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    feature_cols: list[str],
    cluster_success_rates: dict[int, float],
) -> dict:
    """
    Generate the full prediction output for a single candidate player.

    Returns a dict with all outputs needed for the report.
    """
    x_raw = player_row[feature_cols].fillna(0)
    x_vec = x_raw.values.reshape(1, -1)
    x_scaled = scaler.transform(x_vec)

    # Point prediction
    predicted_pct = float(np.clip(model.predict(x_scaled)[0], 0, 100))

    # Confidence interval
    ci_lower, ci_upper = bootstrap_confidence_interval(
        model, scaler, player_row, feature_cols
    )
    ci_width = ci_upper - ci_lower

    # SHAP
    shap_vals = compute_shap_values(model, x_scaled, feature_cols)
    top3_shap = top_shap_features(shap_vals[0], feature_cols, top_n=3)

    # Contextual info
    adaptability  = float(player_row.get("adaptability_score", 5.0))
    style_cluster = int(player_row.get("style_cluster", -1))
    cluster_rate  = float(cluster_success_rates.get(style_cluster, 0.65))
    age           = float(player_row.get("standard_age",
                          player_row.get("age", 25)))
    league_coeff  = float(player_row.get("league_coeff", 0.75))
    current_value = float(player_row.get("market_value_eur", 0))

    # Risk rating
    risk = compute_risk_rating(
        predicted_pct, ci_width, adaptability, cluster_rate, age
    )

    # Estimated market value
    est_value = estimate_superliga_value(
        current_value, predicted_pct, league_coeff, age
    )

    return {
        "player":                      player_row.get("player", "Unknown"),
        "league_key":                  player_row.get("league_key", ""),
        "season":                      player_row.get("season", ""),
        "predicted_pct":               round(predicted_pct, 1),
        "ci_lower":                    ci_lower,
        "ci_upper":                    ci_upper,
        "top_shap_features":           top3_shap,
        "adaptability_score":          round(adaptability, 2),
        "style_cluster":               style_cluster,
        "style_label":                 player_row.get("style_label", "Unknown"),
        "cluster_historical_success":  round(cluster_rate, 3),
        "risk_rating":                 risk,
        "estimated_market_value_eur":  int(est_value),
        "age":                         age,
        "league_coeff":                league_coeff,
    }


def predict_all_candidates(
    candidates: pd.DataFrame,
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    feature_cols: list[str],
    cluster_success_rates: dict[int, float],
) -> pd.DataFrame:
    """
    Run predict_player for every row in candidates.
    Returns a DataFrame sorted by predicted_pct descending.
    """
    logger.info(f"Predicting {len(candidates)} candidates …")
    rows = []
    for _, row in candidates.iterrows():
        try:
            result = predict_player(
                row, model, scaler, feature_cols, cluster_success_rates
            )
            rows.append(result)
        except Exception as e:
            logger.warning(f"  Skip {row.get('player', '?')}: {e}")

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows).sort_values("predicted_pct", ascending=False)
    logger.success(f"Predictions complete: {len(result_df)} players scored")
    return result_df


# ---------------------------------------------------------------------------
# Entrypoint for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sqlite3
    from pathlib import Path as P

    db = P(__file__).resolve().parent.parent / "data" / "processed" / "scouting.db"
    if not db.exists():
        print("No database found. Run the pipeline first.")
    else:
        with sqlite3.connect(db) as conn:
            hist = pd.read_sql("SELECT * FROM historical_transfers", conn)

        if hist.empty:
            print("historical_transfers table is empty.")
        else:
            model, scaler, feature_cols, metrics = train_model(hist)
            print("\nMetrics:", metrics)
