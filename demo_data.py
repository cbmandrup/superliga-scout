"""
demo_data.py
------------
Generates realistic synthetic scouting data for the Streamlit Cloud demo.

When no real scouting.db is present (e.g. on Streamlit Cloud where the
filesystem is ephemeral), this module creates plausible fake players so
the dashboard is fully interactive without requiring a real data pull.

Run standalone:  python demo_data.py
Or imported by:  app.py  (auto-detected when DB is missing)
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "processed" / "scouting.db"

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

EASTERN_EU_LEAGUES = [
    "ekstraklasa", "czech", "croatia", "romania", "bulgaria", "serbia"
]

LEAGUE_COEFFS = {
    "ekstraklasa": 0.82,
    "czech":       0.80,
    "croatia":     0.78,
    "romania":     0.74,
    "serbia":      0.72,
    "bulgaria":    0.70,
}

POSITIONS = ["FW", "MF", "DF", "AM", "WM"]

STYLE_LABELS = {
    0: "Pressing Forward",
    1: "Wide Creator",
    2: "Deep-Lying Playmaker",
    3: "Box-to-Box Midfielder",
    4: "Defensive Anchor",
    5: "Advanced Playmaker",
}

CLUSTER_SUCCESS = {
    0: 0.62, 1: 0.55, 2: 0.71, 3: 0.65, 4: 0.68, 5: 0.58,
}

FIRST_NAMES = [
    "Jakub", "Tomáš", "Luka", "Andrei", "Ivan", "Mihail", "Stefan",
    "Patrik", "Matej", "Nikola", "Aleksandar", "Bogdan", "Radoslav",
    "Przemysław", "Krzysztof", "Wojciech", "Łukasz", "Mateusz",
    "Ondřej", "Vladimír", "Petar", "Marko", "Dario", "Tin",
    "Cosmin", "Florin", "Ionuț", "Valentin", "Claudiu",
    "Georgi", "Dimitar", "Hristo", "Bozhidar", "Yavor",
    "Nemanja", "Aleksandar", "Lazar", "Dušan", "Filip",
]

LAST_NAMES = [
    "Novák", "Kowalski", "Horvat", "Popescu", "Petrov", "Jovanović",
    "Wysocki", "Blažević", "Ionescu", "Dimitrov", "Stojanović",
    "Krejčí", "Wiśniewski", "Marković", "Dobre", "Georgiev",
    "Piotrowski", "Perić", "Munteanu", "Nikolić", "Zieliński",
    "Vlček", "Simonović", "Constantin", "Stoyanov", "Lazović",
    "Wróblewski", "Jurić", "Ungureanu", "Todorov", "Mihajlović",
]

SUPERLIGA_CLUBS = [
    "FC København", "FC Midtjylland", "Brøndby IF", "AGF",
    "Randers FC", "OB", "AaB", "FC Nordsjælland",
]


def _random_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Player profile generator
# ---------------------------------------------------------------------------

def generate_player_profiles(n: int = 200) -> pd.DataFrame:
    rows = []
    seasons = ["2021", "2022", "2023", "2024"]

    for i in range(n):
        league  = random.choice(EASTERN_EU_LEAGUES)
        coeff   = LEAGUE_COEFFS[league]
        pos     = random.choice(POSITIONS)
        age     = random.randint(18, 32)
        season  = random.choice(seasons)
        cluster = random.randint(0, 5)
        mins    = random.uniform(900, 3400)
        nineties = mins / 90

        # Base stats — vary by position
        is_attacker = pos in ("FW", "AM")
        is_defender = pos == "DF"

        xg   = _clip(np.random.normal(0.35 if is_attacker else 0.08, 0.15), 0, 1.2)
        xa   = _clip(np.random.normal(0.22 if is_attacker else 0.10, 0.12), 0, 0.9)
        gls  = _clip(np.random.normal(0.28 if is_attacker else 0.05, 0.12), 0, 1.0)
        ast  = _clip(np.random.normal(0.18, 0.10), 0, 0.7)
        sca  = _clip(np.random.normal(2.8 if is_attacker else 1.5, 0.8), 0, 6.0)
        prog_p = _clip(np.random.normal(5.5 if not is_defender else 3.0, 2.0), 0, 14)
        prog_c = _clip(np.random.normal(3.5 if is_attacker else 1.5, 1.5), 0, 9)
        press  = _clip(np.random.normal(8.0 if not is_defender else 5.0, 3.0), 0, 20)
        tkl    = _clip(np.random.normal(1.5 if is_defender else 1.0, 0.7), 0, 5)
        intercept = _clip(np.random.normal(1.2 if is_defender else 0.7, 0.5), 0, 4)

        # League adjustment
        xg_adj   = xg   / coeff
        xa_adj   = xa   / coeff
        sca_adj  = sca  / coeff
        prog_p_adj = prog_p / coeff
        prog_c_adj = prog_c / coeff
        press_adj  = press  / coeff
        tkl_adj    = tkl    / coeff

        # Adaptability (0-10)
        adapt = _clip(np.random.normal(6.0, 2.0), 0, 10)

        # Market value (EUR)
        base_val = {
            "ekstraklasa": 1_200_000,
            "czech":       1_000_000,
            "croatia":     900_000,
            "romania":     700_000,
            "serbia":      600_000,
            "bulgaria":    500_000,
        }[league]
        value = _clip(np.random.lognormal(np.log(base_val), 0.6), 100_000, 8_000_000)

        # Predicted Superliga percentile
        raw_score = (
            xg_adj * 25 +
            xa_adj * 20 +
            sca_adj * 5 +
            prog_p_adj * 3 +
            adapt * 2 +
            random.gauss(0, 8)
        )
        pred_pct   = _clip(raw_score + 30, 5, 95)
        ci_lower   = _clip(pred_pct - random.uniform(8, 18), 0, 100)
        ci_upper   = _clip(pred_pct + random.uniform(8, 18), 0, 100)

        # Risk
        ci_width = ci_upper - ci_lower
        risk_pts = (
            (2 if ci_width > 30 else 1 if ci_width > 20 else 0) +
            (2 if adapt < 4 else 1 if adapt < 6 else 0) +
            (1 if age > 28 or age < 20 else 0)
        )
        risk = "Low" if risk_pts <= 1 else "High" if risk_pts >= 4 else "Medium"

        # Estimated Superliga value
        perf_factor  = 1.0 + (pred_pct - 50) / 200
        league_factor = 0.9 + (coeff - 0.75) * 0.4
        age_factor   = 1.10 if 24 <= age <= 27 else 1.0 if 22 <= age <= 29 else 0.85
        est_val = round(value * perf_factor * league_factor * age_factor / 50_000) * 50_000

        rows.append({
            "player":                     _random_name(),
            "league_key":                 league,
            "season":                     season,
            "pos":                        pos,
            "pos_broad":                  pos,
            "age":                        age,
            "standard_90s":               round(nineties, 1),
            "standard_xg":                round(xg, 3),
            "standard_xa":                round(xa, 3),
            "standard_gls":               round(gls, 3),
            "standard_ast":               round(ast, 3),
            "goal_shot_creation_sca90":   round(sca, 3),
            "passing_prog_p":             round(prog_p, 2),
            "possession_prg_c":           round(prog_c, 2),
            "defense_press":              round(press, 2),
            "defense_tkl":                round(tkl, 2),
            "defense_int":                round(intercept, 2),
            # Adjusted
            "standard_xg_adj":            round(xg_adj, 3),
            "standard_xa_adj":            round(xa_adj, 3),
            "goal_shot_creation_sca90_adj": round(sca_adj, 3),
            "passing_prog_p_adj":         round(prog_p_adj, 2),
            "possession_prg_c_adj":       round(prog_c_adj, 2),
            "defense_press_adj":          round(press_adj, 2),
            "defense_tkl_adj":            round(tkl_adj, 2),
            # Percentiles (simulated)
            "standard_xg_adj_pct":        round(_clip(pred_pct + random.gauss(0, 10), 5, 95), 1),
            "standard_xa_adj_pct":        round(_clip(pred_pct + random.gauss(0, 10), 5, 95), 1),
            "goal_shot_creation_sca90_adj_pct": round(_clip(pred_pct + random.gauss(0, 10), 5, 95), 1),
            "passing_prog_p_adj_pct":     round(_clip(pred_pct + random.gauss(0, 12), 5, 95), 1),
            "possession_prg_c_adj_pct":   round(_clip(pred_pct + random.gauss(0, 12), 5, 95), 1),
            "defense_press_adj_pct":      round(_clip(50 + random.gauss(0, 20), 5, 95), 1),
            "defense_tkl_adj_pct":        round(_clip(50 + random.gauss(0, 20), 5, 95), 1),
            # Features
            "adaptability_score":         round(adapt, 2),
            "style_cluster":              cluster,
            "style_label":                STYLE_LABELS[cluster],
            "cluster_historical_success_rate": CLUSTER_SUCCESS[cluster],
            "league_coeff":               coeff,
            "market_value_eur":           round(value),
            # Predictions
            "predicted_pct":              round(pred_pct, 1),
            "ci_lower":                   round(ci_lower, 1),
            "ci_upper":                   round(ci_upper, 1),
            "risk_rating":                risk,
            "estimated_market_value_eur": int(est_val),
            "top_shap_features": str([
                ("standard_xg_adj",    round(random.gauss(0.3, 0.2), 4)),
                ("adaptability_score", round(random.gauss(0.2, 0.15), 4)),
                ("league_coeff",       round(random.gauss(-0.1, 0.1), 4)),
            ]),
            "network_centrality": round(_clip(np.random.exponential(3), 0, 10), 2),
        })

    return pd.DataFrame(rows)


def generate_historical_transfers(players_df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """Simulate historical Eastern EU → Superliga transfers for training data."""
    sample = players_df.sample(min(n, len(players_df)), random_state=42).copy()
    sample["transfer_season"] = [
        str(random.randint(2017, 2021)) for _ in range(len(sample))
    ]
    sample["target_club"] = [random.choice(SUPERLIGA_CLUBS) for _ in range(len(sample))]
    # Outcome: Superliga performance percentile
    sample["target_performance_pct"] = sample["predicted_pct"].apply(
        lambda p: _clip(p + random.gauss(0, 12), 5, 95)
    )
    return sample


def generate_market_values(players_df: pd.DataFrame) -> pd.DataFrame:
    mv = players_df[["player", "league_key", "season", "market_value_eur"]].copy()
    mv["currency"] = "EUR"
    return mv


# ---------------------------------------------------------------------------
# Write to SQLite
# ---------------------------------------------------------------------------

def seed_demo_database(db_path: Path = DB_PATH, n_players: int = 200) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_players} synthetic player profiles …")
    profiles = generate_player_profiles(n_players)
    hist     = generate_historical_transfers(profiles)
    mkt      = generate_market_values(profiles)
    preds    = profiles[[c for c in profiles.columns if c not in
                          {"standard_xg","standard_xa","standard_gls","standard_ast",
                           "goal_shot_creation_sca90","passing_prog_p","possession_prg_c",
                           "defense_press","defense_tkl","defense_int"}]].copy()

    with sqlite3.connect(db_path) as conn:
        profiles.to_sql("player_profiles",      conn, if_exists="replace", index=False)
        hist.to_sql(    "historical_transfers",  conn, if_exists="replace", index=False)
        mkt.to_sql(     "market_values",         conn, if_exists="replace", index=False)
        preds.to_sql(   "predictions",           conn, if_exists="replace", index=False)

    print(f"Demo database written to: {db_path}")
    print(f"  player_profiles     : {len(profiles)} rows")
    print(f"  historical_transfers: {len(hist)} rows")
    print(f"  predictions         : {len(preds)} rows")


if __name__ == "__main__":
    seed_demo_database()
    print("\nDone. Run:  streamlit run app.py")
