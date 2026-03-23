"""
app.py
------
Streamlit scouting dashboard for the Superliga Scout system.

Run with:
  streamlit run app.py

Features
--------
  - Sidebar filters: position, age, price range, origin league, risk
  - Ranked shortlist table with key metrics
  - Player detail panel: radar, adaptability, similarity, SHAP
  - Club fit analysis
  - PDF export for selected players
"""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path

import math

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Auto-seed demo data if no real database exists (e.g. Streamlit Cloud)
if not (Path(__file__).resolve().parent / "data" / "processed" / "scouting.db").exists():
    from demo_data import seed_demo_database
    seed_demo_database()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "processed" / "scouting.db"
FEATURE_COLS_PATH = ROOT / "data" / "models" / "feature_cols.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Superliga Scout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0D1B40;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #888;
        margin-top: 0;
    }
    .metric-card {
        background: #F5F5F5;
        border-left: 4px solid #C8102E;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .risk-low    { color: #2ECC71; font-weight: bold; }
    .risk-medium { color: #E67E22; font-weight: bold; }
    .risk-high   { color: #C8102E; font-weight: bold; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0D1B40;
        border-bottom: 2px solid #E8A000;
        padding-bottom: 4px;
        margin-top: 16px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load predictions, player profiles, and historical transfers from SQLite."""
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()

        predictions = pd.read_sql("SELECT * FROM predictions", conn) \
            if "predictions" in tables else pd.DataFrame()
        profiles = pd.read_sql("SELECT * FROM player_profiles", conn) \
            if "player_profiles" in tables else pd.DataFrame()
        hist_transfers = pd.read_sql("SELECT * FROM historical_transfers", conn) \
            if "historical_transfers" in tables else pd.DataFrame()

    return predictions, profiles, hist_transfers


@st.cache_data(ttl=300)
def load_model_metrics() -> dict:
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH) as f:
            return json.load(f).get("metrics", {})
    return {}


# ---------------------------------------------------------------------------
# Helper: radar chart
# ---------------------------------------------------------------------------

RADAR_METRICS = [
    ("standard_xg_adj",              "xG/90"),
    ("standard_xa_adj",              "xA/90"),
    ("goal_shot_creation_sca90_adj", "SCA/90"),
    ("passing_prog_p_adj",           "Prog Pass"),
    ("possession_prg_c_adj",         "Prog Carry"),
    ("defense_press_adj",            "Press"),
    ("defense_tkl_adj",              "Tackles"),
]


def radar_chart(player_vals, avg_vals, labels, player_name):
    N = len(labels)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    p_vals = list(player_vals) + [player_vals[0]]
    a_vals = list(avg_vals) + [avg_vals[0]]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True), facecolor="white")
    ax.set_facecolor("#F8F8F8")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], size=6, color="grey")
    ax.grid(color="lightgrey", linewidth=0.5)

    ax.fill(angles, a_vals, alpha=0.12, color="grey")
    ax.plot(angles, a_vals, color="grey", linewidth=1.5, linestyle="--", label="Superliga avg")
    ax.fill(angles, p_vals, alpha=0.25, color="#C8102E")
    ax.plot(angles, p_vals, color="#C8102E", linewidth=2, label=player_name)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=7)
    return fig


def adaptability_gauge(score: float):
    fig, ax = plt.subplots(figsize=(4, 1.2), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(mpatches.FancyBboxPatch((0, 0.3), 10, 0.4,
                                          boxstyle="round,pad=0.05",
                                          facecolor="#EEEEEE", edgecolor="none"))
    cmap  = plt.get_cmap("RdYlGn")
    color = cmap(score / 10)
    ax.add_patch(mpatches.FancyBboxPatch((0, 0.3), score, 0.4,
                                          boxstyle="round,pad=0.05",
                                          facecolor=color, edgecolor="none"))
    ax.text(5, 0.9, "Adaptability Score", ha="center", va="top", fontsize=8, color="grey")
    ax.text(score, 0.08, f"{score:.1f} / 10", ha="center", va="bottom", fontsize=9, fontweight="bold")
    return fig


def ci_bar(pred, lower, upper):
    fig, ax = plt.subplots(figsize=(5, 0.9), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.barh(0.5, upper - lower, left=lower, height=0.3, color="#AED6F1", align="center")
    ax.plot([pred, pred], [0.28, 0.72], color="#C8102E", linewidth=2.5)
    ax.text(pred, 0.85, f"{pred:.0f}th pct", ha="center", fontsize=9, fontweight="bold", color="#C8102E")
    ax.text(lower, 0.1, f"{lower:.0f}", ha="center", fontsize=7, color="grey")
    ax.text(upper, 0.1, f"{upper:.0f}", ha="center", fontsize=7, color="grey")
    for v in [25, 50, 75]:
        ax.axvline(v, color="#DDDDDD", linewidth=0.5, ymin=0.2, ymax=0.8)
    ax.text(50, -0.12, "Predicted Superliga performance percentile", ha="center", fontsize=7, color="grey")
    return fig


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## Filters")

    # Position
    positions = ["All"] + sorted(df["pos_broad"].dropna().unique().tolist()) \
        if "pos_broad" in df.columns else ["All"]
    pos_filter = st.sidebar.selectbox("Position", positions)

    # Age
    age_col = next((c for c in ["age", "standard_age"] if c in df.columns), None)
    if age_col:
        min_age = int(df[age_col].min()) if not df.empty else 16
        max_age = int(df[age_col].max()) if not df.empty else 35
        age_range = st.sidebar.slider("Age range", min_age, max_age, (18, 28))
    else:
        age_range = (16, 35)
        age_col = None

    # Origin league
    leagues = ["All"] + sorted(df["league_key"].dropna().unique().tolist()) \
        if "league_key" in df.columns else ["All"]
    league_filter = st.sidebar.multiselect("Origin league", leagues[1:], default=leagues[1:])

    # Risk
    risk_opts = ["All", "Low", "Medium", "High"]
    risk_filter = st.sidebar.selectbox("Risk rating", risk_opts)

    # Predicted percentile threshold
    min_pct = st.sidebar.slider("Min predicted percentile", 0, 100, 40)

    # Market value
    val_col = "estimated_market_value_eur"
    if val_col in df.columns and df[val_col].max() > 0:
        max_val = int(df[val_col].max())
        val_range = st.sidebar.slider(
            "Market value (EUR M)",
            0.0, max_val / 1e6, (0.0, max_val / 1e6), step=0.1
        )
    else:
        val_range = None

    # Apply filters
    mask = pd.Series([True] * len(df), index=df.index)

    if pos_filter != "All" and "pos_broad" in df.columns:
        mask &= df["pos_broad"] == pos_filter

    if age_col:
        mask &= df[age_col].between(*age_range)

    if league_filter and "league_key" in df.columns:
        mask &= df["league_key"].isin(league_filter)

    if risk_filter != "All" and "risk_rating" in df.columns:
        mask &= df["risk_rating"] == risk_filter

    if "predicted_pct" in df.columns:
        mask &= df["predicted_pct"] >= min_pct

    if val_range is not None and val_col in df.columns:
        mask &= df[val_col].between(val_range[0] * 1e6, val_range[1] * 1e6)

    return df[mask].copy()


# ---------------------------------------------------------------------------
# Shortlist table
# ---------------------------------------------------------------------------

def display_shortlist(df: pd.DataFrame) -> str | None:
    """Display ranked shortlist; return selected player name or None."""
    st.markdown('<div class="section-header">Ranked Shortlist</div>', unsafe_allow_html=True)

    display_cols_map = {
        "player":                     "Player",
        "league_key":                 "League",
        "pos_broad":                  "Pos",
        "age":                        "Age",
        "predicted_pct":              "Pred. Pct",
        "ci_lower":                   "CI Low",
        "ci_upper":                   "CI High",
        "adaptability_score":         "Adaptability",
        "style_label":                "Style",
        "risk_rating":                "Risk",
        "estimated_market_value_eur": "Est. Value (€)",
    }

    available = {k: v for k, v in display_cols_map.items() if k in df.columns}
    show_df = df[list(available.keys())].rename(columns=available)

    if "Est. Value (€)" in show_df.columns:
        show_df["Est. Value (€)"] = show_df["Est. Value (€)"].apply(
            lambda x: f"€{int(x):,}" if pd.notna(x) else "N/A"
        )
    if "Pred. Pct" in show_df.columns:
        show_df["Pred. Pct"] = show_df["Pred. Pct"].apply(
            lambda x: f"{x:.0f}th" if pd.notna(x) else "N/A"
        )

    st.dataframe(show_df, use_container_width=True, height=300)

    if df.empty:
        return None

    player_names = df["player"].dropna().tolist() if "player" in df.columns else []
    if not player_names:
        return None

    selected = st.selectbox("Select player for detailed profile", ["— select —"] + player_names)
    return selected if selected != "— select —" else None


# ---------------------------------------------------------------------------
# Player detail panel
# ---------------------------------------------------------------------------

def display_player_detail(
    player_name: str,
    predictions: pd.DataFrame,
    profiles: pd.DataFrame,
    hist_transfers: pd.DataFrame,
    club_name: str,
) -> None:
    if predictions.empty or "player" not in predictions.columns:
        st.warning("No prediction data available.")
        return

    pred_row = predictions[predictions["player"] == player_name]
    if pred_row.empty:
        st.warning(f"No predictions found for {player_name}")
        return
    pred_row = pred_row.iloc[0]

    # Basic info
    league    = pred_row.get("league_key", "?").replace("_", " ").title()
    age       = pred_row.get("age", "?")
    predicted = float(pred_row.get("predicted_pct", 0))
    ci_lower  = float(pred_row.get("ci_lower", predicted - 10))
    ci_upper  = float(pred_row.get("ci_upper", predicted + 10))
    adapt     = float(pred_row.get("adaptability_score", 5))
    risk      = pred_row.get("risk_rating", "Medium")
    style_lbl = pred_row.get("style_label", "Unknown")
    est_val   = int(pred_row.get("estimated_market_value_eur", 0))
    cluster_sr = float(pred_row.get("cluster_historical_success", 0.65))
    shap_feat  = pred_row.get("top_shap_features", [])
    if isinstance(shap_feat, str):
        import ast
        try:
            shap_feat = ast.literal_eval(shap_feat)
        except Exception:
            shap_feat = []

    st.markdown(f"## {player_name}")
    st.markdown(f"**League:** {league} &nbsp;|&nbsp; **Age:** {age} &nbsp;|&nbsp; **Style:** {style_lbl}")

    risk_color = {"Low": "#2ECC71", "Medium": "#E67E22", "High": "#C8102E"}.get(risk, "#888")
    st.markdown(
        f'<span style="background:{risk_color};color:white;padding:3px 10px;'
        f'border-radius:4px;font-weight:bold;">{risk} RISK</span>',
        unsafe_allow_html=True,
    )

    # Row 1: radar + CI + adaptability
    col_radar, col_right = st.columns([1, 1.1])

    with col_radar:
        st.markdown('<div class="section-header">Radar vs Superliga Average</div>', unsafe_allow_html=True)
        # Build radar values from profiles
        profile_row = None
        if not profiles.empty and "player" in profiles.columns:
            pf = profiles[profiles["player"] == player_name]
            if not pf.empty:
                profile_row = pf.iloc[0]

        sl_profiles = profiles[profiles["league_key"] == "superliga"] \
            if not profiles.empty and "league_key" in profiles.columns else pd.DataFrame()

        player_vals, avg_vals, radar_labels = [], [], []
        for col, label in RADAR_METRICS:
            pct_col = col.replace("_adj", "_adj_pct") if "_adj" in col else col + "_pct"
            p_val = float(profile_row[pct_col]) if profile_row is not None and pct_col in profile_row.index \
                else float(profile_row[col]) if profile_row is not None and col in profile_row.index else 50
            s_val = float(sl_profiles[pct_col].median()) if not sl_profiles.empty and pct_col in sl_profiles.columns else 50
            player_vals.append(np.clip(p_val, 0, 100))
            avg_vals.append(np.clip(s_val, 0, 100))
            radar_labels.append(label)

        fig = radar_chart(player_vals, avg_vals, radar_labels, player_name)
        st.pyplot(fig, use_container_width=False)

    with col_right:
        st.markdown('<div class="section-header">Predicted Performance</div>', unsafe_allow_html=True)
        fig_ci = ci_bar(predicted, ci_lower, ci_upper)
        st.pyplot(fig_ci, use_container_width=True)

        st.markdown('<div class="section-header">Adaptability</div>', unsafe_allow_html=True)
        fig_adapt = adaptability_gauge(adapt)
        st.pyplot(fig_adapt, use_container_width=True)

        # Key stats row
        c1, c2, c3 = st.columns(3)
        c1.metric("Est. Value", f"€{est_val:,.0f}")
        c2.metric("Style success rate", f"{cluster_sr:.0%}")
        c3.metric("Adaptability", f"{adapt:.1f}/10")

    # SHAP features
    if shap_feat:
        st.markdown('<div class="section-header">Top SHAP Features</div>', unsafe_allow_html=True)
        shap_df = pd.DataFrame(
            [(f.replace("_", " ").title(), v, "▲" if v > 0 else "▼")
             for f, v in shap_feat],
            columns=["Feature", "SHAP Value", "Direction"]
        )
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

    # Similar players
    st.markdown('<div class="section-header">Similar Historical Transfers (Eastern EU → Superliga)</div>', unsafe_allow_html=True)
    if not hist_transfers.empty and "player" in hist_transfers.columns:
        from src.features import build_similarity_index, find_similar_players

        ref_df, feat_cols = build_similarity_index(hist_transfers)
        if profile_row is not None and not ref_df.empty:
            sim_df = find_similar_players(profile_row, ref_df, feat_cols, top_n=3)
            if not sim_df.empty:
                st.dataframe(
                    sim_df[["player", "league_key", "transfer_season",
                             "target_performance_pct", "similarity_score"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No similar historical transfers found.")
        else:
            st.info("Profile data needed for similarity search.")
    else:
        st.info("No historical transfer data loaded.")

    # Club fit
    st.markdown(f'<div class="section-header">Club Fit: {club_name}</div>', unsafe_allow_html=True)
    adapt_desc = (
        "høj robusthed over for systemskift" if adapt >= 7
        else "moderat tilpasningsevne" if adapt >= 4
        else "potentiel tilpasningsudfordring"
    )
    st.write(
        f"**{player_name}** plays in the **{style_lbl}** style archetype. "
        f"This style historically adapts to the Superliga with a **{cluster_sr:.0%}** success rate. "
        f"An adaptability score of **{adapt:.1f}/10** indicates {adapt_desc}. "
        f"Predicted percentile in {club_name}'s system: **{predicted:.0f}th**."
    )


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def export_pdf_section(
    selected_names: list[str],
    predictions: pd.DataFrame,
    profiles: pd.DataFrame,
    club_name: str,
) -> None:
    st.markdown('<div class="section-header">Export PDF Report</div>', unsafe_allow_html=True)
    if st.button("Generate PDF for selected players"):
        if not selected_names:
            st.warning("Select players first.")
            return

        from src.report import generate_report

        subset_preds = predictions[predictions["player"].isin(selected_names)] \
            if "player" in predictions.columns else pd.DataFrame()

        with st.spinner("Generating PDF …"):
            path = generate_report(
                predictions=subset_preds,
                player_profiles=profiles,
                similar_players={},
                club_name=club_name,
            )
        st.success(f"Report saved: {path}")
        with open(path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name=Path(path).name,
                mime="application/pdf",
            )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # Header
    st.markdown('<div class="main-title">⚽ Superliga Scout</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Eastern European talent identification for the Danish Superliga</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Load data
    predictions, profiles, hist_transfers = load_data()
    model_metrics = load_model_metrics()

    # Model status banner
    if predictions.empty:
        st.warning(
            "No prediction data found. Run the pipeline first:\n"
            "```\npython -m src.scraper\npython -m src.pipeline\n"
            "python -m src.model\n```"
        )

    if model_metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model MAE",      f"{model_metrics.get('mae', '?')}")
        c2.metric("Model RMSE",     f"{model_metrics.get('rmse', '?')}")
        c3.metric("R²",             f"{model_metrics.get('r2', '?')}")
        c4.metric("Training samples", f"{model_metrics.get('n_train', '?')}")
        st.markdown("---")

    # Club name input
    club_name = st.text_input("Target club name (for fit analysis)", value="FC København")

    # Sidebar filters
    if not predictions.empty:
        # Merge pos_broad from profiles if missing
        if "pos_broad" not in predictions.columns and not profiles.empty:
            if "pos_broad" in profiles.columns and "player" in profiles.columns:
                pb = profiles[["player", "pos_broad"]].drop_duplicates("player")
                predictions = predictions.merge(pb, on="player", how="left")

        if "age" not in predictions.columns and "standard_age" in predictions.columns:
            predictions["age"] = predictions["standard_age"]

        filtered = sidebar_filters(predictions)
    else:
        filtered = pd.DataFrame()

    # Main layout: shortlist + detail
    col_list, col_detail = st.columns([1, 1.4])

    with col_list:
        selected_player = display_shortlist(filtered)

        # Multi-select for PDF export
        if not filtered.empty and "player" in filtered.columns:
            st.markdown("---")
            export_names = st.multiselect(
                "Select players for PDF export",
                filtered["player"].tolist(),
                default=[],
            )
            export_pdf_section(export_names, predictions, profiles, club_name)

    with col_detail:
        if selected_player:
            display_player_detail(
                selected_player,
                predictions,
                profiles,
                hist_transfers,
                club_name,
            )
        else:
            st.info("Select a player from the shortlist to see the full profile.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<small>Data: FBref / soccerdata · Transfermarkt · StatsBomb Open Data  |  "
        "Model: XGBoost + SHAP  |  "
        "Research: Ribeiro et al. 2025 · Bonetti et al. 2025 · Frontiers 2025 · "
        "Malikov & Kim 2024 · TacticAI (DeepMind 2024)</small>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
