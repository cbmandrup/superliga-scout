"""
Folketing Tracker — Streamlit dashboard for tracking Danish Parliament
meetings (møder) and votes (afstemninger).

Data source: Folketing Open Data API (ODA) — https://oda.ft.dk/api/
Falls back to synthetic demo data automatically if the API is unreachable.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.folketing_api import (
    get_periods,
    get_meetings,
    get_votes,
    get_actors,
    get_individual_votes,
    get_vote_summary,
    build_politician_scoreboard,
    get_meetings_by_month,
    get_votes_by_month,
    FOLKETING_ANNUAL_SALARY_DKK,
)
from folketing_demo import get_demo_data

# ─── Color palette (ft.dk inspired) ─────────────────────────────────────────
TEAL = "#00686E"
DARK = "#1A1A42"
GREEN = "#2E7D32"
RED_VOTE = "#C62828"
LIGHT_BG = "#F5F5F5"
WHITE = "#FFFFFF"

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Folketing Tracker",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    /* Main background */
    .stApp {{ background-color: {WHITE}; }}

    /* Header bar */
    .ft-header {{
        background-color: {TEAL};
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }}
    .ft-header h1 {{
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    .ft-header p {{
        margin: 0.2rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.85;
    }}

    /* KPI cards */
    .kpi-card {{
        background-color: {LIGHT_BG};
        border-left: 4px solid {TEAL};
        border-radius: 4px;
        padding: 1rem 1.2rem;
        text-align: left;
    }}
    .kpi-card .label {{
        font-size: 0.8rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }}
    .kpi-card .value {{
        font-size: 2rem;
        font-weight: 700;
        color: {DARK};
    }}

    /* Vedtaget / forkastet cards */
    .kpi-green {{ border-left-color: {GREEN}; }}
    .kpi-red   {{ border-left-color: {RED_VOTE}; }}

    /* Section titles */
    .section-title {{
        font-size: 1rem;
        font-weight: 600;
        color: {DARK};
        border-bottom: 2px solid {TEAL};
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {DARK};
        color: white;
    }}
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {{
        color: #ccc !important;
        font-size: 0.85rem;
    }}

    /* Table tweaks */
    .dataframe thead tr th {{
        background-color: {TEAL} !important;
        color: white !important;
    }}

    /* Hide Streamlit branding */
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ft-header">
        <h1>Folketing Tracker</h1>
        <p>Overblik over møder og afstemninger i det danske Folketing</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _try_load_periods() -> pd.DataFrame:
    """Try to load real periods; return empty DataFrame on failure."""
    try:
        df = get_periods()
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_demo(periode_id: int | None) -> dict:
    return get_demo_data(periode_id)


# ─── Detect live vs demo mode ─────────────────────────────────────────────────
_probe = _try_load_periods()
IS_DEMO = _probe.empty

if IS_DEMO:
    st.warning(
        "**Demo-tilstand** — Folketing API er ikke tilgængeligt. "
        "Viser syntetiske data til demonstration.",
        icon="ℹ️",
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_periods() -> pd.DataFrame:
    if IS_DEMO:
        return _load_demo(None)["periods"]
    return _probe


def load_meetings(periode_id: int | None) -> pd.DataFrame:
    if IS_DEMO:
        return _load_demo(periode_id)["meetings"]
    try:
        return get_meetings(periode_id=periode_id, max_records=2000)
    except Exception:
        return _load_demo(periode_id)["meetings"]


def load_votes(periode_id: int | None) -> pd.DataFrame:
    if IS_DEMO:
        return _load_demo(periode_id)["votes"]
    try:
        return get_votes(periode_id=periode_id, max_records=5000)
    except Exception:
        return _load_demo(periode_id)["votes"]


def load_actors() -> pd.DataFrame:
    if IS_DEMO:
        return _load_demo(None)["actors"]
    try:
        return get_actors(typeid=5)
    except Exception:
        return _load_demo(None)["actors"]


def load_individual_votes(periode_id: int | None) -> pd.DataFrame:
    if IS_DEMO:
        return _load_demo(periode_id)["individual_votes"]
    try:
        return get_individual_votes(periode_id=periode_id, max_records=50000)
    except Exception:
        return _load_demo(periode_id)["individual_votes"]


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filtrer data")

    if IS_DEMO:
        st.info("Demo-tilstand: viser syntetiske data (API ikke tilgængeligt)")

    periods_df = load_periods()

    if periods_df.empty:
        st.error("Kunne ikke hente samlings-data.")
        selected_periode_id = None
        selected_periode_label = "Alle"
    else:
        periode_options = {
            row["titel"]: row["id"]
            for _, row in periods_df.iterrows()
            if pd.notna(row.get("titel"))
        }
        selected_periode_label = st.selectbox(
            "Valgperiode / samling",
            options=list(periode_options.keys()),
            index=0,
        )
        selected_periode_id = periode_options.get(selected_periode_label)

    st.markdown("---")
    if not IS_DEMO:
        st.markdown(
            "<small style='color:#aaa'>Data: <a href='https://oda.ft.dk' "
            "style='color:#6ec6c8'>Folketing Open Data</a></small>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<small style='color:#aaa'>Grundvederlag: {FOLKETING_ANNUAL_SALARY_DKK:,} DKK/år</small>",
        unsafe_allow_html=True,
    )


# ─── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Henter data..." if not IS_DEMO else "Indlæser demo-data..."):
    meetings_df = load_meetings(selected_periode_id)
    votes_df = load_votes(selected_periode_id)

vote_summary = get_vote_summary(votes_df)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overblik", "Møder", "Afstemninger", "Scoreboard"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERBLIK
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI row
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(
        f'<div class="kpi-card"><div class="label">Antal møder</div>'
        f'<div class="value">{len(meetings_df):,}</div></div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="kpi-card"><div class="label">Antal afstemninger</div>'
        f'<div class="value">{vote_summary["total"]:,}</div></div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        f'<div class="kpi-card kpi-green"><div class="label">Vedtagne forslag</div>'
        f'<div class="value">{vote_summary["vedtaget"]:,}</div></div>',
        unsafe_allow_html=True,
    )
    col4.markdown(
        f'<div class="kpi-card kpi-red"><div class="label">Forkastede forslag</div>'
        f'<div class="value">{vote_summary["forkastet"]:,}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    meetings_by_month = get_meetings_by_month(meetings_df)
    votes_by_month = get_votes_by_month(votes_df, meetings_df)

    with chart_col1:
        st.markdown('<div class="section-title">Møder per måned</div>', unsafe_allow_html=True)
        if not meetings_by_month.empty:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            x_labels = [str(p) for p in meetings_by_month["måned"]]
            ax.bar(x_labels, meetings_by_month["antal_møder"], color=TEAL, width=0.6)
            ax.set_xlabel("")
            ax.set_ylabel("Antal møder")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=45, ha="right", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Ingen mødedata tilgængelig.")

    with chart_col2:
        st.markdown('<div class="section-title">Afstemninger per måned</div>', unsafe_allow_html=True)
        if not votes_by_month.empty:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            x_labels = [str(p) for p in votes_by_month["måned"]]
            ax.bar(x_labels, votes_by_month["antal_afstemninger"], color=DARK, width=0.6)
            ax.set_xlabel("")
            ax.set_ylabel("Antal afstemninger")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=45, ha="right", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Ingen afstemningsdata tilgængelig.")

    # ── Vedtaget vs forkastet donut ────────────────────────────────────────────
    if vote_summary["total"] > 0:
        st.markdown('<div class="section-title">Vedtaget vs. forkastet</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        sizes = [vote_summary["vedtaget"], vote_summary["forkastet"]]
        labels = [
            f"Vedtaget ({vote_summary['vedtaget']:,})",
            f"Forkastet ({vote_summary['forkastet']:,})",
        ]
        colors = [GREEN, RED_VOTE]
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops={"width": 0.55},
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_color("white")
        ax.set_title(f"{selected_periode_label}", fontsize=9, color=DARK)
        plt.tight_layout()
        _, donut_col, _ = st.columns([1, 2, 1])
        donut_col.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MØDER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        f'<div class="section-title">Møder — {selected_periode_label} ({len(meetings_df):,} møder)</div>',
        unsafe_allow_html=True,
    )

    if meetings_df.empty:
        st.info("Ingen møder fundet for den valgte periode.")
    else:
        # Display columns
        show_cols = []
        for c in ["mødenummer", "dato", "titel", "starttidspunkt", "sluttidspunkt", "status"]:
            if c in meetings_df.columns:
                show_cols.append(c)

        display_df = meetings_df[show_cols].copy() if show_cols else meetings_df.copy()

        # Format dates
        for date_col in ["dato", "starttidspunkt", "sluttidspunkt"]:
            if date_col in display_df.columns:
                display_df[date_col] = display_df[date_col].dt.strftime("%d-%m-%Y %H:%M").str.replace(" 00:00", "").str.strip()

        col_labels = {
            "mødenummer": "Nr.",
            "dato": "Dato",
            "titel": "Titel",
            "starttidspunkt": "Start",
            "sluttidspunkt": "Slut",
            "status": "Status",
        }
        display_df = display_df.rename(columns={k: v for k, v in col_labels.items() if k in display_df.columns})

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=500,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AFSTEMNINGER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f'<div class="section-title">Afstemninger — {selected_periode_label} ({vote_summary["total"]:,} afstemninger)</div>',
        unsafe_allow_html=True,
    )

    if votes_df.empty:
        st.info("Ingen afstemninger fundet for den valgte periode.")
    else:
        # Filter
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col2:
            result_filter = st.selectbox(
                "Resultat",
                options=["Alle", "Vedtaget", "Forkastet"],
                index=0,
            )

        filtered_votes = votes_df.copy()
        if result_filter == "Vedtaget" and "vedtaget" in filtered_votes.columns:
            filtered_votes = filtered_votes[filtered_votes["vedtaget"] == True]
        elif result_filter == "Forkastet" and "vedtaget" in filtered_votes.columns:
            filtered_votes = filtered_votes[filtered_votes["vedtaget"] == False]

        # Display columns — prefer konklusion (raw string) for display, fall back to vedtaget bool
        vote_show_cols = []
        for c in ["nummer", "konklusion", "forslag_kortttitel", "vedtaget", "ja", "nej", "hverken", "type"]:
            if c in filtered_votes.columns:
                vote_show_cols.append(c)
        # Don't show both konklusion and vedtaget — prefer konklusion
        if "konklusion" in vote_show_cols and "vedtaget" in vote_show_cols:
            vote_show_cols.remove("vedtaget")

        display_votes = filtered_votes[vote_show_cols].copy() if vote_show_cols else filtered_votes.copy()

        if "vedtaget" in display_votes.columns:
            display_votes["vedtaget"] = display_votes["vedtaget"].map(
                {True: "Vedtaget", False: "Forkastet", 1: "Vedtaget", 0: "Forkastet"}
            ).fillna("Ukendt")

        col_labels_v = {
            "nummer": "Nr.",
            "konklusion": "Resultat",
            "forslag_kortttitel": "Forslag",
            "vedtaget": "Resultat",
            "ja": "Ja",
            "nej": "Nej",
            "hverken": "Hverken",
            "type": "Type",
        }
        display_votes = display_votes.rename(
            columns={k: v for k, v in col_labels_v.items() if k in display_votes.columns}
        )

        st.markdown(f"**{len(display_votes):,} afstemninger vist**")
        st.dataframe(
            display_votes,
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                "Resultat": st.column_config.TextColumn(
                    "Resultat",
                    help="Om forslaget blev vedtaget eller forkastet",
                ),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCOREBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        f'<div class="section-title">Scoreboard — Politikernes aktivitet ({selected_periode_label})</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Beregnet ud fra antal stemmer afgivet og Folketing grundvederlag på "
        f"**{FOLKETING_ANNUAL_SALARY_DKK:,} DKK/år**.",
    )

    with st.spinner("Henter individuelle stemmer og politikerdata..."):
        actors_df = load_actors()
        individual_votes_df = load_individual_votes(selected_periode_id)

    scoreboard_df = build_politician_scoreboard(
        individual_votes_df, actors_df, periode_id=selected_periode_id
    )

    if scoreboard_df.empty:
        st.info("Ingen scoreboard-data tilgængelig for denne periode.")
    else:
        # Filters
        sb_col1, sb_col2, sb_col3 = st.columns([2, 2, 2])
        with sb_col1:
            all_parties = sorted(scoreboard_df["parti"].dropna().unique().tolist())
            selected_parties = st.multiselect(
                "Filtrer på parti",
                options=all_parties,
                default=[],
                placeholder="Alle partier",
            )
        with sb_col2:
            sort_options = {
                "Mest aktiv (flest stemmer)": ("antal_stemmer", False),
                "Mindst aktiv (færrest stemmer)": ("antal_stemmer", True),
                "Højest løn per dag": ("løn_per_dag", False),
                "Lavest løn per dag": ("løn_per_dag", True),
            }
            sort_choice = st.selectbox("Sortér efter", list(sort_options.keys()), index=0)
        with sb_col3:
            navn_search = st.text_input("Søg på navn", placeholder="f.eks. Mette Frederiksen")

        # Apply filters
        sb_display = scoreboard_df.copy()
        if selected_parties:
            sb_display = sb_display[sb_display["parti"].isin(selected_parties)]
        if navn_search:
            sb_display = sb_display[
                sb_display["navn"].str.contains(navn_search, case=False, na=False)
            ]
        sort_col, sort_asc = sort_options[sort_choice]
        sb_display = sb_display.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
        sb_display.index += 1

        # Format løn_per_dag nicely
        sb_display["løn_per_dag"] = sb_display["løn_per_dag"].apply(lambda x: f"{x:,} DKK")

        col_labels_sb = {
            "navn": "Navn",
            "parti": "Parti",
            "antal_stemmer": "Antal stemmer",
            "arbejdsdage": "Arbejdsdage (est.)",
            "løn_per_dag": "Løn per arbejdsdag",
        }
        sb_display = sb_display.rename(columns=col_labels_sb)

        st.markdown(f"**{len(sb_display):,} politikere vist**")
        st.dataframe(
            sb_display,
            use_container_width=True,
            height=550,
            column_config={
                "Navn": st.column_config.TextColumn("Navn", width="medium"),
                "Parti": st.column_config.TextColumn("Parti", width="small"),
                "Antal stemmer": st.column_config.NumberColumn(
                    "Antal stemmer", help="Antal registrerede stemmer i perioden", format="%d"
                ),
                "Arbejdsdage (est.)": st.column_config.NumberColumn(
                    "Arbejdsdage (est.)", help="Estimeret antal dage med aktivitet", format="%d"
                ),
                "Løn per arbejdsdag": st.column_config.TextColumn(
                    "Løn per arbejdsdag",
                    help=f"Grundvederlag ({FOLKETING_ANNUAL_SALARY_DKK:,} DKK/år) divideret med antal arbejdsdage",
                ),
            },
        )

        # Top 10 bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top 10 mest aktive politikere</div>', unsafe_allow_html=True)
        top10 = scoreboard_df.head(10)
        if not top10.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            bars = ax.barh(top10["navn"][::-1], top10["antal_stemmer"][::-1], color=TEAL)
            ax.set_xlabel("Antal stemmer")
            ax.spines[["top", "right"]].set_visible(False)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{int(w):,}", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
