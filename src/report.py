"""
report.py
---------
Generates professional PDF scouting reports using ReportLab + Matplotlib.

Output structure
----------------
  Page 1: Cover + Executive Summary (top 3 recommendations)
  Pages 2+: One page per player:
    - Radar chart (vs Superliga position average)
    - Predicted performance + CI bar
    - Adaptability score gauge
    - "Players like him" similarity table
    - Club fit analysis
    - Contract / market value info
    - Risk/reward summary
  Final page: Methodology appendix (for sporting directors)

Usage
-----
  from src.report import generate_report
  generate_report(
      predictions=predictions_df,        # from model.predict_all_candidates
      player_profiles=player_profiles_df,
      similar_players=similarity_dict,   # {player_name: sim_df}
      club_name="FC København",
      output_path="reports/fck_scouting_2025.pdf",
  )
"""

from __future__ import annotations

import io
import math
import textwrap
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = A4          # 595 × 842 pt
MARGIN        = 1.8 * cm
CONTENT_W     = PAGE_W - 2 * MARGIN

# Brand colours (Danish Superliga feel)
C_RED    = colors.HexColor("#C8102E")
C_NAVY   = colors.HexColor("#0D1B40")
C_GOLD   = colors.HexColor("#E8A000")
C_LIGHT  = colors.HexColor("#F5F5F5")
C_MID    = colors.HexColor("#CCCCCC")
C_WHITE  = colors.white
C_BLACK  = colors.black
C_GREEN  = colors.HexColor("#2ECC71")
C_ORANGE = colors.HexColor("#E67E22")

RISK_COLORS = {"Low": C_GREEN, "Medium": C_ORANGE, "High": C_RED}

RADAR_METRICS = [
    ("standard_xg_adj",          "xG/90"),
    ("standard_xa_adj",          "xA/90"),
    ("goal_shot_creation_sca90_adj", "SCA/90"),
    ("passing_prog_p_adj",       "Prog Passes"),
    ("possession_prg_c_adj",     "Prog Carries"),
    ("defense_press_adj",        "Pressures"),
    ("defense_tkl_adj",          "Tackles"),
]

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title", fontSize=22, textColor=C_WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontSize=13, textColor=C_GOLD,
            fontName="Helvetica", alignment=TA_CENTER, spaceAfter=6
        ),
        "section": ParagraphStyle(
            "section", fontSize=11, textColor=C_NAVY,
            fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4,
            borderPadding=(2, 0, 2, 0),
        ),
        "body": ParagraphStyle(
            "body", fontSize=9, textColor=C_BLACK,
            fontName="Helvetica", leading=14,
        ),
        "small": ParagraphStyle(
            "small", fontSize=7.5, textColor=colors.grey,
            fontName="Helvetica", leading=11,
        ),
        "player_name": ParagraphStyle(
            "player_name", fontSize=16, textColor=C_NAVY,
            fontName="Helvetica-Bold", spaceAfter=2,
        ),
        "stat_label": ParagraphStyle(
            "stat_label", fontSize=8, textColor=colors.grey,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "stat_value": ParagraphStyle(
            "stat_value", fontSize=14, textColor=C_NAVY,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "risk_low":    ParagraphStyle("risk_low",    fontSize=10, textColor=C_GREEN,  fontName="Helvetica-Bold"),
        "risk_medium": ParagraphStyle("risk_medium", fontSize=10, textColor=C_ORANGE, fontName="Helvetica-Bold"),
        "risk_high":   ParagraphStyle("risk_high",   fontSize=10, textColor=C_RED,    fontName="Helvetica-Bold"),
        "toc_entry": ParagraphStyle(
            "toc_entry", fontSize=9, textColor=C_NAVY,
            fontName="Helvetica", leading=14,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Matplotlib figures → ReportLab Image
# ---------------------------------------------------------------------------

def _fig_to_image(fig: plt.Figure, width: float, height: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width, height=height)


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

def _radar_chart(
    player_vals: list[float],
    superliga_avg: list[float],
    labels: list[str],
    player_name: str,
) -> plt.Figure:
    N = len(labels)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    p_vals = player_vals + player_vals[:1]
    s_vals = superliga_avg + superliga_avg[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True),
                           facecolor="white")
    ax.set_facecolor("#F8F8F8")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7, color=C_NAVY.hexval()[1:] if hasattr(C_NAVY, 'hexval') else "#0D1B40")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], size=5, color="grey")
    ax.grid(color="lightgrey", linewidth=0.5)

    # Superliga average
    ax.fill(angles, s_vals, alpha=0.15, color="#AAAAAA")
    ax.plot(angles, s_vals, color="#AAAAAA", linewidth=1.5, linestyle="--", label="Superliga avg")

    # Player
    ax.fill(angles, p_vals, alpha=0.25, color="#C8102E")
    ax.plot(angles, p_vals, color="#C8102E", linewidth=2, label=player_name)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    ax.set_title(f"{player_name}\nvs Superliga Average", size=8,
                 fontweight="bold", color="#0D1B40", pad=15)
    return fig


# ---------------------------------------------------------------------------
# Adaptability gauge
# ---------------------------------------------------------------------------

def _adaptability_gauge(score: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2.5, 1.4), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background bar
    bar_bg = mpatches.FancyBboxPatch((0, 0.3), 10, 0.4,
                                      boxstyle="round,pad=0.05",
                                      facecolor="#EEEEEE", edgecolor="none")
    ax.add_patch(bar_bg)

    # Colour gradient: red → yellow → green
    cmap = plt.get_cmap("RdYlGn")
    color = cmap(score / 10)
    bar_fill = mpatches.FancyBboxPatch((0, 0.3), score, 0.4,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor="none")
    ax.add_patch(bar_fill)

    # Score text
    ax.text(5, 0.9, "Adaptability Score", ha="center", va="top",
            fontsize=7, color="grey")
    ax.text(score, 0.05, f"{score:.1f}/10", ha="center", va="bottom",
            fontsize=8, fontweight="bold",
            color="#2ECC71" if score >= 7 else "#E67E22" if score >= 4 else "#C8102E")
    return fig


# ---------------------------------------------------------------------------
# Confidence interval bar
# ---------------------------------------------------------------------------

def _ci_bar(predicted: float, lower: float, upper: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.5, 1.0), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # CI range
    ax.barh(0.5, upper - lower, left=lower, height=0.35,
            color="#AED6F1", align="center")
    # Point estimate
    ax.plot([predicted, predicted], [0.3, 0.7], color="#C8102E", linewidth=2)

    ax.text(predicted, 0.9, f"{predicted:.0f}th pct", ha="center", va="top",
            fontsize=8, fontweight="bold", color="#C8102E")
    ax.text(lower, 0.05, f"{lower:.0f}", ha="center", va="bottom", fontsize=7, color="grey")
    ax.text(upper, 0.05, f"{upper:.0f}", ha="center", va="bottom", fontsize=7, color="grey")
    ax.text(50, -0.2, "Predicted performance percentile in Superliga",
            ha="center", va="top", fontsize=6, color="grey")

    # Tick lines
    for v in [25, 50, 75]:
        ax.axvline(v, color="#DDDDDD", linewidth=0.5, ymin=0.2, ymax=0.8)

    return fig


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _cover_page(styles: dict, club_name: str, n_players: int, report_date: str) -> list:
    elements = []

    # Header block (navy background — faked with a wide Table)
    header_data = [[Paragraph(
        f"<b>SUPERLIGA SCOUT</b>", styles["title"]
    )]]
    header_table = Table(header_data, colWidths=[CONTENT_W])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph(
        f"Spillerrapport — Østeuropæiske Talenter", styles["subtitle"]
    ))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(HRFlowable(width=CONTENT_W, thickness=2, color=C_GOLD))
    elements.append(Spacer(1, 0.5 * cm))

    meta = [
        ["Klub:", club_name],
        ["Rapport dato:", report_date],
        ["Antal kandidater:", str(n_players)],
        ["Datakilde:", "FBref / Transfermarkt / StatsBomb"],
        ["Model version:", "XGBoost v2 + SHAP"],
    ]
    meta_table = Table(meta, colWidths=[4 * cm, CONTENT_W - 4 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), C_NAVY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(meta_table)
    elements.append(PageBreak())
    return elements


def _exec_summary(
    styles: dict,
    top_players: pd.DataFrame,
) -> list:
    elements = []
    elements.append(Paragraph("Direktionsresumé", styles["section"]))
    elements.append(HRFlowable(width=CONTENT_W, thickness=1, color=C_GOLD))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Følgende tre spillere anbefales som primære mål baseret på "
        "forudsagt præstation, tilpasningsdygtighed og stilmæssigt match "
        "med Superligaens krav:", styles["body"]
    ))
    elements.append(Spacer(1, 0.4 * cm))

    for rank, (_, row) in enumerate(top_players.head(3).iterrows(), start=1):
        risk_style = f"risk_{row.get('risk_rating', 'medium').lower()}"
        risk_label = row.get("risk_rating", "Medium")
        pred       = row.get("predicted_pct", 0)
        name       = row.get("player", "Ukendt")
        league     = row.get("league_key", "").replace("_", " ").title()
        adapt      = row.get("adaptability_score", 5)
        style_lbl  = row.get("style_label", "Ukendt")
        val        = row.get("estimated_market_value_eur", 0)

        card_data = [
            [Paragraph(f"#{rank}  {name}", styles["player_name"]),
             Paragraph(f"Risiko: <b>{risk_label}</b>", styles.get(risk_style, styles["body"]))],
            [Paragraph(f"Liga: {league}  |  Stil: {style_lbl}", styles["small"]), ""],
            [Paragraph(
                f"Forudsagt percentil: <b>{pred:.0f}th</b>  |  "
                f"Tilpasning: <b>{adapt:.1f}/10</b>  |  "
                f"Est. markedsværdi: <b>€{val:,.0f}</b>",
                styles["body"]
            ), ""],
        ]
        card_table = Table(
            card_data,
            colWidths=[CONTENT_W * 0.72, CONTENT_W * 0.28]
        )
        card_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_LIGHT),
            ("BOX",           (0, 0), (-1, -1), 1, C_MID),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("SPAN",          (0, 2), (1, 2)),
            ("SPAN",          (0, 1), (1, 1)),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",         (1, 0), (1, 0), "RIGHT"),
        ]))
        elements.append(card_table)
        elements.append(Spacer(1, 0.25 * cm))

    elements.append(PageBreak())
    return elements


def _player_page(
    styles: dict,
    pred_row: pd.Series,
    profile_row: Optional[pd.Series],
    similar_df: Optional[pd.DataFrame],
    superliga_averages: dict[str, float],
    club_name: str,
) -> list:
    elements = []

    name       = pred_row.get("player", "Ukendt")
    league     = pred_row.get("league_key", "").replace("_", " ").title()
    season     = pred_row.get("season", "")
    age        = pred_row.get("age", "?")
    pos        = profile_row.get("pos_broad", "?") if profile_row is not None else "?"
    predicted  = float(pred_row.get("predicted_pct", 0))
    ci_lower   = float(pred_row.get("ci_lower", predicted - 10))
    ci_upper   = float(pred_row.get("ci_upper", predicted + 10))
    adapt      = float(pred_row.get("adaptability_score", 5))
    style_lbl  = pred_row.get("style_label", "Ukendt")
    cluster_sr = float(pred_row.get("cluster_historical_success", 0.65))
    risk       = pred_row.get("risk_rating", "Medium")
    est_val    = int(pred_row.get("estimated_market_value_eur", 0))
    coeff      = float(pred_row.get("league_coeff", 0.75))
    shap_feat  = pred_row.get("top_shap_features", [])

    # --- Header ---
    risk_color = RISK_COLORS.get(risk, C_ORANGE)
    header_data = [[
        Paragraph(f"<b>{name}</b>", styles["player_name"]),
        Paragraph(
            f"<font color='white'><b> {risk} RISK </b></font>",
            ParagraphStyle("rh", fontSize=9, fontName="Helvetica-Bold",
                           textColor=C_WHITE, alignment=TA_CENTER,
                           backColor=risk_color, borderPadding=4)
        ),
    ]]
    hdr = Table(header_data, colWidths=[CONTENT_W * 0.78, CONTENT_W * 0.22])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_NAVY),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (1, 0), (1, 0), "RIGHT"),
    ]))
    elements.append(hdr)

    # Sub-header: league, age, position
    sub_data = [[Paragraph(
        f"Liga: <b>{league}</b>  |  Sæson: <b>{season}</b>  |  "
        f"Alder: <b>{age}</b>  |  Position: <b>{pos}</b>  |  "
        f"Ligakoefficient: <b>{coeff:.2f}</b>",
        styles["small"]
    )]]
    sub_tbl = Table(sub_data, colWidths=[CONTENT_W])
    sub_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_LIGHT),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(sub_tbl)
    elements.append(Spacer(1, 0.3 * cm))

    # --- Two-column layout: radar | prediction + adapt ---
    # Build radar chart
    if profile_row is not None:
        player_vals = []
        sl_vals     = []
        radar_labels = []
        for col, label in RADAR_METRICS:
            pct_col = col.replace("_adj", "_adj_pct") if "_adj" in col else col + "_pct"
            # Use the percentile value (already in 0-100)
            p_val = float(profile_row.get(pct_col, profile_row.get(col, 50) or 50))
            s_val = float(superliga_averages.get(pct_col, 50))
            player_vals.append(np.clip(p_val, 0, 100))
            sl_vals.append(np.clip(s_val, 0, 100))
            radar_labels.append(label)
    else:
        player_vals = [50] * len(RADAR_METRICS)
        sl_vals     = [50] * len(RADAR_METRICS)
        radar_labels = [lbl for _, lbl in RADAR_METRICS]

    radar_fig = _radar_chart(player_vals, sl_vals, radar_labels, name)
    radar_img = _fig_to_image(radar_fig, width=7 * cm, height=7 * cm)

    ci_fig  = _ci_bar(predicted, ci_lower, ci_upper)
    ci_img  = _fig_to_image(ci_fig, width=8.5 * cm, height=2.2 * cm)

    adapt_fig = _adaptability_gauge(adapt)
    adapt_img = _fig_to_image(adapt_fig, width=8.5 * cm, height=1.8 * cm)

    right_content = [
        Paragraph("Forudsagt Superliga-præstation", styles["section"]),
        ci_img,
        Spacer(1, 0.3 * cm),
        Paragraph("Tilpasningsdygtighed", styles["section"]),
        adapt_img,
        Spacer(1, 0.2 * cm),
        Paragraph(
            f"Stilprofil: <b>{style_lbl}</b><br/>"
            f"Historisk succesrate for stiltype: <b>{cluster_sr:.0%}</b>",
            styles["body"]
        ),
    ]

    two_col = Table(
        [[radar_img, right_content]],
        colWidths=[7.2 * cm, CONTENT_W - 7.2 * cm],
    )
    two_col.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (1, 0), (1, 0), 12),
    ]))
    elements.append(two_col)
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C_MID))

    # --- SHAP explanation ---
    elements.append(Paragraph("Vigtigste forudsigelsesdrivere (SHAP)", styles["section"]))
    if shap_feat:
        shap_rows = [["Feature", "SHAP-bidrag", "Retning"]]
        for feat, val in shap_feat:
            direction = "▲ Positiv" if val > 0 else "▼ Negativ"
            shap_rows.append([
                feat.replace("_", " ").title(),
                f"{val:+.4f}",
                direction,
            ])
        shap_tbl = Table(shap_rows, colWidths=[CONTENT_W * 0.55, CONTENT_W * 0.20, CONTENT_W * 0.25])
        shap_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
            ("GRID",          (0, 0), (-1, -1), 0.25, C_MID),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(shap_tbl)
    elements.append(Spacer(1, 0.3 * cm))

    # --- Similar players ---
    if similar_df is not None and not similar_df.empty:
        elements.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C_MID))
        elements.append(Paragraph("Sammenlignelige spillere (Eastern EU → Superliga)", styles["section"]))
        sim_rows = [["Spiller", "Liga", "Sæson", "Superliga percentil", "Lighed"]]
        for _, sr in similar_df.head(3).iterrows():
            sim_rows.append([
                sr.get("player", "?"),
                sr.get("league_key", "?").replace("_", " ").title(),
                str(sr.get("transfer_season", "?")),
                f"{sr.get('target_performance_pct', 0):.0f}th",
                f"{sr.get('similarity_score', 0):.2f}",
            ])
        sim_tbl = Table(sim_rows, colWidths=[
            CONTENT_W * 0.28, CONTENT_W * 0.18, CONTENT_W * 0.14,
            CONTENT_W * 0.22, CONTENT_W * 0.18
        ])
        sim_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
            ("GRID",          (0, 0), (-1, -1), 0.25, C_MID),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(sim_tbl)
        elements.append(Spacer(1, 0.3 * cm))

    # --- Contract / value + Risk/Reward ---
    elements.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C_MID))
    summary_data = [
        [
            Paragraph("Estimeret markedsværdi (Superliga)", styles["stat_label"]),
            Paragraph("Risikovurdering",                    styles["stat_label"]),
            Paragraph("Tilpasnings-score",                  styles["stat_label"]),
        ],
        [
            Paragraph(f"€{est_val:,.0f}", styles["stat_value"]),
            Paragraph(f"<b>{risk}</b>",
                      styles.get(f"risk_{risk.lower()}", styles["body"])),
            Paragraph(f"{adapt:.1f} / 10", styles["stat_value"]),
        ],
    ]
    sum_tbl = Table(summary_data, colWidths=[CONTENT_W / 3] * 3)
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_LIGHT),
        ("BOX",           (0, 0), (-1, -1), 1, C_MID),
        ("INNERGRID",     (0, 0), (-1, -1), 0.25, C_MID),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(sum_tbl)

    # --- Club fit ---
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C_MID))
    elements.append(Paragraph(f"Klubmatch: {club_name}", styles["section"]))
    fit_text = (
        f"{name} spiller i stilen <b>{style_lbl}</b>, som historisk set "
        f"tilpasser sig med <b>{cluster_sr:.0%}</b> succesrate til Superligaen. "
        f"Spillerens tilpasningsdygtighed på <b>{adapt:.1f}/10</b> indikerer "
        f"{'høj robusthed over for systemskift' if adapt >= 7 else 'moderat tilpasningsevne' if adapt >= 4 else 'potentiel tilpasningsudfordring'}. "
        f"Estimeret præstationspercentil i {club_name}s system: <b>{predicted:.0f}th</b>."
    )
    elements.append(Paragraph(fit_text, styles["body"]))
    elements.append(PageBreak())
    return elements


def _methodology_appendix(styles: dict) -> list:
    elements = []
    elements.append(Paragraph("Metodologi — For Sportslige Direktører", styles["section"]))
    elements.append(HRFlowable(width=CONTENT_W, thickness=1, color=C_GOLD))
    elements.append(Spacer(1, 0.4 * cm))

    sections = [
        ("Datakilder", (
            "Spillerstatistik indsamles fra FBref via soccerdata-biblioteket "
            "(standard-, skud-, afleverings-, besiddelses- og forsvarsmålinger). "
            "Markedsværdier og transferhistorik hentes fra Transfermarkt. "
            "StatsBomb Open Data bruges til kalibrering af hændelsesniveaumodellen."
        )),
        ("Ligakoefficient", (
            "En ligajusteringsfaktor beregnes for hver Østeuropæisk liga baseret på "
            "historiske transferer: spillere som gik fra liga L til Superligaen. "
            "Koefficienten er medianen af (Superliga-præstation / præ-transfer-præstation). "
            "Dette normaliserer statistikker til Superliga-niveau."
        )),
        ("Tilpasningsdygtighed (Ribeiro et al. 2025 / Bonetti et al. 2025)", (
            "Baseret på Ribeiro et al. 2025 scoping review om kontekstuel tilpasning "
            "og Bonetti et al. 2025 (PNAS) om kognitiv fleksibilitet som prædiktor for "
            "elite-præstation. Scoren måler konsistens på tværs af sæsoner (lav variance = høj tilpasning)."
        )),
        ("Stilklynger (Frontiers 2025)", (
            "Spillere grupperes i 6 stilarkityper via K-means clustering. "
            "Baseret på Frontiers 2025-studiet (NBA/CBA krydsligaanalyse): "
            "indre og defensive roller viser mere stabil adaptation end perimeterroller."
        )),
        ("Forudsigelsesmodel (XGBoost + SHAP)", (
            "XGBoost-regressionsmodel trænet på historiske Eastern EU → Superliga-transferer "
            "(2015-2022). Valideret på 2022-2025. SHAP-værdier forklarer de 3 vigtigste drivere "
            "for hver spiller. Dual-prediction tilgang inspireret af Malikov & Kim 2024."
        )),
        ("Lighedssøgning", (
            "Cosinus-similaritet på normaliserede featurevektorer identificerer de 3 "
            "mest lignende spillere som allerede har foretaget springet fra Østeuropa til Superligaen."
        )),
        ("TacticAI-netværkscentralitet (DeepMind 2024)", (
            "Inspireret af TacticAI (Google DeepMind 2024): en forenklet passing-netværks "
            "centralitetsscore beregnes fra StatsBomb event-data for at modellere "
            "spillerrelationer som en graf."
        )),
        ("Risikovurdering", (
            "Low/Medium/High baseret på: modelusikkerhed (CI-bredde), "
            "tilpasningsdygtighed, stilklyngens historiske succesrate, og alder "
            "(spillere udenfor 20-28 aldersvinduet får højere risiko)."
        )),
    ]

    for title, text in sections:
        elements.append(Paragraph(title, ParagraphStyle(
            "meth_title", fontSize=9, fontName="Helvetica-Bold",
            textColor=C_NAVY, spaceBefore=8, spaceAfter=2
        )))
        elements.append(Paragraph(text, styles["body"]))

    return elements


# ---------------------------------------------------------------------------
# Main PDF generator
# ---------------------------------------------------------------------------

def generate_report(
    predictions: pd.DataFrame,
    player_profiles: pd.DataFrame,
    similar_players: dict[str, pd.DataFrame],
    club_name: str = "Danish Superliga Club",
    output_path: Optional[str] = None,
    max_players: int = 10,
) -> str:
    """
    Generate a full scouting PDF report.

    Parameters
    ----------
    predictions     : output of model.predict_all_candidates
    player_profiles : enriched player profiles (from features.build_all_features)
    similar_players : {player_name: similarity_df} from features.find_similar_players
    club_name       : name of the requesting club
    output_path     : where to save the PDF (default: reports/<date>_<club>.pdf)
    max_players     : max number of player pages to include

    Returns
    -------
    str : path to generated PDF
    """
    if output_path is None:
        reports_dir = Path(__file__).resolve().parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        safe_club = club_name.replace(" ", "_").replace("/", "-")
        output_path = str(reports_dir / f"{date.today()}_{safe_club}.pdf")

    logger.info(f"Generating PDF report → {output_path}")

    styles = _build_styles()
    report_date = date.today().strftime("%d. %B %Y")

    # Compute Superliga averages for radar chart baseline
    superliga_profiles = player_profiles[
        player_profiles.get("league_key", pd.Series()) == "superliga"
    ] if "league_key" in player_profiles.columns else pd.DataFrame()

    superliga_avg: dict[str, float] = {}
    for col, _ in RADAR_METRICS:
        pct_col = col.replace("_adj", "_adj_pct") if "_adj" in col else col + "_pct"
        if not superliga_profiles.empty and pct_col in superliga_profiles.columns:
            superliga_avg[pct_col] = float(superliga_profiles[pct_col].median())
        else:
            superliga_avg[pct_col] = 50.0  # default: median

    # Build document
    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )
    frame = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    all_elements = []

    # Cover
    all_elements.extend(_cover_page(styles, club_name, min(max_players, len(predictions)), report_date))

    # Executive summary
    all_elements.extend(_exec_summary(styles, predictions.head(max_players)))

    # Player pages
    top_players = predictions.head(max_players)
    for _, pred_row in top_players.iterrows():
        player_name = pred_row.get("player", "")
        league_key  = pred_row.get("league_key", "")
        season      = pred_row.get("season", "")

        # Look up the full profile row
        mask = (
            (player_profiles["player"] == player_name) &
            (player_profiles["league_key"] == league_key)
        ) if "player" in player_profiles.columns else pd.Series([False] * len(player_profiles))

        if "season" in player_profiles.columns:
            mask &= (player_profiles["season"] == season)

        profile_rows = player_profiles[mask]
        profile_row  = profile_rows.iloc[0] if not profile_rows.empty else None

        sim_df = similar_players.get(player_name, pd.DataFrame())

        all_elements.extend(
            _player_page(
                styles, pred_row, profile_row,
                sim_df, superliga_avg, club_name
            )
        )

    # Methodology appendix
    all_elements.extend(_methodology_appendix(styles))

    doc.build(all_elements)
    logger.success(f"Report saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sqlite3

    db = Path(__file__).resolve().parent.parent / "data" / "processed" / "scouting.db"
    if not db.exists():
        print("No database found. Run the pipeline first.")
    else:
        with sqlite3.connect(db) as conn:
            try:
                preds    = pd.read_sql("SELECT * FROM predictions", conn)
                profiles = pd.read_sql("SELECT * FROM player_profiles", conn)
            except Exception as e:
                print(f"Table missing: {e}. Run model.py first.")
                preds = profiles = pd.DataFrame()

        if not preds.empty:
            path = generate_report(
                predictions=preds,
                player_profiles=profiles,
                similar_players={},
                club_name="Test FC",
                max_players=3,
            )
            print(f"Report: {path}")
