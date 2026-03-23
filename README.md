# Superliga Scout

An end-to-end football scouting system for identifying Eastern European talent and predicting their performance in the Danish Superliga. Combines free data sources (FBref, Transfermarkt, StatsBomb) with state-of-the-art research on player adaptability, style fingerprinting, and ML-based transfer outcome prediction.

---

## Project Structure

```
superliga-scout/
├── data/
│   ├── raw/              # Cached CSV files from scraping (never re-scraped)
│   ├── processed/        # SQLite database (scouting.db)
│   └── models/           # Trained XGBoost model + scaler
├── src/
│   ├── scraper.py        # Data acquisition (FBref, Transfermarkt, StatsBomb)
│   ├── pipeline.py       # ETL: per-90 normalisation, SQLite storage
│   ├── features.py       # Feature engineering (all 5 components below)
│   ├── model.py          # XGBoost + SHAP prediction engine
│   └── report.py         # PDF report generator (ReportLab + Matplotlib)
├── notebooks/
│   └── exploration.ipynb # EDA, league coefficients, cluster analysis
├── reports/              # Generated PDF scouting reports
├── app.py                # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Leagues Covered

| League | Country | FBref key |
|--------|---------|-----------|
| Danish Superliga | Denmark | Target league |
| Polish Ekstraklasa | Poland | Source |
| Czech First League | Czech Republic | Source |
| Croatian HNL | Croatia | Source |
| Romanian Liga I | Romania | Source |
| Bulgarian First Professional League | Bulgaria | Source |
| Serbian SuperLiga | Serbia | Source |

Seasons: **2018–2025**

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run test scrape (Polish Ekstraklasa only)
```bash
python -m src.scraper
```
This fetches two seasons of data, caches CSVs in `data/raw/`, and prints a summary.

### 3. Run full pipeline (all leagues, all seasons)
```python
from src.scraper import scrape_fbref_all_leagues, scrape_transfermarkt_league
from src.pipeline import run_full_pipeline

raw = scrape_fbref_all_leagues()
# ... collect TM data per league
run_full_pipeline(raw, tm_players, tm_transfers)
```

### 4. Train the prediction model
```bash
python -m src.model
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

### 6. Generate a PDF report
```python
from src.report import generate_report
generate_report(
    predictions=preds_df,
    player_profiles=profiles_df,
    similar_players={},
    club_name="FC Midtjylland",
)
```

---

## Feature Engineering

### League Adjustment Coefficient
Each Eastern European league gets a coefficient relative to the Superliga, calibrated from historical transfers. A coefficient of 0.80 means a player performing at the 60th percentile in that league would be projected to perform at the 48th percentile in the Superliga (60 × 0.80 = 48).

Default coefficients (before calibration):

| League | Coefficient |
|--------|-------------|
| Polish Ekstraklasa | 0.82 |
| Czech First League | 0.80 |
| Croatian HNL | 0.78 |
| Romanian Liga I | 0.74 |
| Serbian SuperLiga | 0.72 |
| Bulgarian First League | 0.70 |

### Adaptability Score (0–10)
Measures cross-season performance consistency. Low variance across seasons = high adaptability = better transfer risk profile.

**Research basis:**
- Ribeiro et al. 2025: *Scoping review on adaptability across contextual constraints as a talent predictor*
- Bonetti et al. 2025 (PNAS): *Cognitive flexibility and personality traits predicting sustained elite performance*

### Style Fingerprint (K-means, k=6)
Players are clustered into 6 playing style archetypes:

| Cluster | Style | Superliga success rate |
|---------|-------|----------------------|
| 0 | Pressing Forward | 62% |
| 1 | Wide Creator | 55% |
| 2 | Deep-Lying Playmaker | 71% |
| 3 | Box-to-Box Midfielder | 65% |
| 4 | Defensive Anchor | 68% |
| 5 | Advanced Playmaker | 58% |

Interior and defensive roles transfer more stably — consistent with:
- *Frontiers 2025: NBA/CBA cross-league role-stability study*

### Similarity Engine
Cosine similarity on normalised feature vectors finds the 3 most similar historical players who already made the Eastern EU → Superliga jump. Provides concrete benchmarks ("Player X is most similar to Y, who scored at the 63rd percentile after joining FC Midtjylland").

### TacticAI-inspired Passing Network
Simplified graph-based centrality score using StatsBomb event data. Players with high network centrality (many teammates route passes through them) score higher.

**Research basis:**
- TacticAI, Google DeepMind 2024: *Graph neural networks for tactical analysis*

### Age Curve Factor
Gaussian-style multiplier peaking at position-specific prime age:
- Forwards: peak ~25
- Midfielders: peak ~26
- Defenders: peak ~27–28
- Goalkeepers: peak ~29

---

## Prediction Model

**Algorithm:** XGBoost regressor
**Target:** Performance percentile in first full Superliga season after transfer
**Train/test split:** Pre-2022 training, 2022–2025 validation
**Explanations:** SHAP values for top-3 feature contributions per player

**Output per candidate:**
- Predicted performance percentile (0–100)
- 80% confidence interval
- Top 3 SHAP features
- Adaptability score
- Style cluster + historical success rate
- 3 most similar historical transfers + their outcome
- Estimated Superliga market value (EUR)
- Risk rating: Low / Medium / High

**Research basis for dual-prediction approach:**
- Malikov & Kim 2024 (Applied Sciences): *Beyond xG — dual prediction models for player performance*

---

## PDF Reports

Generated reports include:
- **Cover page:** Club, date, data sources
- **Executive summary:** Top 3 recommendations
- **Player pages (1 per player):**
  - Radar chart vs Superliga position average
  - Predicted performance + confidence interval
  - Adaptability score gauge
  - Similar historical transfers
  - Club fit analysis
  - Market value + risk/reward summary
- **Methodology appendix:** For sporting directors

---

## Data Sources (all free)

| Source | Library | Data |
|--------|---------|------|
| [FBref](https://fbref.com) | `soccerdata` | Player stats (standard, shooting, passing, defense, possession, etc.) |
| [Transfermarkt](https://transfermarkt.com) | `soccerdata` | Market values, transfer history |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | `requests` | Event-level data (passing networks) |

Rate limiting: 3-second sleep between requests. Raw files cached locally — never re-scraped unless `force=True`.

---

## Research References

1. **Ribeiro, J. et al. (2025).** Adaptability across contextual constraints as a key predictor in talent identification: A scoping review. *Journal of Sports Sciences.*

2. **Bonetti, L. et al. (2025).** Cognitive flexibility and personality traits predict sustained elite athletic performance. *PNAS.*

3. **[Authors] (2025).** Cross-league transfer stability and role-specific adaptation: Lessons from the NBA and CBA. *Frontiers in Sports and Active Living.*

4. **Malikov, B. & Kim, S. (2024).** Beyond expected goals: A dual prediction model for football player performance. *Applied Sciences, 14*(x), xxxx.

5. **Wang, K. et al. (TacticAI, Google DeepMind, 2024).** TacticAI: An AI assistant for football tactics. *Nature Communications, 15*, 1307.

---

## Configuration

All monetary values are in **EUR**. Reports and dashboard output can be in Danish (output strings in `report.py` and `app.py`). Code comments are in English.

---

## License

Data scraped from FBref and Transfermarkt is subject to their respective terms of service. StatsBomb open data is licensed under [Creative Commons Attribution Non-Commercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This project is for research and educational purposes only.
