"""
folketing_demo.py
-----------------
Generates realistic synthetic data for the Folketing Tracker dashboard.

Used automatically by folketing_app.py when the Folketing ODA API is
unreachable (e.g. in sandbox environments or offline use).
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ── Reference data ────────────────────────────────────────────────────────────

PARTIER = {
    "S":   "Socialdemokratiet",
    "V":   "Venstre",
    "M":   "Moderaterne",
    "SF":  "SF",
    "DF":  "Dansk Folkeparti",
    "EL":  "Enhedslisten",
    "RV":  "Radikale Venstre",
    "KF":  "Det Konservative Folkeparti",
    "LA":  "Liberal Alliance",
    "ALT": "Alternativet",
    "DD":  "Danmarksdemokraterne",
    "NB":  "Nye Borgerlige",
}

# Approximate seat counts per party (179 total)
PARTI_SEATS = {
    "S": 50, "V": 23, "M": 16, "SF": 15, "DF": 7, "EL": 13,
    "RV": 7, "KF": 10, "LA": 14, "ALT": 7, "DD": 14, "NB": 3,
}

DANISH_FIRST_NAMES = [
    "Mette", "Lars", "Søren", "Pia", "Jakob", "Rasmus", "Sofie",
    "Peter", "Anne", "Helle", "Henrik", "Jeppe", "Ida", "Christian",
    "Maria", "Torsten", "Trine", "Carsten", "Louise", "Martin",
    "Camilla", "Thomas", "Katrine", "Nicolaj", "Emma", "Frederik",
    "Kirsten", "Jesper", "Birgit", "Ole", "Dorte", "Mikkel",
    "Ane", "Claus", "Stine", "Hans", "Gitte", "Bent", "Lise", "Niels",
]

DANISH_LAST_NAMES = [
    "Jensen", "Nielsen", "Hansen", "Pedersen", "Andersen", "Christensen",
    "Larsen", "Sørensen", "Rasmussen", "Jørgensen", "Petersen", "Madsen",
    "Kristensen", "Olsen", "Thomsen", "Poulsen", "Johansen", "Knudsen",
    "Mortensen", "Møller", "Lund", "Holm", "Dahl", "Kjær", "Vestager",
    "Frederiksen", "Ellemann", "Wammen", "Bæk", "Skaarup",
]

MØDE_TITLER = [
    "Spørgetime", "Aktuel debat", "1. behandling af lovforslag",
    "2. behandling af lovforslag", "3. behandling af lovforslag",
    "Hasteforespørgsel", "Forespørgselsdebat", "Samråd",
    "Redegørelse fra statsministeren", "Åbningsdebat",
    "Finanslovsbehandling", "Udvalgsmøde", "Grundlovsdag",
]

FORSLAG_EMNER = [
    "Lov om ændring af sundhedsloven",
    "Lov om klimaindsatsen",
    "Lov om ændring af udlændingeloven",
    "Lov om børnepasning",
    "Lov om trafikinfrastruktur",
    "Lov om skattereform",
    "Lov om ændring af folkeskoleloven",
    "Lov om pensionsreform",
    "Lov om digitalisering af den offentlige sektor",
    "Lov om boligstøtte",
    "Lov om energiforsyning",
    "Lov om erhvervsfremme",
    "Lov om forsvarsbudget",
    "Lov om integration",
    "Lov om arbejdsmarkedsreform",
    "Beslutningsforslag om udviklingsbistand",
    "Beslutningsforslag om NATO-bidrag",
    "Beslutningsforslag om EU-samarbejde",
    "Lov om dyrevelfærd",
    "Lov om miljøbeskyttelse",
]

AFSTEMNING_TYPER = ["Lovforslag", "Beslutningsforslag", "Hastevedtagelse", "Ændringsforslag"]


# ── Generators ────────────────────────────────────────────────────────────────

def _make_name() -> str:
    return f"{random.choice(DANISH_FIRST_NAMES)} {random.choice(DANISH_LAST_NAMES)}"


def make_periods() -> pd.DataFrame:
    """Return synthetic parliamentary periods (samlinger)."""
    rows = []
    start = date(2019, 10, 1)
    for i in range(6):
        end = start + timedelta(days=364)
        periode_year = start.year
        rows.append({
            "id": 100 + i,
            "titel": f"{periode_year}-{str(periode_year + 1)[-2:]}",
            "startdato": pd.Timestamp(start),
            "slutdato": pd.Timestamp(end),
        })
        start = end + timedelta(days=1)
    return pd.DataFrame(rows).sort_values("startdato", ascending=False).reset_index(drop=True)


def make_meetings(periode_id: int, periods_df: pd.DataFrame) -> pd.DataFrame:
    """Return synthetic meetings for a given period."""
    period_row = periods_df[periods_df["id"] == periode_id]
    if period_row.empty:
        return pd.DataFrame()

    start = period_row["startdato"].iloc[0].date()
    end = min(period_row["slutdato"].iloc[0].date(), date.today())

    rows = []
    current = start
    møde_nr = 1
    while current <= end:
        # Folketing meets ~3 days/week, skip summer recess (Jul–Sep)
        if current.weekday() < 5 and current.month not in (7, 8):
            if random.random() < 0.6:
                rows.append({
                    "id": int(f"{periode_id}{møde_nr:04d}"),
                    "periode_id": periode_id,
                    "mødenummer": møde_nr,
                    "dato": pd.Timestamp(current),
                    "titel": random.choice(MØDE_TITLER),
                    "starttidspunkt": pd.Timestamp(f"{current} 10:00"),
                    "sluttidspunkt": pd.Timestamp(f"{current} {random.randint(13,20):02d}:00"),
                    "status": "Afholdt",
                })
                møde_nr += 1
        current += timedelta(days=1)

    return pd.DataFrame(rows)


def make_votes(meetings_df: pd.DataFrame) -> pd.DataFrame:
    """Return synthetic votes linked to meetings."""
    if meetings_df.empty:
        return pd.DataFrame()

    rows = []
    vote_id = 1
    for _, møde in meetings_df.iterrows():
        n_votes = random.randint(0, 8)
        for i in range(n_votes):
            vedtaget = random.random() < 0.72
            ja = random.randint(90, 120) if vedtaget else random.randint(40, 88)
            nej = 179 - ja - random.randint(0, 10)
            hverken = max(0, 179 - ja - nej)
            rows.append({
                "id": vote_id,
                "møde_id": møde["id"],
                "møde_dato": møde["dato"],
                "nummer": i + 1,
                "konklusion": "Vedtaget" if vedtaget else "Forkastet",
                "vedtaget": vedtaget,
                "forslag_kortttitel": random.choice(FORSLAG_EMNER),
                "type": random.choice(AFSTEMNING_TYPER),
                "ja": ja,
                "nej": nej,
                "hverken": hverken,
            })
            vote_id += 1

    return pd.DataFrame(rows)


def make_actors() -> pd.DataFrame:
    """Return synthetic Folketing members."""
    rows = []
    actor_id = 1
    for kort, _ in PARTIER.items():
        seats = PARTI_SEATS.get(kort, 5)
        for _ in range(seats):
            rows.append({
                "id": actor_id,
                "navn": _make_name(),
                "gruppenavnkort": kort,
                "typeid": 5,
            })
            actor_id += 1
    return pd.DataFrame(rows)


def make_individual_votes(votes_df: pd.DataFrame, actors_df: pd.DataFrame) -> pd.DataFrame:
    """Return synthetic individual politician votes (Stemme)."""
    if votes_df.empty or actors_df.empty:
        return pd.DataFrame()

    rows = []
    stemme_id = 1
    for _, afstemning in votes_df.iterrows():
        # ~85–95% of members vote each time
        participating = actors_df.sample(frac=random.uniform(0.85, 0.98))
        for _, aktør in participating.iterrows():
            # Party line voting with some deviation
            if afstemning["vedtaget"]:
                vote_type = random.choices([1, 2, 3], weights=[75, 18, 7])[0]
            else:
                vote_type = random.choices([1, 2, 3], weights=[22, 68, 10])[0]
            rows.append({
                "id": stemme_id,
                "afstemning_id": afstemning["id"],
                "aktør_id": aktør["id"],
                "typeid": vote_type,  # 1=Ja, 2=Nej, 3=Hverken
                "møde_dato": afstemning.get("møde_dato"),
            })
            stemme_id += 1

    return pd.DataFrame(rows)


# ── Public interface ──────────────────────────────────────────────────────────

def get_demo_data(periode_id: int | None = None) -> dict:
    """
    Return a dict of DataFrames with synthetic Folketing demo data.

    Keys: periods, meetings, votes, actors, individual_votes
    """
    periods = make_periods()
    if periode_id is None and not periods.empty:
        periode_id = int(periods["id"].iloc[0])

    meetings = make_meetings(periode_id, periods)
    votes = make_votes(meetings)
    actors = make_actors()
    individual_votes = make_individual_votes(votes, actors)

    return {
        "periods": periods,
        "meetings": meetings,
        "votes": votes,
        "actors": actors,
        "individual_votes": individual_votes,
        "demo": True,
    }
