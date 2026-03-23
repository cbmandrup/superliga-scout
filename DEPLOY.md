# Deploy til Streamlit Cloud — Trin-for-trin

## Hvad du skal bruge
- En **GitHub konto** (gratis)
- En **Streamlit Cloud konto** (gratis på share.streamlit.io)

---

## Trin 1: Opret et GitHub repository

1. Gå til [github.com](https://github.com) og log ind
2. Klik **"New repository"** (grøn knap øverst til højre)
3. Navngiv det f.eks. `superliga-scout`
4. Vælg **Private** (så kun du kan se koden)
5. Klik **"Create repository"**

---

## Trin 2: Upload projektet til GitHub

Åbn en terminal i projektmappen og kør:

```bash
cd "C:\Users\cbman\OneDrive\Desktop\Code test\superliga-scout"

git init
git add .
git commit -m "Initial commit: Superliga Scout"
git branch -M main
git remote add origin https://github.com/DIT-BRUGERNAVN/superliga-scout.git
git push -u origin main
```

> Erstat `DIT-BRUGERNAVN` med dit GitHub-brugernavn.

---

## Trin 3: Deploy på Streamlit Cloud

1. Gå til [share.streamlit.io](https://share.streamlit.io)
2. Log ind med din GitHub konto
3. Klik **"New app"**
4. Udfyld formularen:
   - **Repository:** `DIT-BRUGERNAVN/superliga-scout`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Klik **"Deploy!"**

Streamlit installerer automatisk alt fra `requirements.txt` og `packages.txt`.

---

## Hvad sker der ved første opstart?

Siden der ikke er en rigtig database på Streamlit Cloud (filsystemet er midlertidigt),
kører `demo_data.py` automatisk og genererer **200 realistiske syntetiske spillere**
så dashboardet er fuldt interaktivt fra dag 1.

Når du har scraped rigtige data lokalt, kan du:
1. Køre `python -m src.scraper` + `python -m src.pipeline` lokalt
2. Uploade `data/processed/scouting.db` til et cloud storage (f.eks. Supabase gratis tier)
3. Opdatere `app.py` til at hente fra cloud storage

---

## Din app URL

Når den er deployed vil den have en URL som:
```
https://DIT-BRUGERNAVN-superliga-scout-app-XXXXX.streamlit.app
```

Du kan dele dette link med alle i klubben — ingen installation nødvendig.

---

## Opdater appen

Hver gang du pusher til GitHub opdateres appen automatisk:

```bash
git add .
git commit -m "Opdatering"
git push
```

---

## Adgangskontrol (valgfrit)

Vil du have at kun bestemte personer kan se appen?

1. På Streamlit Cloud: gå til app settings → **"Sharing"**
2. Vælg **"Only specific people can view this app"**
3. Tilføj email-adresser

Eller tilføj simpel password-beskyttelse via `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml  (aldrig commit denne fil!)
[auth]
password = "ditHemmeligeKodeord"
```

Og i `app.py` tilføj øverst:
```python
if st.text_input("Password", type="password") != st.secrets["auth"]["password"]:
    st.stop()
```
