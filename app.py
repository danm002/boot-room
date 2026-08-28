"""
The Boot Room — browser UI version.

Run locally with:  python -m streamlit run app.py
(after: pip install streamlit requests)

Locally: set FOOTBALL_DATA_KEY as an environment variable.
On Streamlit Community Cloud: add FOOTBALL_DATA_KEY under app Settings > Secrets instead.
"""

import os
import csv
import math
from datetime import datetime

import requests
import streamlit as st

try:
    API_KEY = st.secrets["FOOTBALL_DATA_KEY"]
except Exception:
    API_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")

BASE_URL = "https://api.football-data.org/v4"
LIVERPOOL_ID = 64
RECENT_MATCHES_N = 6
LOG_FILE = "value_bet_log.csv"

MARKETS = [
    ("home", "Liverpool Win"),
    ("draw", "Draw"),
    ("away", "Opponent Win"),
    ("over25", "Over 2.5 Goals"),
    ("under25", "Under 2.5 Goals"),
    ("bttsY", "BTTS - Yes"),
    ("bttsN", "BTTS - No"),
]

# ---------- data + model (same logic as the terminal version) ----------

def api_get(path, params=None):
    headers = {"X-Auth-Token": API_KEY}
    resp = requests.get(f"{BASE_URL}{path}", headers=headers, params=params or {})
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600)
def get_next_fixture(team_id):
    data = api_get(f"/teams/{team_id}/matches", params={"status": "SCHEDULED", "limit": 15})
    matches = data.get("matches", [])
    if not matches:
        return None
    matches.sort(key=lambda m: m["utcDate"])
    m = matches[0]
    is_home = m["homeTeam"]["id"] == team_id
    opponent = m["awayTeam"]["name"] if is_home else m["homeTeam"]["name"]
    opponent_id = m["awayTeam"]["id"] if is_home else m["homeTeam"]["id"]
    return {
        "opponent": opponent,
        "opponent_id": opponent_id,
        "venue": "H" if is_home else "A",
        "kickoff": m["utcDate"],
        "competition": m.get("competition", {}).get("name", "Unknown"),
    }


@st.cache_data(ttl=3600)
def recent_scoring_form(team_id, n=RECENT_MATCHES_N):
    data = api_get(f"/teams/{team_id}/matches", params={"status": "FINISHED", "limit": n})
    matches = data.get("matches", [])
    if not matches:
        return 1.3, 1.3
    scored, conceded = [], []
    for m in matches:
        is_home = m["homeTeam"]["id"] == team_id
        gs = m["score"]["fullTime"]["home"] if is_home else m["score"]["fullTime"]["away"]
        gc = m["score"]["fullTime"]["away"] if is_home else m["score"]["fullTime"]["home"]
        if gs is not None and gc is not None:
            scored.append(gs)
            conceded.append(gc)
    if not scored:
        return 1.3, 1.3
    return sum(scored) / len(scored), sum(conceded) / len(conceded)


def estimate_match_xg(liv_scored, liv_conceded, opp_scored, opp_conceded):
    liv_xg = (liv_scored + opp_conceded) / 2
    opp_xg = (opp_scored + liv_conceded) / 2
    return round(liv_xg, 2), round(opp_xg, 2)


def poisson_pmf(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def compute_model(xg_home, xg_away, max_goals=10):
    home = draw = away = over25 = under25 = bttsY = bttsN = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson_pmf(xg_home, i) * poisson_pmf(xg_away, j)
            if i > j: home += p
            elif i == j: draw += p
            else: away += p
            if i + j >= 3: over25 += p
            else: under25 += p
            if i >= 1 and j >= 1: bttsY += p
            else: bttsN += p
    return {"home": home, "draw": draw, "away": away, "over25": over25,
            "under25": under25, "bttsY": bttsY, "bttsN": bttsN}


def verdict(edge):
    if edge > 0.03: return "VALUE"
    if edge < -0.03: return "AVOID"
    return "marginal"


def log_row(fixture, xg_liv, xg_opp, market_label, model_p, odds, verdict_str):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date_logged", "opponent", "venue", "xg_liv", "xg_opp",
                              "market", "model_prob", "odds", "implied_prob", "edge", "verdict"])
        implied = 1 / odds if odds else ""
        edge = (model_p - implied) if odds else ""
        writer.writerow([datetime.now().isoformat(timespec="seconds"), fixture["opponent"],
                          fixture["venue"], xg_liv, xg_opp, market_label, f"{model_p:.4f}",
                          odds or "", f"{implied:.4f}" if odds else "", f"{edge:.4f}" if odds else "",
                          verdict_str])

MARKET_GROUPS = [
    ("Match Result", ["home", "draw", "away"]),
    ("Goals", ["over25", "under25"]),
    ("Both Teams to Score", ["bttsY", "bttsN"]),
]
LABELS = dict(MARKETS)

# ---------- page ----------

st.set_page_config(page_title="The Boot Room", page_icon="⚽", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0C0F0A; }
    .badge {
        display:inline-block; padding:3px 10px; border-radius:12px;
        font-size:12px; font-weight:700; letter-spacing:0.03em;
    }
    .badge-value { background:rgba(76,175,125,0.18); color:#4CAF7D; }
    .badge-avoid { background:rgba(193,99,63,0.18); color:#C1633F; }
    .badge-marginal { background:rgba(139,148,138,0.18); color:#8B948A; }
    .market-card {
        background:#171B14; border:1px solid #2A332A; border-radius:8px;
        padding:12px 16px; margin-bottom:8px;
    }
    .market-name { font-weight:600; font-size:14px; }
    .model-pct { color:#8B948A; font-size:12px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Weekly routine")
    st.markdown(
        "1. Open PowerShell\n"
        "2. Set your API key\n"
        "3. `cd` into betting-tool\n"
        "4. `python -m streamlit run app.py`\n"
        "5. Enter this week's odds\n"
        "6. Click **Log this analysis**"
    )
    st.divider()
    st.caption(f"Results are appended to `{LOG_FILE}` in this folder — open it in Excel any time to review your history.")

st.title("⚽ The Boot Room")
st.caption("Liverpool value-bet model — expected goals vs. the odds")

if not API_KEY:
    st.error("FOOTBALL_DATA_KEY isn't set. Close this, set it in your terminal the same way as before, then run `python -m streamlit run app.py` again.")
    st.stop()

fixture = get_next_fixture(LIVERPOOL_ID)
if not fixture:
    st.error("Couldn't find an upcoming fixture. Try again shortly.")
    st.stop()

kickoff_dt = datetime.fromisoformat(fixture["kickoff"].replace("Z", "+00:00"))
venue_word = "vs" if fixture["venue"] == "H" else "at"
st.subheader(f"Liverpool {venue_word} {fixture['opponent']}")
st.write(f"{fixture['competition']} · {kickoff_dt.strftime('%a %d %b, %H:%M UTC')}")

liv_scored, liv_conceded = recent_scoring_form(LIVERPOOL_ID)
opp_scored, opp_conceded = recent_scoring_form(fixture["opponent_id"])

if fixture["venue"] == "H":
    xg_liv, xg_opp = estimate_match_xg(liv_scored, liv_conceded, opp_scored, opp_conceded)
else:
    xg_opp, xg_liv = estimate_match_xg(opp_scored, opp_conceded, liv_scored, liv_conceded)

with st.expander("Estimated goals (auto-calculated — click to adjust)", expanded=False):
    col1, col2 = st.columns(2)
    xg_liv = col1.number_input("Liverpool est. goals", min_value=0.0, max_value=6.0,
                                value=float(xg_liv), step=0.05, key="xg_liv_input")
    xg_opp = col2.number_input(f"{fixture['opponent']} est. goals", min_value=0.0, max_value=6.0,
                                value=float(xg_opp), step=0.05, key="xg_opp_input")
    st.caption("Scoring-form proxy, not true xG — adjust if you know about an injury or other team news.")

model = compute_model(xg_liv, xg_opp)

st.divider()
st.subheader("Odds & edge")
st.caption("Enter the price for any market — fractions like 5/2, or decimals like 2.10. Leave blank to skip.")

results = []
odds_by_key = {}

for group_name, keys in MARKET_GROUPS:
    st.markdown(f"**{group_name}**")
    for key in keys:
        label = LABELS[key]
        c1, c2, c3 = st.columns([2.2, 1.3, 1.6])
        c1.markdown(f'<div class="market-card"><span class="market-name">{label}</span><br>'
                     f'<span class="model-pct">Model: {model[key]*100:.1f}%</span></div>', unsafe_allow_html=True)
        odds_str = c2.text_input(f"{label} odds", key=f"odds_{key}", label_visibility="collapsed", placeholder="2.10 or 5/2")
        badge_html = '<span class="badge badge-marginal">—</span>'
        if odds_str:
            try:
                odds_str = odds_str.strip()
                if "/" in odds_str:
                    num, den = odds_str.split("/")
                    odds = (float(num) / float(den)) + 1
                else:
                    odds = float(odds_str)
                implied = 1 / odds
                edge = model[key] - implied
                v = verdict(edge)
                cls = {"VALUE": "badge-value", "AVOID": "badge-avoid", "marginal": "badge-marginal"}[v]
                badge_html = f'<span class="badge {cls}">{edge*100:+.1f}% · {v}</span>'
                results.append((fixture, xg_liv, xg_opp, label, model[key], odds, v))
                odds_by_key[key] = (odds, implied, edge, v)
            except ValueError:
                pass
        c3.markdown(f'<div style="padding-top:10px">{badge_html}</div>', unsafe_allow_html=True)

value_bets = [(k, data) for k, data in odds_by_key.items() if data[3] == "VALUE"]
if value_bets:
    best_key, best_data = max(value_bets, key=lambda x: x[1][2])
    st.success(f"Best value found: **{LABELS[best_key]}** — edge {best_data[2]*100:+.1f}% at odds {best_data[0]:.2f}")

st.divider()
if st.button("Log this analysis to CSV", type="primary"):
    logged, skipped = 0, 0
    for key, label in MARKETS:
        match = next((r for r in results if r[3] == label), None)
        if match:
            log_row(*match)
            logged += 1
        else:
            log_row(fixture, xg_liv, xg_opp, label, model[key], None, "skipped")
            skipped += 1
    st.success(f"Logged {logged} priced market(s) and {skipped} skipped to {LOG_FILE}.")

st.caption("This models likely outcomes from expected-goals inputs — it's not a prediction, and no model beats a well-priced market consistently. Bet only what you can afford to lose.")
