"""
The Boot Room — browser UI version.

Run locally with:  python -m streamlit run app.py
(after: pip install streamlit requests)

Locally: set FOOTBALL_DATA_KEY as an environment variable.
On Streamlit Community Cloud: add FOOTBALL_DATA_KEY under app Settings > Secrets instead.
"""

import os
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
SUMMARY_HEADER = ["date", "opponent", "venue", "xg_liv", "xg_opp",
                   "market", "model_prob", "odds", "implied_prob", "edge", "verdict"]

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


def build_row(fixture, xg_liv, xg_opp, market_label, model_p, odds, verdict_str):
    implied = 1 / odds if odds else ""
    edge = (model_p - implied) if odds else ""
    return [datetime.now().strftime("%Y-%m-%d"), fixture["opponent"],
            fixture["venue"], xg_liv, xg_opp, market_label, f"{model_p:.4f}",
            odds or "", f"{implied:.4f}" if odds else "", f"{edge:.4f}" if odds else "",
            verdict_str]

MARKET_GROUPS = [
    ("Match Result", "🥅", ["home", "draw", "away"]),
    ("Goals", "⚽", ["over25", "under25"]),
    ("Both Teams to Score", "🎯", ["bttsY", "bttsN"]),
    ("Discipline", "🟨", ["cardsOver", "cardsUnder"]),
]
LABELS = dict(MARKETS)

CARDS_THRESHOLD = 3.5
LABELS["cardsOver"] = f"Over {CARDS_THRESHOLD} Match Cards"
LABELS["cardsUnder"] = f"Under {CARDS_THRESHOLD} Match Cards"
ALL_MARKET_KEYS = MARKETS + [("cardsOver", LABELS["cardsOver"]), ("cardsUnder", LABELS["cardsUnder"])]


def cards_over_under(lam, threshold=CARDS_THRESHOLD):
    floor_t = int(math.floor(threshold))
    under = sum(poisson_pmf(lam, k) for k in range(floor_t + 1))
    return 1 - under, under

# ---------- page ----------

st.set_page_config(page_title="The Boot Room", page_icon="⚽", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;700&display=swap');

    .stApp { background-color: #0C0F0A; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #1A0508 0%, #0C0F0A 70%);
        border: 1px solid #2A332A; border-left: 4px solid #C8102E;
        border-radius: 10px; padding: 22px 24px; margin-bottom: 18px;
    }
    .hero-title {
        font-family: 'Bebas Neue', sans-serif; font-size: 40px; letter-spacing: 1px;
        color: #EDEDE6; line-height: 1;
    }
    .hero-title span { color: #C8102E; }
    .hero-sub { color: #8B948A; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }
    .hero-fixture { font-family: 'Bebas Neue', sans-serif; font-size: 26px; color: #E8B33D; margin-top: 14px; letter-spacing: 0.4px; }
    .hero-meta { color: #8B948A; font-size: 13px; margin-top: 2px; }

    .badge {
        display:inline-block; padding:4px 11px; border-radius:12px;
        font-size:12px; font-weight:700; letter-spacing:0.03em; font-family:'IBM Plex Mono',monospace;
    }
    .badge-value { background:rgba(76,175,125,0.18); color:#4CAF7D; }
    .badge-avoid { background:rgba(193,99,63,0.18); color:#C1633F; }
    .badge-marginal { background:rgba(139,148,138,0.18); color:#8B948A; }

    .market-card {
        background:#171B14; border:1px solid #2A332A; border-radius:8px;
        padding:12px 16px; margin-bottom:8px; transition: border-color 0.15s;
    }
    .market-name { font-weight:600; font-size:14px; color:#EDEDE6; }
    .model-pct { color:#8B948A; font-size:12px; font-family:'IBM Plex Mono',monospace; }

    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: #E8B33D; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Weekly routine")
    st.markdown(
        "1. Open your bookmarked app URL\n"
        "2. Check the fixture and estimated goals\n"
        "3. Enter this week's odds\n"
        "4. Copy the summary and paste it into your own spreadsheet"
    )

if not API_KEY:
    st.error("FOOTBALL_DATA_KEY isn't set. Add it under app Settings > Secrets (hosted) or as an environment variable (local).")
    st.stop()

fixture = get_next_fixture(LIVERPOOL_ID)
if not fixture:
    st.error("Couldn't find an upcoming fixture. Try again shortly.")
    st.stop()

kickoff_dt = datetime.fromisoformat(fixture["kickoff"].replace("Z", "+00:00"))
venue_word = "vs" if fixture["venue"] == "H" else "at"
now_utc = datetime.now(kickoff_dt.tzinfo)
delta = kickoff_dt - now_utc
days_to_ko = max(delta.days, 0)
hours_to_ko = max(delta.seconds // 3600, 0) if delta.total_seconds() > 0 else 0

st.markdown(f"""
<div class="hero">
    <div class="hero-title">The Boot <span>Room</span></div>
    <div class="hero-sub">Liverpool value-bet model &middot; expected goals vs. the odds</div>
    <div class="hero-fixture">Liverpool {venue_word} {fixture['opponent']}</div>
    <div class="hero-meta">{fixture['competition']} &middot; {kickoff_dt.strftime('%a %d %b, %H:%M UTC')}</div>
</div>
""", unsafe_allow_html=True)

if delta.total_seconds() > 0:
    c1, c2 = st.columns(2)
    c1.metric("Days to kickoff", days_to_ko)
    c2.metric("Hours (remainder)", hours_to_ko)

liv_scored, liv_conceded = recent_scoring_form(LIVERPOOL_ID)
opp_scored, opp_conceded = recent_scoring_form(fixture["opponent_id"])

if fixture["venue"] == "H":
    xg_liv, xg_opp = estimate_match_xg(liv_scored, liv_conceded, opp_scored, opp_conceded)
else:
    xg_opp, xg_liv = estimate_match_xg(opp_scored, opp_conceded, liv_scored, liv_conceded)

with st.expander("⚽ Estimated goals (auto-calculated — click to adjust)", expanded=False):
    col1, col2 = st.columns(2)
    xg_liv = col1.number_input("Liverpool est. goals", min_value=0.0, max_value=6.0,
                                value=float(xg_liv), step=0.05, key="xg_liv_input")
    xg_opp = col2.number_input(f"{fixture['opponent']} est. goals", min_value=0.0, max_value=6.0,
                                value=float(xg_opp), step=0.05, key="xg_opp_input")
    st.caption("Scoring-form proxy, not true xG — adjust if you know about an injury or other team news.")

with st.expander("🟨 Match cards estimate (click to adjust)", expanded=False):
    cards_lambda = st.number_input(
        "Expected total match cards", min_value=0.0, max_value=10.0, value=4.0, step=0.25, key="cards_lambda_input"
    )
    st.caption("No reliable free data source for this, so it's manual — set it based on the referee's average "
               "cards/game and how feisty this fixture tends to be. Premier League matches typically run 3\u20135.")

model = compute_model(xg_liv, xg_opp)
cards_over, cards_under = cards_over_under(cards_lambda)
model["cardsOver"] = cards_over
model["cardsUnder"] = cards_under

st.divider()
st.subheader("Odds & edge")
st.caption("Enter the price for any market — fractions like 5/2, or decimals like 2.10. Leave blank to skip.")

results = []
odds_by_key = {}

tab_labels = [f"{icon} {name}" for name, icon, _ in MARKET_GROUPS]
tabs = st.tabs(tab_labels)

for tab, (group_name, icon, keys) in zip(tabs, MARKET_GROUPS):
    with tab:
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
if st.button("Log this analysis", type="primary"):
    logged, skipped = 0, 0
    dest = "airtable"
    for key, label in ALL_MARKET_KEYS:
        match = next((r for r in results if r[3] == label), None)
        if match:
            dest = log_row(*match)
            logged += 1
        else:
            dest = log_row(fixture, xg_liv, xg_opp, label, model[key], None, "skipped")
            skipped += 1
    where = "your Airtable base" if dest == "airtable" else f"the local file `{LOG_FILE}`"
    st.success(f"Logged {logged} priced market(s) and {skipped} skipped to {where}.")

st.caption("This models likely outcomes from expected-goals/cards inputs — it's not a prediction, and no model beats a well-priced market consistently. Bet only what you can afford to lose.")
