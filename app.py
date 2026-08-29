"""
The Boot Room — browser UI version.

Run locally with:
    python -m streamlit run app.py

Install:
    pip install streamlit requests

Locally:
    set FOOTBALL_DATA_KEY as an environment variable.

Streamlit Community Cloud:
    Add FOOTBALL_DATA_KEY under App Settings > Secrets.

============================================================
MODEL
============================================================

The goal model is deliberately transparent rather than pretending
to be a sophisticated commercial xG model.

It combines:

1. League-relative attacking strength
2. League-relative defensive strength
3. Home/away context
4. Recent form
5. Longer-term form
6. League home advantage
7. Independent Poisson goal distributions

The model does NOT use provider xG because the current free data
source does not provide the required xG dataset.

The result should therefore be thought of as:

    "modelled expected goals"

rather than:

    "true xG"

Cards remain a manually supplied Poisson estimate.

============================================================
RELIABILITY NOTE
============================================================

api_get() never halts the app on a failed request — it returns
None and lets each caller decide what to do. Non-critical data
(league standings) falls back to sane defaults; only a missing
fixture (nothing to show at all) stops the page.

============================================================
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import math
from datetime import datetime, timezone

import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://api.football-data.org/v4"

LIVERPOOL_ID = 64

RECENT_MATCHES_N = 6
LONG_TERM_MATCHES_N = 30

RECENT_WEIGHT = 0.35
LONG_TERM_WEIGHT = 0.65

VENUE_WEIGHT = 0.30

TEAM_STRENGTH_WEIGHT = 0.75
LEAGUE_MEAN_WEIGHT = 0.25

MAX_GOALS = 12

VALUE_EDGE_THRESHOLD = 0.05

CARDS_THRESHOLD = 3.5
DEFAULT_CARDS_LAMBDA = 4.0

REQUEST_TIMEOUT = 10


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="The Boot Room",
    page_icon="⚽",
    layout="centered",
)


# ============================================================
# API KEY
# ============================================================

try:
    API_KEY = st.secrets["FOOTBALL_DATA_KEY"]
except Exception:
    API_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")


# ============================================================
# MARKET DEFINITIONS
# ============================================================

MARKETS = [
    ("home", "Liverpool Win"),
    ("draw", "Draw"),
    ("away", "Opponent Win"),
    ("over25", "Over 2.5 Goals"),
    ("under25", "Under 2.5 Goals"),
    ("bttsY", "BTTS - Yes"),
    ("bttsN", "BTTS - No"),
]

MARKET_GROUPS = [
    ("Match Result", "🥅", ["home", "draw", "away"]),
    ("Goals", "⚽", ["over25", "under25"]),
    ("Both Teams to Score", "🎯", ["bttsY", "bttsN"]),
    ("Discipline", "🟨", ["cardsOver", "cardsUnder"]),
]

LABELS = dict(MARKETS)
LABELS["cardsOver"] = f"Over {CARDS_THRESHOLD:g} Match Cards"
LABELS["cardsUnder"] = f"Under {CARDS_THRESHOLD:g} Match Cards"

ALL_MARKET_KEYS = list(LABELS.keys())

# Kept at 11 columns, same order as the original tracker spreadsheet expects.
# (fair_market_prob still improves the edge calc internally — it's just not
# added as a 12th column here, so pasting into the existing tracker doesn't
# shift verdict/actual_result/stake/profit out of alignment.)
SUMMARY_HEADER = [
    "date", "opponent", "venue", "xg_liv", "xg_opp",
    "market", "model_prob", "odds", "implied_prob", "edge", "verdict",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;700&display=swap');
    .stApp { background-color: #0C0F0A; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .hero {
        background: linear-gradient(135deg, #1A0508 0%, #0C0F0A 70%);
        border: 1px solid #2A332A; border-left: 4px solid #C8102E;
        border-radius: 10px; padding: 22px 24px; margin-bottom: 18px;
    }
    .hero-title { font-family: 'Bebas Neue', sans-serif; font-size: 40px; letter-spacing: 1px; color: #EDEDE6; line-height: 1; }
    .hero-title span { color: #C8102E; }
    .hero-sub { color: #8B948A; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }
    .hero-fixture { font-family: 'Bebas Neue', sans-serif; font-size: 26px; color: #E8B33D; margin-top: 14px; letter-spacing: 0.4px; }
    .hero-meta { color: #8B948A; font-size: 13px; margin-top: 2px; }
    .section-label { color: #8B948A; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px; }
    .model-card { background: #171B14; border: 1px solid #2A332A; border-radius: 8px; padding: 14px 16px; margin-bottom: 8px; }
    .model-card-title { color: #EDEDE6; font-weight: 600; font-size: 14px; }
    .model-card-value { color: #E8B33D; font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 700; margin-top: 3px; }
    .model-card-sub { color: #8B948A; font-size: 11px; margin-top: 3px; }
    .market-card { background: #171B14; border: 1px solid #2A332A; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
    .market-name { font-weight: 600; font-size: 14px; color: #EDEDE6; }
    .model-pct { color: #8B948A; font-size: 12px; font-family: 'IBM Plex Mono', monospace; }
    .badge { display: inline-block; padding: 4px 11px; border-radius: 12px; font-size: 12px; font-weight: 700; letter-spacing: 0.03em; font-family: 'IBM Plex Mono', monospace; }
    .badge-value { background: rgba(76,175,125,0.18); color: #4CAF7D; }
    .badge-avoid { background: rgba(193,99,63,0.18); color: #C1633F; }
    .badge-marginal { background: rgba(139,148,138,0.18); color: #8B948A; }
    .best-value {
        background: linear-gradient(135deg, #172319 0%, #111710 100%);
        border: 1px solid #31543D; border-left: 4px solid #4CAF7D; border-radius: 9px;
        padding: 16px 18px; margin: 10px 0 18px 0;
    }
    .best-value-title { color: #8B948A; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }
    .best-value-market { color: #EDEDE6; font-size: 21px; font-weight: 700; margin-top: 4px; }
    .best-value-edge { color: #4CAF7D; font-family: 'IBM Plex Mono', monospace; font-size: 16px; margin-top: 4px; }
    .warning-box { background: #201C12; border: 1px solid #4A4025; border-radius: 8px; padding: 12px 15px; color: #C9B77A; font-size: 12px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: #E8B33D; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# API  (fix #1: never halts the app — returns None on any failure)
# ============================================================

def api_get(path, params=None):
    """
    Makes a Football-Data.org API request. Returns the parsed JSON on
    success, or None on any failure (timeout, rate limit, HTTP error,
    network error). Callers decide what None means for them — critical
    data (the fixture itself) should stop the page; everything else
    should fall back gracefully.
    """
    if not API_KEY:
        return None

    headers = {"X-Auth-Token": API_KEY}

    try:
        response = requests.get(
            f"{BASE_URL}{path}", headers=headers, params=params or {}, timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            st.toast("Football data API rate limit reached — some data may be using fallback estimates.", icon="⚠️")
            return None
        response.raise_for_status()
        return response.json()

    except requests.Timeout:
        st.toast("Football data service timed out — using fallback estimates where needed.", icon="⚠️")
        return None
    except requests.HTTPError:
        st.toast(f"Football data service returned an error ({response.status_code}).", icon="⚠️")
        return None
    except requests.RequestException:
        st.toast("Couldn't reach the football data service — using fallback estimates where needed.", icon="⚠️")
        return None


# ============================================================
# FIXTURE  (the one place that's still allowed to stop the page —
# there's genuinely nothing useful to show without a fixture)
# ============================================================

@st.cache_data(ttl=3600)
def get_next_fixture(team_id):
    data = api_get(f"/teams/{team_id}/matches", params={"status": "SCHEDULED", "limit": 15})
    if data is None:
        return None

    matches = data.get("matches", [])
    if not matches:
        return None

    matches.sort(key=lambda match: match["utcDate"])
    match = matches[0]
    is_home = match["homeTeam"]["id"] == team_id
    opponent = match["awayTeam"]["name"] if is_home else match["homeTeam"]["name"]
    opponent_id = match["awayTeam"]["id"] if is_home else match["homeTeam"]["id"]
    competition = match.get("competition", {})

    return {
        "opponent": opponent,
        "opponent_id": opponent_id,
        "venue": "H" if is_home else "A",
        "kickoff": match["utcDate"],
        "competition": competition.get("name", "Unknown"),
        "competition_code": competition.get("code"),
        "competition_id": competition.get("id"),
    }


# ============================================================
# TEAM MATCH DATA  (fix #1 continued: empty list on failure,
# summarise_matches already has fallback averages for this case)
# ============================================================

@st.cache_data(ttl=3600)
def get_finished_matches(team_id, limit=LONG_TERM_MATCHES_N):
    data = api_get(f"/teams/{team_id}/matches", params={"status": "FINISHED", "limit": limit})
    if data is None:
        return []
    matches = data.get("matches", [])
    matches.sort(key=lambda match: match["utcDate"], reverse=True)
    return matches


def extract_team_result(match, team_id):
    home_id = match["homeTeam"]["id"]
    away_id = match["awayTeam"]["id"]

    if team_id == home_id:
        venue = "home"
        scored = match["score"]["fullTime"]["home"]
        conceded = match["score"]["fullTime"]["away"]
    elif team_id == away_id:
        venue = "away"
        scored = match["score"]["fullTime"]["away"]
        conceded = match["score"]["fullTime"]["home"]
    else:
        return None

    if scored is None or conceded is None:
        return None

    return {"scored": scored, "conceded": conceded, "venue": venue, "date": match["utcDate"]}


def summarise_matches(matches, team_id):
    rows = [r for r in (extract_team_result(m, team_id) for m in matches) if r]

    if not rows:
        return {
            "matches": 0, "scored": 1.3, "conceded": 1.3,
            "home_matches": 0, "away_matches": 0,
            "home_scored": None, "home_conceded": None,
            "away_scored": None, "away_conceded": None,
        }

    scored = [row["scored"] for row in rows]
    conceded = [row["conceded"] for row in rows]
    home_rows = [row for row in rows if row["venue"] == "home"]
    away_rows = [row for row in rows if row["venue"] == "away"]

    def average(values):
        return sum(values) / len(values) if values else None

    return {
        "matches": len(rows),
        "scored": average(scored),
        "conceded": average(conceded),
        "home_matches": len(home_rows),
        "away_matches": len(away_rows),
        "home_scored": average([r["scored"] for r in home_rows]),
        "home_conceded": average([r["conceded"] for r in home_rows]),
        "away_scored": average([r["scored"] for r in away_rows]),
        "away_conceded": average([r["conceded"] for r in away_rows]),
    }


@st.cache_data(ttl=3600)
def get_team_form(team_id):
    matches = get_finished_matches(team_id, limit=LONG_TERM_MATCHES_N)
    recent_matches = matches[:RECENT_MATCHES_N]
    return {
        "recent": summarise_matches(recent_matches, team_id),
        "long_term": summarise_matches(matches, team_id),
    }


# ============================================================
# LEAGUE DATA  (fix #1 continued: this is the exact function whose
# fallback never used to run — now it actually gets the chance to)
# ============================================================

@st.cache_data(ttl=3600)
def get_competition_standings(competition_code):
    if not competition_code:
        return []
    data = api_get(f"/competitions/{competition_code}/standings")
    if data is None:
        return []
    standings = data.get("standings", [])
    if not standings:
        return []
    return standings[0].get("table", [])


def calculate_league_averages(standings):
    fallback = {"goals_per_team_per_match": 1.35, "home_goals": 1.50, "away_goals": 1.15}
    if not standings:
        return fallback

    total_goals_for = 0
    total_matches = 0
    for row in standings:
        played = row.get("playedGames", 0)
        goals_for = row.get("goalsFor", 0)
        if played > 0:
            total_goals_for += goals_for
            total_matches += played

    if total_matches <= 0:
        return fallback

    goals_per_team_per_match = total_goals_for / total_matches
    return {
        "goals_per_team_per_match": goals_per_team_per_match,
        "home_goals": goals_per_team_per_match * 1.08,
        "away_goals": goals_per_team_per_match * 0.92,
    }


# ============================================================
# STRENGTH MODEL
# ============================================================

def safe_divide(numerator, denominator, fallback=1.0):
    if denominator is None or denominator <= 0:
        return fallback
    return numerator / denominator


def weighted_average(recent, long_term):
    if recent is None:
        return long_term
    if long_term is None:
        return recent
    return recent * RECENT_WEIGHT + long_term * LONG_TERM_WEIGHT


def contextual_average(overall, venue_specific):
    if venue_specific is None:
        return overall
    return overall * (1 - VENUE_WEIGHT) + venue_specific * VENUE_WEIGHT


def build_team_profile(form, league_average):
    recent = form["recent"]
    long_term = form["long_term"]

    overall_scored = weighted_average(recent["scored"], long_term["scored"])
    overall_conceded = weighted_average(recent["conceded"], long_term["conceded"])

    home_scored = contextual_average(overall_scored, long_term["home_scored"])
    home_conceded = contextual_average(overall_conceded, long_term["home_conceded"])
    away_scored = contextual_average(overall_scored, long_term["away_scored"])
    away_conceded = contextual_average(overall_conceded, long_term["away_conceded"])

    league_mean = league_average["goals_per_team_per_match"]

    def shrink(value):
        return value * TEAM_STRENGTH_WEIGHT + 1.0 * LEAGUE_MEAN_WEIGHT

    return {
        "overall_scored": overall_scored, "overall_conceded": overall_conceded,
        "home_scored": home_scored, "home_conceded": home_conceded,
        "away_scored": away_scored, "away_conceded": away_conceded,
        "attack_overall": shrink(safe_divide(overall_scored, league_mean)),
        "defence_overall": shrink(safe_divide(overall_conceded, league_mean)),
        "home_attack": shrink(safe_divide(home_scored, league_mean)),
        "home_defence": shrink(safe_divide(home_conceded, league_mean)),
        "away_attack": shrink(safe_divide(away_scored, league_mean)),
        "away_defence": shrink(safe_divide(away_conceded, league_mean)),
    }


def calculate_expected_goals(team_profile, opponent_profile, team_venue, league_average):
    league_home = league_average["home_goals"]
    league_away = league_average["away_goals"]

    if team_venue == "home":
        team_expected = league_home * team_profile["home_attack"] * opponent_profile["away_defence"]
        opponent_expected = league_away * opponent_profile["away_attack"] * team_profile["home_defence"]
    else:
        team_expected = league_away * team_profile["away_attack"] * opponent_profile["home_defence"]
        opponent_expected = league_home * opponent_profile["home_attack"] * team_profile["away_defence"]

    team_expected = max(0.05, min(team_expected, 6.0))
    opponent_expected = max(0.05, min(opponent_expected, 6.0))
    return round(team_expected, 2), round(opponent_expected, 2)


# ============================================================
# POISSON MODEL
# ============================================================

def poisson_pmf(lam, k):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def compute_model(xg_home, xg_away, max_goals=MAX_GOALS):
    raw_matrix = []
    total_probability = 0.0

    for home_goals in range(max_goals + 1):
        row = []
        for away_goals in range(max_goals + 1):
            probability = poisson_pmf(xg_home, home_goals) * poisson_pmf(xg_away, away_goals)
            row.append(probability)
            total_probability += probability
        raw_matrix.append(row)

    if total_probability <= 0:
        raise ValueError("Could not construct probability model.")

    probabilities = {"home": 0.0, "draw": 0.0, "away": 0.0, "over25": 0.0, "under25": 0.0, "bttsY": 0.0, "bttsN": 0.0}

    for home_goals, row in enumerate(raw_matrix):
        for away_goals, raw_probability in enumerate(row):
            p = raw_probability / total_probability
            if home_goals > away_goals:
                probabilities["home"] += p
            elif home_goals == away_goals:
                probabilities["draw"] += p
            else:
                probabilities["away"] += p
            if home_goals + away_goals >= 3:
                probabilities["over25"] += p
            else:
                probabilities["under25"] += p
            if home_goals >= 1 and away_goals >= 1:
                probabilities["bttsY"] += p
            else:
                probabilities["bttsN"] += p

    return probabilities


# ============================================================
# CARDS
# ============================================================

def cards_over_under(lam, threshold=CARDS_THRESHOLD):
    floor_threshold = int(math.floor(threshold))
    under = sum(poisson_pmf(lam, k) for k in range(floor_threshold + 1))
    return 1 - under, under


# ============================================================
# ODDS
# ============================================================

def parse_odds(value):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        if "/" in value:
            parts = value.split("/")
            if len(parts) != 2:
                return None
            numerator = float(parts[0].strip())
            denominator = float(parts[1].strip())
            if numerator < 0 or denominator <= 0:
                return None
            decimal_odds = (numerator / denominator) + 1
        else:
            decimal_odds = float(value)

        if not math.isfinite(decimal_odds) or decimal_odds <= 1:
            return None
        return decimal_odds
    except (ValueError, ZeroDivisionError):
        return None


def implied_probability(odds):
    if odds is None or odds <= 1:
        return None
    return 1 / odds


def calculate_fair_probabilities(odds_by_key, market_keys):
    available = [key for key in market_keys if key in odds_by_key]
    if len(available) != len(market_keys):
        return {}

    probabilities = {key: implied_probability(odds_by_key[key]) for key in market_keys}
    overround = sum(probabilities.values())
    if overround <= 0:
        return {}

    return {key: probabilities[key] / overround for key in market_keys}


def verdict(edge):
    if edge >= VALUE_EDGE_THRESHOLD:
        return "VALUE"
    if edge <= -VALUE_EDGE_THRESHOLD:
        return "AVOID"
    return "marginal"


# ============================================================
# SUMMARY ROW  (fix #3: back to the 11-column format the tracker
# spreadsheet expects — edge still benefits from the fair-market
# de-vig internally, it's just not added as an extra column)
# ============================================================

def build_summary_row(fixture, xg_liv, xg_opp, market_label, model_p, odds, edge, verdict_str):
    implied = implied_probability(odds)
    return [
        datetime.now().strftime("%Y-%m-%d"),
        fixture["opponent"],
        fixture["venue"],
        xg_liv,
        xg_opp,
        market_label,
        f"{model_p:.4f}",
        f"{odds:.2f}",
        f"{implied:.4f}" if implied is not None else "",
        f"{edge:.4f}",
        verdict_str,
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Weekly routine")
    st.markdown(
        "1. Open your bookmarked app URL\n"
        "2. Check the fixture and model inputs\n"
        "3. Enter this week's odds\n"
        "4. Copy the summary into your spreadsheet"
    )
    st.divider()
    st.markdown("### Model settings")
    st.caption(f"Recent form: {RECENT_MATCHES_N} matches · Long-term: {LONG_TERM_MATCHES_N} matches")
    st.caption(f"Recent weighting: {RECENT_WEIGHT:.0%} · Long-term: {LONG_TERM_WEIGHT:.0%}")
    st.caption(f"VALUE threshold: {VALUE_EDGE_THRESHOLD:.0%}")


# ============================================================
# API KEY CHECK
# ============================================================

if not API_KEY:
    st.error("FOOTBALL_DATA_KEY isn't set. Add it under App Settings > Secrets (hosted) or as an environment variable (local).")
    st.stop()


# ============================================================
# FIXTURE
# ============================================================

fixture = get_next_fixture(LIVERPOOL_ID)
if not fixture:
    st.error("Couldn't find an upcoming fixture, or couldn't reach the football data service. Try refreshing shortly.")
    st.stop()

kickoff_dt = datetime.fromisoformat(fixture["kickoff"].replace("Z", "+00:00"))
now_utc = datetime.now(timezone.utc)
delta = kickoff_dt - now_utc

if delta.total_seconds() > 0:
    total_hours = int(delta.total_seconds() // 3600)
    countdown_text = f"{total_hours // 24}d {total_hours % 24}h to kickoff"
else:
    countdown_text = "Kickoff passed"

venue_word = "vs" if fixture["venue"] == "H" else "at"

st.markdown(f"""
<div class="hero">
    <div class="hero-title">The Boot <span>Room</span></div>
    <div class="hero-sub">Liverpool value-bet model &middot; league-relative expected goals vs. the odds</div>
    <div class="hero-fixture">Liverpool {venue_word} {fixture['opponent']}</div>
    <div class="hero-meta">{fixture['competition']} &middot; {kickoff_dt.strftime('%a %d %b, %H:%M UTC')} &middot; {countdown_text}</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LEAGUE + FORM + PROFILES
# ============================================================

standings = get_competition_standings(fixture["competition_code"])
league_average = calculate_league_averages(standings)

if not standings:
    st.caption("⚠️ League standings weren't available this run — using typical league-average baselines instead of live ones.")

liv_form = get_team_form(LIVERPOOL_ID)
opp_form = get_team_form(fixture["opponent_id"])
liv_profile = build_team_profile(liv_form, league_average)
opp_profile = build_team_profile(opp_form, league_average)

auto_xg_liv, auto_xg_opp = calculate_expected_goals(liv_profile, opp_profile, fixture["venue"], league_average)


# ============================================================
# MODEL INPUT INSPECTOR
# ============================================================

with st.expander("⚽ Model inputs & league-relative strengths", expanded=False):
    st.caption("The model compares each team's scoring and conceding rates against the league baseline. A strength of 1.00 means league average.")
    st.markdown('<div class="section-label">League baseline</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Goals / team / match", f"{league_average['goals_per_team_per_match']:.2f}")
    c2.metric("Home baseline", f"{league_average['home_goals']:.2f}")
    c3.metric("Away baseline", f"{league_average['away_goals']:.2f}")
    st.divider()

    st.markdown('<div class="section-label">Liverpool</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Recent scored", f"{liv_form['recent']['scored']:.2f}")
    c2.metric("Long-term scored", f"{liv_form['long_term']['scored']:.2f}")
    c3.metric("Long-term conceded", f"{liv_form['long_term']['conceded']:.2f}")
    st.write("")
    c1, c2, c3 = st.columns(3)
    if fixture["venue"] == "H":
        c1.metric("Home attack", f"{liv_profile['home_attack']:.2f}")
        c2.metric("Home defence", f"{liv_profile['home_defence']:.2f}")
        c3.metric("Home matches", liv_form["long_term"]["home_matches"])
    else:
        c1.metric("Away attack", f"{liv_profile['away_attack']:.2f}")
        c2.metric("Away defence", f"{liv_profile['away_defence']:.2f}")
        c3.metric("Away matches", liv_form["long_term"]["away_matches"])
    st.divider()

    st.markdown(f'<div class="section-label">{fixture["opponent"]}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Recent scored", f"{opp_form['recent']['scored']:.2f}")
    c2.metric("Long-term scored", f"{opp_form['long_term']['scored']:.2f}")
    c3.metric("Long-term conceded", f"{opp_form['long_term']['conceded']:.2f}")
    st.write("")
    c1, c2, c3 = st.columns(3)
    if fixture["venue"] == "H":
        c1.metric("Away attack", f"{opp_profile['away_attack']:.2f}")
        c2.metric("Away defence", f"{opp_profile['away_defence']:.2f}")
        c3.metric("Away matches", opp_form["long_term"]["away_matches"])
    else:
        c1.metric("Home attack", f"{opp_profile['home_attack']:.2f}")
        c2.metric("Home defence", f"{opp_profile['home_defence']:.2f}")
        c3.metric("Home matches", opp_form["long_term"]["home_matches"])
    st.divider()
    st.caption("Strength interpretation: attack > 1.00 means the team scores more than the league baseline. Defence > 1.00 means the team concedes more than the league baseline, so lower defensive strength is better.")


# ============================================================
# MANUAL EXPECTED GOALS ADJUSTMENT
# ============================================================

with st.expander("🎛 Expected goals — inspect or manually adjust", expanded=False):
    col1, col2 = st.columns(2)
    xg_liv = col1.number_input("Liverpool expected goals", min_value=0.05, max_value=6.0, value=float(auto_xg_liv), step=0.05, key="xg_liv_input")
    xg_opp = col2.number_input(f"{fixture['opponent']} expected goals", min_value=0.05, max_value=6.0, value=float(auto_xg_opp), step=0.05, key="xg_opp_input")
    if abs(xg_liv - auto_xg_liv) > 0.001 or abs(xg_opp - auto_xg_opp) > 0.001:
        st.warning("Manual adjustment active. Model probabilities now use your adjusted figures rather than the automatically calculated values.")
    st.caption("These are modelled expected-goal estimates, not provider xG. Manual adjustments should only be made when you have specific information not reflected in historical data, such as a major injury or suspension.")

with st.expander("🟨 Match cards estimate — manual input", expanded=False):
    cards_lambda = st.number_input("Expected total match cards", min_value=0.0, max_value=10.0, value=DEFAULT_CARDS_LAMBDA, step=0.25, key="cards_lambda_input")
    st.caption("Cards are not currently data-driven. Set this manually using referee tendencies, team discipline and fixture context.")


# ============================================================
# MODEL
# ============================================================

model = compute_model(xg_liv, xg_opp)
cards_over, cards_under = cards_over_under(cards_lambda)
model["cardsOver"] = cards_over
model["cardsUnder"] = cards_under


# ============================================================
# MODEL SNAPSHOT
# ============================================================

st.divider()
st.subheader("Model snapshot")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""<div class="model-card"><div class="model-card-title">Liverpool expected goals</div><div class="model-card-value">{xg_liv:.2f}</div><div class="model-card-sub">League-relative attack/defence model</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="model-card"><div class="model-card-title">{fixture['opponent']} expected goals</div><div class="model-card-value">{xg_opp:.2f}</div><div class="model-card-sub">League-relative attack/defence model</div></div>""", unsafe_allow_html=True)


# ============================================================
# ODDS  (fix #2: collect inputs + placeholders first, compute
# everything ONCE afterward, then fill in the badges — so the
# live badge and the summary table below always agree)
# ============================================================

st.divider()
st.subheader("Odds & edge")
st.caption("Enter decimal odds such as 2.10 or fractional odds such as 5/2. Leave blank to skip. Enter all three 1X2 prices to remove the bookmaker's overround for a fairer edge on those markets.")

odds_strs = {}
badge_placeholders = {}

tabs = st.tabs([f"{icon} {name}" for name, icon, _ in MARKET_GROUPS])

for tab, (group_name, icon, keys) in zip(tabs, MARKET_GROUPS):
    with tab:
        if group_name == "Discipline":
            st.markdown('<div class="warning-box">Cards markets use a manually supplied expected-card estimate. Treat these probabilities as substantially less reliable than the goal model.</div>', unsafe_allow_html=True)
            st.write("")

        for key in keys:
            label = LABELS[key]
            c1, c2, c3 = st.columns([2.2, 1.3, 1.6])
            c1.markdown(f'<div class="market-card"><span class="market-name">{label}</span><br><span class="model-pct">Model: {model[key]*100:.1f}%</span></div>', unsafe_allow_html=True)
            odds_strs[key] = c2.text_input(f"{label} odds", key=f"odds_{key}", label_visibility="collapsed", placeholder="2.10 or 5/2")
            badge_placeholders[key] = c3.empty()
            badge_placeholders[key].markdown('<div style="padding-top:10px"><span class="badge badge-marginal">—</span></div>', unsafe_allow_html=True)

# --- single, unified pass: parse odds, de-vig 1X2, compute verdicts ---

parsed_odds = {}
for key, odds_str in odds_strs.items():
    if odds_str:
        odds = parse_odds(odds_str)
        if odds is not None:
            parsed_odds[key] = odds
        else:
            badge_placeholders[key].markdown('<div style="padding-top:10px;color:#8B948A;font-size:12px;">Invalid odds</div>', unsafe_allow_html=True)

fair_1x2 = calculate_fair_probabilities(parsed_odds, ["home", "draw", "away"])

results = []
value_bets = []

for key, odds in parsed_odds.items():
    model_probability = model[key]
    implied = implied_probability(odds)
    fair_probability = fair_1x2.get(key)
    comparison_probability = fair_probability if fair_probability is not None else implied
    edge = model_probability - comparison_probability
    v = verdict(edge)

    cls = {"VALUE": "badge-value", "AVOID": "badge-avoid", "marginal": "badge-marginal"}[v]
    badge_placeholders[key].markdown(
        f'<div style="padding-top:10px"><span class="badge {cls}">{edge*100:+.1f}% · {v}</span></div>',
        unsafe_allow_html=True,
    )

    result = {
        "key": key, "label": LABELS[key], "model_probability": model_probability,
        "odds": odds, "implied_probability": implied, "fair_probability": fair_probability,
        "edge": edge, "verdict": v,
    }
    results.append(result)
    if v == "VALUE":
        value_bets.append(result)

if fair_1x2:
    overround = sum(implied_probability(parsed_odds[k]) for k in ["home", "draw", "away"])
    st.caption(f"Bookmaker overround on 1X2: {(overround - 1) * 100:.1f}% · edges above use the de-vigged fair-market probability.")


# ============================================================
# BEST VALUE
# ============================================================

if value_bets:
    best = max(value_bets, key=lambda r: r["edge"])
    st.markdown(f"""
<div class="best-value">
    <div class="best-value-title">Best available value</div>
    <div class="best-value-market">{best['label']}</div>
    <div class="best-value-edge">{best['edge']*100:+.1f}% edge &middot; odds {best['odds']:.2f} &middot; model {best['model_probability']*100:.1f}%</div>
</div>
""", unsafe_allow_html=True)
elif results:
    st.info(f"No market currently clears the {VALUE_EDGE_THRESHOLD:.0%} VALUE threshold.")


# ============================================================
# MARKET DETAIL
# ============================================================

if results:
    st.subheader("Market detail")
    display_rows = [{
        "Market": r["label"],
        "Model": f"{r['model_probability']*100:.1f}%",
        "Odds": f"{r['odds']:.2f}",
        "Implied": f"{r['implied_probability']*100:.1f}%",
        "Fair market": f"{r['fair_probability']*100:.1f}%" if r["fair_probability"] is not None else "—",
        "Edge": f"{r['edge']*100:+.1f}%",
        "Verdict": r["verdict"],
    } for r in results]
    st.dataframe(display_rows, use_container_width=True, hide_index=True)


# ============================================================
# COPY SUMMARY
# ============================================================

st.divider()
st.subheader("Copy this week's summary")

if results:
    lines = ["\t".join(SUMMARY_HEADER)]
    for r in results:
        row = build_summary_row(fixture, xg_liv, xg_opp, r["label"], r["model_probability"], r["odds"], r["edge"], r["verdict"])
        lines.append("\t".join(str(v) for v in row))
    summary_text = "\n".join(lines)

    st.caption("Click the copy icon in the top-right of the box below, then paste into your tracker spreadsheet — tab-separated, so it lands in the right columns.")
    st.code(summary_text, language=None)
else:
    st.caption("Enter at least one price above and it will appear here, ready to copy.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "The Boot Room uses a simplified league-relative Poisson model. Expected goals are model estimates, "
    "not provider xG, and don't fully account for injuries, line-ups, tactical changes, transfers, fixture "
    "congestion, or other qualitative information. Cards are manually estimated. Model edge is an estimate, "
    "not a guarantee of value or profit. No model beats a well-priced market consistently. Bet only what "
    "you can afford to lose."
)
