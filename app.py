"""
THE BOOT ROOM
Football match probability and value-bet analysis dashboard.

IMPORTANT MODEL DESIGN
----------------------
- xG is entered manually by the user. football-data.org is NOT treated as an
  xG source.
- Historical results are shown as context only; they do not silently replace
  or modify the manually entered xG.
- Match probabilities are generated from a Poisson score model with a
  conservative Dixon-Coles correction for low-scoring results.
- Odds are analysed by market. A complete market is required before a
  de-vigged market probability is used.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Optional

import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="The Boot Room",
    page_icon="⚽",
    layout="wide",
)

BASE_URL = "https://api.football-data.org/v4"
API_TIMEOUT = 20

LIVERPOOL_ID = 64

# Supported competitions requested for the app.
SUPPORTED_COMPETITIONS = {
    "Premier League": "PL",
    "League Cup": "FLC",
    "FA Cup": "FAC",
    "Champions League": "CL",
}

RECENT_MATCHES_N = 6
LONG_TERM_MATCHES_N = 20
MAX_GOALS = 12
CARDS_THRESHOLD = 3.5
DIXON_COLES_RHO = -0.08

# Betting thresholds. Edge is expressed as a probability-point difference.
VALUE_EDGE = 0.05
AVOID_EDGE = -0.03
MIN_VALUE_EV = 0.05


# ============================================================
# API KEY
# ============================================================

def get_api_key() -> str:
    try:
        return str(st.secrets["FOOTBALL_DATA_KEY"]).strip()
    except Exception:
        return os.environ.get("FOOTBALL_DATA_KEY", "").strip()


API_KEY = get_api_key()


# ============================================================
# API
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def api_get(path: str, params: Optional[dict] = None) -> dict:
    """GET a football-data.org endpoint with useful Streamlit-safe errors."""
    if not API_KEY:
        raise RuntimeError("FOOTBALL_DATA_KEY is not configured.")

    try:
        response = requests.get(
            f"{BASE_URL}{path}",
            headers={"X-Auth-Token": API_KEY},
            params=params or {},
            timeout=API_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("The football-data.org API timed out. Please try again.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not connect to football-data.org: {exc}")

    if response.status_code == 401:
        raise RuntimeError("Your FOOTBALL_DATA_KEY was rejected by the API.")
    if response.status_code == 403:
        raise RuntimeError("Your API plan does not have access to this endpoint.")
    if response.status_code == 429:
        raise RuntimeError("The football-data.org API rate limit has been reached.")
    if not response.ok:
        raise RuntimeError(f"Football API error ({response.status_code}).")

    try:
        return response.json()
    except ValueError:
        raise RuntimeError("The football-data.org API returned invalid JSON.")


# ============================================================
# HELPERS
# ============================================================

def parse_datetime(value: Optional[str]):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def poisson_pmf(lam: float, k: int) -> float:
    if lam < 0 or k < 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def parse_odds(value) -> Optional[float]:
    """Accept decimal odds (2.10) or fractional odds (11/10)."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        if "/" in text:
            parts = text.split("/")
            if len(parts) != 2:
                return None
            numerator = float(parts[0].strip())
            denominator = float(parts[1].strip())
            if numerator < 0 or denominator <= 0:
                return None
            decimal_odds = numerator / denominator + 1.0
        else:
            decimal_odds = float(text)

        return decimal_odds if decimal_odds > 1.0 else None
    except (TypeError, ValueError):
        return None


def implied_probability(odds: Optional[float]) -> Optional[float]:
    if odds is None or odds <= 1.0:
        return None
    return 1.0 / odds


def devig_probabilities(odds_map: dict[str, Optional[float]]) -> dict[str, float]:
    """
    Normalise inverse decimal odds to remove the bookmaker overround.

    A complete market is required. This avoids the original behaviour where
    an incomplete market could be treated as if it were a fair market.
    """
    if not odds_map or any(odds is None or odds <= 1 for odds in odds_map.values()):
        return {}

    raw = {key: 1.0 / odds for key, odds in odds_map.items()}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in raw.items()}


def verdict(edge: Optional[float], expected_value: Optional[float]) -> str:
    if edge is None:
        return "NEEDS FULL MARKET"
    if edge >= VALUE_EDGE and expected_value is not None and expected_value >= MIN_VALUE_EV:
        return "VALUE"
    if edge <= AVOID_EDGE or (expected_value is not None and expected_value < 0):
        return "AVOID"
    return "MARGINAL"


def verdict_class(verdict_name: str) -> str:
    return {
        "VALUE": "badge-value",
        "AVOID": "badge-avoid",
        "MARGINAL": "badge-marginal",
        "NEEDS FULL MARKET": "badge-info",
    }.get(verdict_name, "badge-marginal")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# ============================================================
# FIXTURES
# ============================================================

def normalise_fixture(match: dict) -> dict:
    competition = match.get("competition", {})
    home = match.get("homeTeam", {})
    away = match.get("awayTeam", {})

    return {
        "match_id": match.get("id"),
        "competition": competition.get("name", "Unknown Competition"),
        "competition_code": competition.get("code", ""),
        "kickoff": match.get("utcDate"),
        "home_team_id": home.get("id"),
        "home_team_name": home.get("name", "Home Team"),
        "away_team_id": away.get("id"),
        "away_team_name": away.get("name", "Away Team"),
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_next_fixture(team_id: int) -> Optional[dict]:
    data = api_get(
        f"/teams/{team_id}/matches",
        params={"status": "SCHEDULED", "limit": 20},
    )
    matches = data.get("matches", [])
    matches.sort(key=lambda match: match.get("utcDate", ""))
    return normalise_fixture(matches[0]) if matches else None


@st.cache_data(ttl=900, show_spinner=False)
def get_competition_fixtures(competition_code: str, limit: int = 100) -> list[dict]:
    data = api_get(
        f"/competitions/{competition_code}/matches",
        params={"status": "SCHEDULED", "limit": limit},
    )

    fixtures = []
    for match in data.get("matches", []):
        home = match.get("homeTeam", {})
        away = match.get("awayTeam", {})
        if not home.get("id") or not away.get("id"):
            continue
        fixtures.append(normalise_fixture(match))

    fixtures.sort(key=lambda fixture: fixture.get("kickoff") or "")
    return fixtures


@st.cache_data(ttl=86400, show_spinner=False)
def get_competition_teams(competition_code: str) -> list[dict]:
    data = api_get(f"/competitions/{competition_code}/teams")
    teams = []

    for team in data.get("teams", []):
        if team.get("id"):
            teams.append({
                "id": team["id"],
                "name": team.get("name", "Unknown Team"),
            })

    return sorted(teams, key=lambda team: team["name"].lower())


def fixture_label(fixture: dict) -> str:
    kickoff = parse_datetime(fixture.get("kickoff"))
    if kickoff:
        local_kickoff = kickoff.astimezone()
        date_text = local_kickoff.strftime("%a %d %b %Y %H:%M")
    else:
        date_text = "Date TBC"
    return f"{fixture['home_team_name']} vs {fixture['away_team_name']} — {date_text}"


# ============================================================
# HISTORICAL CONTEXT
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_finished_matches(team_id: int, limit: int = 100) -> list[dict]:
    """Return recent completed matches for display/context only."""
    data = api_get(
        f"/teams/{team_id}/matches",
        params={"status": "FINISHED", "limit": limit},
    )
    matches = data.get("matches", [])
    matches.sort(key=lambda match: match.get("utcDate", ""), reverse=True)
    return matches


def extract_team_match_result(match: dict, team_id: int) -> Optional[dict]:
    home = match.get("homeTeam", {})
    away = match.get("awayTeam", {})
    score = match.get("score", {}).get("fullTime", {})

    home_goals = score.get("home")
    away_goals = score.get("away")

    if home_goals is None or away_goals is None:
        return None

    if home.get("id") == team_id:
        scored, conceded, venue = home_goals, away_goals, "H"
    elif away.get("id") == team_id:
        scored, conceded, venue = away_goals, home_goals, "A"
    else:
        return None

    if scored > conceded:
        result = "W"
    elif scored == conceded:
        result = "D"
    else:
        result = "L"

    return {
        "venue": venue,
        "scored": float(scored),
        "conceded": float(conceded),
        "result": result,
    }


def calculate_form(
    matches: list[dict],
    team_id: int,
    venue: Optional[str] = None,
    max_matches: Optional[int] = None,
) -> dict:
    rows = []

    for match in matches:
        result = extract_team_match_result(match, team_id)
        if not result:
            continue
        if venue and result["venue"] != venue:
            continue

        rows.append(result)
        if max_matches and len(rows) >= max_matches:
            break

    if not rows:
        return {
            "matches": 0,
            "scored": None,
            "conceded": None,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }

    return {
        "matches": len(rows),
        "scored": sum(row["scored"] for row in rows) / len(rows),
        "conceded": sum(row["conceded"] for row in rows) / len(rows),
        "wins": sum(row["result"] == "W" for row in rows),
        "draws": sum(row["result"] == "D" for row in rows),
        "losses": sum(row["result"] == "L" for row in rows),
    }


def build_team_context(team_id: int) -> dict:
    matches = get_finished_matches(team_id)
    return {
        "recent": calculate_form(matches, team_id, max_matches=RECENT_MATCHES_N),
        "overall": calculate_form(matches, team_id, max_matches=LONG_TERM_MATCHES_N),
        "home": calculate_form(matches, team_id, venue="H", max_matches=LONG_TERM_MATCHES_N),
        "away": calculate_form(matches, team_id, venue="A", max_matches=LONG_TERM_MATCHES_N),
    }


# ============================================================
# PROBABILITY MODEL
# ============================================================

def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_xg: float,
    away_xg: float,
    rho: float = DIXON_COLES_RHO,
) -> float:
    """Dixon-Coles adjustment for 0-0, 0-1, 1-0 and 1-1."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - (home_xg * away_xg * rho)
    if home_goals == 0 and away_goals == 1:
        return 1.0 + (home_xg * rho)
    if home_goals == 1 and away_goals == 0:
        return 1.0 + (away_xg * rho)
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def compute_model(
    home_xg: float,
    away_xg: float,
    max_goals: int = MAX_GOALS,
) -> tuple[dict[str, float], dict[tuple[int, int], float]]:
    """Build a normalised Poisson/Dixon-Coles score matrix and market probabilities."""
    if home_xg <= 0 or away_xg <= 0:
        raise ValueError("xG values must be greater than zero.")

    raw_matrix = {}
    total = 0.0

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = (
                poisson_pmf(home_xg, home_goals)
                * poisson_pmf(away_xg, away_goals)
                * dixon_coles_tau(home_goals, away_goals, home_xg, away_xg)
            )
            probability = max(0.0, probability)
            raw_matrix[(home_goals, away_goals)] = probability
            total += probability

    if total <= 0:
        raise ValueError("Could not construct the score probability matrix.")

    matrix = {score: probability / total for score, probability in raw_matrix.items()}

    model = {
        "home": 0.0,
        "draw": 0.0,
        "away": 0.0,
        "over25": 0.0,
        "under25": 0.0,
        "bttsY": 0.0,
        "bttsN": 0.0,
    }

    for (home_goals, away_goals), probability in matrix.items():
        if home_goals > away_goals:
            model["home"] += probability
        elif home_goals == away_goals:
            model["draw"] += probability
        else:
            model["away"] += probability

        if home_goals + away_goals >= 3:
            model["over25"] += probability
        else:
            model["under25"] += probability

        if home_goals >= 1 and away_goals >= 1:
            model["bttsY"] += probability
        else:
            model["bttsN"] += probability

    return model, matrix


def cards_over_under(
    expected_cards: float,
    threshold: float = CARDS_THRESHOLD,
) -> tuple[float, float]:
    """Poisson cards model from a manually supplied expected-card total."""
    if expected_cards <= 0:
        raise ValueError("Expected cards must be greater than zero.")

    # For a 3.5 line, under = 0..3 and over = 4+.
    under_max = math.floor(threshold)
    under = sum(poisson_pmf(expected_cards, k) for k in range(under_max + 1))
    over = 1.0 - under
    return max(0.0, over), max(0.0, under)


def most_likely_prediction(model: dict[str, float], home_name: str, away_name: str) -> str:
    outcome = max(
        [(home_name, model["home"]), ("Draw", model["draw"]), (away_name, model["away"])],
        key=lambda item: item[1],
    )
    return outcome[0]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

.stApp { background-color: #0C0F0A; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero { background: linear-gradient(135deg, #1A0508 0%, #0C0F0A 70%); border: 1px solid #2A332A; border-left: 4px solid #C8102E; border-radius: 12px; padding: 24px; margin-bottom: 18px; }
.hero-title { font-family: 'Bebas Neue', sans-serif; font-size: 42px; color: #EDEDE6; line-height: 1; }
.hero-title span { color: #C8102E; }
.hero-sub { color: #8B948A; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 7px; }
.hero-fixture { font-family: 'Bebas Neue', sans-serif; font-size: 30px; color: #E8B33D; margin-top: 16px; }
.hero-meta { color: #8B948A; font-size: 13px; margin-top: 4px; }
.badge { display: inline-block; padding: 5px 11px; border-radius: 12px; font-size: 12px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.badge-value { background: rgba(76, 175, 125, 0.18); color: #4CAF7D; }
.badge-avoid { background: rgba(193, 99, 63, 0.18); color: #C1633F; }
.badge-marginal { background: rgba(139, 148, 138, 0.18); color: #8B948A; }
.badge-info { background: rgba(232, 179, 61, 0.14); color: #E8B33D; }
.market-card { background: #171B14; border: 1px solid #2A332A; border-radius: 9px; padding: 13px 16px; margin-bottom: 8px; }
.market-name { font-weight: 600; font-size: 14px; color: #EDEDE6; }
.model-pct { color: #8B948A; font-size: 12px; font-family: 'IBM Plex Mono', monospace; }
.inspector-card { background: #171B14; border: 1px solid #2A332A; border-radius: 9px; padding: 14px; margin-bottom: 10px; }
.inspector-title { color: #E8B33D; font-weight: 700; font-size: 13px; margin-bottom: 8px; }
.inspector-row { color: #B8C0B5; font-family: 'IBM Plex Mono', monospace; font-size: 12px; margin: 3px 0; }
div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: #E8B33D; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("# ⚽ The Boot Room")
    page = st.radio(
        "Choose analysis",
        ["🏠 Liverpool Next Match", "🔎 Analyse Any Match"],
    )
    st.divider()
    st.markdown(
        "### How to use it\n"
        "1. Choose a match\n"
        "2. Enter home & away xG manually\n"
        "3. Enter odds\n"
        "4. Review probability, edge and EV"
    )
    st.divider()
    st.caption("xG is manual. Historical results are context only.")


# ============================================================
# FIXTURE SELECTION
# ============================================================

def choose_fixture() -> Optional[dict]:
    if page == "🏠 Liverpool Next Match":
        fixture = get_next_fixture(LIVERPOOL_ID)
        if not fixture:
            st.error("Could not find Liverpool's next scheduled fixture.")
            return None
        return fixture

    st.title("Analyse Any Match")

    selection_mode = st.radio(
        "Selection mode",
        ["Upcoming fixture", "Choose teams manually"],
        horizontal=True,
    )

    competition_name = st.selectbox(
        "Competition",
        list(SUPPORTED_COMPETITIONS.keys()),
    )
    competition_code = SUPPORTED_COMPETITIONS[competition_name]

    if selection_mode == "Upcoming fixture":
        fixtures = get_competition_fixtures(competition_code)
        if not fixtures:
            st.warning("No upcoming fixtures are available for this competition.")
            return None

        labels = [fixture_label(fixture) for fixture in fixtures]
        selected_label = st.selectbox("Fixture", labels)
        return fixtures[labels.index(selected_label)]

    teams = get_competition_teams(competition_code)
    if len(teams) < 2:
        st.warning("Not enough teams were returned by the API.")
        return None

    team_by_name = {team["name"]: team for team in teams}
    team_names = list(team_by_name.keys())

    c1, c2 = st.columns(2)
    home_name = c1.selectbox("Home team", team_names)
    away_options = [name for name in team_names if name != home_name]
    away_name = c2.selectbox("Away team", away_options)

    home_team = team_by_name[home_name]
    away_team = team_by_name[away_name]

    return {
        "match_id": None,
        "competition": competition_name,
        "competition_code": competition_code,
        "kickoff": None,
        "home_team_id": home_team["id"],
        "home_team_name": home_team["name"],
        "away_team_id": away_team["id"],
        "away_team_name": away_team["name"],
    }


# ============================================================
# HISTORICAL UI
# ============================================================

def render_team_context(team_name: str, context: dict, venue_key: str) -> None:
    recent = context["recent"]
    overall = context["overall"]
    venue = context[venue_key]

    recent_goals = (
        f"{recent['scored']:.2f} / {recent['conceded']:.2f}"
        if recent["matches"] else "N/A"
    )

    st.markdown(
        f"""
<div class="inspector-card">
<div class="inspector-title">{team_name} — historical context</div>
<div class="inspector-row">Last {recent['matches']}: {recent['wins']}W {recent['draws']}D {recent['losses']}L</div>
<div class="inspector-row">Recent goals: {recent_goals} scored / conceded</div>
<div class="inspector-row">Overall sample: {overall['matches']} matches</div>
<div class="inspector-row">Relevant venue sample: {venue['matches']} matches</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard(fixture: dict) -> None:
    kickoff = parse_datetime(fixture.get("kickoff"))
    kickoff_text = (
        kickoff.astimezone().strftime("%a %d %b %Y, %H:%M")
        if kickoff
        else "Manual match selection"
    )

    st.markdown(
        f"""
<div class="hero">
<div class="hero-title">The Boot <span>Room</span></div>
<div class="hero-sub">Football probability & value analysis</div>
<div class="hero-fixture">{fixture['home_team_name']} vs {fixture['away_team_name']}</div>
<div class="hero-meta">{fixture['competition']} · {kickoff_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # MANUAL MODEL INPUTS
    # --------------------------------------------------------
    st.subheader("Model inputs")
    st.info(
        "xG is entered manually. The data source does not provide fixture-level "
        "xG, so the app does not manufacture xG from historical goals."
    )

    c1, c2 = st.columns(2)
    home_xg = c1.number_input(
        f"{fixture['home_team_name']} xG",
        min_value=0.05,
        max_value=6.00,
        value=1.50,
        step=0.05,
        format="%.2f",
        key=f"xg_home_{fixture['home_team_id']}_{fixture['away_team_id']}",
    )
    away_xg = c2.number_input(
        f"{fixture['away_team_name']} xG",
        min_value=0.05,
        max_value=6.00,
        value=1.20,
        step=0.05,
        format="%.2f",
        key=f"xg_away_{fixture['home_team_id']}_{fixture['away_team_id']}",
    )

    cards_lambda = st.number_input(
        "Expected total cards",
        min_value=0.05,
        max_value=12.0,
        value=4.00,
        step=0.25,
        format="%.2f",
        help="Manual expected total cards. Used only for the cards Poisson model.",
    )

    # --------------------------------------------------------
    # HISTORICAL CONTEXT — deliberately NOT part of xG model
    # --------------------------------------------------------
    with st.expander("Historical context", expanded=False):
        try:
            with st.spinner("Loading recent results..."):
                home_context = build_team_context(fixture["home_team_id"])
                away_context = build_team_context(fixture["away_team_id"])

            c1, c2 = st.columns(2)
            with c1:
                render_team_context(
                    fixture["home_team_name"],
                    home_context,
                    "home",
                )
            with c2:
                render_team_context(
                    fixture["away_team_name"],
                    away_context,
                    "away",
                )

            st.caption(
                "Historical results are diagnostic context only. They do not "
                "change the manually entered xG."
            )
        except RuntimeError as exc:
            st.warning(f"Historical context unavailable: {exc}")

    # --------------------------------------------------------
    # MODEL OUTPUT
    # --------------------------------------------------------
    model, score_matrix = compute_model(home_xg, away_xg)
    cards_over, cards_under = cards_over_under(cards_lambda)
    model["cardsOver"] = cards_over
    model["cardsUnder"] = cards_under

    prediction = most_likely_prediction(
        model,
        fixture["home_team_name"],
        fixture["away_team_name"],
    )

    st.divider()
    st.subheader("Prediction")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Most likely result", prediction)
    c2.metric("Home win", pct(model["home"]))
    c3.metric("Draw", pct(model["draw"]))
    c4.metric("Away win", pct(model["away"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Over 2.5", pct(model["over25"]))
    c2.metric("Under 2.5", pct(model["under25"]))
    c3.metric("BTTS Yes", pct(model["bttsY"]))
    c4.metric("BTTS No", pct(model["bttsN"]))

    # --------------------------------------------------------
    # ODDS / EDGE
    # --------------------------------------------------------
    st.divider()
    st.subheader("Odds & Edge")
    st.caption(
        "Enter all selections in a market for a de-vigged market probability. "
        "Edge = model probability minus de-vigged market probability. "
        "EV = model probability × decimal odds − 1."
    )

    market_groups = {
        "1X2": ("home", "draw", "away"),
        "Goals": ("over25", "under25"),
        "BTTS": ("bttsY", "bttsN"),
        "Cards": ("cardsOver", "cardsUnder"),
    }

    labels = {
        "home": f"{fixture['home_team_name']} Win",
        "draw": "Draw",
        "away": f"{fixture['away_team_name']} Win",
        "over25": "Over 2.5 Goals",
        "under25": "Under 2.5 Goals",
        "bttsY": "BTTS — Yes",
        "bttsN": "BTTS — No",
        "cardsOver": f"Over {CARDS_THRESHOLD} Cards",
        "cardsUnder": f"Under {CARDS_THRESHOLD} Cards",
    }

    results = []

    for group_name, market_keys in market_groups.items():
        st.markdown(f"#### {group_name}")

        odds_map = {}
        input_columns = st.columns(len(market_keys))

        for index, key in enumerate(market_keys):
            odds_map[key] = parse_odds(
                input_columns[index].text_input(
                    labels[key],
                    placeholder="2.10",
                    key=(
                        f"odds_{key}_"
                        f"{fixture['home_team_id']}_"
                        f"{fixture['away_team_id']}"
                    ),
                )
            )

        devigged = devig_probabilities(odds_map)

        for key in market_keys:
            odds = odds_map[key]
            if odds is None:
                continue

            market_probability = devigged.get(key)
            model_probability = model[key]
            edge = (
                model_probability - market_probability
                if market_probability is not None
                else None
            )
            expected_value = model_probability * odds - 1.0
            verdict_name = verdict(edge, expected_value)

            c1, c2, c3, c4, c5 = st.columns([2.3, 1.0, 1.2, 1.1, 1.4])

            c1.markdown(
                f"""
<div class="market-card">
<span class="market-name">{labels[key]}</span><br>
<span class="model-pct">Model: {pct(model_probability)}</span>
</div>
""",
                unsafe_allow_html=True,
            )
            c2.metric("Odds", f"{odds:.2f}")

            if market_probability is None:
                c3.markdown(
                    '<span class="badge badge-info">Complete market needed</span>',
                    unsafe_allow_html=True,
                )
                c4.write("—")
            else:
                c3.metric("De-vig", pct(market_probability))
                c4.metric("Edge", f"{edge * 100:+.1f}%")

            c5.markdown(
                f'<span class="badge {verdict_class(verdict_name)}">'
                f"{verdict_name}</span>",
                unsafe_allow_html=True,
            )

            results.append({
                "key": key,
                "label": labels[key],
                "model_probability": model_probability,
                "odds": odds,
                "market_probability": market_probability,
                "edge": edge,
                "expected_value": expected_value,
                "verdict": verdict_name,
            })

            if market_probability is not None:
                st.caption(
                    f"{labels[key]} · model {pct(model_probability)} · "
                    f"de-vig {pct(market_probability)} · "
                    f"EV {expected_value * 100:+.1f}%"
                )

    value_bets = [result for result in results if result["verdict"] == "VALUE"]

    if value_bets:
        best = max(value_bets, key=lambda result: result["edge"])
        st.success(
            f"Best qualifying edge: **{best['label']}** — "
            f"{best['edge'] * 100:+.1f}% edge, "
            f"{best['expected_value'] * 100:+.1f}% EV at odds {best['odds']:.2f}."
        )
    elif results:
        st.warning("No qualifying VALUE bet found from the supplied odds.")

    # --------------------------------------------------------
    # SCORELINES
    # --------------------------------------------------------
    with st.expander("Most likely scorelines"):
        ranked = sorted(
            score_matrix.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]

        st.dataframe(
            [
                {
                    "Score": f"{home_goals}-{away_goals}",
                    "Probability": pct(probability),
                }
                for (home_goals, away_goals), probability in ranked
            ],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MAIN
# ============================================================

if not API_KEY:
    st.error("FOOTBALL_DATA_KEY isn't set.")
    st.stop()

try:
    fixture = choose_fixture()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if fixture:
    try:
        render_dashboard(fixture)
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
