"""
THE BOOT ROOM
A football match probability and value-bet analysis dashboard.

Run locally:
    pip install streamlit requests
    python -m streamlit run app.py

Environment variable:
    FOOTBALL_DATA_KEY=your_api_key

Or Streamlit secrets:
    FOOTBALL_DATA_KEY = "your_api_key"

Important:
This is a probability model, not a prediction engine or guaranteed betting system.
"""

import os
import math
from datetime import datetime, timezone

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

LIVERPOOL_ID = 64

RECENT_MATCHES_N = 6
LONG_TERM_MATCHES_N = 20

# Regression strength:
# Higher values = more conservative estimates.
REGRESSION_MATCHES = 8

HOME_ADVANTAGE_FALLBACK = 1.08

MAX_GOALS = 12

CARDS_THRESHOLD = 3.5

SUMMARY_HEADER = [
    "date",
    "competition",
    "home_team",
    "away_team",
    "kickoff_utc",
    "home_xg",
    "away_xg",
    "market",
    "model_probability",
    "odds",
    "market_probability",
    "edge",
    "verdict",
]

MARKET_GROUPS = [
    (
        "Match Result",
        "🥅",
        ["home", "draw", "away"],
    ),
    (
        "Goals",
        "⚽",
        ["over25", "under25"],
    ),
    (
        "Both Teams to Score",
        "🎯",
        ["bttsY", "bttsN"],
    ),
    (
        "Discipline",
        "🟨",
        ["cardsOver", "cardsUnder"],
    ),
]


# ============================================================
# API KEY
# ============================================================

try:
    API_KEY = st.secrets["FOOTBALL_DATA_KEY"]
except Exception:
    API_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")


# ============================================================
# API
# ============================================================

def api_get(path, params=None):
    """
    Generic football-data.org GET request.
    """

    if not API_KEY:
        raise RuntimeError(
            "FOOTBALL_DATA_KEY is not configured."
        )

    headers = {
        "X-Auth-Token": API_KEY,
    }

    try:
        response = requests.get(
            f"{BASE_URL}{path}",
            headers=headers,
            params=params or {},
            timeout=20,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "The football data API timed out. Please try again."
        )

    except requests.exceptions.HTTPError as exc:

        status = exc.response.status_code

        if status == 401:
            raise RuntimeError(
                "Your FOOTBALL_DATA_KEY was rejected by the API."
            )

        if status == 403:
            raise RuntimeError(
                "Your football-data.org API plan does not appear to have access "
                "to this competition or endpoint."
            )

        if status == 429:
            raise RuntimeError(
                "The football data API rate limit has been reached. "
                "Please wait and try again."
            )

        raise RuntimeError(
            f"Football API error ({status})."
        )

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to the football API: {exc}"
        )


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_datetime(value):
    """
    Convert API UTC timestamp into timezone-aware datetime.
    """

    if not value:
        return None

    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def weighted_average(values, weights=None):
    if not values:
        return None

    if weights is None:
        return sum(values) / len(values)

    total_weight = sum(weights)

    if total_weight == 0:
        return None

    return sum(
        value * weight
        for value, weight in zip(values, weights)
    ) / total_weight


def poisson_pmf(lam, k):
    """
    Poisson probability mass function.
    """

    if lam < 0:
        return 0.0

    return (
        math.exp(-lam)
        * (lam ** k)
        / math.factorial(k)
    )


def verdict(edge):
    """
    Conservative edge classification.
    """

    if edge >= 0.05:
        return "VALUE"

    if edge <= -0.03:
        return "AVOID"

    return "MARGINAL"


def verdict_class(verdict_name):
    return {
        "VALUE": "badge-value",
        "AVOID": "badge-avoid",
        "MARGINAL": "badge-marginal",
    }.get(
        verdict_name,
        "badge-marginal",
    )


# ============================================================
# ODDS
# ============================================================

def parse_odds(value):
    """
    Supports:
        2.10
        5/2

    Returns decimal odds.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:

        if "/" in value:

            parts = value.split("/")

            if len(parts) != 2:
                return None

            numerator = float(parts[0].strip())
            denominator = float(parts[1].strip())

            if denominator <= 0:
                return None

            decimal_odds = (
                numerator / denominator
            ) + 1

        else:

            decimal_odds = float(value)

        if decimal_odds <= 1:
            return None

        return decimal_odds

    except (ValueError, TypeError):
        return None


def implied_probability(odds):
    if odds is None or odds <= 1:
        return None

    return 1 / odds


def fair_probabilities_from_market(odds_map):
    """
    Removes bookmaker margin from a complete 1X2 market.

    odds_map:
        {
            "home": 2.10,
            "draw": 3.60,
            "away": 3.40
        }
    """

    required = ["home", "draw", "away"]

    if not all(
        odds_map.get(key)
        for key in required
    ):
        return {}

    raw_probs = {
        key: 1 / odds_map[key]
        for key in required
    }

    total = sum(raw_probs.values())

    if total <= 0:
        return {}

    return {
        key: probability / total
        for key, probability in raw_probs.items()
    }


# ============================================================
# COMPETITIONS
# ============================================================

@st.cache_data(ttl=86400)
def get_competitions():
    """
    Fetch available competitions.
    """

    data = api_get("/competitions")

    competitions = data.get(
        "competitions",
        [],
    )

    cleaned = []

    for competition in competitions:

        code = competition.get("code")

        if not code:
            continue

        cleaned.append(
            {
                "id": competition.get("id"),
                "code": code,
                "name": competition.get(
                    "name",
                    code,
                ),
                "area": competition.get(
                    "area",
                    {},
                ).get(
                    "name",
                    "",
                ),
            }
        )

    return sorted(
        cleaned,
        key=lambda x: x["name"],
    )


# ============================================================
# FIXTURES
# ============================================================

@st.cache_data(ttl=1800)
def get_next_fixture(team_id):
    """
    Finds the next scheduled fixture for a team.
    """

    data = api_get(
        f"/teams/{team_id}/matches",
        params={
            "status": "SCHEDULED",
            "limit": 20,
        },
    )

    matches = data.get(
        "matches",
        [],
    )

    if not matches:
        return None

    matches.sort(
        key=lambda m: m.get(
            "utcDate",
            "",
        )
    )

    match = matches[0]

    return normalise_fixture(match)


@st.cache_data(ttl=1800)
def get_competition_fixtures(
    competition_code,
    limit=80,
):
    """
    Returns upcoming fixtures for a competition.
    """

    data = api_get(
        f"/competitions/{competition_code}/matches",
        params={
            "status": "SCHEDULED",
            "limit": limit,
        },
    )

    matches = data.get(
        "matches",
        [],
    )

    fixtures = []

    for match in matches:

        home = match.get(
            "homeTeam",
            {},
        )

        away = match.get(
            "awayTeam",
            {},
        )

        if not home.get("id"):
            continue

        if not away.get("id"):
            continue

        fixtures.append(
            normalise_fixture(match)
        )

    fixtures.sort(
        key=lambda x: x.get(
            "kickoff",
            "",
        )
    )

    return fixtures


def normalise_fixture(match):
    """
    Converts every fixture into one consistent structure.

    This is the core architectural improvement:
    everything always explicitly has a home team
    and away team.
    """

    competition = match.get(
        "competition",
        {},
    )

    home = match.get(
        "homeTeam",
        {},
    )

    away = match.get(
        "awayTeam",
        {},
    )

    return {
        "match_id": match.get("id"),

        "competition": competition.get(
            "name",
            "Unknown Competition",
        ),

        "competition_code": competition.get(
            "code",
            "",
        ),

        "kickoff": match.get(
            "utcDate",
        ),

        "home_team_id": home.get("id"),

        "home_team_name": home.get(
            "name",
            "Home Team",
        ),

        "away_team_id": away.get("id"),

        "away_team_name": away.get(
            "name",
            "Away Team",
        ),
    }


# ============================================================
# TEAMS
# ============================================================

@st.cache_data(ttl=86400)
def get_competition_teams(
    competition_code,
):
    """
    Fetch teams in a competition.
    """

    data = api_get(
        f"/competitions/{competition_code}/teams"
    )

    teams = data.get(
        "teams",
        [],
    )

    cleaned = []

    for team in teams:

        if not team.get("id"):
            continue

        cleaned.append(
            {
                "id": team["id"],
                "name": team.get(
                    "name",
                    "Unknown Team",
                ),
            }
        )

    return sorted(
        cleaned,
        key=lambda x: x["name"],
    )


# ============================================================
# TEAM MATCH DATA
# ============================================================

@st.cache_data(ttl=1800)
def get_finished_matches(
    team_id,
    limit=30,
):
    """
    Fetch finished matches for a team.
    """

    data = api_get(
        f"/teams/{team_id}/matches",
        params={
            "status": "FINISHED",
            "limit": limit,
        },
    )

    matches = data.get(
        "matches",
        [],
    )

    matches.sort(
        key=lambda m: m.get(
            "utcDate",
            ""
        ),
        reverse=True,
    )

    return matches


def extract_team_match_result(
    match,
    team_id,
):
    """
    Returns a team's scored/conceded goals
    and venue for one match.
    """

    home = match.get(
        "homeTeam",
        {},
    )

    away = match.get(
        "awayTeam",
        {},
    )

    score = match.get(
        "score",
        {},
    ).get(
        "fullTime",
        {},
    )

    home_goals = score.get("home")
    away_goals = score.get("away")

    if (
        home_goals is None
        or away_goals is None
    ):
        return None

    if home.get("id") == team_id:

        return {
            "venue": "H",
            "scored": home_goals,
            "conceded": away_goals,
        }

    if away.get("id") == team_id:

        return {
            "venue": "A",
            "scored": away_goals,
            "conceded": home_goals,
        }

    return None


def calculate_form(
    matches,
    team_id,
    venue=None,
    max_matches=None,
):
    """
    Calculates scored/conceded averages.

    Optional:
        venue="H"
        venue="A"
    """

    scored = []
    conceded = []

    for match in matches:

        result = extract_team_match_result(
            match,
            team_id,
        )

        if not result:
            continue

        if (
            venue
            and result["venue"] != venue
        ):
            continue

        scored.append(
            result["scored"]
        )

        conceded.append(
            result["conceded"]
        )

        if (
            max_matches
            and len(scored) >= max_matches
        ):
            break

    if not scored:

        return {
            "matches": 0,
            "scored": None,
            "conceded": None,
        }

    return {
        "matches": len(scored),

        "scored": (
            sum(scored)
            / len(scored)
        ),

        "conceded": (
            sum(conceded)
            / len(conceded)
        ),
    }


def blended_form(
    recent,
    long_term,
    recent_weight=0.65,
):
    """
    Blends recent form with longer-term form.
    """

    if (
        recent["matches"] == 0
        and long_term["matches"] == 0
    ):
        return {
            "scored": None,
            "conceded": None,
        }

    if recent["matches"] == 0:
        return {
            "scored": long_term["scored"],
            "conceded": long_term["conceded"],
        }

    if long_term["matches"] == 0:
        return {
            "scored": recent["scored"],
            "conceded": recent["conceded"],
        }

    return {
        "scored": (
            recent_weight
            * recent["scored"]
            + (1 - recent_weight)
            * long_term["scored"]
        ),

        "conceded": (
            recent_weight
            * recent["conceded"]
            + (1 - recent_weight)
            * long_term["conceded"]
        ),
    }


def build_team_profile(
    team_id,
    venue,
):
    """
    Builds a team profile for the venue
    they will play in.
    """

    matches = get_finished_matches(
        team_id,
        limit=LONG_TERM_MATCHES_N + 10,
    )

    overall_recent = calculate_form(
        matches,
        team_id,
        max_matches=RECENT_MATCHES_N,
    )

    overall_long = calculate_form(
        matches,
        team_id,
        max_matches=LONG_TERM_MATCHES_N,
    )

    venue_recent = calculate_form(
        matches,
        team_id,
        venue=venue,
        max_matches=RECENT_MATCHES_N,
    )

    venue_long = calculate_form(
        matches,
        team_id,
        venue=venue,
        max_matches=LONG_TERM_MATCHES_N,
    )

    overall = blended_form(
        overall_recent,
        overall_long,
    )

    venue_form = blended_form(
        venue_recent,
        venue_long,
    )

    # Blend venue-specific and overall data.
    # Venue gets more importance when enough matches exist.

    if (
        venue_form["scored"] is None
        or venue_form["conceded"] is None
    ):

        combined = overall

    else:

        venue_sample = min(
            venue_long["matches"],
            10,
        )

        venue_weight = min(
            0.70,
            0.30 + venue_sample * 0.04,
        )

        overall_weight = (
            1 - venue_weight
        )

        combined = {
            "scored": (
                venue_weight
                * venue_form["scored"]
                + overall_weight
                * overall["scored"]
            ),

            "conceded": (
                venue_weight
                * venue_form["conceded"]
                + overall_weight
                * overall["conceded"]
            ),
        }

    return {
        "team_id": team_id,

        "venue": venue,

        "recent": overall_recent,

        "long_term": overall_long,

        "venue_recent": venue_recent,

        "venue_long": venue_long,

        "blended_scored": combined[
            "scored"
        ],

        "blended_conceded": combined[
            "conceded"
        ],
    }


# ============================================================
# LEAGUE BASELINES
# ============================================================

@st.cache_data(ttl=3600)
def get_league_baselines(
    competition_code,
):
    """
    Calculates actual home and away scoring
    averages from finished competition matches.

    This is preferable to simply assuming:
        home = league_average * 1.08
        away = league_average * 0.92
    """

    data = api_get(
        f"/competitions/{competition_code}/matches",
        params={
            "status": "FINISHED",
        },
    )

    matches = data.get(
        "matches",
        [],
    )

    home_goals = []
    away_goals = []

    for match in matches:

        score = match.get(
            "score",
            {},
        ).get(
            "fullTime",
            {},
        )

        home = score.get("home")
        away = score.get("away")

        if (
            home is None
            or away is None
        ):
            continue

        home_goals.append(home)
        away_goals.append(away)

    if not home_goals:

        return {
            "home_goals": 1.50,
            "away_goals": (
                1.50
                / HOME_ADVANTAGE_FALLBACK
            ),
            "matches": 0,
        }

    return {
        "home_goals": (
            sum(home_goals)
            / len(home_goals)
        ),

        "away_goals": (
            sum(away_goals)
            / len(away_goals)
        ),

        "matches": len(home_goals),
    }


# ============================================================
# EXPECTED GOALS MODEL
# ============================================================

def shrink_to_baseline(
    value,
    sample_size,
    baseline,
    regression_matches=REGRESSION_MATCHES,
):
    """
    Bayesian-style regression toward league average.

    Example:
    A team averaging 3.5 goals over only two matches
    should not be treated as a true 3.5-goal attack.
    """

    if value is None:
        return baseline

    sample_size = max(
        sample_size,
        0,
    )

    weight = (
        sample_size
        / (
            sample_size
            + regression_matches
        )
    )

    return (
        weight * value
        + (1 - weight) * baseline
    )


def calculate_match_expected_goals(
    home_profile,
    away_profile,
    league,
):
    """
    Calculates expected goals from explicit:

        HOME ATTACK
        HOME DEFENCE

        AWAY ATTACK
        AWAY DEFENCE

    Every estimate is league-relative.
    """

    league_home = league[
        "home_goals"
    ]

    league_away = league[
        "away_goals"
    ]

    home_sample = max(
        home_profile[
            "venue_long"
        ]["matches"],
        home_profile[
            "long_term"
        ]["matches"],
    )

    away_sample = max(
        away_profile[
            "venue_long"
        ]["matches"],
        away_profile[
            "long_term"
        ]["matches"],
    )

    # HOME TEAM ATTACK

    home_attack_goals = shrink_to_baseline(
        home_profile[
            "blended_scored"
        ],
        home_sample,
        league_home,
    )

    # HOME TEAM DEFENSIVE BASELINE:
    # away teams score at league_away on average

    home_defence_conceded = shrink_to_baseline(
        home_profile[
            "blended_conceded"
        ],
        home_sample,
        league_away,
    )

    # AWAY TEAM ATTACK

    away_attack_goals = shrink_to_baseline(
        away_profile[
            "blended_scored"
        ],
        away_sample,
        league_away,
    )

    # AWAY TEAM DEFENCE:
    # home teams score at league_home on average

    away_defence_conceded = shrink_to_baseline(
        away_profile[
            "blended_conceded"
        ],
        away_sample,
        league_home,
    )

    # Attack strengths

    home_attack_strength = (
        home_attack_goals
        / league_home
        if league_home > 0
        else 1
    )

    away_attack_strength = (
        away_attack_goals
        / league_away
        if league_away > 0
        else 1
    )

    # Defensive weakness strengths:
    # > 1 means conceding more than baseline

    home_defence_weakness = (
        home_defence_conceded
        / league_away
        if league_away > 0
        else 1
    )

    away_defence_weakness = (
        away_defence_conceded
        / league_home
        if league_home > 0
        else 1
    )

    # Multiplicative attack × opponent defensive weakness model.

    home_xg = (
        league_home
        * home_attack_strength
        * away_defence_weakness
    )

    away_xg = (
        league_away
        * away_attack_strength
        * home_defence_weakness
    )

    # Reasonable safety limits.

    home_xg = max(
        0.10,
        min(home_xg, 5.50),
    )

    away_xg = max(
        0.10,
        min(away_xg, 5.50),
    )

    diagnostics = {
        "league_home": league_home,
        "league_away": league_away,

        "home_attack_goals": (
            home_attack_goals
        ),

        "away_attack_goals": (
            away_attack_goals
        ),

        "home_defence_conceded": (
            home_defence_conceded
        ),

        "away_defence_conceded": (
            away_defence_conceded
        ),

        "home_attack_strength": (
            home_attack_strength
        ),

        "away_attack_strength": (
            away_attack_strength
        ),

        "home_defence_weakness": (
            home_defence_weakness
        ),

        "away_defence_weakness": (
            away_defence_weakness
        ),
    }

    return (
        round(home_xg, 2),
        round(away_xg, 2),
        diagnostics,
    )


# ============================================================
# POISSON MODEL
# ============================================================

def compute_model(
    home_xg,
    away_xg,
    max_goals=MAX_GOALS,
):
    """
    Creates score probabilities and
    derives market probabilities.
    """

    matrix = {}

    total_probability = 0.0

    for home_goals in range(
        max_goals + 1
    ):

        for away_goals in range(
            max_goals + 1
        ):

            probability = (
                poisson_pmf(
                    home_xg,
                    home_goals,
                )
                * poisson_pmf(
                    away_xg,
                    away_goals,
                )
            )

            matrix[
                (
                    home_goals,
                    away_goals,
                )
            ] = probability

            total_probability += probability

    # Normalise truncated matrix.

    if total_probability > 0:

        matrix = {
            score: probability
            / total_probability

            for score, probability
            in matrix.items()
        }

    model = {
        "home": 0.0,
        "draw": 0.0,
        "away": 0.0,
        "over25": 0.0,
        "under25": 0.0,
        "bttsY": 0.0,
        "bttsN": 0.0,
    }

    for (
        home_goals,
        away_goals,
    ), probability in matrix.items():

        if home_goals > away_goals:
            model["home"] += probability

        elif home_goals == away_goals:
            model["draw"] += probability

        else:
            model["away"] += probability

        if (
            home_goals
            + away_goals
            >= 3
        ):
            model["over25"] += probability

        else:
            model["under25"] += probability

        if (
            home_goals >= 1
            and away_goals >= 1
        ):
            model["bttsY"] += probability

        else:
            model["bttsN"] += probability

    return (
        model,
        matrix,
    )


def cards_over_under(
    lam,
    threshold=CARDS_THRESHOLD,
):
    """
    Poisson approximation for total cards.

    For 3.5:
        Under = 0,1,2,3
        Over  = 4+
    """

    floor_threshold = int(
        math.floor(threshold)
    )

    under = sum(
        poisson_pmf(lam, k)
        for k in range(
            floor_threshold + 1
        )
    )

    over = 1 - under

    return over, under


# ============================================================
# SUMMARY
# ============================================================

def build_summary_row(
    fixture,
    home_xg,
    away_xg,
    market_label,
    model_probability,
    odds,
    market_probability,
    edge,
    verdict_name,
):
    return [
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d"
        ),

        fixture["competition"],

        fixture["home_team_name"],

        fixture["away_team_name"],

        fixture["kickoff"],

        f"{home_xg:.2f}",

        f"{away_xg:.2f}",

        market_label,

        f"{model_probability:.4f}",

        f"{odds:.2f}",

        f"{market_probability:.4f}",

        f"{edge:.4f}",

        verdict_name,
    ]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap'
);

.stApp {
    background-color: #0C0F0A;
}

html,
body,
[class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero {
    background:
        linear-gradient(
            135deg,
            #1A0508 0%,
            #0C0F0A 70%
        );

    border: 1px solid #2A332A;
    border-left: 4px solid #C8102E;

    border-radius: 12px;

    padding: 24px;

    margin-bottom: 18px;
}

.hero-title {
    font-family:
        'Bebas Neue',
        sans-serif;

    font-size: 42px;

    letter-spacing: 1px;

    color: #EDEDE6;

    line-height: 1;
}

.hero-title span {
    color: #C8102E;
}

.hero-sub {
    color: #8B948A;

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 0.08em;

    margin-top: 7px;
}

.hero-fixture {
    font-family:
        'Bebas Neue',
        sans-serif;

    font-size: 30px;

    color: #E8B33D;

    margin-top: 16px;

    letter-spacing: 0.4px;
}

.hero-meta {
    color: #8B948A;

    font-size: 13px;

    margin-top: 4px;
}

.badge {
    display: inline-block;

    padding: 5px 11px;

    border-radius: 12px;

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 0.03em;

    font-family:
        'IBM Plex Mono',
        monospace;
}

.badge-value {
    background:
        rgba(
            76,
            175,
            125,
            0.18
        );

    color: #4CAF7D;
}

.badge-avoid {
    background:
        rgba(
            193,
            99,
            63,
            0.18
        );

    color: #C1633F;
}

.badge-marginal {
    background:
        rgba(
            139,
            148,
            138,
            0.18
        );

    color: #8B948A;
}

.market-card {
    background: #171B14;

    border:
        1px solid #2A332A;

    border-radius: 9px;

    padding: 13px 16px;

    margin-bottom: 8px;
}

.market-name {
    font-weight: 600;

    font-size: 14px;

    color: #EDEDE6;
}

.model-pct {
    color: #8B948A;

    font-size: 12px;

    font-family:
        'IBM Plex Mono',
        monospace;
}

.inspector-card {
    background: #171B14;

    border:
        1px solid #2A332A;

    border-radius: 9px;

    padding: 14px;

    margin-bottom: 10px;
}

.inspector-title {
    color: #E8B33D;

    font-weight: 700;

    font-size: 13px;

    margin-bottom: 8px;
}

.inspector-row {
    color: #B8C0B5;

    font-family:
        'IBM Plex Mono',
        monospace;

    font-size: 12px;

    margin: 3px 0;
}

div[data-testid="stMetricValue"] {
    font-family:
        'IBM Plex Mono',
        monospace;

    color: #E8B33D;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# ⚽ The Boot Room"
    )

    page = st.radio(
        "Choose analysis",
        [
            "🏠 Liverpool Next Match",
            "🔎 Analyse Any Match",
        ],
    )

    st.divider()

    st.markdown(
        "### How to use it"
    )

    st.markdown(
        """
1. Choose a match
2. Check the model's estimated goals
3. Adjust if you have information the model cannot know
4. Enter bookmaker odds
5. Compare model probability with market probability
"""
    )

    st.divider()

    st.caption(
        "The model estimates probabilities from historical scoring patterns. "
        "It does not guarantee outcomes."
    )


# ============================================================
# FIXTURE SELECTION
# ============================================================

def fixture_label(fixture):

    kickoff = parse_datetime(
        fixture["kickoff"]
    )

    if kickoff:

        date_text = kickoff.strftime(
            "%a %d %b %H:%M UTC"
        )

    else:
        date_text = "Date TBC"

    return (
        f"{fixture['home_team_name']} vs "
        f"{fixture['away_team_name']} "
        f"— {date_text}"
    )


def choose_fixture():

    # --------------------------------------------------------
    # LIVERPOOL MODE
    # --------------------------------------------------------

    if page == "🏠 Liverpool Next Match":

        fixture = get_next_fixture(
            LIVERPOOL_ID
        )

        if not fixture:
            st.error(
                "Could not find Liverpool's next scheduled fixture."
            )
            return None

        return fixture

    # --------------------------------------------------------
    # ANY MATCH MODE
    # --------------------------------------------------------

    st.title(
        "Analyse Any Match"
    )

    selection_mode = st.radio(
        "How do you want to choose the match?",
        [
            "Upcoming fixture",
            "Choose teams manually",
        ],
        horizontal=True,
    )

    competitions = get_competitions()

    if not competitions:
        st.error(
            "No competitions were returned by the API."
        )
        return None

    competition_names = [
        competition["name"]
        for competition in competitions
    ]

    selected_name = st.selectbox(
        "Competition",
        competition_names,
        key="competition_selector",
    )

    selected_competition = next(
        competition
        for competition in competitions
        if competition["name"]
        == selected_name
    )

    competition_code = selected_competition[
        "code"
    ]

    # --------------------------------------------------------
    # UPCOMING FIXTURE
    # --------------------------------------------------------

    if selection_mode == "Upcoming fixture":

        fixtures = get_competition_fixtures(
            competition_code
        )

        if not fixtures:

            st.warning(
                "No upcoming fixtures were available for this competition."
            )

            return None

        labels = [
            fixture_label(fixture)
            for fixture in fixtures
        ]

        selected_label = st.selectbox(
            "Fixture",
            labels,
            key="fixture_selector",
        )

        index = labels.index(
            selected_label
        )

        return fixtures[index]

    # --------------------------------------------------------
    # MANUAL TEAMS
    # --------------------------------------------------------

    teams = get_competition_teams(
        competition_code
    )

    if len(teams) < 2:

        st.warning(
            "Not enough teams were returned for this competition."
        )

        return None

    team_names = [
        team["name"]
        for team in teams
    ]

    col1, col2 = st.columns(2)

    with col1:

        home_name = st.selectbox(
            "Home team",
            team_names,
            key="manual_home_team",
        )

    with col2:

        away_options = [
            name
            for name in team_names
            if name != home_name
        ]

        away_name = st.selectbox(
            "Away team",
            away_options,
            key="manual_away_team",
        )

    home_team = next(
        team
        for team in teams
        if team["name"]
        == home_name
    )

    away_team = next(
        team
        for team in teams
        if team["name"]
        == away_name
    )

    return {
        "match_id": None,

        "competition": selected_competition[
            "name"
        ],

        "competition_code": competition_code,

        "kickoff": None,

        "home_team_id": home_team[
            "id"
        ],

        "home_team_name": home_team[
            "name"
        ],

        "away_team_id": away_team[
            "id"
        ],

        "away_team_name": away_team[
            "name"
        ],
    }


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard(fixture):

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    kickoff = parse_datetime(
        fixture.get("kickoff")
    )

    if kickoff:

        now = datetime.now(
            timezone.utc
        )

        delta = kickoff - now

        if delta.total_seconds() > 0:

            total_hours = int(
                delta.total_seconds()
                // 3600
            )

            days = total_hours // 24
            hours = total_hours % 24

        else:

            days = None
            hours = None

        kickoff_text = kickoff.strftime(
            "%a %d %b, %H:%M UTC"
        )

    else:

        days = None
        hours = None

        kickoff_text = (
            "Manual match selection"
        )

    st.markdown(
        f"""
<div class="hero">

<div class="hero-title">
The Boot <span>Room</span>
</div>

<div class="hero-sub">
Football probability model · expected goals vs market odds
</div>

<div class="hero-fixture">
{fixture["home_team_name"]}
vs
{fixture["away_team_name"]}
</div>

<div class="hero-meta">
{fixture["competition"]}
·
{kickoff_text}
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    if days is not None:

        col1, col2 = st.columns(2)

        col1.metric(
            "Days to kickoff",
            days,
        )

        col2.metric(
            "Hours remaining",
            hours,
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    with st.spinner(
        "Building team profiles..."
    ):

        league = get_league_baselines(
            fixture[
                "competition_code"
            ]
        )

        home_profile = build_team_profile(
            fixture[
                "home_team_id"
            ],
            venue="H",
        )

        away_profile = build_team_profile(
            fixture[
                "away_team_id"
            ],
            venue="A",
        )

        auto_home_xg, auto_away_xg, diagnostics = (
            calculate_match_expected_goals(
                home_profile,
                away_profile,
                league,
            )
        )

    # --------------------------------------------------------
    # XG CONTROLS
    # --------------------------------------------------------

    with st.expander(
        "⚽ Estimated goals "
        "(auto-calculated — click to adjust)",
        expanded=False,
    ):

        col1, col2 = st.columns(2)

        home_xg = col1.number_input(
            f"{fixture['home_team_name']} estimated goals",
            min_value=0.0,
            max_value=6.0,
            value=float(
                auto_home_xg
            ),
            step=0.05,
            key=(
                f"xg_home_"
                f"{fixture['home_team_id']}_"
                f"{fixture['away_team_id']}"
            ),
        )

        away_xg = col2.number_input(
            f"{fixture['away_team_name']} estimated goals",
            min_value=0.0,
            max_value=6.0,
            value=float(
                auto_away_xg
            ),
            step=0.05,
            key=(
                f"xg_away_"
                f"{fixture['home_team_id']}_"
                f"{fixture['away_team_id']}"
            ),
        )

        st.caption(
            "The automatic estimate uses recent form, longer-term form, "
            "venue-specific performance and league-relative regression. "
            "You can manually adjust for injuries, rotation, tactical changes "
            "or other information the model cannot see."
        )

    # --------------------------------------------------------
    # CARDS
    # --------------------------------------------------------

    with st.expander(
        "🟨 Match cards estimate "
        "(optional manual input)",
        expanded=False,
    ):

        cards_lambda = st.number_input(
            "Expected total match cards",
            min_value=0.0,
            max_value=12.0,
            value=4.0,
            step=0.25,
            key=(
                f"cards_"
                f"{fixture['home_team_id']}_"
                f"{fixture['away_team_id']}"
            ),
        )

        st.caption(
            "This market remains manual. "
            "Consider referee history, league averages, rivalry intensity "
            "and team disciplinary records."
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model, score_matrix = compute_model(
        home_xg,
        away_xg,
    )

    cards_over, cards_under = (
        cards_over_under(
            cards_lambda
        )
    )

    model["cardsOver"] = cards_over
    model["cardsUnder"] = cards_under

    market_labels = {
        "home": (
            f"{fixture['home_team_name']} Win"
        ),

        "draw": "Draw",

        "away": (
            f"{fixture['away_team_name']} Win"
        ),

        "over25": (
            "Over 2.5 Goals"
        ),

        "under25": (
            "Under 2.5 Goals"
        ),

        "bttsY": (
            "BTTS — Yes"
        ),

        "bttsN": (
            "BTTS — No"
        ),

        "cardsOver": (
            f"Over {CARDS_THRESHOLD} Match Cards"
        ),

        "cardsUnder": (
            f"Under {CARDS_THRESHOLD} Match Cards"
        ),
    }

    # --------------------------------------------------------
    # MODEL INSPECTOR
    # --------------------------------------------------------

    with st.expander(
        "🔬 Model inspector",
        expanded=False,
    ):

        st.caption(
            "These are the main inputs used to build the automatic expected-goals estimate."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
<div class="inspector-card">

<div class="inspector-title">
{fixture["home_team_name"]} — Home profile
</div>

<div class="inspector-row">
Recent goals scored:
{home_profile["recent"]["scored"] if home_profile["recent"]["scored"] is not None else "N/A"}
</div>

<div class="inspector-row">
Recent goals conceded:
{home_profile["recent"]["conceded"] if home_profile["recent"]["conceded"] is not None else "N/A"}
</div>

<div class="inspector-row">
Long-term goals scored:
{home_profile["long_term"]["scored"] if home_profile["long_term"]["scored"] is not None else "N/A"}
</div>

<div class="inspector-row">
Long-term goals conceded:
{home_profile["long_term"]["conceded"] if home_profile["long_term"]["conceded"] is not None else "N/A"}
</div>

<div class="inspector-row">
Attack strength:
{diagnostics["home_attack_strength"]:.2f}
</div>

<div class="inspector-row">
Defensive weakness:
{diagnostics["home_defence_weakness"]:.2f}
</div>

</div>
""",
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
<div class="inspector-card">

<div class="inspector-title">
{fixture["away_team_name"]} — Away profile
</div>

<div class="inspector-row">
Recent goals scored:
{away_profile["recent"]["scored"] if away_profile["recent"]["scored"] is not None else "N/A"}
</div>

<div class="inspector-row">
Recent goals conceded:
{away_profile["recent"]["conceded"] if away_profile["recent"]["conceded"] is not None else "N/A"}
</div>

<div class="inspector-row">
Long-term goals scored:
{away_profile["long_term"]["scored"] if away_profile["long_term"]["scored"] is not None else "N/A"}
</div>

<div class="inspector-row">
Long-term goals conceded:
{away_profile["long_term"]["conceded"] if away_profile["long_term"]["conceded"] is not None else "N/A"}
</div>

<div class="inspector-row">
Attack strength:
{diagnostics["away_attack_strength"]:.2f}
</div>

<div class="inspector-row">
Defensive weakness:
{diagnostics["away_defence_weakness"]:.2f}
</div>

</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
<div class="inspector-card">

<div class="inspector-title">
League baseline
</div>

<div class="inspector-row">
Average home goals:
{league["home_goals"]:.2f}
</div>

<div class="inspector-row">
Average away goals:
{league["away_goals"]:.2f}
</div>

<div class="inspector-row">
Finished matches sampled:
{league["matches"]}
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # SCORE PROBABILITIES
    # --------------------------------------------------------

    with st.expander(
        "📊 Most likely scores",
        expanded=False,
    ):

        top_scores = sorted(
            score_matrix.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:8]

        columns = st.columns(4)

        for index, (
            score,
            probability,
        ) in enumerate(top_scores):

            home_goals, away_goals = score

            columns[
                index % 4
            ].metric(
                f"{home_goals}–{away_goals}",
                f"{probability * 100:.1f}%",
            )

    # --------------------------------------------------------
    # ODDS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Odds & edge"
    )

    st.caption(
        "Enter decimal odds such as 2.10 or fractional odds such as 5/2. "
        "For the 1X2 market, entering all three prices automatically removes "
        "the bookmaker margin before calculating the edge."
    )

    odds_inputs = {}

    # 1X2 first so we can devig.

    result_keys = [
        "home",
        "draw",
        "away",
    ]

    st.markdown(
        "### 🥅 Match Result"
    )

    for key in result_keys:

        c1, c2, c3 = st.columns(
            [2.2, 1.3, 1.6]
        )

        label = market_labels[key]

        c1.markdown(
            f"""
<div class="market-card">

<span class="market-name">
{label}
</span>

<br>

<span class="model-pct">
Model: {model[key] * 100:.1f}%
</span>

</div>
""",
            unsafe_allow_html=True,
        )

        odds_text = c2.text_input(
            f"{label} odds",
            key=(
                f"odds_{key}_"
                f"{fixture['home_team_id']}_"
                f"{fixture['away_team_id']}"
            ),
            label_visibility="collapsed",
            placeholder="2.10 or 5/2",
        )

        odds = parse_odds(
            odds_text
        )

        odds_inputs[key] = odds

        c3.markdown(
            """
<div style="padding-top:10px">
<span class="badge badge-marginal">
—
</span>
</div>
""",
            unsafe_allow_html=True,
        )

    devig_probs = fair_probabilities_from_market(
        odds_inputs
    )

    # Display corrected result badges.

    for key in result_keys:

        odds = odds_inputs.get(
            key
        )

        if odds is None:
            continue

        if devig_probs:

            market_prob = devig_probs[
                key
            ]

        else:

            market_prob = implied_probability(
                odds
            )

        edge = (
            model[key]
            - market_prob
        )

        odds_inputs[key] = {
            "odds": odds,
            "market_probability": market_prob,
            "edge": edge,
            "verdict": verdict(edge),
        }

    # --------------------------------------------------------
    # OTHER MARKETS
    # --------------------------------------------------------

    other_groups = [
        (
            "⚽ Goals",
            ["over25", "under25"],
        ),

        (
            "🎯 Both Teams to Score",
            ["bttsY", "bttsN"],
        ),

        (
            "🟨 Discipline",
            ["cardsOver", "cardsUnder"],
        ),
    ]

    results = []

    # Add result markets.

    st.divider()

    st.subheader(
        "Match result edge"
    )

    for key in result_keys:

        data = odds_inputs.get(
            key
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        edge = data["edge"]
        verdict_name = data["verdict"]

        c1, c2, c3 = st.columns(
            [2.2, 1.3, 1.6]
        )

        c1.markdown(
            f"""
<div class="market-card">

<span class="market-name">
{market_labels[key]}
</span>

<br>

<span class="model-pct">
Model: {model[key] * 100:.1f}%
</span>

</div>
""",
            unsafe_allow_html=True,
        )

        c2.metric(
            "Odds",
            f"{data['odds']:.2f}",
        )

        c3.markdown(
            f"""
<div style="padding-top:10px">

<span class="badge {verdict_class(verdict_name)}">
{edge * 100:+.1f}% · {verdict_name}
</span>

</div>
""",
            unsafe_allow_html=True,
        )

        results.append(
            {
                "key": key,

                "label": market_labels[
                    key
                ],

                "model_probability": model[
                    key
                ],

                "odds": data[
                    "odds"
                ],

                "market_probability": data[
                    "market_probability"
                ],

                "edge": edge,

                "verdict": verdict_name,
            }
        )

    # --------------------------------------------------------
    # OTHER MARKET INPUTS
    # --------------------------------------------------------

    for (
        group_name,
        keys,
    ) in other_groups:

        st.divider()

        st.subheader(
            group_name
        )

        for key in keys:

            c1, c2, c3 = st.columns(
                [2.2, 1.3, 1.6]
            )

            label = market_labels[key]

            c1.markdown(
                f"""
<div class="market-card">

<span class="market-name">
{label}
</span>

<br>

<span class="model-pct">
Model: {model[key] * 100:.1f}%
</span>

</div>
""",
                unsafe_allow_html=True,
            )

            odds_text = c2.text_input(
                f"{label} odds",
                key=(
                    f"odds_{key}_"
                    f"{fixture['home_team_id']}_"
                    f"{fixture['away_team_id']}"
                ),
                label_visibility="collapsed",
                placeholder="2.10 or 5/2",
            )

            odds = parse_odds(
                odds_text
            )

            if odds is None:

                c3.markdown(
                    """
<div style="padding-top:10px">
<span class="badge badge-marginal">
—
</span>
</div>
""",
                    unsafe_allow_html=True,
                )

                continue

            market_prob = implied_probability(
                odds
            )

            edge = (
                model[key]
                - market_prob
            )

            verdict_name = verdict(
                edge
            )

            c3.markdown(
                f"""
<div style="padding-top:10px">

<span class="badge {verdict_class(verdict_name)}">
{edge * 100:+.1f}% · {verdict_name}
</span>

</div>
""",
                unsafe_allow_html=True,
            )

            results.append(
                {
                    "key": key,

                    "label": label,

                    "model_probability": model[
                        key
                    ],

                    "odds": odds,

                    "market_probability": market_prob,

                    "edge": edge,

                    "verdict": verdict_name,
                }
            )

    # --------------------------------------------------------
    # BEST VALUE
    # --------------------------------------------------------

    value_bets = [
        result
        for result in results
        if result["verdict"]
        == "VALUE"
    ]

    if value_bets:

        best = max(
            value_bets,
            key=lambda x: x["edge"],
        )

        st.success(
            f"Best model edge found: "
            f"**{best['label']}** "
            f"— {best['edge'] * 100:+.1f}% edge "
            f"at odds {best['odds']:.2f}"
        )

    elif results:

        st.info(
            "No market currently reaches the model's VALUE threshold."
        )

    # --------------------------------------------------------
    # COPY SUMMARY
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Copy this analysis"
    )

    if results:

        lines = [
            "\t".join(
                SUMMARY_HEADER
            )
        ]

        for result in results:

            row = build_summary_row(
                fixture=fixture,

                home_xg=home_xg,

                away_xg=away_xg,

                market_label=result[
                    "label"
                ],

                model_probability=result[
                    "model_probability"
                ],

                odds=result[
                    "odds"
                ],

                market_probability=result[
                    "market_probability"
                ],

                edge=result[
                    "edge"
                ],

                verdict_name=result[
                    "verdict"
                ],
            )

            lines.append(
                "\t".join(
                    str(value)
                    for value in row
                )
            )

        summary = "\n".join(
            lines
        )

        st.caption(
            "Use the copy icon in the top-right of the box and paste into "
            "Excel, Google Sheets or another tracker."
        )

        st.code(
            summary,
            language=None,
        )

    else:

        st.caption(
            "Enter odds for one or more markets to generate a copyable summary."
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "The Boot Room estimates probabilities from historical scoring data "
        "and manually adjustable assumptions. It is not a guarantee of future "
        "results. A positive model edge can still lose, and bookmaker prices "
        "may contain information this model does not capture. Bet only what "
        "you can afford to lose."
    )


# ============================================================
# MAIN
# ============================================================

if not API_KEY:

    st.error(
        "FOOTBALL_DATA_KEY isn't set."
    )

    st.markdown(
        """
### Local development

Set:

```bash
export FOOTBALL_DATA_KEY="your_key_here"
