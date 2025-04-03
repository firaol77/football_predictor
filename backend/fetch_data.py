import requests
import pandas as pd
from datetime import datetime, timedelta
from ratelimit import limits, sleep_and_retry

API_TOKEN = "a8191890ffa24ae89288882c6c6136f6"
BASE_URL = "http://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_TOKEN}
CALLS = 10
PERIOD = 60  # 60 seconds

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def api_call(url):
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def fetch_matches(league_code="PL", days_back=750):
    url = f"{BASE_URL}competitions/{league_code}/matches?status=FINISHED"
    data = api_call(url)
    
    matches = []
    for match in data["matches"]:
        match_date = datetime.strptime(match["utcDate"], "%Y-%m-%dT%H:%M:%SZ")
        if (datetime.now() - match_date).days <= days_back:
            matches.append({
                "match_id": match["id"],
                "date": match_date,
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_goals": match["score"]["fullTime"]["home"],
                "away_goals": match["score"]["fullTime"]["away"],
                "season": match["season"]["startDate"][:4]
            })
    return pd.DataFrame(matches)

def fetch_upcoming_fixtures(league_code="PL"):
    url = f"{BASE_URL}competitions/{league_code}/matches?status=SCHEDULED"
    data = api_call(url)
    
    fixtures = []
    for match in data["matches"][:10]:
        fixtures.append({
            "match_id": match["id"],
            "date": match["utcDate"],
            "home_team": match["homeTeam"]["name"],
            "away_team": match["awayTeam"]["name"]
        })
    return pd.DataFrame(fixtures)

def get_head_to_head(matches, home_team, away_team, days_back=750):
    h2h = matches[((matches["home_team"] == home_team) & (matches["away_team"] == away_team)) |
                  ((matches["home_team"] == away_team) & (matches["away_team"] == home_team))]
    h2h = h2h[(datetime.now() - h2h["date"]).dt.days <= days_back]
    return h2h