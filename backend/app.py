from flask import Flask, jsonify, request
import requests
from fetch_data import fetch_matches, fetch_upcoming_fixtures
from predictor import FootballPredictor
from models import db, Match, Prediction
from datetime import datetime
import time
from ratelimit import limits, sleep_and_retry
from flask_cors import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///football_db.sqlite"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

predictor = FootballPredictor(app=app)

CALLS = 10
PERIOD = 60

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    response = jsonify({"error": str(e)})
    response.status_code = 500
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
    return response

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def api_call(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

LEAGUES = [
    {"code": "PL", "name": "Premier League"},
    {"code": "CL", "name": "UEFA Champions League"},
    {"code": "PD", "name": "Primera Division"},
    {"code": "SA", "name": "Serie A"},
    {"code": "BL1", "name": "Bundesliga"},
    {"code": "FL1", "name": "Ligue 1"},
    {"code": "ELC", "name": "Championship"},
    {"code": "PPL", "name": "Primeira Liga"},
    {"code": "DED", "name": "Eredivisie"},
    {"code": "BSA", "name": "Campeonato Brasileiro Série A"}
]

@app.route("/api/leagues", methods=["GET"])
def get_leagues():
    logger.info("Fetching leagues")
    return jsonify([{"code": league["code"], "name": league["name"]} for league in LEAGUES])

@app.route("/api/fixtures/<league>", methods=["GET"])
def get_fixtures(league):
    if league not in [l["code"] for l in LEAGUES]:
        logger.error(f"Invalid league code: {league}")
        return jsonify({"error": "Invalid league code"}), 400
    
    logger.info(f"Fetching fixtures for league: {league}")
    fixtures = fetch_upcoming_fixtures(league)
    logger.debug(f"Fixtures fetched: {len(fixtures)} rows")
    matches = fetch_matches(league)
    logger.debug(f"Matches fetched: {len(matches)} rows")
    
    logger.info("Generating predictions")
    predictions = predictor.predict(matches, fixtures)
    logger.debug(f"Predictions generated: {len(predictions)} rows")
    
    with app.app_context():
        logger.info("Updating database with fixtures and predictions")
        for _, row in fixtures.iterrows():
            match = Match.query.filter_by(match_id=row["match_id"]).first()
            if not match:
                match = Match(
                    match_id=row["match_id"], league=league,
                    date=datetime.strptime(row["date"], "%Y-%m-%dT%H:%M:%SZ"),
                    home_team=row["home_team"], away_team=row["away_team"],
                    season="2024"
                )
                db.session.add(match)
        
        for _, pred in predictions.iterrows():
            pred_entry = Prediction(
                match_id=pred["match_id"], league=league,
                predicted_home_goals=pred["predicted_home_goals"],
                predicted_away_goals=pred["predicted_away_goals"],
                confidence=pred["confidence"]
            )
            db.session.add(pred_entry)
        db.session.commit()
        logger.info("Database updated")
    
    return jsonify({
        "fixtures": fixtures.to_dict(orient="records"),
        "predictions": predictions.to_dict(orient="records")
    })

@app.route("/api/past_predictions/<league>", methods=["GET"])
def get_past_predictions(league):
    if league not in [l["code"] for l in LEAGUES]:
        logger.error(f"Invalid league code: {league}")
        return jsonify({"error": "Invalid league code"}), 400
    with app.app_context():
        logger.info(f"Fetching past predictions for league: {league}")
        predictions = Prediction.query.filter_by(league=league).all()
        past_data = [{
            "match_id": p.match_id,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "predicted_home_goals": p.predicted_home_goals,
            "predicted_away_goals": p.predicted_away_goals,
            "actual_home_goals": p.actual_home_goals,
            "actual_away_goals": p.actual_away_goals,
            "date": m.date.isoformat(),
            "confidence": p.confidence
        } for p, m in [(p, Match.query.get(p.match_id)) for p in predictions] if m]
        
        correct = 0
        total = 0
        for p in past_data:
            if p["actual_home_goals"] is not None and p["actual_away_goals"] is not None:
                total += 1
                pred_winner = "home" if p["predicted_home_goals"] > p["predicted_away_goals"] else \
                             "away" if p["predicted_away_goals"] > p["predicted_home_goals"] else "draw"
                actual_winner = "home" if p["actual_home_goals"] > p["actual_away_goals"] else \
                               "away" if p["actual_away_goals"] > p["actual_home_goals"] else "draw"
                if pred_winner == actual_winner:
                    correct += 1
        success_rate = (correct / total * 100) if total > 0 else 0
        
        logger.info(f"Past predictions fetched: {len(past_data)} items")
        return jsonify({"past_predictions": past_data, "success_rate": success_rate})

@app.route("/api/form/<team>", methods=["GET"])
def get_team_form(team):
    league = request.args.get("league", "PL")
    logger.info(f"Fetching form for team: {team} in league: {league}")
    matches = fetch_matches(league)
    form = predictor.calculate_form(matches, team)
    return jsonify({"points_per_game": form[0], "goals_scored_per_game": form[1], "goals_conceded_per_game": form[2]})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)