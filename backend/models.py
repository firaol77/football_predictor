from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.Integer, unique=True)
    league = db.Column(db.String(10))  # e.g., "PL", "CL"
    date = db.Column(db.DateTime)
    home_team = db.Column(db.String(100))
    away_team = db.Column(db.String(100))
    home_goals = db.Column(db.Integer, nullable=True)
    away_goals = db.Column(db.Integer, nullable=True)
    season = db.Column(db.String(4))

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.Integer, db.ForeignKey("match.match_id"))
    league = db.Column(db.String(10))  # Match league for filtering
    predicted_home_goals = db.Column(db.Float)
    predicted_away_goals = db.Column(db.Float)
    confidence = db.Column(db.Float)
    actual_home_goals = db.Column(db.Integer, nullable=True)
    actual_away_goals = db.Column(db.Integer, nullable=True)
    date_predicted = db.Column(db.DateTime, default=datetime.utcnow)