import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os
import tensorflow as tf
from fetch_data import get_head_to_head
from models import db, Match, Prediction
import logging

# Set up logging
logger = logging.getLogger(__name__)

class FootballPredictor:
    def __init__(self, app=None):
        self.app = app  # Store Flask app instance for database context
        self.xgb_home = XGBRegressor()
        self.xgb_away = XGBRegressor()
        self.rf_home = RandomForestRegressor()
        self.rf_away = RandomForestRegressor()
        self.poisson_home = PoissonRegressor()
        self.poisson_away = PoissonRegressor()
        self.lstm_model = None
        self.model_path = "hybrid_model.pkl"
        self.famous_rivalries = {
            "PL": [("Manchester United", "Manchester City"), ("Liverpool", "Everton"), ("Arsenal", "Tottenham Hotspur")],
            "CL": [],
            "PD": [("Real Madrid", "Barcelona"), ("Atlético Madrid", "Real Madrid")],
            "SA": [("Inter", "Milan"), ("Juventus", "Inter"), ("Roma", "Lazio")],
            "BL1": [("Bayern Munich", "Borussia Dortmund"), ("Schalke 04", "Borussia Dortmund")],
            "FL1": [("Paris Saint-Germain", "Olympique Marseille"), ("Olympique Lyonnais", "Saint-Étienne")],
            "ELC": [("Leeds United", "Sheffield Wednesday"), ("Birmingham City", "Aston Villa")],
            "PPL": [("Benfica", "Porto"), ("Sporting CP", "Benfica")],
            "DED": [("Ajax", "Feyenoord"), ("PSV", "Ajax")],
            "BSA": [("Flamengo", "Fluminense"), ("Corinthians", "Palmeiras")]
        }
        logger.info("FootballPredictor initialized")

    def calculate_form(self, matches, team, n=10, decay_factor=0.9):
        logger.debug(f"Calculating form for team: {team}")
        team_matches = matches[(matches["home_team"] == team) | (matches["away_team"] == team)].sort_values("date")
        points = goals_scored = goals_conceded = 0
        weights = [decay_factor ** i for i in range(len(team_matches))]
        for i, (_, row) in enumerate(team_matches.iterrows()):
            weight = weights[i]
            if row["home_team"] == team:
                goals_scored += row["home_goals"] * weight
                goals_conceded += row["away_goals"] * weight
                if row["home_goals"] > row["away_goals"]:
                    points += 3 * weight
                elif row["home_goals"] == row["away_goals"]:
                    points += 1 * weight
            else:
                goals_scored += row["away_goals"] * weight
                goals_conceded += row["home_goals"] * weight
                if row["away_goals"] > row["home_goals"]:
                    points += 3 * weight
                elif row["away_goals"] == row["home_goals"]:
                    points += 1 * weight
        total_weight = sum(weights) or 1
        return points / total_weight, goals_scored / total_weight, goals_conceded / total_weight

    def get_specialness_score(self, matches, home_team, away_team, league):
        logger.debug(f"Calculating specialness score for {home_team} vs {away_team} in {league}")
        if (home_team, away_team) in self.famous_rivalries.get(league, []) or \
           (away_team, home_team) in self.famous_rivalries.get(league, []):
            base_score = 0.8
        else:
            base_score = 0.0
        
        h2h = get_head_to_head(matches, home_team, away_team)
        if not h2h.empty:
            avg_goal_diff = abs((h2h["home_goals"] - h2h["away_goals"]).mean())
            draw_rate = len(h2h[h2h["home_goals"] == h2h["away_goals"]]) / len(h2h)
            base_score += 0.3 * (1 - min(avg_goal_diff / 2, 1)) + 0.2 * draw_rate
        
        if any(word in away_team for word in home_team.split() if len(word) > 3):
            base_score += 0.2
        
        if league == "CL" and matches["date"].max().month in [2, 3, 4, 5]:
            base_score += 0.3
        
        return min(base_score, 1.0)

    def prepare_features(self, matches, upcoming):
        logger.info("Preparing features for prediction")
        features = []
        for _, match in upcoming.iterrows():
            home_team = match["home_team"]
            away_team = match["away_team"]
            league = match.get("league", "")
            
            logger.debug(f"Calculating form for team: {home_team}")
            home_form = self.calculate_form(matches, home_team)
            logger.debug(f"Calculating form for team: {away_team}")
            away_form = self.calculate_form(matches, away_team)
            
            # Get head-to-head history
            h2h = matches[
                ((matches["home_team"] == home_team) & (matches["away_team"] == away_team)) |
                ((matches["home_team"] == away_team) & (matches["away_team"] == home_team))
            ]
            
            h2h_goals_home = h2h[h2h["home_team"] == home_team]["home_goals"].mean() if not h2h.empty else 0
            h2h_goals_away = h2h[h2h["away_team"] == away_team]["away_goals"].mean() if not h2h.empty else 0
            
            logger.debug(f"Calculating specialness score for {home_team} vs {away_team} in {league}")
            special_score = self.get_specialness_score(matches, home_team, away_team, league)
            
            # Create feature array and handle NaN values
            feature_array = [
                home_form[0], home_form[1], home_form[2],
                away_form[0], away_form[1], away_form[2],
                h2h_goals_home, h2h_goals_away,
                special_score, 1, 0
            ]
            
            # Replace NaN values with 0 (for new teams or missing data)
            feature_array = [0 if pd.isna(x) else x for x in feature_array]
            features.append(feature_array)
            
        return np.array(features)

    def prepare_lstm_data(self, matches, n_timesteps=10):
        logger.info("Preparing LSTM data")
        X, y_home, y_away = [], [], []
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce").dt.tz_localize(None)
        teams = matches["home_team"].unique()
        for team in teams:
            team_matches = matches[(matches["home_team"] == team) | (matches["away_team"] == team)].sort_values("date")
            for i in range(len(team_matches) - n_timesteps):
                seq = team_matches.iloc[i:i+n_timesteps]
                next_match = team_matches.iloc[i+n_timesteps]
                seq_features = []
                for _, row in seq.iterrows():
                    is_home = 1 if row["home_team"] == team else 0
                    goals = row["home_goals"] if is_home else row["away_goals"]
                    conceded = row["away_goals"] if is_home else row["home_goals"]
                    seq_features.append([goals, conceded, is_home])
                X.append(seq_features)
                y_home.append(next_match["home_goals"] if next_match["home_team"] == team else 0)
                y_away.append(next_match["away_goals"] if next_match["away_team"] == team else 0)
        return np.array(X), np.array(y_home), np.array(y_away)

    def train(self, matches):
        logger.info("Starting model training")
        if matches.empty:
            logger.warning("No historical data available to train the model.")
            return False
        
        X = self.prepare_features(matches, matches)
        y_home = matches["home_goals"].values
        y_away = matches["away_goals"].values
        
        X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2)
        _, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2)
        
        logger.info("Training XGBoost models")
        self.xgb_home.fit(X_train, y_home_train)
        self.xgb_away.fit(X_train, y_away_train)
        logger.info("Training Random Forest models")
        self.rf_home.fit(X_train, y_home_train)
        self.rf_away.fit(X_train, y_away_train)
        logger.info("Training Poisson models")
        self.poisson_home.fit(X_train, y_home_train)
        self.poisson_away.fit(X_train, y_away_train)
        
        logger.info("Training LSTM model")
        X_lstm, y_lstm_home, y_lstm_away = self.prepare_lstm_data(matches)
        self.lstm_model = Sequential([
            LSTM(50, input_shape=(10, 3), return_sequences=False),
            Dense(20, activation="relu"),
            Dense(2)
        ])
        self.lstm_model.compile(optimizer="adam", loss="mean_squared_error")
        self.lstm_model.fit(X_lstm, np.column_stack((y_lstm_home, y_lstm_away)), epochs=10, batch_size=32, verbose=1)
        
        logger.info("Saving trained models")
        joblib.dump({
            "xgb_home": self.xgb_home, "xgb_away": self.xgb_away,
            "rf_home": self.rf_home, "rf_away": self.rf_away,
            "poisson_home": self.poisson_home, "poisson_away": self.poisson_away
        }, self.model_path)
        self.lstm_model.save("lstm_model.h5")
        
        for model_name, model_home, model_away in [
            ("XGBoost", self.xgb_home, self.xgb_away),
            ("Random Forest", self.rf_home, self.rf_away),
            ("Poisson", self.poisson_home, self.poisson_away)
        ]:
            home_pred = model_home.predict(X_test)
            away_pred = model_away.predict(X_test)
            logger.info(f"{model_name} Home RMSE: {np.sqrt(mean_squared_error(y_home_test, home_pred))}")
            logger.info(f"{model_name} Away RMSE: {np.sqrt(mean_squared_error(y_away_test, away_pred))}")
        return True

    def predict(self, matches, upcoming):
        logger.info("Starting prediction process")
        if not os.path.exists(self.model_path) or not os.path.exists("lstm_model.h5"):
            logger.info("No pre-trained model found. Training with historical data...")
            success = self.train(matches)
            if not success:
                logger.warning("Training failed, returning default predictions")
                return pd.DataFrame({
                    "match_id": upcoming["match_id"],
                    "predicted_home_goals": [0] * len(upcoming),
                    "predicted_away_goals": [0] * len(upcoming),
                    "confidence": [0.5] * len(upcoming),
                    "reasoning": ["Insufficient data for prediction"] * len(upcoming),
                    "upset_potential": [0] * len(upcoming)
                })
        else:
            logger.info("Loading pre-trained models")
            models = joblib.load(self.model_path)
            self.xgb_home, self.xgb_away = models["xgb_home"], models["xgb_away"]
            self.rf_home, self.rf_away = models["rf_home"], models["rf_away"]
            self.poisson_home, self.poisson_away = models["poisson_home"], models["poisson_away"]
            try:
                self.lstm_model = tf.keras.models.load_model("lstm_model.h5")
                logger.info("LSTM model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LSTM model: {e}. Retraining...")
                success = self.train(matches)
                if not success:
                    logger.warning("Retraining failed, returning default predictions")
                    return pd.DataFrame({
                        "match_id": upcoming["match_id"],
                        "predicted_home_goals": [0] * len(upcoming),
                        "predicted_away_goals": [0] * len(upcoming),
                        "confidence": [0.5] * len(upcoming),
                        "reasoning": ["Insufficient data for prediction"] * len(upcoming),
                        "upset_potential": [0] * len(upcoming)
                    })
        
        logger.debug(f"Matches shape: {matches.shape}, Upcoming shape: {upcoming.shape}")
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce").dt.tz_localize(None)
        upcoming["date"] = pd.to_datetime(upcoming["date"], errors="coerce").dt.tz_localize(None)
        
        logger.info("Preparing features")
        X = self.prepare_features(matches, upcoming)
        logger.info("Preparing LSTM data")
        lstm_X = self.prepare_lstm_data(pd.concat([matches, upcoming]), n_timesteps=10)[-len(upcoming):]
        
        logger.info("Making predictions")
        xgb_home_preds = self.xgb_home.predict(X)
        xgb_away_preds = self.xgb_away.predict(X)
        rf_home_preds = self.rf_home.predict(X)
        rf_away_preds = self.rf_away.predict(X)
        poisson_home_preds = self.poisson_home.predict(X)
        poisson_away_preds = self.poisson_away.predict(X)
        lstm_preds = self.lstm_model.predict(lstm_X, verbose=0)
        
        weights = {"xgb": 0.3, "rf": 0.2, "poisson": 0.3, "lstm": 0.2}
        home_preds = (weights["xgb"] * xgb_home_preds + weights["rf"] * rf_home_preds +
                      weights["poisson"] * poisson_home_preds + weights["lstm"] * lstm_preds[:, 0])
        away_preds = (weights["xgb"] * xgb_away_preds + weights["rf"] * rf_away_preds +
                      weights["poisson"] * poisson_away_preds + weights["lstm"] * lstm_preds[:, 1])
        
        confidence = []
        upset_potential = []
        reasoning = []
        for i, (_, fixture) in enumerate(upcoming.iterrows()):
            home_team = fixture["home_team"]
            away_team = fixture["away_team"]
            league = fixture.get("league", "PL")
            home_form = self.calculate_form(matches, home_team)
            away_form = self.calculate_form(matches, away_team)
            h2h = get_head_to_head(matches, home_team, away_team)
            special_score = self.get_specialness_score(matches, home_team, away_team, league)
            
            h2h_summary = f"H2H: {len(h2h)} matches, {home_team} avg {h2h[h2h['home_team'] == home_team]['home_goals'].mean():.1f} goals" if not h2h.empty else "No recent H2H"
            special_text = f" (Special Match: {special_score:.2f})" if special_score > 0.5 else ""
            
            home_var = np.var([xgb_home_preds[i], rf_home_preds[i], poisson_home_preds[i], lstm_preds[i, 0]])
            away_var = np.var([xgb_away_preds[i], rf_away_preds[i], poisson_away_preds[i], lstm_preds[i, 1]])
            conf = 0.9 - (home_var + away_var) * 0.1
            confidence.append(max(0.5, min(conf, 0.95)))
            
            form_diff = abs(home_form[1] - away_form[1])
            underdog = "away" if home_form[1] > away_form[1] else "home"
            upset = 1 if (form_diff < 0.5 and confidence[i] < 0.75 and \
                          ((underdog == "away" and away_preds[i] > home_preds[i]) or \
                           (underdog == "home" and home_preds[i] > away_preds[i]))) else 0
            upset_potential.append(upset)
            
            if upset and underdog == "away":
                away_preds[i] += 0.5
            elif upset and underdog == "home":
                home_preds[i] += 0.5
            
            upset_text = " Potential upset: underdog has a chance due to close form." if upset else ""
            reason = (
                f"{home_team} (Form: {home_form[1]:.1f} goals scored, {home_form[2]:.1f} conceded) "
                f"vs {away_team} (Form: {away_form[1]:.1f} goals scored, {away_form[2]:.1f} conceded). "
                f"{h2h_summary}{special_text}.{upset_text} Ensemble prediction based on form, H2H, and context."
            )
            reasoning.append(reason)
        
        logger.info("Prediction process completed")
        return pd.DataFrame({
            "match_id": upcoming["match_id"],
            "predicted_home_goals": home_preds,
            "predicted_away_goals": away_preds,
            "confidence": confidence,
            "reasoning": reasoning,
            "upset_potential": upset_potential
        })

    def update_and_retrain(self, matches):
        logger.info("Updating and retraining model")
        if self.app is None:
            logger.warning("No Flask app provided. Skipping database update.")
            self.train(matches)
            return
        
        with self.app.app_context():
            predictions = Prediction.query.all()
            for pred in predictions:
                match = Match.query.filter_by(match_id=pred.match_id).first()
                if match and match.home_goals is not None and pred.actual_home_goals is None:
                    pred.actual_home_goals = match.home_goals
                    pred.actual_away_goals = match.away_goals
                    db.session.add(pred)
            db.session.commit()
        self.train(matches)