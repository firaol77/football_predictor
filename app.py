import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
import schedule
import threading
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_cors import cross_origin
from functools import lru_cache
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import traceback
import random
import xgboost as xgb
import numpy as np
random.seed(42)
np.random.seed(42)
from sklearn.calibration import CalibratedClassifierCV
import logging
from scipy.stats import poisson
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report


app = Flask(__name__)
CORS(app, resources={
    r"/*": {  # Allow all routes
        "origins": ["http://localhost:5174"],  # Add your frontend URL
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# API Configurations
FOOTBALL_DATA_TOKEN = "a8191890ffa24ae89288882c6c6136f6"
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
leagues = {
    "PL": "Premier League",
    "PD": "Primera Division (La Liga)",
    "BL1": "Bundesliga",
    "DED": "Eredivisie",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "ELC": "Championship",
    "PPL": "Primeira Liga",
    "CL": "UEFA Champions League",
    "EC": "European Championship",
    "BSA": "Campeonato Brasileiro Série A"
}

PREDICTIONS_FILE = "predictions_feedback.json"

# Rate limiting configuration
RATE_LIMIT = {
    'calls': 0,
    'timestamp': time.time(),
    'limit': 10,  # Maximum calls per minute
    'window': 60  # Time window in seconds
}

def rate_limit_check():
    """Check and handle rate limiting"""
    current_time = time.time()
    if current_time - RATE_LIMIT['timestamp'] > RATE_LIMIT['window']:
        # Reset if window has passed
        RATE_LIMIT['calls'] = 0
        RATE_LIMIT['timestamp'] = current_time
    
    if RATE_LIMIT['calls'] >= RATE_LIMIT['limit']:
        sleep_time = RATE_LIMIT['window'] - (current_time - RATE_LIMIT['timestamp'])
        if sleep_time > 0:
            time.sleep(sleep_time)
            RATE_LIMIT['calls'] = 0
            RATE_LIMIT['timestamp'] = time.time()
    
    RATE_LIMIT['calls'] += 1

# Cache for API responses
CACHE = {}
CACHE_DURATION = timedelta(minutes=30)

def get_cached_data(key):
    """Get data from cache if available and not expired"""
    if key in CACHE:
        data, timestamp = CACHE[key]
        if datetime.now() - timestamp < CACHE_DURATION:
            return data
        del CACHE[key]
    return None

def set_cached_data(key, data):
    """Store data in cache with timestamp"""
    CACHE[key] = (data, datetime.now())

def load_predictions() -> List[Dict]:
    """
    Load predictions from the predictions_feedback.json file with validation and deduplication.
    Returns an empty list if validation fails.
    """ 
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            predictions = json.load(f)

        # Deduplicate predictions
        seen_keys = set()
        deduplicated_predictions = []
        for prediction in predictions:
            prediction_key = f"{prediction['match_id']}_{prediction['home_team']}_{prediction['away_team']}"
            if prediction_key not in seen_keys:
                seen_keys.add(prediction_key)
                deduplicated_predictions.append(prediction)

        return deduplicated_predictions
    return []

def save_prediction(match_id: int, home_team: str, away_team: str, league_id: str, prediction: Dict, actual_result: Optional[Dict] = None):
    try:
        # Load existing predictions
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, "r") as f:
                predictions = json.load(f)
        else:
            predictions = []

        # Create a unique key for the prediction
        prediction_key = f"{match_id}_{home_team}_{away_team}"

        # Check if the prediction already exists
        existing_prediction = next(
            (
                p
                for p in predictions
                if f"{p['match_id']}_{p['home_team']}_{p['away_team']}" == prediction_key
            ),
            None,
        )

        if existing_prediction:
            # Update the existing prediction only if there's new information
            if actual_result and not existing_prediction.get("actual_result"):
                existing_prediction.update({
                    "actual_result": actual_result,
                    "updated_at": datetime.now().isoformat(),
                })
        else:
            # Add a new prediction
            predictions.append({
                "match_id": str(match_id),  # Ensure match_id is stored as a string
                "date": datetime.now().strftime("%Y-%m-%d"),
                "home_team": home_team,
                "away_team": away_team,
                "league_id": league_id,
                "prediction": prediction,
                "actual_result": actual_result,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            })

        # Convert all float32 values to native Python types
        predictions = convert_to_native_types(predictions)

        # Write to a temporary file first to ensure atomicity
        temp_file = PREDICTIONS_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(predictions, f, indent=4)

        # Replace the original file with the temporary file
        os.replace(temp_file, PREDICTIONS_FILE)

        logging.info(f"Prediction saved/updated for match ID {match_id}")
    except Exception as e:
        logging.error(f"Error saving prediction: {str(e)}")
        traceback.print_exc()
def ensure_match_ids_as_strings(file_path: str):
    """
    Ensure all match_id values in the predictions_feedback.json file are stored as strings.
    """
    if not os.path.exists(file_path):
        logging.info(f"File {file_path} does not exist. Skipping validation.")
        return

    try:
        # Load existing predictions
        with open(file_path, "r") as f:
            predictions = json.load(f)

        # Check and convert match_id values to strings
        modified = False
        for entry in predictions:
            if isinstance(entry["match_id"], int):  # If match_id is an integer
                entry["match_id"] = str(entry["match_id"])  # Convert to string
                modified = True

        # Save the updated predictions back to the file if any changes were made
        if modified:
            temp_file = file_path + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(predictions, f, indent=4)
            os.replace(temp_file, file_path)
            logging.info("Converted all match_id values to strings.")
        else:
            logging.info("All match_id values are already strings.")

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {str(e)}")
        logging.info("Resetting the file to an empty list.")
        with open(file_path, "w") as f:
            json.dump([], f)  # Reset to an empty list
    except Exception as e:
        logging.error(f"Error validating {file_path}: {str(e)}")
        with open(file_path, "w") as f:
            json.dump([], f)  # Reset to an empty list        
def validate_predictions_file(file_path: str):
    """
    Validate and initialize the predictions_feedback.json file.
    Also ensures all match_ids are stored as strings.
    """
    if not os.path.exists(file_path):
        logging.info(f"File {file_path} does not exist. Creating an empty list.")
        with open(file_path, "w") as f:
            json.dump([], f)  # Initialize with an empty list
        return

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("File content is not a list.")
            
            # Check and convert match_id values to strings
            modified = False
            for entry in data:
                if isinstance(entry.get("match_id"), int):
                    entry["match_id"] = str(entry["match_id"])
                    modified = True
            
            # Save the updated predictions back to the file if any changes were made
            if modified:
                temp_file = file_path + ".tmp"
                with open(temp_file, "w") as f:
                    json.dump(data, f, indent=4)
                os.replace(temp_file, file_path)
                logging.info("Converted all match_id values to strings.")
            else:
                logging.info("All match_id values are already strings.")

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {str(e)}")
        logging.info("Resetting the file to an empty list.")
        with open(file_path, "w") as f:
            json.dump([], f)  # Reset to an empty list
    except Exception as e:
        logging.error(f"Error validating {file_path}: {str(e)}")
        with open(file_path, "w") as f:
            json.dump([], f)  # Reset to an empty list

def convert_to_native_types(data):
    """
    Recursively converts all float32 and int32 values in a dictionary or list to native Python types.
    """
    if isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, float):  # Handle numpy float types
        return float(data)
    elif isinstance(data, int):  # Handle numpy int types
        return int(data)
    else:
        return data

class FootballDataAPI:
    def __init__(self):
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {"X-Auth-Token": "a8191890ffa24ae89288882c6c6136f6"}
        self.api_call_times = []  # Track API call timestamps
        self.data_dir = "data"  # Directory to store cached JSON files
        self._cache = {}
        os.makedirs(self.data_dir, exist_ok=True)  # Ensure the directory exists
        
    def get_team_name(self, team_id: int) -> str:
        """Fetch the name of a team by its ID."""
        try:
            logging.info(f"Fetching team name for team ID {team_id}")
            url = f"{self.base_url}/teams/{team_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            team_data = response.json()
            return team_data.get("name", "Unknown Team")
        except Exception as e:
            logging.error(f"Error fetching team name for ID {team_id}: {str(e)}")
            traceback.print_exc()
            return "Unknown Team"

    def _check_rate_limit(self):
        """Check and enforce rate limit."""
        current_time = time.time()
        # Remove calls older than 60 seconds
        self.api_call_times = [t for t in self.api_call_times if current_time - t < 60]
        if len(self.api_call_times) >= 10:
            # Calculate wait time until we can make another call
            wait_time = 60 - (current_time - self.api_call_times[0])
            if wait_time > 0:
                logging.info(f"Rate limit reached. Sleeping for {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        # Record the current API call
        self.api_call_times.append(current_time)

    def get_with_retry(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """Make an API request with retries and exponential backoff."""
        for attempt in range(max_retries):
            try:
                self._check_rate_limit()  # Ensure rate limit compliance
                logging.info(f"Attempt {attempt + 1} to fetch data from {url}")
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()  # Raise exception for HTTP errors
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:  # Forbidden (rate limit or permission issue)
                    logging.error(f"403 Forbidden: Check your API token and permissions.")
                    return None
                elif e.response.status_code == 429:  # Rate limit exceeded
                    sleep_time = (2 ** attempt) + np.random.uniform(0, 1)  # Exponential backoff with jitter
                    logging.info(f"Rate limit hit. Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"HTTP error during API call: {str(e)}")
                    return None
            except Exception as e:
                logging.error(f"Unexpected error during API call: {str(e)}")
                return None
        logging.error(f"Failed to fetch data after {max_retries} retries.")
        return None
    def get_new_matches(self, team_id: int, last_update_date: str, days: int = 750) -> List[Dict]:
        """
        Fetch new matches for a team since the last update date.
        Args:
            team_id (int): ID of the team.
            last_update_date (str): The last date when matches were fetched (YYYY-MM-DD).
            days (int): Maximum number of days to look back.
        Returns:
            List[Dict]: A list of new matches involving the team.
        """
        try:
            logging.info(f"Fetching new matches for team {team_id} since {last_update_date}")
            # Calculate the end date (today)
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Ensure the start date is not older than `days` ago
            start_date = max(last_update_date, (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"))
            
            # Build the API URL and parameters
            url = f"{self.base_url}/teams/{team_id}/matches"
            params = {
                "dateFrom": start_date,
                "dateTo": end_date,
                "status": "FINISHED",
                "limit": 100
            }
            
            # Fetch data with retry logic
            data = self.get_with_retry(url, params)
            if not data:
                logging.error("Failed to fetch new matches.")
                return []
            
            matches = data.get("matches", [])
            logging.info(f"Fetched {len(matches)} new matches for team {team_id}")
            return matches
        except Exception as e:
            logging.error(f"Error fetching new matches: {str(e)}")
            traceback.print_exc()
            return []

    def load_data_from_file(self, filename: str) -> Optional[Dict]:
        """Load data from a JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return None

    def save_data_to_file(self, filename: str, data: Dict):
        """Save data to a JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def get_fixtures(self, league_id: str) -> List[Dict]:
        """Get upcoming fixtures for a league."""
        try:
            self._check_rate_limit()  # Ensure rate limit compliance
            today = datetime.now().strftime("%Y-%m-%d")
            next_month = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            url = f"{self.base_url}/competitions/{league_id}/matches"
            params = {
                "dateFrom": today,
                "dateTo": next_month,
                "status": "SCHEDULED",
                "limit": 10
            }
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 429:  # Rate limit exceeded
                logging.info("Rate limit exceeded. Retrying after 10 seconds...")
                time.sleep(10)
                response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            matches = response.json().get('matches', [])
            matches = matches[:10]
            logging.info(f"Retrieved {len(matches)} fixtures")
            return matches
        except Exception as e:
            logging.error(f"Error fetching fixtures: {str(e)}")
            traceback.print_exc()
            return []


    def get_team_matches(self, team_id: int, limit: int = 10) -> List[Dict]:
        """Get recent matches for a team."""
        cache_key = f"team_matches_{team_id}"
        cached_data = get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._check_rate_limit()  # Ensure rate limit compliance
            url = f"{self.base_url}/teams/{team_id}/matches"
            params = {
                "limit": limit,
                "status": "FINISHED"
            }
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 429:
                logging.warning("Rate limit hit, waiting before retry...")
                time.sleep(61)
                return self.get_team_matches(team_id)
                
            response.raise_for_status()
            matches = response.json()['matches']
            
            # Calculate form statistics directly here
            form_data = {
                'matches': matches,
                'wins': sum(1 for match in matches if self._is_team_winner(match, team_id)),
                'draws': sum(1 for match in matches if self._is_match_draw(match)),
                'losses': len(matches) - sum(1 for match in matches if self._is_team_winner(match, team_id) or self._is_match_draw(match)),
                'goals_for': sum(self._get_team_goals(match, team_id) for match in matches),
                'goals_against': sum(self._get_opponent_goals(match, team_id) for match in matches),
                'clean_sheets': sum(1 for match in matches if self._get_opponent_goals(match, team_id) == 0),
                'form_score': 0,  # Will calculate below
                'consistency': 0  # Will calculate below
            }
            
            # Calculate form score and consistency
            total_matches = len(matches)
            if total_matches > 0:
                form_data['form_score'] = (form_data['wins'] * 3 + form_data['draws']) / (total_matches * 3)
                form_data['consistency'] = 1 - (form_data['losses'] / total_matches)
                form_data['goal_difference'] = form_data['goals_for'] - form_data['goals_against']
            
            set_cached_data(cache_key, form_data)
            return form_data
        except Exception as e:
            logging.error(f"Error fetching team matches: {str(e)}")
            return {'matches': [], 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0, 'clean_sheets': 0, 'form_score': 0, 'consistency': 0}

    def get_head_to_head(self, team1_id: int, team2_id: int, limit: int = 10) -> Dict:
        """Get head to head matches between two teams."""
        try:
            # Create cache key for head to head data
            cache_key = f"h2h_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
            cached_data = get_cached_data(cache_key)
            if cached_data:
                return cached_data

            # Get matches for both teams with a longer timeframe
            team1_matches = self.get_historical_matches(team1_id, days=1500)  # Increased timeframe
            team2_matches = self.get_historical_matches(team2_id, days=1500)  # Increased timeframe
            
            # Create sets of match IDs for efficient lookup
            team1_match_ids = {match['id']: match for match in team1_matches}
            team2_match_ids = {match['id']: match for match in team2_matches}
            
            # Find common matches
            common_match_ids = set(team1_match_ids.keys()) & set(team2_match_ids.keys())
            h2h_matches = [team1_match_ids[match_id] for match_id in common_match_ids]
            
            # Sort matches by date (most recent first)
            h2h_matches.sort(key=lambda x: x.get('utcDate', ''), reverse=True)
            h2h_matches = h2h_matches[:limit]
            
            # Calculate statistics
            h2h_data = {
                'matches': h2h_matches,
                'team1_wins': 0,
                'team2_wins': 0,
                'draws': 0,
                'team1_goals': 0,
                'team2_goals': 0
            }
            
            for match in h2h_matches:
                score = match.get('score', {}).get('fullTime', {})
                home_score = score.get('home', 0)
                away_score = score.get('away', 0)
                
                # Determine if team1 was home or away
                team1_is_home = match.get('homeTeam', {}).get('id') == team1_id
                
                # Add goals
                if team1_is_home:
                    h2h_data['team1_goals'] += home_score
                    h2h_data['team2_goals'] += away_score
                else:
                    h2h_data['team1_goals'] += away_score
                    h2h_data['team2_goals'] += home_score
                
                # Calculate result
                if home_score > away_score:
                    if team1_is_home:
                        h2h_data['team1_wins'] += 1
                    else:
                        h2h_data['team2_wins'] += 1
                elif away_score > home_score:
                    if team1_is_home:
                        h2h_data['team2_wins'] += 1
                    else:
                        h2h_data['team1_wins'] += 1
                else:
                    h2h_data['draws'] += 1
            
            # Cache the data
            set_cached_data(cache_key, h2h_data)
            
            if h2h_matches:
                logging.info(f"Found {len(h2h_matches)} head-to-head matches")
                
            return h2h_data
            
        except Exception as e:
            logging.error(f"Error fetching head to head matches: {str(e)}")
            return {
                'matches': [],
                'team1_wins': 0,
                'team2_wins': 0,
                'draws': 0,
                'team1_goals': 0,
                'team2_goals': 0
            }

    def get_historical_matches(self, team_id: int, days: int = 750) -> List[Dict]:
        """Fetch historical matches for a team (last X days)."""
        try:
            self._check_rate_limit()
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            url = f"{self.base_url}/teams/{team_id}/matches"
            params = {
                "dateFrom": start_date,
                "dateTo": end_date,
                "status": "FINISHED",
                "limit": 100
            }
            data = self.get_with_retry(url, params)
            if not data:
                logging.error("Failed to fetch historical matches.")
                return []
            matches = data.get("matches", [])
            logging.info(f"Fetched {len(matches)} historical matches for team {team_id}")
            return matches
        except Exception as e:
            logging.error(f"Error fetching historical matches: {str(e)}")
            traceback.print_exc()
            return []

    def get_selected_fixture(self, fixture_id: int) -> Optional[Dict]:
        """Fetch a specific fixture by ID."""
        filename = f"fixture_{fixture_id}.json"
        cached_data = self.load_data_from_file(filename)
        if cached_data:
            logging.info(f"Using cached fixture data for ID {fixture_id}")
            return cached_data

        try:
            url = f"{self.base_url}/matches/{fixture_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            fixture = response.json()
            self.save_data_to_file(filename, fixture)
            logging.info(f"Fetched and saved fixture data for ID {fixture_id}")
            return fixture
        except Exception as e:
            logging.error(f"Error fetching fixture: {str(e)}")
            traceback.print_exc()
            return None



class FootballPredictor:
    def __init__(self, league_id: str, league_name: str):
        """Initialize the predictor with league information."""
        self.league_id = league_id
        self.league_name = league_name
        self.api = FootballDataAPI()
        self.cache = {}  # Add cache initialization
        self.cache_timeout = 3600  # Cache timeout in seconds (1 hour)
        
        # Add rate limiting parameters
        self.last_request_time = time.time()
        self.min_request_interval = 1.0  # Minimum time between requests in seconds
        
        # Initialize model with consistent parameters
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42
        )
        
        # Start accuracy tracking scheduler
        self.schedule_accuracy_updates()
        # Add caching and API call tracking
        self._cache = {}
        self._api_calls = 0
        self._last_call_time = time.time()
        
        # Train the model during initialization
        self._train_model()
        
        logging.info(f"Enhanced predictor initialized for {league_name} (ID: {league_id})")
    
    def schedule_accuracy_updates(self):
        """
        Set up scheduled tasks for updating predictions and logging accuracy.
        """       
        
        
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour
        
        def update_and_log_accuracy():
            try:
                self.update_predictions_with_results()
                accuracy = self.calculate_accuracy()
                logging.info(f"Scheduled accuracy update: {accuracy}")
                
                # Save accuracy metrics to file
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metrics = {
                    "timestamp": timestamp,
                    "metrics": accuracy
                }
                
                with open("accuracy_history.json", "a+") as f:
                    f.seek(0)
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                    data.append(metrics)
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=4)
                    
            except Exception as e:
                logging.error(f"Error in scheduled accuracy update: {str(e)}")
        
        # Schedule daily updates at midnight
        schedule.every().day.at("00:00").do(update_and_log_accuracy)
        
        # Run the scheduler in a separate thread
        scheduler_thread = threading.Thread(target=run_schedule, daemon=True)
        scheduler_thread.start()
        
        logging.info("Accuracy tracking scheduler initialized")

        
    def calculate_match_probabilities(self, home_strength: float, away_strength: float) -> Dict[str, float]:
        """Calculate match outcome probabilities."""
        try:
            # Base probabilities from team strengths
            home_win_prob = home_strength * (1 - away_strength)
            away_win_prob = away_strength * (1 - home_strength)
            draw_prob = 1 - (home_win_prob + away_win_prob)

            # Normalize probabilities
            total = home_win_prob + away_win_prob + draw_prob
            return {
                'home_win': home_win_prob / total,  # Changed to 'home_win'
                'away_win': away_win_prob / total,  # Changed to 'away_win'
                'draw': draw_prob / total
            }
        except Exception as e:
            logging.error(f"Error calculating match probabilities: {str(e)}")
            return {'home_win': 0.33, 'away_win': 0.33, 'draw': 0.34}

    def _train_model(self):
        """Train the model with available data."""
        try:
            # Load training data
            X, y = self.load_training_data()
            if X is not None and y is not None and len(X) > 0:
                # Train the model
                self.model.fit(X, y)
                logging.info("Model trained successfully")
            else:
                logging.warning("No training data available, model will use default parameters")
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            traceback.print_exc()
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare training data.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features DataFrame and labels Series
        """
        try:
            logging.info("Loading training data...")
            
            # Define feature columns to ensure consistency
            self.feature_columns = [
                'home_wins', 'home_draws', 'home_losses',
                'home_goals_for', 'home_goals_against',
                'home_matches_played', 'home_win_rate',
                'home_form_rating', 'home_points_per_game',
                
                'away_wins', 'away_draws', 'away_losses',
                'away_goals_for', 'away_goals_against',
                'away_matches_played', 'away_win_rate',
                'away_form_rating', 'away_points_per_game',
                
                'h2h_matches', 'h2h_home_wins', 'h2h_away_wins',
                'h2h_draws'
            ]

            # Generate training data
            X, y = self._generate_training_data(None, None)
            
            if len(X) == 0:
                logging.warning("No training data available")
                return pd.DataFrame(), pd.Series()

            # Create DataFrame with correct feature columns
            df = pd.DataFrame(X, columns=self.feature_columns)
            labels = pd.Series(y)

            logging.info(f"Loaded {len(df)} training samples")
            return df, labels

        except Exception as e:
            logging.error(f"Error loading training data: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame(), pd.Series()
        
    def calculate_defensive_strength(self, recent_results: List[Dict]) -> float:
        """Calculate defensive strength based on recent results."""
        try:
            if not recent_results:
                return 0.5

            goals_conceded = 0
            clean_sheets = 0
            matches_analyzed = 0

            for match in recent_results:
                try:
                    goals_against = match.get('goals_against', 0)
                    goals_conceded += goals_against
                    if goals_against == 0:
                        clean_sheets += 1
                    matches_analyzed += 1
                except Exception as e:
                    logging.warning(f"Error processing match in defensive strength: {str(e)}")
                    continue

            if matches_analyzed == 0:
                return 0.5

            # Calculate defensive rating (0-1 scale)
            avg_goals_conceded = goals_conceded / matches_analyzed
            clean_sheet_ratio = clean_sheets / matches_analyzed

            # Weight the factors
            defensive_strength = (
                (1 - min(avg_goals_conceded / 3, 1)) * 0.7 +  # Lower goals conceded is better
                clean_sheet_ratio * 0.3                        # More clean sheets is better
            )

            return defensive_strength

        except Exception as e:
            logging.error(f"Error calculating defensive strength: {str(e)}")
            return 0.5

    def analyze_winning_streak(self, recent_results, team_id):
        """Analyze quality of winning streak"""
        recent_matches = recent_results[-5:]  # Last 5 matches
        if not recent_matches:
            return 0.5  # Default neutral value
        
        streak_metrics = {
            'wins': 0,
            'goal_margin': 0,
            'consecutive_wins': 0
        }
        
        current_streak = 0
        for match in recent_matches:
            if match['result'] == 'W':
                streak_metrics['wins'] += 1
                current_streak += 1
                # Calculate goal margin
                scores = match['score'].split('-')
                team_score = int(scores[0] if match['was_home'] else scores[1])
                opponent_score = int(scores[1] if match['was_home'] else scores[0])
                streak_metrics['goal_margin'] += team_score - opponent_score
            else:
                current_streak = 0
            
            streak_metrics['consecutive_wins'] = max(streak_metrics['consecutive_wins'], current_streak)

        matches_played = len(recent_matches)
        streak_score = (
            (streak_metrics['wins'] / matches_played) * 0.4 +
            (min(streak_metrics['goal_margin'] / matches_played / 2, 1)) * 0.3 +
            (streak_metrics['consecutive_wins'] / matches_played) * 0.3
        )
        
        return streak_score

    def identify_upset_potential(self, underdog_form, favorite_form, h2h_stats):
        """Calculate potential for an upset"""
        # Get team ID from the team object instead of directly from form
        underdog_team_id = underdog_form.get('team', {}).get('id')
        
        # Calculate defensive strength using recent_results and correct team ID
        underdog_defense = self.calculate_defensive_strength(underdog_form['recent_results'])
        favorite_attack = favorite_form.get('goals_for', 0) / max(favorite_form.get('matches_played', 1), 1)
        
        # Check H2H performance
        h2h_factor = 0.5
        if h2h_stats.get('total_matches', 0) > 0:
            underdog_wins = h2h_stats.get('away_wins' if underdog_form.get('is_away') else 'home_wins', 0)
            h2h_factor = underdog_wins / h2h_stats['total_matches']
        
        # Calculate upset potential
        upset_score = (
            underdog_defense * 0.4 +
            (1 - min(favorite_attack / 3, 1)) * 0.4 +
            h2h_factor * 0.2
        )
        
        return upset_score
    
    
    def calculate_accuracy(self) -> Dict[str, float]:
        """
        Calculate prediction accuracy metrics from stored predictions.
        
        Returns:
            Dict[str, float]: Dictionary containing various accuracy metrics
        """
        try:
            predictions = load_predictions()
            if not predictions:
                logging.warning("No predictions found for accuracy calculation")
                return {
                    'overall_accuracy': 0.0,
                    'home_accuracy': 0.0,
                    'draw_accuracy': 0.0,
                    'away_accuracy': 0.0,
                    'total_predictions': 0
                }

            metrics = {
                'correct_predictions': 0,
                'correct_home': 0,
                'correct_draw': 0,
                'correct_away': 0,
                'total_home': 0,
                'total_draw': 0,
                'total_away': 0,
                'total_predictions': 0
            }

            for pred in predictions:
                if not pred.get('actual_result'):
                    continue

                predicted = pred['prediction']['prediction']
                actual = pred['actual_result']['outcome']
                
                # Update total counters
                metrics['total_predictions'] += 1
                if predicted == 'Home Win':
                    metrics['total_home'] += 1
                elif predicted == 'Draw':
                    metrics['total_draw'] += 1
                else:  # Away Win
                    metrics['total_away'] += 1

                # Check if prediction was correct
                if predicted == actual:
                    metrics['correct_predictions'] += 1
                    if predicted == 'Home Win':
                        metrics['correct_home'] += 1
                    elif predicted == 'Draw':
                        metrics['correct_draw'] += 1
                    else:  # Away Win
                        metrics['correct_away'] += 1

            # Calculate accuracy percentages
            results = {
                'overall_accuracy': metrics['correct_predictions'] / metrics['total_predictions'] if metrics['total_predictions'] > 0 else 0.0,
                'home_accuracy': metrics['correct_home'] / metrics['total_home'] if metrics['total_home'] > 0 else 0.0,
                'draw_accuracy': metrics['correct_draw'] / metrics['total_draw'] if metrics['total_draw'] > 0 else 0.0,
                'away_accuracy': metrics['correct_away'] / metrics['total_away'] if metrics['total_away'] > 0 else 0.0,
                'total_predictions': metrics['total_predictions']
            }

            # Log the accuracy metrics
            logging.info(f"""
            Prediction Accuracy Metrics:
            - Overall Accuracy: {results['overall_accuracy']:.2%}
            - Home Win Accuracy: {results['home_accuracy']:.2%}
            - Draw Accuracy: {results['draw_accuracy']:.2%}
            - Away Win Accuracy: {results['away_accuracy']:.2%}
            - Total Predictions: {results['total_predictions']}
            """)

            return results

        except Exception as e:
            logging.error(f"Error calculating accuracy metrics: {str(e)}")
            traceback.print_exc()
            return {
                'overall_accuracy': 0.0,
                'home_accuracy': 0.0,
                'draw_accuracy': 0.0,
                'away_accuracy': 0.0,
                'total_predictions': 0
        }

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model's performance on test data.
        
        Args:
            X_test: Test features
            y_test: True labels
        
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        try:
            if not hasattr(self, 'model') or self.model is None:
                logging.error("Model not initialized for evaluation")
                return {}

            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Get unique classes in test data
            unique_classes = np.unique(y_test)
            n_classes = len(unique_classes)
            
            # Create mapping of actual classes to labels
            class_labels = ['Home Win', 'Draw', 'Away Win']
            present_labels = [class_labels[int(c)] for c in unique_classes if int(c) < len(class_labels)]
            
            # Initialize metrics dictionary
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'roc_auc': None,
                'n_classes_in_test': n_classes
            }

            # Try to calculate ROC AUC if possible
            try:
                y_pred_proba = self.model.predict_proba(X_test)
                if n_classes > 1:  # Only calculate ROC AUC if we have more than one class
                    metrics['roc_auc'] = roc_auc_score(
                        y_test, 
                        y_pred_proba,
                        multi_class='ovr',
                        labels=unique_classes
                    )
                else:
                    logging.warning("ROC AUC calculation skipped: Only one class present in test data")
            except Exception as e:
                logging.warning(f"Could not calculate ROC AUC: {str(e)}")

            # Log evaluation results
            logging.info(f"""
            Model Evaluation Metrics:
            - Number of classes in test data: {n_classes}
            - Classes present: {present_labels}
            - Accuracy: {metrics['accuracy']:.3f}
            - Precision: {metrics['precision']:.3f}
            - Recall: {metrics['recall']:.3f}
            - F1 Score: {metrics['f1']:.3f}
            - ROC AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'Not calculated'}
            """)

            # Only generate classification report if we have more than one class
            if n_classes > 1:
                try:
                    report = classification_report(y_test, y_pred, target_names=present_labels)
                    logging.info(f"Detailed Classification Report:\n{report}")
                except Exception as e:
                    logging.warning(f"Could not generate classification report: {str(e)}")
            else:
                logging.info("Classification report skipped: Only one class present in test data")

            return metrics

        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            traceback.print_exc()
            return {}

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model and evaluate its performance."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train base model
            base_model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=3,
                random_state=42
            )
            base_model.fit(X_train, y_train)

            # Calibrate the model
            calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
            calibrated_model.fit(X_train, y_train)

            # Evaluate the calibrated model
            evaluation_results = self.evaluate_model(calibrated_model, X_test, y_test)
            logging.info(f"Evaluation Results: {evaluation_results}")
        except Exception as e:
            logging.error(f"Error in train_and_evaluate: {str(e)}")
            traceback.print_exc()

    def _fine_tune_model(self, home_team_id: str, away_team_id: str) -> xgb.XGBClassifier:
        """
        Fine-tune model hyperparameters based on specific match context.
        """
        try:
            # Get training data
            X_train, y_train = self._generate_training_data(home_team_id, away_team_id)
            
            if X_train is None or y_train is None or len(X_train) == 0:
                logging.warning("Insufficient data for fine-tuning. Using default model configuration.")
                return xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=3,
                    max_depth=5,
                    learning_rate=0.1,
                    n_estimators=100,
                    random_state=42
                )
            
            # Define parameter grid
            param_grid = {
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [50, 100],
            }
            
            # Initialize base model
            base_model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                random_state=42
            )
            
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            logging.info(f"Best parameters found: {grid_search.best_params_}")
            logging.info(f"Best cross-validation accuracy: {grid_search.best_score_:.3f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logging.error(f"Error in model fine-tuning: {str(e)}")
            traceback.print_exc()
            return self.model  # Fall back to existing model

    def _generate_training_data(self, home_team_id: int, away_team_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data using real matches and optionally synthetic data.
        
        Args:
            home_team_id: ID of home team (can be None)
            away_team_id: ID of away team (can be None)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and labels (y) for model training
        """
        try:
            logging.info("Generating training data...")
            X = []
            y = []
            seen_matches = set()

            # Validate team IDs and create list of valid IDs to process
            team_ids = [tid for tid in [home_team_id, away_team_id] if tid is not None and tid > 0]
            
            if not team_ids:
                logging.debug("Using synthetic data as team IDs not provided")
                return self._generate_synthetic_training_data()

            # Process each team's matches
            for team_id in team_ids:
                try:
                    # Add rate limiting delay
                    time.sleep(6)  # 10 requests per minute = 1 request every 6 seconds
                    
                    matches = self.api.get_historical_matches(team_id, days=750)
                    logging.info(f"Fetched {len(matches)} matches for team {team_id}")
                    
                    for match in matches:
                        try:
                            match_id = match.get("id")
                            if not match_id or match_id in seen_matches:
                                continue
                            seen_matches.add(match_id)

                            # Get team IDs from match
                            home_team_id_match = match["homeTeam"]["id"]
                            away_team_id_match = match["awayTeam"]["id"]
                            
                            # Get match outcome
                            outcome = self._determine_outcome(match)
                            if outcome is None:
                                continue

                            # Calculate team forms
                            home_form = self.calculate_team_form(matches, home_team_id_match)
                            away_form = self.calculate_team_form(matches, away_team_id_match)
                            
                            # Calculate H2H stats
                            h2h_stats = self.calculate_h2h_stats(
                                matches, 
                                home_team_id_match, 
                                away_team_id_match
                            )

                            # Prepare features
                            features = self.prepare_features(home_form, away_form, h2h_stats)
                            if not features:
                                continue

                            X.append(list(features.values()))
                            y.append(outcome)

                        except Exception as match_error:
                            logging.error(f"Error processing match {match.get('id', 'unknown')}: {str(match_error)}")
                            continue

                except Exception as team_error:
                    logging.error(f"Error processing team {team_id}: {str(team_error)}")
                    continue

            # Check if we have enough real data
            if len(X) < 100:  # Reduced minimum threshold due to API limitations
                logging.info(f"Insufficient real data ({len(X)} samples). Adding synthetic data...")
                synthetic_X, synthetic_y = self._generate_synthetic_training_data()
                
                if len(X) > 0:
                    X.extend(synthetic_X)
                    y.extend(synthetic_y)
                    logging.info(f"Combined {len(X)} real and synthetic samples")
                else:
                    X = synthetic_X
                    y = synthetic_y
                    logging.info(f"Using {len(X)} synthetic samples only")

            # Convert to numpy arrays
            X_array = np.array(X)
            y_array = np.array(y)

            # Validate final data
            if X_array.size == 0 or y_array.size == 0:
                logging.error("No valid training data generated")
                return np.array([]), np.array([])

            logging.info(f"Successfully generated {len(X_array)} training samples")
            return X_array, y_array

        except Exception as e:
            logging.error(f"Error in _generate_training_data: {str(e)}")
            traceback.print_exc()
            return np.array([]), np.array([])

    def _determine_outcome(self, match: Dict) -> Optional[int]:
        """
        Determine the outcome of a match (0: Home Win, 1: Draw, 2: Away Win).
        
        Args:
            match: Match data dictionary
        
        Returns:
            Optional[int]: Match outcome or None if invalid
        """
        try:
            score = match.get('score', {})
            if not score:
                return None

            # Try to get score from fullTime first, then extraTime, then penalties
            for score_type in ['fullTime', 'extraTime', 'penalties']:
                score_data = score.get(score_type, {})
                if score_data:
                    home_score = score_data.get('home')
                    away_score = score_data.get('away')
                    if home_score is not None and away_score is not None:
                        if home_score > away_score:
                            return 0  # Home win
                        elif home_score < away_score:
                            return 2  # Away win
                        else:
                            return 1  # Draw

            logging.warning(f"No valid score found for match {match.get('id', 'unknown')}")
            return None

        except Exception as e:
            logging.error(f"Error determining match outcome: {str(e)}")
            return None
        
    def _generate_synthetic_training_data(self) -> Tuple[List, List]:
        """
        Generate synthetic training data with realistic football statistics.
        
        Returns:
            Tuple[List, List]: Features (X) and labels (y) for model training
        """
        try:
            X, y = [], []
            samples_per_outcome = 333  # ~1000 total samples
            
            for outcome in range(3):  # 0: Home Win, 1: Draw, 2: Away Win
                for _ in range(samples_per_outcome):
                    # Generate base stats
                    home_matches = np.random.randint(8, 12)
                    away_matches = np.random.randint(8, 12)
                    
                    # Generate wins/draws/losses based on outcome
                    if outcome == 0:  # Home Win
                        home_wins = np.random.randint(4, 8)
                        home_draws = np.random.randint(1, 3)
                        away_wins = np.random.randint(2, 4)
                        away_draws = np.random.randint(1, 3)
                    elif outcome == 1:  # Draw
                        home_wins = np.random.randint(3, 5)
                        home_draws = np.random.randint(2, 4)
                        away_wins = np.random.randint(3, 5)
                        away_draws = np.random.randint(2, 4)
                    else:  # Away Win
                        home_wins = np.random.randint(2, 4)
                        home_draws = np.random.randint(1, 3)
                        away_wins = np.random.randint(4, 8)
                        away_draws = np.random.randint(1, 3)
                    
                    # Calculate losses
                    home_losses = max(0, home_matches - (home_wins + home_draws))
                    away_losses = max(0, away_matches - (away_wins + away_draws))
                    
                    # Calculate realistic goals
                    home_goals_for = home_wins * 2 + home_draws
                    home_goals_against = home_losses * 2 + home_draws
                    away_goals_for = away_wins * 2 + away_draws
                    away_goals_against = away_losses * 2 + away_draws
                    
                    # Generate H2H stats based on outcome
                    h2h_matches = np.random.randint(2, 6)
                    if outcome == 0:  # Home Win
                        h2h_home_wins = np.random.randint(2, h2h_matches + 1)
                        h2h_away_wins = np.random.randint(0, h2h_matches - h2h_home_wins + 1)
                    elif outcome == 2:  # Away Win
                        h2h_away_wins = np.random.randint(2, h2h_matches + 1)
                        h2h_home_wins = np.random.randint(0, h2h_matches - h2h_away_wins + 1)
                    else:  # Draw
                        h2h_home_wins = np.random.randint(1, 3)
                        h2h_away_wins = np.random.randint(1, 3)
                    
                    h2h_draws = max(0, h2h_matches - (h2h_home_wins + h2h_away_wins))
                    
                    # Create feature vector matching feature_columns order
                    features = [
                        home_wins, home_draws, home_losses,
                        home_goals_for, home_goals_against,
                        home_matches, home_wins/home_matches,
                        np.random.uniform(0.4, 0.8),  # home_form_rating
                        (home_wins * 3 + home_draws)/home_matches,  # home_points_per_game
                        
                        away_wins, away_draws, away_losses,
                        away_goals_for, away_goals_against,
                        away_matches, away_wins/away_matches,
                        np.random.uniform(0.4, 0.8),  # away_form_rating
                        (away_wins * 3 + away_draws)/away_matches,  # away_points_per_game
                        
                        h2h_matches, h2h_home_wins,
                        h2h_away_wins, h2h_draws
                    ]
                    
                    X.append(features)
                    y.append(outcome)
            
            logging.info(f"Generated {len(X)} synthetic training samples")
            return np.array(X), np.array(y)
            
        except Exception as e:
            logging.error(f"Error generating synthetic training data: {str(e)}")
            traceback.print_exc()
            return np.array([]), np.array([])

    
    def get_upcoming_fixtures(self) -> List[Dict]:
        """Fetch next 10 upcoming fixtures"""
        url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{self.league_id}/matches"
        params = {"status": "SCHEDULED,TIMED"}
        try:
            logging.info(f"Fetching fixtures from API: {url}")  # Debugging
            response = requests.get(url, headers=self.api.headers, params=params)
            response.raise_for_status()
            data = response.json()
            matches = data.get('matches', [])
            
            if not matches:
                logging.info("No matches found in the API response")  # Debugging
                return []
            
            # Sort matches by date, most recent first
            matches.sort(key=lambda x: x['utcDate'])
            logging.info(f"Found {len(matches)} matches")  # Debugging
            return matches[:10]  # Take only first 10 matches
        
        except requests.exceptions.RequestException as e:
            logging.info(f"Error fetching fixtures: {e}")  # Debugging
            return []
        except Exception as e:
            logging.info(f"Unexpected error: {e}")  # Debugging
            return []
    
    

    def analyze_match(self, home_team_id: int, away_team_id: int, league_id: str) -> Dict:
        """
        Analyze a match between two teams and provide comprehensive prediction.
        """
        try:
            logging.info(f"Analyzing match: {home_team_id} (Home) vs {away_team_id} (Away)")
            
            # Get team data and recent matches
            home_recent = self.api.get_team_matches(home_team_id) or []
            away_recent = self.api.get_team_matches(away_team_id) or []
            head_to_head_matches = self.get_head_to_head(home_team_id, away_team_id) or []
            
            # Calculate team forms
            home_form = self._get_team_form(home_recent, home_team_id)
            away_form = self._get_team_form(away_recent, away_team_id)
            
            # Add team information to forms
            home_form['team'] = {'id': home_team_id, 'name': self.api.get_team_name(home_team_id)}
            away_form['team'] = {'id': away_team_id, 'name': self.api.get_team_name(away_team_id)}
            
            # Calculate head-to-head stats
            head_to_head = self._get_team_form(head_to_head_matches, home_team_id)
            
            # Calculate prediction
            home_strength = home_form['wins'] * 3 + home_form['draws']
            away_strength = away_form['wins'] * 3 + away_form['draws']
            total_strength = home_strength + away_strength
            
            if total_strength == 0:
                home_prob = away_prob = 0.33
                draw_prob = 0.34
            else:
                home_prob = (home_strength / total_strength) * 0.8
                away_prob = (away_strength / total_strength) * 0.8
                draw_prob = 1 - home_prob - away_prob
            
            prediction = {
                'probabilities': {
                    'home': round(home_prob * 100, 2),
                    'away': round(away_prob * 100, 2),
                    'draw': round(draw_prob * 100, 2)
                },
                'predicted_winner': 'HOME_TEAM' if home_prob > away_prob else 'AWAY_TEAM' if away_prob > home_prob else 'DRAW',
                'confidence': round(max(home_prob, away_prob, draw_prob) * 100, 2),
                'score_prediction': [
                    round((home_form['goals_for'] / max(len(home_recent), 1)) * 1.1),
                    round(away_form['goals_for'] / max(len(away_recent), 1))
                ],
                'xg': {
                    'home': round((home_form['goals_for'] / max(len(home_recent), 1)) * 1.1, 2),
                    'away': round(away_form['goals_for'] / max(len(away_recent), 1), 2)
                }
            }
            
            analysis = {
                'analysis': {
                    'home_form': home_form,
                    'away_form': away_form,
                    'head_to_head': head_to_head,
                    'prediction': prediction
                }
            }
            
            print("\n=== ANALYSIS RESULT ===")
            print(json.dumps(analysis, indent=2))
            print("\n=== PREDICTION ONLY ===")
            print(json.dumps(prediction, indent=2))
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in analyze_match: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

    def _calculate_prediction_confidence(self, features: Dict) -> float:
        """Calculate confidence level of prediction based on available data."""
        try:
            confidence_factors = {
                'data_completeness': 0.0,
                'form_consistency': 0.0,
                'prediction_margin': 0.0
            }
            
            # Check data completeness
            expected_features = len(self.feature_columns)
            actual_features = len([v for v in features.values() if v is not None])
            confidence_factors['data_completeness'] = actual_features / expected_features
            
            # Check form consistency
            home_matches = features.get('home_matches_played', 0)
            away_matches = features.get('away_matches_played', 0)
            min_matches = 5
            confidence_factors['form_consistency'] = (
                min(home_matches, min_matches) + min(away_matches, min_matches)
            ) / (min_matches * 2)
            
            # Calculate prediction margin
            probabilities = sorted([
                features.get('home_win_rate', 0),
                features.get('away_win_rate', 0)
            ], reverse=True)
            confidence_factors['prediction_margin'] = (probabilities[0] - probabilities[1]) / probabilities[0] if probabilities[0] > 0 else 0
            
            # Calculate weighted confidence
            weights = {
                'data_completeness': 0.4,
                'form_consistency': 0.4,
                'prediction_margin': 0.2
            }
            
            confidence = sum(
                factor * weights[name]
                for name, factor in confidence_factors.items()
            )
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logging.error(f"Error calculating prediction confidence: {str(e)}")
            return 0.5

    def _identify_key_factors(self, home_form: Dict, away_form: Dict, h2h: Dict) -> List[str]:
        """Identify key factors influencing the prediction."""
        try:
            factors = []
            
            # Form comparison
            home_ppg = home_form.get('points', 0) / max(home_form.get('matches_played', 1), 1)
            away_ppg = away_form.get('points', 0) / max(away_form.get('matches_played', 1), 1)
            
            if abs(home_ppg - away_ppg) > 0.5:
                factors.append(
                    f"{'Home' if home_ppg > away_ppg else 'Away'} team showing significantly better form"
                )
            
            # Scoring trends
            home_gpm = home_form.get('goals_for', 0) / max(home_form.get('matches_played', 1), 1)
            away_gpm = away_form.get('goals_for', 0) / max(away_form.get('matches_played', 1), 1)
            
            if max(home_gpm, away_gpm) > 2:
                factors.append(
                    f"{'Home' if home_gpm > away_gpm else 'Away'} team showing strong attacking form"
                )
            
            # H2H dominance
            if h2h.get('total_matches', 0) >= 3:
                home_wins = h2h.get('home_wins', 0)
                away_wins = h2h.get('away_wins', 0)
                if abs(home_wins - away_wins) >= 2:
                    factors.append(
                        f"{'Home' if home_wins > away_wins else 'Away'} team has H2H advantage"
                    )
            
            # Recent momentum
            home_recent = home_form.get('recent_results', [])[-3:]
            away_recent = away_form.get('recent_results', [])[-3:]
            
            home_momentum = sum(1 for m in home_recent if m.get('result') == 'W')
            away_momentum = sum(1 for m in away_recent if m.get('result') == 'W')
            
            if max(home_momentum, away_momentum) >= 2:
                factors.append(
                    f"{'Home' if home_momentum > away_momentum else 'Away'} team has good momentum"
                )
            
            return factors[:3]  # Return top 3 most significant factors
            
        except Exception as e:
            logging.error(f"Error identifying key factors: {str(e)}")
            return ["Insufficient data for factor analysis"]
            
    def get_team_recent_matches(self, team_id: int, limit: int = 10) -> List[Dict]:
        """Get recent matches for a team"""
        return self.api.get_team_matches(team_id, limit)

    def get_head_to_head(self, team1_id: int, team2_id: int) -> List[Dict]:
        """Get head to head matches between two teams."""
        try:
            matches = []
            response = requests.get(
                f"{FOOTBALL_DATA_BASE_URL}/teams/{team1_id}/matches",
                headers={'X-Auth-Token': FOOTBALL_DATA_TOKEN},
                params={'limit': 100}
            )
            response.raise_for_status()
            
            all_matches = response.json().get('matches', [])
            
            # Filter for matches between the two teams
            for match in all_matches:
                if (match['homeTeam']['id'] == team1_id and match['awayTeam']['id'] == team2_id) or \
                   (match['homeTeam']['id'] == team2_id and match['awayTeam']['id'] == team1_id):
                    matches.append(match)
            
            return matches[:10]  # Return last 10 H2H matches
            
        except Exception as e:
            logging.error(f"Error fetching H2H matches: {str(e)}")
            return []

    def calculate_h2h_stats(self, matches: List[Dict], team1_id: int, team2_id: int) -> Dict:
        """Calculate head to head statistics."""
        try:
            stats = {
                'total_matches': 0,
                'team1_wins': 0,
                'team2_wins': 0,
                'draws': 0,
                'team1_goals': 0,
                'team2_goals': 0
            }
            
            for match in matches:
                home_id = match['homeTeam']['id']
                away_id = match['awayTeam']['id']
                score = match.get('score', {}).get('fullTime', {})
                home_score = score.get('home')
                away_score = score.get('away')
                
                if home_score is None or away_score is None:
                    continue
                
                stats['total_matches'] += 1
                
                if home_id == team1_id:
                    stats['team1_goals'] += home_score
                    stats['team2_goals'] += away_score
                    if home_score > away_score:
                        stats['team1_wins'] += 1
                    elif away_score > home_score:
                        stats['team2_wins'] += 1
                    else:
                        stats['draws'] += 1
                else:
                    stats['team1_goals'] += away_score
                    stats['team2_goals'] += home_score
                    if away_score > home_score:
                        stats['team1_wins'] += 1
                    elif home_score > away_score:
                        stats['team2_wins'] += 1
                    else:
                        stats['draws'] += 1
            
            # Calculate averages
            stats['team1_goals_avg'] = stats['team1_goals'] / max(stats['total_matches'], 1)
            stats['team2_goals_avg'] = stats['team2_goals'] / max(stats['total_matches'], 1)
            
            return stats
            
        except Exception as e:
            logging.error(f"Error calculating H2H stats: {str(e)}")
            return {
                'total_matches': 0,
                'team1_wins': 0,
                'team2_wins': 0,
                'draws': 0,
                'team1_goals': 0,
                'team2_goals': 0,
                'team1_goals_avg': 0,
                'team2_goals_avg': 0
            }

    def prepare_features(self, home_form: Dict, away_form: Dict, h2h_stats: Dict) -> Dict:
        """
        Prepare features for prediction.
        
        Args:
            home_form: Dictionary containing home team form data
            away_form: Dictionary containing away team form data
            h2h_stats: Dictionary containing head-to-head statistics
        
        Returns:
            Dict: Prepared features for prediction
        """
        try:
            # Calculate points per game
            home_points = (home_form.get('wins', 0) * 3 + home_form.get('draws', 0))
            away_points = (away_form.get('wins', 0) * 3 + away_form.get('draws', 0))
            home_matches = max(home_form.get('matches_played', 1), 1)  # Avoid division by zero
            away_matches = max(away_form.get('matches_played', 1), 1)
            
            features = {
                'home_wins': home_form.get('wins', 0),
                'home_draws': home_form.get('draws', 0),
                'home_losses': home_form.get('losses', 0),
                'home_goals_for': home_form.get('goals_for', 0),
                'home_goals_against': home_form.get('goals_against', 0),
                'home_matches_played': home_matches,
                'home_win_rate': home_form.get('win_rate', 0),
                'home_form_rating': home_form.get('form_rating', 0.5),
                'home_points_per_game': home_points / home_matches,
                
                'away_wins': away_form.get('wins', 0),
                'away_draws': away_form.get('draws', 0),
                'away_losses': away_form.get('losses', 0),
                'away_goals_for': away_form.get('goals_for', 0),
                'away_goals_against': away_form.get('goals_against', 0),
                'away_matches_played': away_matches,
                'away_win_rate': away_form.get('win_rate', 0),
                'away_form_rating': away_form.get('form_rating', 0.5),
                'away_points_per_game': away_points / away_matches,
                
                'h2h_matches': h2h_stats.get('total_matches', 0),
                'h2h_home_wins': h2h_stats.get('home_wins', 0),
                'h2h_away_wins': h2h_stats.get('away_wins', 0),
                'h2h_draws': h2h_stats.get('draws', 0)
            }

            # Validate all features are present
            missing_features = set(self.feature_columns) - set(features.keys())
            if missing_features:
                logging.error(f"Missing features: {missing_features}")
                raise ValueError(f"Missing required features: {missing_features}")
                
            return features

        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            traceback.print_exc()
            return {}

    def calculate_xg(self, team_form, opponent_form, h2h_stats, is_home):
        """Calculate more realistic xG based on form and H2H"""
        # Base xG from recent scoring rate
        base_goals = team_form.get('goals_for', 0)
        matches_played = max(team_form.get('matches_played', 1), 1)
        base_xg = base_goals / matches_played
        
        # Adjust for opponent's defensive strength
        opponent_goals_against = opponent_form.get('goals_against', 0)
        opponent_matches = max(opponent_form.get('matches_played', 1), 1)
        opponent_defense = 1 - (opponent_goals_against / (opponent_matches * 3))
        
        # Enhanced home/away factor
        venue_factor = 1.2 if is_home else 0.9
        
        # Consider team's win rate and form
        win_rate = team_form.get('win_rate', 0.5)
        form_factor = 1 + (win_rate - 0.5)
        
        # H2H scoring factor
        h2h_factor = 1.0
        if h2h_stats.get('total_matches', 0) > 0:
            if is_home:
                h2h_goals_avg = h2h_stats.get('home_goals_avg', 1.0)
            else:
                h2h_goals_avg = h2h_stats.get('away_goals_avg', 1.0)
            h2h_factor = max(0.8, min(1.3, h2h_goals_avg / 1.5))
        
        # Calculate final xG
        final_xg = base_xg * opponent_defense * venue_factor * form_factor * h2h_factor
        
        # Ensure minimum xG for strong teams
        if win_rate > 0.6:  # For teams in good form
            final_xg = max(final_xg, 1.0)
        
        return min(final_xg, 4.0)  # Cap at 4.0

    def adjust_probabilities(self, base_probs, home_form, away_form, h2h_stats):
        """Adjust probabilities based on form and team quality"""
        # Calculate form difference
        home_points = home_form.get('points', 0)
        away_points = away_form.get('points', 0)
        form_diff = (home_points - away_points) / max(home_points + away_points, 1)
        
        # Calculate quality gap
        quality_gap = abs(form_diff)
        
        # Adjust draw probability cap based on quality gap
        max_draw_prob = 0.35 - (quality_gap * 0.2)  # Reduces draw probability when teams are mismatched
        
        # Adjust base probabilities
        adjusted = base_probs.copy()
        
        # Apply home advantage
        home_advantage = 0.1 if home_form.get('win_rate', 0) > 0.5 else 0.05
        adjusted[0] += home_advantage
        adjusted[2] -= home_advantage
        
        # Apply form difference
        form_impact = form_diff * 0.2
        adjusted[0] += form_impact
        adjusted[2] -= form_impact
        
        # Cap probabilities
        adjusted = np.clip(adjusted, 0.15, 0.70)
        
        # Adjust draw probability
        if adjusted[1] > max_draw_prob:
            excess = adjusted[1] - max_draw_prob
            adjusted[1] = max_draw_prob
            # Distribute excess to home/away based on form
            if form_diff > 0:
                adjusted[0] += excess * 0.7
                adjusted[2] += excess * 0.3
            else:
                adjusted[0] += excess * 0.3
                adjusted[2] += excess * 0.7
        
        # Normalize
        adjusted /= adjusted.sum()
        
        return adjusted
    
    
    def predict_match(self, home_form: Dict, away_form: Dict, h2h_stats: Dict) -> Dict:
        """Predict match outcome with comprehensive analysis."""
        try:
            # Calculate team strengths
            home_attack = self.calculate_attacking_strength(home_form['recent_results'])
            home_defense = self.calculate_defensive_strength(home_form['recent_results'])
            away_attack = self.calculate_attacking_strength(away_form['recent_results'])
            away_defense = self.calculate_defensive_strength(away_form['recent_results'])

            # Calculate expected goals
            home_xg = self.calculate_xg(home_form, away_form, h2h_stats, is_home=True)
            away_xg = self.calculate_xg(away_form, home_form, h2h_stats, is_home=False)

            # Calculate win probabilities
            probabilities = self.calculate_match_probabilities(
                (home_attack + home_defense) / 2,
                (away_attack + away_defense) / 2
            )

            # Predict score
            score_prediction = self.predict_realistic_score(home_xg, away_xg)

            # Calculate confidence level
            confidence = self.calculate_prediction_confidence(
                home_form, away_form, h2h_stats,
                probabilities['home_win'], probabilities['away_win'], probabilities['draw']
            )

            return {
                'prediction': 'Home Win' if probabilities['home_win'] > max(probabilities['away_win'], probabilities['draw'])
                             else 'Away Win' if probabilities['away_win'] > max(probabilities['home_win'], probabilities['draw'])
                             else 'Draw',
                'probabilities': {
                    'home': round(probabilities['home_win'] * 100, 1),
                    'draw': round(probabilities['draw'] * 100, 1),
                    'away': round(probabilities['away_win'] * 100, 1)
                },
                'xg': {
                    'home': round(home_xg, 2),
                    'away': round(away_xg, 2)
                },
                'score_prediction': {
                    'home': score_prediction[0],
                    'away': score_prediction[1]
                },
                'confidence': confidence
            }

        except Exception as e:
            logging.error(f"Error in predict_match: {str(e)}")
            return {
                'prediction': 'Error',
                'probabilities': {'home': 33.3, 'draw': 33.4, 'away': 33.3},
                'xg': {'home': 1.0, 'away': 1.0},
                'score_prediction': {'home': 1, 'away': 1},
                'confidence': 0.5
            }

    def calculate_xg(self, team_form: Dict, opponent_form: Dict, h2h_stats: Dict, is_home: bool) -> float:
        """Calculate expected goals for a team."""
        try:
            # Base xG from recent scoring form
            base_xg = team_form.get('goals_per_game', 0)
            
            # Adjust for opponent's defensive strength
            opponent_defense = opponent_form.get('goals_against_per_game', 1)
            defensive_factor = min(opponent_defense / 1.5, 1.5)  # Cap the adjustment
            
            # Adjust for home/away factor
            venue_factor = 1.1 if is_home else 0.9
            
            # Consider H2H scoring history
            h2h_goals = h2h_stats.get('home_goals_avg' if is_home else 'away_goals_avg', base_xg)
            h2h_factor = 0.2  # Weight given to H2H history
            
            # Calculate final xG
            xg = (base_xg * defensive_factor * venue_factor * 0.8 +
                  h2h_goals * h2h_factor)
            
            return max(min(xg, 5.0), 0.1)  # Cap between 0.1 and 5.0
            
        except Exception as e:
            logging.error(f"Error calculating xG: {str(e)}")
            return 1.0

    def predict_realistic_score(self, home_xg: float, away_xg: float) -> Tuple[int, int]:
        """Predict a realistic score based on expected goals."""
        try:
            # Generate Poisson distributions for both teams
            max_goals = 5  # Maximum goals to consider
            home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
            away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
            
            # Find most likely score combination
            max_prob = 0
            best_score = (1, 1)  # Default score
            
            for home_goals in range(max_goals + 1):
                for away_goals in range(max_goals + 1):
                    prob = home_probs[home_goals] * away_probs[away_goals]
                    if prob > max_prob:
                        max_prob = prob
                        best_score = (home_goals, away_goals)
            
            return best_score
            
        except Exception as e:
            logging.error(f"Error predicting realistic score: {str(e)}")
            return (1, 1)

    def calculate_prediction_confidence(self, home_form: Dict, away_form: Dict, 
                                     h2h_stats: Dict, home_prob: float, 
                                     away_prob: float, draw_prob: float) -> float:
        """Calculate confidence level in prediction."""
        try:
            # Factors affecting confidence
            form_sample_size = min(home_form.get('matches_played', 0), 
                                 away_form.get('matches_played', 0)) / 10  # Scale to 0-1
            
            h2h_sample_size = min(h2h_stats.get('total_matches', 0), 10) / 10
            
            probability_margin = max(home_prob, away_prob, draw_prob) - min(home_prob, away_prob, draw_prob)
            
            consistency_factor = (home_form.get('consistency', 0) + 
                                away_form.get('consistency', 0)) / 2
            
            # Weight the factors
            confidence = (
                form_sample_size * 0.3 +
                h2h_sample_size * 0.2 +
                probability_margin * 0.3 +
                consistency_factor * 0.2
            )
            
            return min(max(confidence, 0.1), 0.9)  # Cap between 0.1 and 0.9
            
        except Exception as e:
            logging.error(f"Error calculating prediction confidence: {str(e)}")
            return 0.5

    def determine_outcome(self, probabilities):
        """Determine the match outcome based on probabilities"""
        max_prob_index = np.argmax(probabilities)
        return ['Home Win', 'Draw', 'Away Win'][max_prob_index]

    def log_prediction(self, match_id: int, home_team: str, away_team: str, league_id: str, prediction: Dict, actual_result: Optional[Dict] = None):
        """
        Log a prediction and optionally the actual result to the predictions_feedback.json file.
        """
        validate_predictions_file(PREDICTIONS_FILE)  # Ensure the file is valid
        try:
            # Load existing predictions
            if os.path.exists(PREDICTIONS_FILE):
                with open(PREDICTIONS_FILE, "r") as f:
                    predictions = json.load(f)
            else:
                predictions = []

            # Check if a prediction for this match already exists
            for entry in predictions:
                if entry.get("match_id") == match_id:
                    logging.info(f"Prediction for match ID {match_id} already exists. Skipping...")
                    return

            # Save the new prediction
            save_prediction(
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                league_id=league_id,
                prediction=prediction,
                actual_result=actual_result,
            )
        except Exception as e:
            logging.error(f"Error logging prediction: {str(e)}")
            traceback.print_exc()
    def update_predictions_with_results(self):
        """
        Fetch actual results for completed matches and update the predictions file.
        """
        try:
            # Load existing predictions
            if not os.path.exists(PREDICTIONS_FILE):
                logging.info("No predictions file found. Skipping update.")
                return

            with open(PREDICTIONS_FILE, "r") as f:
                predictions = json.load(f)

            # Track updated predictions
            updated_predictions = []
            for prediction in predictions:
                match_id = prediction["match_id"]
                if prediction["actual_result"] is not None:
                    # Skip already updated predictions
                    updated_predictions.append(prediction)
                    continue

                # Fetch the match result using the API
                match = self.api.get_selected_fixture(match_id)
                if not match:
                    logging.warning(f"Match ID {match_id} not found in API. Skipping...")
                    updated_predictions.append(prediction)
                    continue

                # Extract actual result
                score = match.get("score", {}).get("fullTime", {})
                home_score = score.get("home", None)
                away_score = score.get("away", None)
                if home_score is None or away_score is None:
                    logging.warning(f"Incomplete result for match ID {match_id}. Skipping...")
                    updated_predictions.append(prediction)
                    continue

                # Determine actual outcome
                if home_score > away_score:
                    actual_result = "Home Win"
                elif home_score < away_score:
                    actual_result = "Away Win"
                else:
                    actual_result = "Draw"

                # Update the prediction with the actual result
                prediction["actual_result"] = {
                    "home_score": home_score,
                    "away_score": away_score,
                    "outcome": actual_result
                }
                updated_predictions.append(prediction)

            # Save the updated predictions back to the file
            with open(PREDICTIONS_FILE, "w") as f:
                json.dump(updated_predictions, f, indent=4)

            logging.info("Updated predictions with actual results.")
        except Exception as e:
            logging.error(f"Error updating predictions with results: {str(e)}")
            traceback.print_exc()

    def calculate_underdog_potential(self, home_team_id: int, away_team_id: int, match_data: Dict) -> Dict:
        """Calculate potential for an upset."""
        try:
            
            logging.info(f"Received match_data: {match_data}")
            
            if 'home_form' not in match_data:
                 match_data['home_form'] = self._get_team_form(home_team_id)
            if 'away_form' not in match_data:
                match_data['away_form'] = self._get_team_form(away_team_id)
            
            if not match_data or 'home_form' not in match_data or 'away_form' not in match_data:
                logging.error(f"Invalid match_data structure: {match_data}")
                return {
                    'underdog': {'id': None, 'name': 'Unknown'},
                    'favorite': {'id': None, 'name': 'Unknown'},
                    'is_home_underdog': False,
                    'upset_probability': 0,
                    'factors': {}
                }
            # Calculate team strengths
            home_strength = self._calculate_team_strength(match_data['home_form'])
            away_strength = self._calculate_team_strength(match_data['away_form'])
            
            # Determine underdog
            is_home_underdog = home_strength < away_strength
            
            # Calculate factors
            factors = {
                'recent_form': self._analyze_recent_form_for_upset(match_data, is_home_underdog),
                'defensive_resilience': self._calculate_defensive_resilience(match_data, is_home_underdog),
                'scoring_potential': self._analyze_scoring_potential(match_data, is_home_underdog),
                'momentum_factor': self._calculate_momentum_factor(match_data, is_home_underdog)
            }
            
            # Calculate weighted upset probability
            weights = {
                'recent_form': 0.3,
                'defensive_resilience': 0.3,
                'scoring_potential': 0.2,
                'momentum_factor': 0.2
            }
            
            upset_probability = sum(factors[k] * weights[k] for k in factors.keys())
            
            return {
                'underdog': {
                    'id': home_team_id if is_home_underdog else away_team_id,
                    'name': self.api.get_team_name(home_team_id if is_home_underdog else away_team_id)
                },
                'favorite': {
                    'id': away_team_id if is_home_underdog else home_team_id,
                    'name': self.api.get_team_name(away_team_id if is_home_underdog else home_team_id)
                },
                'is_home_underdog': is_home_underdog,
                'upset_probability': min(upset_probability, 1.0),
                'factors': factors
            }
            
        except Exception as e:
            logging.error(f"Error calculating underdog potential: {str(e)}")
            return {
                'underdog': {'id': None, 'name': 'Unknown'},
                'favorite': {'id': None, 'name': 'Unknown'},
                'is_home_underdog': False,
                'upset_probability': 0,
                'factors': {}
            }

    def _calculate_team_strength(self, form: Dict) -> float:
        """Calculate team strength from form data"""
        try:
            matches_played = max(form.get('matches_played', 1), 1)
            return (
                form.get('win_rate', 0) * 0.4 +
                form.get('goals_for', 0) / matches_played * 0.3 +
                (1 - form.get('goals_against', 0) / matches_played) * 0.3
            )
        except Exception as e:
            logging.error(f"Error calculating team strength: {str(e)}")
            return 0.0

    def _analyze_recent_form_for_upset(self, match_data: Dict, is_home_underdog: bool) -> float:
        """Analyze recent form for upset potential."""
        try:
            underdog_form = match_data['home_form'] if is_home_underdog else match_data['away_form']
            favorite_form = match_data['away_form'] if is_home_underdog else match_data['home_form']
            
            # Calculate form metrics
            underdog_ppg = underdog_form.get('points_per_game', 0)
            favorite_ppg = favorite_form.get('points_per_game', 0)
            
            # Recent results analysis
            underdog_recent = underdog_form.get('recent_results', [])[:5]
            favorite_recent = favorite_form.get('recent_results', [])[:5]
            
            underdog_recent_points = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 
                                       for m in underdog_recent)
            favorite_recent_points = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 
                                       for m in favorite_recent)
            
            # Calculate form factors
            form_gap = max(0, favorite_ppg - underdog_ppg)
            recent_form_ratio = (underdog_recent_points / max(favorite_recent_points, 1))
            
            # Weight recent form more heavily
            upset_potential = (
                (1 - min(form_gap, 2) / 2) * 0.4 +  # Overall form gap
                recent_form_ratio * 0.6              # Recent form comparison
            )
            
            return min(max(upset_potential, 0.0), 1.0)
            
        except Exception as e:
            logging.error(f"Error analyzing recent form for upset: {str(e)}")
            return 0.0

    def _calculate_defensive_resilience(self, match_data: Dict, is_home_underdog: bool) -> float:
        """Calculate defensive resilience for upset potential."""
        try:
            underdog_form = match_data['home_form'] if is_home_underdog else match_data['away_form']
            favorite_form = match_data['away_form'] if is_home_underdog else match_data['home_form']
            
            # Calculate defensive metrics
            underdog_clean_sheets = underdog_form.get('clean_sheets', 0)
            underdog_matches = max(underdog_form.get('matches_played', 1), 1)
            underdog_goals_against = underdog_form.get('goals_against', 0)
            
            favorite_goals_for = favorite_form.get('goals_for', 0)
            favorite_matches = max(favorite_form.get('matches_played', 1), 1)
            
            # Calculate ratios
            clean_sheet_ratio = underdog_clean_sheets / underdog_matches
            goals_conceded_ratio = underdog_goals_against / underdog_matches
            favorite_scoring_ratio = favorite_goals_for / favorite_matches
            
            # Calculate defensive resilience
            resilience = (
                clean_sheet_ratio * 0.4 +
                (1 - min(goals_conceded_ratio / 2, 1)) * 0.4 +
                (1 - min(favorite_scoring_ratio / 3, 1)) * 0.2
            )
            
            return min(max(resilience, 0.0), 1.0)
            
        except Exception as e:
            logging.error(f"Error calculating defensive resilience: {str(e)}")
            return 0.0

    def _analyze_scoring_potential(self, match_data: Dict, is_home_underdog: bool) -> float:
        """Analyze scoring potential for upset."""
        try:
            underdog_form = match_data['home_form'] if is_home_underdog else match_data['away_form']
            favorite_form = match_data['away_form'] if is_home_underdog else match_data['home_form']
            
            # Calculate scoring metrics
            underdog_goals = underdog_form.get('goals_for', 0)
            underdog_matches = max(underdog_form.get('matches_played', 1), 1)
            
            favorite_goals_against = favorite_form.get('goals_against', 0)
            favorite_matches = max(favorite_form.get('matches_played', 1), 1)
            
            # Calculate scoring rates
            underdog_scoring_rate = underdog_goals / underdog_matches
            favorite_conceding_rate = favorite_goals_against / favorite_matches
            
            # Recent scoring form
            recent_matches = underdog_form.get('recent_results', [])[:5]
            scoring_streak = sum(1 for m in recent_matches if m.get('goals_for', 0) > 0)
            
            # Calculate scoring potential
            potential = (
                min(underdog_scoring_rate / 2, 1) * 0.4 +
                min(favorite_conceding_rate / 2, 1) * 0.3 +
                (scoring_streak / 5) * 0.3
            )
            
            return min(max(potential, 0.0), 1.0)
            
        except Exception as e:
            logging.error(f"Error analyzing scoring potential: {str(e)}")
            return 0.0

    def _calculate_momentum_factor(self, match_data: Dict, is_home_underdog: bool) -> float:
        """Calculate momentum factor for upset potential."""
        try:
            underdog_form = match_data['home_form'] if is_home_underdog else match_data['away_form']
            favorite_form = match_data['away_form'] if is_home_underdog else match_data['home_form']
            
            # Get recent results
            underdog_recent = underdog_form.get('recent_results', [])[:5]
            favorite_recent = favorite_form.get('recent_results', [])[:5]
            
            # Calculate weighted momentum (more recent matches count more)
            weights = [1.0, 0.8, 0.6, 0.4, 0.2]
            
            underdog_momentum = 0
            favorite_momentum = 0
            
            for i, match in enumerate(underdog_recent):
                if i >= len(weights):
                    break
                result = match.get('result', '')
                underdog_momentum += weights[i] * (3 if result == 'W' else 1 if result == 'D' else 0)
                
            for i, match in enumerate(favorite_recent):
                if i >= len(weights):
                    break
                result = match.get('result', '')
                favorite_momentum += weights[i] * (3 if result == 'W' else 1 if result == 'D' else 0)
                
            # Calculate momentum difference
            max_possible_momentum = sum(w * 3 for w in weights[:min(5, len(underdog_recent))])
            momentum_ratio = underdog_momentum / max(favorite_momentum, 1)
            
            return min(max(momentum_ratio, 0.0), 1.0)
            
        except Exception as e:
            logging.error(f"Error calculating momentum factor: {str(e)}")
            return 0.0

    def _generate_upset_analysis(self, factors: Dict, is_home_underdog: bool) -> str:
        """Generate detailed analysis of upset potential."""
        try:
            analysis_points = []
            
            # Analyze recent form
            if factors['recent_form'] > 0.7:
                analysis_points.append("Strong recent form indicates upset potential")
            elif factors['recent_form'] < 0.3:
                analysis_points.append("Poor recent form reduces upset chances")
                
            # Analyze H2H
            if factors['h2h_advantage'] > 0.6:
                analysis_points.append("Favorable head-to-head history")
            
            # Analyze defensive resilience
            if factors['defensive_resilience'] > 0.7:
                analysis_points.append("Strong defensive record supports upset potential")
                
            # Analyze scoring potential
            if factors['scoring_potential'] > 0.6:
                analysis_points.append("Good scoring potential against favorite's defense")
                
            # Analyze momentum
            if factors['momentum_factor'] > 0.7:
                analysis_points.append("Strong momentum heading into the match")
                
            # Add home/away context
            if is_home_underdog:
                analysis_points.append("Home advantage could help upset chances")
                
            return " | ".join(analysis_points) if analysis_points else "No significant upset factors identified"
            
        except Exception as e:
            logging.error(f"Error generating upset analysis: {str(e)}")
            return "Unable to generate analysis"

    def calculate_team_form(self, matches: List[Dict], team_id: int) -> Dict:
        """Calculate team form based on recent matches."""
        try:
            if not matches:
                return {
                    'wins': 0,
                    'draws': 0,
                    'losses': 0,
                    'goals_for': 0,
                    'goals_against': 0,
                    'clean_sheets': 0,
                    'matches_played': 0,
                    'win_rate': 0,
                    'recent_results': [],
                    'form_score': 0,
                    'consistency': 0,
                    'momentum': 0,
                    'goal_difference': 0,
                    'points': 0,
                    'points_per_game': 0,
                    'goals_per_game': 0,
                    'goals_against_per_game': 0
                }

            form = {
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_for': 0,
                'goals_against': 0,
                'clean_sheets': 0,
                'matches_played': 0,
                'recent_results': []
            }

            # Process each match
            for match in matches:
                score = match.get('score', {}).get('fullTime', {})
                home_score = score.get('home')
                away_score = score.get('away')

                if home_score is None or away_score is None:
                    continue

                is_home = match['homeTeam']['id'] == team_id
                team_score = home_score if is_home else away_score
                opponent_score = away_score if is_home else home_score

                # Update goals
                form['goals_for'] += team_score
                form['goals_against'] += opponent_score

                # Update results
                if team_score > opponent_score:
                    form['wins'] += 1
                    result = 'W'
                elif team_score < opponent_score:
                    form['losses'] += 1
                    result = 'L'
                else:
                    form['draws'] += 1
                    result = 'D'

                # Update clean sheets
                if opponent_score == 0:
                    form['clean_sheets'] += 1

                # Add to recent results
                form['recent_results'].append({
                    'date': match.get('utcDate'),
                    'result': result,
                    'goals_for': team_score,
                    'goals_against': opponent_score,
                    'opponent_id': match['awayTeam']['id'] if is_home else match['homeTeam']['id']
                })

                form['matches_played'] += 1

            # Calculate win rate
            form['win_rate'] = form['wins'] / max(form['matches_played'], 1)

            # Sort recent results by date
            form['recent_results'].sort(key=lambda x: x['date'], reverse=True)

            # Calculate form metrics
            form['goal_difference'] = form['goals_for'] - form['goals_against']
            form['points'] = (form['wins'] * 3) + form['draws']
            form['points_per_game'] = form['points'] / max(form['matches_played'], 1)
            form['goals_per_game'] = form['goals_for'] / max(form['matches_played'], 1)
            form['goals_against_per_game'] = form['goals_against'] / max(form['matches_played'], 1)

            # Calculate form score (weighted recent performance)
            recent_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Most recent matches weighted higher
            form_score = 0
            for i, match in enumerate(form['recent_results'][:5]):  # Last 5 matches
                if i >= len(recent_weights):
                    break
                if match['result'] == 'W':
                    form_score += 3 * recent_weights[i]
                elif match['result'] == 'D':
                    form_score += 1 * recent_weights[i]

            form['form_score'] = form_score / sum(recent_weights[:min(5, len(form['recent_results']))])

            # Calculate consistency metrics
            results = [m['result'] for m in form['recent_results'][:5]]
            form['consistency'] = len(set(results)) / len(results) if results else 0
            
            # Calculate momentum (trend in last 5 matches)
            points_trend = []
            for result in results:
                points_trend.append(3 if result == 'W' else 1 if result == 'D' else 0)
            
            form['momentum'] = sum((i + 1) * points for i, points in enumerate(points_trend)) / sum(range(1, len(points_trend) + 1)) if points_trend else 0

            return form

        except Exception as e:
            logging.error(f"Error calculating team form: {str(e)}")
            return {
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_for': 0,
                'goals_against': 0,
                'clean_sheets': 0,
                'matches_played': 0,
                'win_rate': 0,
                'recent_results': [],
                'form_score': 0,
                'consistency': 0,
                'momentum': 0,
                'goal_difference': 0,
                'points': 0,
                'points_per_game': 0,
                'goals_per_game': 0,
                'goals_against_per_game': 0
            }

    def calculate_attacking_strength(self, recent_results: List[Dict]) -> float:
        """Calculate attacking strength based on recent results."""
        try:
            if not recent_results:
                return 0.5

            goals_scored = 0
            matches_analyzed = 0
            scoring_streak = 0
            current_streak = 0

            for match in recent_results:
                try:
                    goals = match.get('goals_for', 0)
                    goals_scored += goals
                    matches_analyzed += 1
                    
                    # Track scoring streak
                    if goals > 0:
                        current_streak += 1
                        scoring_streak = max(scoring_streak, current_streak)
                    else:
                        current_streak = 0
                        
                except Exception as e:
                    logging.warning(f"Error processing match in attacking strength: {str(e)}")
                    continue

            if matches_analyzed == 0:
                return 0.5

            # Calculate attacking metrics
            avg_goals_scored = goals_scored / matches_analyzed
            scoring_rate = sum(1 for m in recent_results if m.get('goals_for', 0) > 0) / matches_analyzed
            
            # Calculate streak factor (0-1)
            streak_factor = min(scoring_streak / 5, 1.0)  # Cap at 5 match streak

            # Weight the factors
            attacking_strength = (
                min(avg_goals_scored / 3, 1) * 0.5 +  # Goals per game (capped at 3)
                scoring_rate * 0.3 +                   # Consistency in scoring
                streak_factor * 0.2                    # Recent scoring streak
            )

            return attacking_strength

        except Exception as e:
            logging.error(f"Error calculating attacking strength: {str(e)}")
            return 0.5

    def _get_team_form(self, team_id: int) -> Dict:
        """Get team form data."""
        try:
            # Get recent matches for the team
            recent_matches = self.get_recent_matches(team_id, limit=10)
            
            # Calculate form statistics
            wins = sum(1 for match in recent_matches if self._is_team_winner(match, team_id))
            draws = sum(1 for match in recent_matches if self._is_match_draw(match))
            losses = len(recent_matches) - wins - draws
            
            goals_for = sum(self._get_team_goals(match, team_id) for match in recent_matches)
            goals_against = sum(self._get_opponent_goals(match, team_id) for match in recent_matches)
            
            return {
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_difference': goals_for - goals_against,
                'recent_matches': recent_matches
            }
        except Exception as e:
            logging.error(f"Error getting team form: {str(e)}")
            return None

    def _is_team_winner(self, match: Dict, team_id: int) -> bool:
        """Check if team won the match."""
        if match['score']['winner'] == 'HOME_TEAM' and match['homeTeam']['id'] == team_id:
            return True
        if match['score']['winner'] == 'AWAY_TEAM' and match['awayTeam']['id'] == team_id:
            return True
        return False

    def _is_match_draw(self, match: Dict) -> bool:
        """Check if match was a draw."""
        return match['score']['winner'] == 'DRAW' or not match['score']['winner']

    def _get_team_goals(self, match: Dict, team_id: int) -> int:
        """Get goals scored by team in match."""
        if match['homeTeam']['id'] == team_id:
            return match['score']['fullTime']['home'] or 0
        return match['score']['fullTime']['away'] or 0

    def _get_opponent_goals(self, match: Dict, team_id: int) -> int:
        """Get goals scored against team in match."""
        if match['homeTeam']['id'] == team_id:
            return match['score']['fullTime']['away'] or 0
        return match['score']['fullTime']['home'] or 0

def create_predictor(league_id: str) -> Optional[FootballPredictor]:
    """Create a predictor for a league"""
    try:
        league_name = leagues.get(league_id, 'Unknown League')
        logging.info(f"Creating predictor for {league_name} (ID: {league_id})")
        predictor = FootballPredictor(league_id, league_name)
        return predictor
    except Exception as e:
        logging.info(f"Error creating predictor: {str(e)}")
        traceback.print_exc()
        return None

# API Routes
@app.route('/api/leagues', methods=['GET'])
@cross_origin()
def get_leagues():
    """Get available leagues."""
    try:
        return jsonify({
            'leagues': [
                {'code': code, 'name': name}
                for code, name in leagues.items()
            ]
        })
    except Exception as e:
        logging.error(f"Error getting leagues: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/fixtures/<league_id>', methods=['GET'])
def get_fixtures(league_id):
    """Get upcoming fixtures for a league with caching."""
    cache_key = f'fixtures_{league_id}'
    cached = get_cached_data(cache_key)
    if cached:
        logging.info("Returning cached fixtures data")
        return jsonify(cached)
    try:
        api = FootballDataAPI()
        fixtures = api.get_fixtures(league_id)
        if not fixtures:
            return jsonify({'error': 'No fixtures found'}), 404
        set_cached_data(cache_key, {'fixtures': fixtures})
        return jsonify({'fixtures': fixtures})
    except Exception as e:
        logging.error(f"Error fetching fixtures: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_predictions', methods=['POST'])
def update_predictions():
    logging.info("Entering update_predictions function")
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")
        match_id = data.get('match_id')
        actual_result = data.get('actual_result')

        if not match_id or not actual_result:
            return jsonify({"error": "Invalid request data"}), 400

        if not os.path.exists(PREDICTIONS_FILE):
            return jsonify({"error": "Predictions file not found"}), 404

        # Load predictions
        with open(PREDICTIONS_FILE, "r") as f:
            predictions = json.load(f)

        # Find and update the match
        updated = False
        for entry in predictions:
            logging.info(f"Checking match_id: {entry['match_id']} against received: {match_id}")
            if entry["match_id"] == str(match_id):  # Ensure consistent comparison
                entry["actual_result"] = actual_result
                updated = True
                break

        if not updated:
            logging.error(f"Match ID {match_id} not found in predictions")
            return jsonify({"error": "Match not found"}), 404

        # Save updated predictions back to the file
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(predictions, f, indent=4)

        logging.info(f"Successfully updated result for match ID {match_id}")
        return jsonify({"message": "Match result updated successfully"}), 200

    except Exception as e:
        logging.error(f"Error updating result: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/predictions/<league_id>', methods=['GET'])
@cross_origin()
def get_predictions_by_league(league_id):
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            logging.error("Predictions file not found.")
            return jsonify({"error": "Predictions file not found"}), 404

        # Load predictions from the file
        with open(PREDICTIONS_FILE, "r") as f:
            predictions = json.load(f)

        # Filter predictions by league and actual_result == null
        filtered_predictions = [
            entry for entry in predictions
            if entry["league_id"] == league_id and entry["actual_result"] is None
        ]

        # Sort predictions by date (most recent first)
        filtered_predictions.sort(key=lambda x: x["date"], reverse=True)

        # Limit to 10 predictions
        limited_predictions = filtered_predictions[:10]

        return jsonify({"predictions": limited_predictions}), 200

    except Exception as e:
        logging.error(f"Error fetching predictions: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/update-result', methods=['POST'])
def update_result():    
    try:
        data = request.get_json()
        match_id = data.get("match_id")
        actual_result = data.get("actual_result")

        if not match_id or not actual_result:
            return jsonify({"error": "Invalid request data"}), 400

        if not os.path.exists(PREDICTIONS_FILE):
            return jsonify({"error": "Predictions file not found"}), 404

        # Load predictions
        with open(PREDICTIONS_FILE, "r") as f:
            predictions = json.load(f)

        # Find and update the match
        updated = False
        for entry in predictions:
            if entry["match_id"] == match_id:
                entry["actual_result"] = actual_result
                updated = True
                break

        if not updated:
            return jsonify({"error": "Match not found"}), 404

        # Save updated predictions back to the file
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(predictions, f, indent=4)

        return jsonify({"message": "Match result updated successfully"}), 200

    except Exception as e:
        logging.error(f"Error updating result: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_match():
    try:
        data = request.json
        print("\n=== ANALYZE MATCH REQUEST ===")
        print(f"Received data: {json.dumps(data, indent=2)}")
        
        home_team = data['homeTeam']
        away_team = data['awayTeam']
        league_id = data.get('competition', {}).get('code', 'CL')
        
        print(f"\nAnalyzing match: {home_team['name']} vs {away_team['name']}")
        
        # Get form data first
        home_form = get_team_matches(home_team['id'])
        away_form = get_team_matches(away_team['id'])
        head_to_head = get_head_to_head(home_team['id'], away_team['id'])
        
        print("\n=== FORM DATA ===")
        print(f"Home form: {json.dumps(home_form, indent=2)}")
        print(f"Away form: {json.dumps(away_form, indent=2)}")
        print(f"Head to head: {json.dumps(head_to_head, indent=2)}")
        
        # Calculate prediction
        home_strength = home_form['wins'] * 3 + home_form['draws']
        away_strength = away_form['wins'] * 3 + away_form['draws']
        total_strength = home_strength + away_strength
        
        print(f"\nStrength calculation:")
        print(f"Home strength: {home_strength}")
        print(f"Away strength: {away_strength}")
        print(f"Total strength: {total_strength}")
        
        if total_strength == 0:
            home_prob = away_prob = 0.33
            draw_prob = 0.34
        else:
            home_prob = (home_strength / total_strength) * 0.8
            away_prob = (away_strength / total_strength) * 0.8
            draw_prob = 1 - home_prob - away_prob
        
        prediction = {
            'homeWinProbability': round(home_prob * 100, 2),
            'awayWinProbability': round(away_prob * 100, 2),
            'drawProbability': round(draw_prob * 100, 2),
            'predictedWinner': 'HOME_TEAM' if home_prob > away_prob else 'AWAY_TEAM' if away_prob > home_prob else 'DRAW',
            'confidence': round(max(home_prob, away_prob, draw_prob) * 100, 2),
            'scorePrediction': {
                'home': round((home_form['goals_for'] / max(len(home_form['matches']), 1)) * 1.1),
                'away': round(away_form['goals_for'] / max(len(away_form['matches']), 1))
            },
            'recommendedBet': 'Home Win' if home_prob > max(away_prob, draw_prob) else 'Away Win' if away_prob > max(home_prob, draw_prob) else 'Draw'
        }
        
        analysis = {
            'homeForm': home_form,
            'awayForm': away_form,
            'headToHead': head_to_head,
            'prediction': prediction
        }
        
        print("\n=== FINAL PREDICTION ===")
        print(json.dumps(prediction, indent=2))
        print("\n=== SENDING RESPONSE ===")
        print(json.dumps(analysis, indent=2))
        
        return jsonify(analysis)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def get_team_matches(team_id):
    """Get team matches with rate limiting and caching"""
    api = FootballDataAPI()
    cache_key = f"team_matches_{team_id}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    try:
        # Get raw matches data
        response = requests.get(
            f"{FOOTBALL_DATA_BASE_URL}/teams/{team_id}/matches",
            headers={"X-Auth-Token": FOOTBALL_DATA_TOKEN},
            params={"status": "FINISHED", "limit": 10}
        )
        response.raise_for_status()
        matches = response.json().get('matches', [])
        
        # Calculate form data
        form_data = {
            'matches': matches,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'clean_sheets': 0
        }
        
        for match in matches:
            if not isinstance(match, dict):
                continue
                
            score = match.get('score', {})
            full_time = score.get('fullTime', {})
            if not isinstance(full_time, dict):
                continue
                
            home_score = full_time.get('home', 0) or 0
            away_score = full_time.get('away', 0) or 0
            is_home = match.get('homeTeam', {}).get('id') == team_id
            
            # Calculate goals
            team_score = home_score if is_home else away_score
            opponent_score = away_score if is_home else home_score
            form_data['goals_for'] += team_score
            form_data['goals_against'] += opponent_score
            
            # Calculate result
            winner = score.get('winner')
            if winner == 'HOME_TEAM' and is_home or winner == 'AWAY_TEAM' and not is_home:
                form_data['wins'] += 1
            elif winner == 'DRAW' or not winner:
                form_data['draws'] += 1
            else:
                form_data['losses'] += 1
            
            # Clean sheets
            if opponent_score == 0:
                form_data['clean_sheets'] += 1
        
        # Calculate additional stats
        matches_played = len(matches)
        if matches_played > 0:
            form_data['win_rate'] = form_data['wins'] / matches_played
            form_data['points'] = (form_data['wins'] * 3) + form_data['draws']
            form_data['points_per_game'] = form_data['points'] / matches_played
            form_data['goal_difference'] = form_data['goals_for'] - form_data['goals_against']
        else:
            form_data['win_rate'] = 0
            form_data['points'] = 0
            form_data['points_per_game'] = 0
            form_data['goal_difference'] = 0
        
        set_cached_data(cache_key, form_data)
        return form_data
        
    except Exception as e:
        logging.error(f"Error fetching team matches: {str(e)}")
        traceback.print_exc()  # Add full stack trace
        return {'matches': [], 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0, 'clean_sheets': 0}

def get_head_to_head(team1_id, team2_id):
    """Get head to head matches with rate limiting and caching"""
    cache_key = f"h2h_{team1_id}_{team2_id}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    try:
        # Get raw matches data
        response = requests.get(
            f"{FOOTBALL_DATA_BASE_URL}/matches",
            headers={"X-Auth-Token": FOOTBALL_DATA_TOKEN},
            params={
                "homeTeam": team1_id,
                "awayTeam": team2_id,
                "status": "FINISHED",
                "limit": 10
            }
        )
        response.raise_for_status()
        matches = response.json().get('matches', [])
        
        h2h_data = {
            'matches': matches,
            'team1_wins': 0,
            'team2_wins': 0,
            'draws': 0,
            'team1_goals': 0,
            'team2_goals': 0
        }
        
        for match in matches:
            if not isinstance(match, dict):
                continue
                
            score = match.get('score', {})
            full_time = score.get('fullTime', {})
            if not isinstance(full_time, dict):
                continue
                
            home_score = full_time.get('home', 0) or 0
            away_score = full_time.get('away', 0) or 0
            team1_is_home = match.get('homeTeam', {}).get('id') == team1_id
            
            # Calculate goals
            team1_score = home_score if team1_is_home else away_score
            team2_score = away_score if team1_is_home else home_score
            h2h_data['team1_goals'] += team1_score
            h2h_data['team2_goals'] += team2_score
            
            # Calculate result
            winner = score.get('winner')
            if winner == 'HOME_TEAM':
                if team1_is_home:
                    h2h_data['team1_wins'] += 1
                else:
                    h2h_data['team2_wins'] += 1
            elif winner == 'AWAY_TEAM':
                if team1_is_home:
                    h2h_data['team2_wins'] += 1
                else:
                    h2h_data['team1_wins'] += 1
            else:
                h2h_data['draws'] += 1
        
        set_cached_data(cache_key, h2h_data)
        return h2h_data
        
    except Exception as e:
        logging.error(f"Error fetching head to head matches: {str(e)}")
        traceback.print_exc()  # Add full stack trace
        return {'matches': [], 'team1_wins': 0, 'team2_wins': 0, 'draws': 0}

@app.route('/api/analyze/underdog', methods=['POST'])
@cross_origin()
def analyze_underdog():
    """Analyze underdog potential for a match."""
    try:
        data = request.get_json()
        logging.info(f"Received underdog analysis request: {data}")
        
        home_team_id = data.get('homeTeam', {}).get('id')
        away_team_id = data.get('awayTeam', {}).get('id')
        league_id = data.get('competition', {}).get('code')
        
        if not all([home_team_id, away_team_id, league_id]):
            return jsonify({
                'error': 'Missing required data',
                'analysis': None
            }), 400

        predictor = create_predictor(league_id)
        if not predictor:
            return jsonify({
                'error': 'Failed to create predictor',
                'analysis': None
            }), 500

        # Get team matches with rate limiting
        home_matches = predictor.api.get_team_matches(home_team_id)
        if not home_matches:
            # Return a default analysis if no matches are found
            return jsonify({
                'analysis': {
                    'underdog': {'id': None, 'name': 'Unknown'},
                    'favorite': {'id': None, 'name': 'Unknown'},
                    'is_home_underdog': False,
                    'upset_probability': 0,
                    'factors': {
                        'defensive_resilience': 0,
                        'momentum_factor': 0,
                        'recent_form': 0,
                        'scoring_potential': 0
                    }
                }
            })

        # Wait before making the second request
        time.sleep(1)
        
        away_matches = predictor.api.get_team_matches(away_team_id)
        if not away_matches:
            # Return a default analysis if no matches are found
            return jsonify({
                'analysis': {
                    'underdog': {'id': None, 'name': 'Unknown'},
                    'favorite': {'id': None, 'name': 'Unknown'},
                    'is_home_underdog': False,
                    'upset_probability': 0,
                    'factors': {
                        'defensive_resilience': 0,
                        'momentum_factor': 0,
                        'recent_form': 0,
                        'scoring_potential': 0
                    }
                }
            })

        # Calculate form data
        home_form = predictor.calculate_team_form(home_matches, home_team_id)
        away_form = predictor.calculate_team_form(away_matches, away_team_id)

        # Prepare match data with forms
        match_data = {
            'home_form': home_form,
            'away_form': away_form
        }

        # Calculate underdog potential
        underdog_analysis = predictor.calculate_underdog_potential(
            home_team_id,
            away_team_id,
            match_data
        )
        
        return jsonify({'analysis': underdog_analysis})

    except Exception as e:
        logging.error(f"Error in underdog analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'analysis': None
        }), 500

if __name__ == '__main__':
    validate_predictions_file(PREDICTIONS_FILE)
    ensure_match_ids_as_strings(PREDICTIONS_FILE)
    app.run(debug=True, port=5000)