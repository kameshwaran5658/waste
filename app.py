from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, g
import mysql.connector
import bcrypt
from flask_wtf import CSRFProtect
from flask_socketio import SocketIO
from flask_apscheduler import APScheduler
import os
import joblib
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score # Import r2_score
from statistics import variance
from flask import jsonify
import io
import csv

# Load environment variables from .env file
load_dotenv()

# Flask Application Setup
app = Flask(__name__)
# Set a secret key for session management and CSRF protection
app.secret_key = os.getenv('SECRET_KEY') or 'a_very_secret_key_for_development'
# Enable CSRF protection for all POST requests
csrf = CSRFProtect(app)
# Initialize Flask-SocketIO for real-time communication (if needed)
socketio = SocketIO(app)
# Initialize APScheduler for scheduled tasks
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Database Configuration
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306)) # Default to 3306 if not specified
}

# Model and Application Configuration
MODEL_FILENAME = "food_quantity_model.pkl"
FEATURES_FILENAME = "feature_columns.pkl"
DEFAULT_COMPARE_DAYS = 7
COST_PER_KG = 3.50 # Cost per kilogram for waste calculation
PER_PAGE = 5 # Items per page for food history pagination
ACTIVITY_PER_PAGE = 10 # Items per page for activity log pagination
INGREDIENT_LOW_STOCK_THRESHOLD = 5.0 # kg - Threshold for low ingredient stock alerts
WASTE_PERCENTAGE_THRESHOLD = 10.0 # % - Threshold for high waste alerts

# Define units for various food items
ITEM_UNITS = {
    "idli": "pieces", "vada": "pieces", "pongal": "kg", "dosa": "pieces", "kal dosa": "pieces",
    "poori": "pieces", "chapati": "pieces", "parotta": "pieces", "rava kichadi": "kg",
    "rava uppuma": "kg", "kesari": "kg", "bread": "slices", "jam": "grams",
    "bonda": "pieces", "sweet bonda": "pieces", "sundal": "kg", "valaikai baji": "pieces",
    "groundnut": "kg", "biscuit": "pieces", "white rice": "kg", "sambar": "liters",
    "rasam": "liters", "mor kulambu": "liters", "kara kulambu": "liters", "butter milk": "liters",
    "poriyal": "kg", "kootu": "kg", "variety rice": "kg", "egg": "pieces", "pickle": "grams",
    "papad": "pieces", "gobi manjurian": "kg", "brinjal thalicha": "kg", "channa masala": "kg",
    "tomato thokku": "kg", "chicken gravy with piece": "kg", "chicken kulambu": "liters",
    "chicken biryani": "kg", "veg biryani": "kg", "onion raita": "liters",
    "tea": "liters", "milk": "liters", "juice": "liters"
}

# Define ingredients and their proportions per unit of food item (in kg or liters per piece/kg/liter of food item)
# These are example proportions and should be adjusted based on actual recipes.
ITEM_INGREDIENTS = {
    "idli": {"rice":0.05,"urad dal":0.02,"salt":0.005,"water":0.1},
    "vada": {"urad dal":0.03,"onion":0.01,"green chili":0.005,"salt":0.005,"oil":0.02},
    "pongal": {"rice":0.08,"moong dal":0.02,"pepper":0.002,"cumin seeds":0.002,"ghee":0.01,"salt":0.005},
    "dosa": {"rice":0.06,"urad dal":0.02,"salt":0.005,"oil":0.02},
    "kal dosa": {"rice":0.07,"urad dal":0.02,"salt":0.005,"oil":0.02},
    "poori": {"wheat flour":0.06,"oil":0.015,"salt":0.005,"water":0.1},
    "chapati": {"wheat flour":0.05,"oil":0.005,"salt":0.005,"water":0.1},
    "parotta": {"maida":0.07,"oil":0.02,"curd":0.01,"salt":0.005},
    "rava kichadi": {"rava":0.05,"vegetables mixed":0.03,"salt":0.005,"mustard seeds":0.002,"curry leaves":0.003,"oil":0.02},
    "rava uppuma": {"rava":0.05,"vegetables mixed":0.02,"salt":0.005,"mustard seeds":0.002,"curry leaves":0.003,"oil":0.02},
    "kesari": {"rava":0.04,"sugar":0.04,"ghee":0.015,"cardamom":0.001,"salt":0.002},
    "bread": {"bread":0.06,"butter":0.01}, "jam": {"jam":0.02},
    "bonda": {"besan flour":0.03,"potato":0.02,"onion":0.01,"green chili":0.005,"salt":0.005,"oil":0.02},
    "sweet bonda": {"maida":0.04,"sugar":0.03,"salt":0.005,"oil":0.02},
    "sundal": {"white channa":0.04,"coconut":0.01,"salt":0.005,"mustard seeds":0.002,"curry leaves":0.003,"oil":0.02},
    "valaikai baji": {"banana flower":0.03,"besan flour":0.02,"salt":0.005,"oil":0.02},
    "groundnut": {"groundnut":0.04,"salt":0.005}, "biscuit": {"biscuit":0.03,"sugar":0.01},
    "white rice": {"rice":0.10,"salt":0.005},
    "sambar": {"toor dal":0.03,"vegetables mixed":0.04,"tamarind":0.01,"salt":0.005,"mustard seeds":0.002,"curry leaves":0.003,"oil":0.02,"sambar powder":0.01},
    "rasam": {"tamarind":0.01,"tomato":0.02,"rasam powder":0.005,"garlic":0.005,"cumin":0.002,"coriander leaves":0.003,"salt":0.005},
    "mor kulambu": {"curd":0.1,"vegetables mixed":0.03,"salt":0.005,"mustard seeds":0.002,"curry leaves":0.003,"oil":0.02},
    "kara kulambu": {"tamarind":0.015,"brinjal":0.03,"drumstick":0.02,"onion":0.01,"tomato":0.02,"salt":0.005,"mustard seeds":0.002,"curry leaves":0.003,"oil":0.02},
    "butter milk": {"curd":0.08,"salt":0.005,"cumin powder":0.002},
    "poriyal": {"vegetables mixed":0.06,"coconut":0.01,"salt":0.005,"mustard seeds":0.002,"curry leaves":0.003,"oil":0.02},
    "kootu": {"moong dal":0.03,"vegetables mixed":0.05,"salt":0.005,"mustard seeds":0.002,"curry leaves":0.003,"oil":0.02},
    "variety rice": {"rice":0.10,"mixed spices":0.02,"salt":0.005,"oil":0.02},
    "egg": {"egg":1}, "pickle": {"pickle":0.01,"salt":0.005}, "papad": {"papad":0.02},
    "gobi manjurian": {"cauliflower":0.08,"maida":0.02,"cornflour":0.01,"salt":0.005,"oil":0.02},
    "brinjal thalicha": {"brinjal":0.05,"spices":0.01,"salt":0.005,"oil":0.02},
    "channa masala": {"white channa":0.06,"onion":0.02,"tomato":0.02,"garlic":0.005,"salt":0.005,"oil":0.02},
    "tomato thokku": {"tomato":0.08,"spices":0.01,"salt":0.005,"oil":0.02},
    "chicken gravy with piece": {"chicken":0.15,"onion":0.03,"tomato":0.02,"masala powder":0.02,"salt":0.005,"oil":0.02,"garlic":0.005,"ginger":0.005},
    "chicken kulambu": {"chicken":0.12,"onion":0.03,"tomato":0.02,"salt":0.005,"oil":0.02,"garlic":0.005,"ginger":0.005},
    "chicken biryani": {"chicken":0.15,"biryani rice":0.1,"onion":0.04,"curd":0.02,"spices":0.02,"salt":0.005,"oil":0.02,"garlic":0.005},
    "veg biryani": {"biryani rice":0.1,"vegetables mixed":0.05,"curd":0.02,"spices":0.02,"salt":0.005,"oil":0.02,"garlic":0.005},
    "onion raita": {"curd":0.1,"onion":0.02,"salt":0.005,"cumin powder":0.002,"coriander leaves":0.003},
    "tea": {"milk":0.1,"tea powder":0.005,"sugar":0.02,"water":0.05},
    "milk": {"milk":0.2}, "juice": {"juice":0.2}
}

# --- Database Connection Helper ---
def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None

# --- Activity Logging Function ---
def get_ip():
    """Fetches the user's IP address dynamically, safely handling non-request contexts."""
    try:
        # Check if a request context is active
        if request:
            return request.remote_addr
        else:
            return "N/A (No Request Context)"
    except RuntimeError:
        # This occurs if request is accessed outside an application context (e.g., in a scheduled task)
        return "N/A (Outside Request Context)"

def log_activity(admin_id: int, activity_type: str, details: str):
    """Logs an activity performed by an admin."""
    conn = get_db_connection()
    if not conn:
        print("Database connection error. Could not log activity.")
        return

    cursor = conn.cursor()
    try:
        # Assuming 'id' is AUTO_INCREMENT in activity_log table
        # Ensure 'ip_address' column exists in your activity_log table
        cursor.execute(
            "INSERT INTO activity_log (admin_id, timestamp, type, details, ip_address) VALUES (%s, NOW(), %s, %s, %s)",
            (admin_id, activity_type, details, get_ip())
        )
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Database error logging activity: {err}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

# --- Attendance and Student Count Functions ---
def get_today_student_count() -> int:
    """Fetches today's student count from the database."""
    conn = get_db_connection()
    if not conn:
        return 0 # Return 0 if DB connection fails
    cursor = conn.cursor()
    try:
        today = date.today()
        cursor.execute("SELECT student_count FROM attendance WHERE date = %s", (today,))
        row = cursor.fetchone()
        return row[0] if row else 0 # Return 0 if no entry found
    except Exception as e:
        print(f"Error fetching today's student count: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()

def get_student_count_by_date(target_date) -> int:
    """Fetches student count for a specific date from the database."""
    conn = get_db_connection()
    if not conn:
        return 0 # Return 0 if DB connection fails
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT student_count FROM attendance WHERE date=%s', (target_date,))
        row = cursor.fetchone()
        return row[0] if row else 0 # Return 0 if no entry found
    except Exception as e:
        print(f"Error fetching student count for {target_date}: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()

def get_meals_served_count(for_date=None) -> int:
    """Calculates the total number of meals served for a given date."""
    if for_date is None:
        for_date = date.today()
    conn = get_db_connection()
    if not conn:
        return 0
    cursor = conn.cursor(dictionary=True)
    try:
        weekday = for_date.strftime('%A')
        # Count distinct meal types for the day, assuming each distinct type represents a "meal period"
        cursor.execute('SELECT COUNT(DISTINCT meal_type) AS meal_count FROM menu WHERE day=%s', (weekday,))
        periods_row = cursor.fetchone()
        periods = periods_row['meal_count'] if periods_row and 'meal_count' in periods_row and periods_row['meal_count'] is not None else 0
    except Exception as e:
        print(f"Error fetching meal periods: {e}")
        periods = 0
    finally:
        cursor.close()
        conn.close()
    
    student_count = get_student_count_by_date(for_date)
    return student_count * periods if student_count is not None else 0

# --- Machine Learning Model Functions ---
def fetch_history_data() -> pd.DataFrame:
    """Fetches historical food data from the database for model training."""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to DB for fetching history data.")
        return pd.DataFrame()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT day, meal_type, item_name, actual_quantity_used, food_waste, student_count FROM food_history"
        )
        rows = cursor.fetchall()
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error fetching history data: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()
        conn.close()

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Generates features and targets for the machine learning model."""
    # Ensure 'day' column is treated as a string for dummy variable creation
    df['day'] = df['day'].apply(lambda x: x.strftime('%A') if isinstance(x, (date, datetime)) else x)

    # Convert categorical columns to dummy variables
    # We keep all dummy variables (drop_first=False) to ensure consistent feature sets
    # during training and prediction.
    df_enc = pd.get_dummies(df, columns=['day', 'meal_type', 'item_name'], drop_first=False)

    # Define target variables: actual_quantity_used and food_waste
    y = df_enc[['actual_quantity_used', 'food_waste']].values
    # Define features by dropping target columns
    X = df_enc.drop(columns=['actual_quantity_used', 'food_waste'])
    return X, y

def train_model():
    """Trains or re-trains the RandomForestRegressor model."""
    df = fetch_history_data()
    
    # Check if data is empty
    if df.empty:
        print("No data available for training. Model not trained.")
        return

    # Ensure 'student_count' is a numeric type, handle potential errors
    if 'student_count' in df.columns:
        df['student_count'] = pd.to_numeric(df['student_count'], errors='coerce').fillna(0)
        
        # --- DIAGNOSTIC: Check variance of student_count ---
        if df['student_count'].nunique() < 2: # Check if there's at least 2 unique values
            print("Warning: 'student_count' has very low variance in training data. Model might not learn its impact effectively.")
            print("To improve: Enter food history data with varying student counts.")
        # --- END DIAGNOSTIC ---
    else:
        df['student_count'] = 0 # Add if missing
        print("Warning: 'student_count' column missing from food_history data. Model will not use it for prediction.")

    # --- DIAGNOSTIC: Check variance of target variables ---
    if df['actual_quantity_used'].nunique() < 2:
        print("Warning: 'actual_quantity_used' has very low variance in training data. Model might struggle to predict it accurately.")
        print("To improve: Enter food history data with varying actual quantities used.")
    if df['food_waste'].nunique() < 2:
        print("Warning: 'food_waste' has very low variance in training data. Model might struggle to predict it accurately.")
        print("To improve: Enter food history data with varying food waste amounts.")
    # --- END DIAGNOSTIC ---


    # If data is insufficient, no model is saved.
    if df.shape[0] < 20: # Increased threshold for more robust training
        print(f"Not enough data ({df.shape[0]} entries) in food_history to train a robust model. Model not saved.")
        print("To improve: Add more food history entries (minimum 20 recommended).")
        return # Exit without saving a model

    X, y = prepare_features(df)
    
    # Ensure X is not empty after feature preparation
    if X.empty:
        print("Prepared features DataFrame is empty. Cannot train model.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # --- DIAGNOSTIC: Evaluate model on training data (simple R-squared) ---
    train_preds = model.predict(X_train)
    r2_actual_qty = r2_score(y_train[:, 0], train_preds[:, 0])
    r2_food_waste = r2_score(y_train[:, 1], train_preds[:, 1])
    print(f"Model training R-squared (Actual Quantity Used): {r2_actual_qty:.2f}")
    print(f"Model training R-squared (Food Waste): {r2_food_waste:.2f}")
    
    if r2_actual_qty < 0.5 or r2_food_waste < 0.5:
        print("Warning: Low R-squared score suggests the model is not learning well from the current data.")
        print("To improve: Consider adding more diverse and accurate historical food data.")
    # --- END DIAGNOSTIC ---

    # Save the trained model and feature columns
    with open(MODEL_FILENAME, 'wb') as mf:
        pickle.dump(model, mf)
    with open(FEATURES_FILENAME, 'wb') as ff:
        pickle.dump(X.columns.tolist(), ff)
    print("Model trained and saved successfully.")

@scheduler.task('cron', id='retrain_model', hour=3, minute=0) # Schedule for 3 AM daily
def scheduled_training():
    """Scheduled task to retrain the model daily."""
    print("Starting scheduled model retraining...")
    with app.app_context(): # Ensure app context for database operations
        train_model()
    print("Scheduled model retraining finished.")

def load_model() -> tuple:
    """Loads the pre-trained model and feature columns. Raises FileNotFoundError if not found."""
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(FEATURES_FILENAME):
        raise FileNotFoundError("Model or feature columns files not found. Please ensure the model has been trained successfully.")
    
    with open(MODEL_FILENAME, 'rb') as mf:
        model = pickle.load(mf)
    with open(FEATURES_FILENAME, 'rb') as ff:
        feature_cols = pickle.load(ff)
    return model, feature_cols

def build_feature_vector(day_str: str, meal: str, item: str, student_count: int, feature_cols: list) -> pd.DataFrame:
    """Builds a feature vector for prediction based on input parameters.
    Ensures the feature vector matches the structure the model was trained on."""
    
    # Initialize a dictionary with all expected feature columns set to 0
    # This is crucial for handling dummy variables correctly
    vec = {col: 0 for col in feature_cols}
    
    # Set the student_count
    vec['student_count'] = int(student_count)
    
    # Set the appropriate dummy variables to 1
    # Check if the specific dummy variable column exists in the trained feature columns
    day_col = f'day_{day_str}'
    if day_col in vec:
        vec[day_col] = 1
            
    meal_col = f'meal_type_{meal}'
    if meal_col in vec:
        vec[meal_col] = 1

    item_col = f'item_name_{item}'
    if item_col in vec:
        vec[item_col] = 1
            
    # Create DataFrame from the dictionary, ensuring column order matches training data
    # The .reindex() method is key here to ensure consistency
    fv = pd.DataFrame([vec]).reindex(columns=feature_cols, fill_value=0)
    
    return fv

def suggest_today(student_count: int) -> tuple[list, list]:
    """Generates food quantity and ingredient suggestions for today's menu."""
    if student_count <= 0:
        print("Student count is zero or less, cannot generate food suggestions.")
        return [], [] # Cannot make suggestions without a valid student count

    try:
        model, feature_cols = load_model()
    except FileNotFoundError as e: # Catch FileNotFoundError specifically
        print(f"Warning: Model not available for suggestions: {e}. Please ensure enough data for training and that the model is saved.")
        flash("Warning: Prediction model not available. Please add more food history data to enable predictions.", 'warning')
        return [], []
    except Exception as e:
        print(f"Error loading model: {e}")
        flash(f"Error loading prediction model: {e}", 'danger')
        return [], []

    today_day = datetime.now().strftime('%A')
    conn = get_db_connection()
    if not conn:
        print("Database connection error for fetching menu, skipping suggestions.")
        return [], []
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT meal_type, items FROM menu WHERE day=%s", (today_day,))
        menu = cursor.fetchall()
    except Exception as e:
        print(f"Error fetching today's menu: {e}")
        menu = []
    finally:
        cursor.close()
        conn.close()

    suggestions = []
    total_ingredients = {}
    for row in menu:
        meal = row['meal_type']
        items = [i.strip() for i in row['items'].split(',') if i.strip()]
        for itm in items:
            # Ensure item name is in lowercase for consistent lookup in ITEM_UNITS and ITEM_INGREDIENTS
            lower_itm = itm.lower()
            
            # Build feature vector for prediction
            fv = build_feature_vector(today_day, meal, lower_itm, student_count, feature_cols)
            
            # Make prediction
            preds = model.predict(fv)
            # Ensure predictions are non-negative
            pred_qty = max(0.0, float(preds[0][0]))
            pred_waste = max(0.0, float(preds[0][1]))
            
            qty = int(round(pred_qty)) # Quantity as integer
            waste = round(pred_waste, 2) # Waste rounded to 2 decimal places
            unit = ITEM_UNITS.get(lower_itm, 'kg') # Get unit, default to 'kg'
            
            suggestions.append({
                'meal_type': meal,
                'item_name': itm,
                'suggested_quantity': qty,
                'predicted_waste': waste,
                'unit': unit
            })
            # Calculate total ingredients required
            for ing, per in ITEM_INGREDIENTS.get(lower_itm, {}).items():
                # Accumulate required quantity for each ingredient
                total_ingredients[ing] = total_ingredients.get(ing, 0) + (per * qty)

    ing_list = [{'ingredient': k, 'required_quantity': round(v, 2)} for k, v in total_ingredients.items()]
    return suggestions, ing_list

def get_predicted_waste(for_date=None) -> float:
    """Predicts total food waste (in kg) for all menu items on a given date."""
    if for_date is None:
        for_date = date.today()
    
    students = get_student_count_by_date(for_date)
    if students <= 0: # Check if students is 0 after fetching
        return 0.0 # Cannot predict waste without a valid student count

    suggestions, _ = suggest_today(students)
    total_waste = sum(item['predicted_waste'] for item in suggestions)
    return round(total_waste, 2)

# --- Metric Calculation Functions ---
def calculate_savings_potential(for_date=None) -> float:
    """Calculates the potential cost savings based on predicted waste."""
    waste_kg = get_predicted_waste(for_date)
    return round(waste_kg * COST_PER_KG, 2)

def calculate_change(metric: str, days: int = DEFAULT_COMPARE_DAYS) -> float:
    """Calculates the percentage change for a given metric over a specified number of days."""
    today = date.today()
    prev_date = today - timedelta(days=days)
    
    current = 0.0
    previous = 0.0

    if metric == 'students':
        current = get_today_student_count() # Now returns 0 if None
        previous = get_student_count_by_date(prev_date) # Now returns 0 if None
    elif metric == 'meals':
        current = get_meals_served_count(today)
        previous = get_meals_served_count(prev_date)
    elif metric == 'waste':
        current = get_predicted_waste(today)
        previous = get_predicted_waste(prev_date)
    elif metric == 'savings':
        current = calculate_savings_potential(today)
        previous = calculate_savings_potential(prev_date)
    else:
        raise ValueError(f"Unknown metric '{metric}' for change calculation")
    
    if previous == 0:
        # If previous was 0: 0% change if current is also 0, otherwise 100% (or more) increase
        return 0.0 if current == 0 else 100.0
    
    return round(((current - previous) / previous) * 100, 2)

def calculate_weekly_average(attendance_records: list) -> int:
    """Calculates average student count from the last 7 days from provided records."""
    past_week_counts = []
    today = datetime.today().date()
    for record in attendance_records:
        record_date = record['date']
        # Ensure record_date is a date object for comparison
        if isinstance(record_date, datetime):
            record_date = record_date.date()
        elif isinstance(record_date, str):
            try:
                record_date = datetime.strptime(record_date, '%Y-%m-%d').date()
            except ValueError:
                continue # Skip if date format is incorrect
        
        # Check if the record date is within the last 7 days (inclusive of today, exclusive of 7 days ago)
        if 0 <= (today - record_date).days < 7:
            past_week_counts.append(record['student_count'])
    return round(sum(past_week_counts) / len(past_week_counts)) if past_week_counts else 0

def calculate_change_from_last_week(attendance_records: list) -> float:
    """Calculates percentage change between this week's average and the previous week's."""
    today = datetime.today().date()
    this_week_counts, last_week_counts = [], []
    for record in attendance_records:
        record_date = record['date']
        if isinstance(record_date, datetime):
            record_date = record_date.date()
        elif isinstance(record_date, str):
            try:
                record_date = datetime.strptime(record_date, '%Y-%m-%d').date()
            except ValueError:
                continue # Skip if date format is incorrect

        days_ago = (today - record_date).days
        if 0 <= days_ago < 7: # This week (today to 6 days ago)
            this_week_counts.append(record['student_count'])
        elif 7 <= days_ago < 14: # Last week (7 days ago to 13 days ago)
            last_week_counts.append(record['student_count'])

    avg_this = sum(this_week_counts) / len(this_week_counts) if this_week_counts else 0
    avg_last = sum(last_week_counts) / len(last_week_counts) if last_week_counts else 0

    if avg_last == 0:
        return 0.0 if avg_this == 0 else 100.0 # If last week was 0, and this week is not 0, it's a 100% increase
    return round(((avg_this - avg_last) / avg_last) * 100, 2)

def calculate_monthly_high(attendance_records: list) -> int:
    """Finds the highest attendance count in the last 30 days from provided records."""
    today = datetime.today().date()
    max_count = 0
    for record in attendance_records:
        record_date = record['date']
        if isinstance(record_date, datetime):
            record_date = record_date.date()
        elif isinstance(record_date, str):
            try:
                record_date = datetime.strptime(record_date, '%Y-%m-%d').date()
            except ValueError:
                continue # Skip if date format is incorrect

        if 0 <= (today - record_date).days < 30: # Last 30 days (today to 29 days ago)
            max_count = max(max_count, record['student_count'])
    return max_count

def calculate_change_from_peak(attendance_records: list) -> float:
    """Calculates the drop percentage from peak (highest in last 30 days) to today's attendance."""
    today = datetime.today().date()
    today_count = 0 # Initialize to 0
    max_count = 0
    for record in attendance_records:
        record_date = record['date']
        if isinstance(record_date, datetime):
            record_date = record_date.date()
        elif isinstance(record_date, str):
            try:
                record_date = datetime.strptime(record_date, '%Y-%m-%d').date()
            except ValueError:
                continue # Skip if date format is incorrect
                
        if record_date == today:
            today_count = record['student_count']
        if 0 <= (today - record_date).days < 30:
            max_count = max(max_count, record['student_count'])

    if max_count == 0: # If there's no peak or today's count is 0
        return 0.0
    return round(((today_count - max_count) / max_count) * 100, 2)

# --- Ingredient Inventory Functions ---
def get_ingredient_inventory() -> list:
    """Fetches all ingredients and their current stock from the database."""
    conn = get_db_connection()
    if not conn:
        print("Database connection error. Could not fetch ingredient inventory.")
        return []
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, ingredient_name, current_stock_kg, last_updated FROM ingredients_inventory ORDER BY ingredient_name")
        return cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Database error fetching ingredient inventory: {err}")
        return []
    finally:
        cursor.close()
        conn.close()

def update_ingredient_stock(ingredient_id: int, new_stock: float, admin_id: int):
    """Updates the stock of a specific ingredient."""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error. Could not update ingredient stock.', 'danger')
        return False
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE ingredients_inventory SET current_stock_kg=%s, last_updated=NOW() WHERE id=%s",
            (new_stock, ingredient_id)
        )
        conn.commit()
        flash('Ingredient stock updated successfully.', 'success')
        log_activity(admin_id, 'Inventory Update', f'Updated stock for ingredient ID {ingredient_id} to {new_stock} kg.')
        return True
    except mysql.connector.Error as err:
        flash(f"Database error updating ingredient stock: {err}", 'danger')
        conn.rollback()
        log_activity(admin_id, 'Database Error', f'DB error updating ingredient ID {ingredient_id}: {err}.')
        return False
    finally:
        cursor.close()
        conn.close()

def add_ingredient(ingredient_name: str, initial_stock: float, admin_id: int):
    """Adds a new ingredient to the inventory."""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error. Could not add ingredient.', 'danger')
        return False
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO ingredients_inventory (ingredient_name, current_stock_kg, last_updated) VALUES (%s, %s, NOW())",
            (ingredient_name, initial_stock)
        )
        conn.commit()
        flash(f'Ingredient "{ingredient_name}" added successfully.', 'success')
        log_activity(admin_id, 'Inventory Add', f'Added new ingredient: {ingredient_name} with {initial_stock} kg.')
        return True
    except mysql.connector.Error as err:
        flash(f"Database error adding ingredient: {err}", 'danger')
        conn.rollback()
        log_activity(admin_id, 'Database Error', f'DB error adding ingredient {ingredient_name}: {err}.')
        return False
    finally:
        cursor.close()
        conn.close()

def delete_ingredient(ingredient_id: int, admin_id: int):
    """Deletes an ingredient from the inventory."""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error. Could not delete ingredient.', 'danger')
        return False
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM ingredients_inventory WHERE id=%s", (ingredient_id,))
        conn.commit()
        flash('Ingredient deleted successfully.', 'success')
        log_activity(admin_id, 'Inventory Delete', f'Deleted ingredient ID {ingredient_id}.')
        return True
    except mysql.connector.Error as err:
        flash(f"Database error deleting ingredient: {err}", 'danger')
        conn.rollback()
        log_activity(admin_id, 'Database Error', f'DB error deleting ingredient ID {ingredient_id}: {err}.')
        return False
    finally:
        cursor.close()
        conn.close()

def generate_daily_summary() -> dict:
    """Generates a summary of daily activities, waste, and savings."""
    today = date.today()
    summary = {
        'date': today.strftime('%Y-%m-%d'),
        'total_students': get_today_student_count(),
        'meals_served': get_meals_served_count(today),
        'predicted_waste_kg': get_predicted_waste(today),
        'potential_savings_usd': calculate_savings_potential(today),
        'waste_alerts': [],
        'low_stock_ingredients': []
    }

    # Check for high waste items
    suggestions, _ = suggest_today(summary['total_students'])
    for item in suggestions:
        if item['suggested_quantity'] > 0: # Avoid division by zero
            waste_percentage = (item['predicted_waste'] / item['suggested_quantity']) * 100
            if waste_percentage > WASTE_PERCENTAGE_THRESHOLD:
                summary['waste_alerts'].append(
                    f"High waste for {item['item_name']} ({item['predicted_waste']:.2f} {item['unit']} wasted, {waste_percentage:.2f}% of suggested quantity)."
                )
    
    # Check for low stock ingredients
    inventory = get_ingredient_inventory()
    for ing in inventory:
        if ing['current_stock_kg'] < INGREDIENT_LOW_STOCK_THRESHOLD:
            summary['low_stock_ingredients'].append(
                f"{ing['ingredient_name']} is low in stock ({ing['current_stock_kg']:.2f} kg remaining)."
            )
    
    return summary

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Handles admin login."""
    if 'admin' in session:
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        conn = get_db_connection()
        if not conn:
            flash('Database connection error. Please try again later.', 'danger')
            return render_template('admin_login.html')
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute('SELECT * FROM admins WHERE username=%s', (username,))
            admin = cursor.fetchone()
            if admin and bcrypt.checkpw(password, admin['password_hash'].encode('utf-8')):
                session['admin'] = {
                    'id': admin['id'], 
                    'username': admin['username'], 
                    'full_name': admin['full_name'],
                    'email': admin.get('email', ''), # Include email
                    'email_alerts': bool(admin.get('email_alerts', 0)) # Include email alerts
                }
                flash('Login Successful', 'success')
                log_activity(admin['id'], 'Login', f'Admin {username} logged in.')
                return redirect(url_for('admin_dashboard'))
            flash('Invalid credentials', 'danger')
        except mysql.connector.Error as err:
            flash(f"Database error during login: {err}", 'danger')
        finally:
            cursor.close()
            conn.close()
    return render_template('admin_login.html')

@app.route('/admin/dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    """Renders the admin dashboard with metrics, suggestions, and forms."""
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    conn = get_db_connection()
    if not conn:
        flash('Database connection error. Please try again later.', 'danger')
        # Return a minimal template with error message if DB connection fails
        return render_template('admin_dashboard.html', admin=session.get('admin', {}), 
                               # Pass empty lists for chart data to prevent errors in template
                               waste_trend_labels=[], waste_trend_data=[],
                               savings_labels=[], savings_data=[],
                               waste_items_labels=[], waste_items_data=[],
                               attendance_labels=[], attendance_data=[], waste_data=[],
                               meal_waste_labels=[], meal_waste_data=[],
                               # Pass empty lists for paginated data
                               recent_waste_entries=[], activity_logs=[],
                               # Ensure pagination variables are initialized
                               total_entries=0, current_page=1, total_pages=1, start_entry=0, end_entry=0,
                               total_activity_pages=1, activity_current_page=1,
                               ingredient_inventory=[], daily_summary={}) 

    cursor = conn.cursor(dictionary=True, buffered=True)

    # --- Handle POST requests (attendance, menu, waste, inventory forms) ---
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        admin_id = session['admin']['id']
        anchor_map = {
            'attendance':'#attendance',
            'menu':'#menus',
            'waste':'#addwaste',
            'add_ingredient':'#inventory',
            'update_ingredient':'#inventory'
        }

        try:
            if form_type == 'attendance':
                day = request.form['date']
                student_count_val_str = request.form['student_count']
                
                try:
                    student_count_val = int(student_count_val_str)
                    if student_count_val < 0:
                        flash('Student count cannot be negative.', 'danger')
                        log_activity(admin_id, 'Attendance', f'Failed to record attendance for {day}: Negative student count.')
                    else:
                        cursor.execute("SELECT COUNT(*) AS cnt FROM attendance WHERE DATE(date)=%s", (day,))
                        cnt_result = cursor.fetchone()
                        cnt = cnt_result['cnt'] if cnt_result and 'cnt' in cnt_result and cnt_result['cnt'] is not None else 0
                        
                        if cnt:
                            flash(f'Attendance already recorded for {day}', 'warning')
                            log_activity(admin_id, 'Attendance', f'Attempted to record attendance for {day} (already exists).')
                        else:
                            cursor.execute(
                                "INSERT INTO attendance (date, student_count) VALUES (%s,%s)",
                                (day, student_count_val)
                            )
                            conn.commit() # Commit after attendance insert
                            flash(f'Attendance recorded for {day}', 'success')
                            log_activity(admin_id, 'Attendance', f'Recorded attendance for {day} with {student_count_val} students.')
                except ValueError:
                    flash('Student count must be a valid integer.', 'danger')
                    log_activity(admin_id, 'Attendance', f'Failed to record attendance for {day}: Invalid student count format.')

            elif form_type == 'menu':
                day = request.form['day']
                meal_type = request.form['meal_type']
                items = request.form['items'].strip()
                if not items:
                    flash('Menu items cannot be empty.', 'danger')
                    log_activity(admin_id, 'Menu Update', 'Failed to add menu: Empty items field.')
                else:
                    cursor.execute(
                        "INSERT INTO menu (day, meal_type, items) VALUES (%s,%s,%s)",
                        (day, meal_type, items)
                    )
                    conn.commit() # Commit after menu insert
                    flash('Menu updated successfully!', 'success')
                    log_activity(admin_id, 'Menu Update', f'Added menu for {day}, {meal_type}: {items}.')

            elif form_type == 'waste':
                actual_qty_str = request.form['actual_qty']
                waste_qty_str = request.form['waste_qty']
                student_count_waste_str = request.form['student_count']
                item_name = request.form['item_name'].strip()
                meal_type_waste = request.form['meal_type_waste']
                waste_date = request.form['date']

                try:
                    actual_qty = float(actual_qty_str)
                    waste_qty = float(waste_qty_str)
                    student_count_waste = int(student_count_waste_str)

                    if actual_qty < 0 or waste_qty < 0 or student_count_waste < 0:
                        flash('Quantities and student count cannot be negative.', 'danger')
                        log_activity(admin_id, 'Waste Entry', f'Failed to add waste for {item_name}: Negative quantity/count.')
                    elif waste_qty > actual_qty:
                        flash('Waste quantity cannot exceed actual quantity used.', 'danger')
                        log_activity(admin_id, 'Waste Entry', f'Failed to add waste for {item_name}: Waste > Actual.')
                    elif not item_name:
                        flash('Food item name cannot be empty.', 'danger')
                        log_activity(admin_id, 'Waste Entry', 'Failed to add waste: Empty item name.')
                    else:
                        cursor.execute(
                            """INSERT INTO food_history
                               (day, meal_type, item_name, actual_quantity_used, food_waste, student_count)
                               VALUES (%s,%s,%s,%s,%s,%s)""",
                            (
                                waste_date,
                                meal_type_waste,
                                item_name,
                                actual_qty,
                                waste_qty,
                                student_count_waste
                            )
                        )
                        conn.commit() # Commit after waste entry
                        flash('Waste entry added successfully!', 'success')
                        log_activity(admin_id, 'Waste Entry', f'Added waste entry for {item_name} ({actual_qty} used, {waste_qty} wasted).')
                except ValueError:
                    flash('Actual quantity, waste quantity, and student count must be valid numbers.', 'danger')
                    log_activity(admin_id, 'Waste Entry', f'Failed to add waste for {item_name}: Invalid numeric input.')
            
            elif form_type == 'add_ingredient':
                ingredient_name = request.form['ingredient_name'].strip()
                initial_stock_str = request.form['initial_stock']
                try:
                    initial_stock = float(initial_stock_str)
                    if not ingredient_name:
                        flash('Ingredient name cannot be empty.', 'danger')
                    elif initial_stock < 0:
                        flash('Initial stock cannot be negative.', 'danger')
                    else:
                        add_ingredient(ingredient_name, initial_stock, admin_id)
                except ValueError:
                    flash('Initial stock must be a valid number.', 'danger')
                    log_activity(admin_id, 'Inventory Add', f'Failed to add ingredient {ingredient_name}: Invalid stock format.')

            elif form_type == 'update_ingredient':
                ingredient_id = request.form['ingredient_id']
                new_stock_str = request.form['new_stock']
                try:
                    new_stock = float(new_stock_str)
                    if new_stock < 0:
                        flash('New stock cannot be negative.', 'danger')
                    else:
                        update_ingredient_stock(ingredient_id, new_stock, admin_id)
                except ValueError:
                    flash('New stock must be a valid number.', 'danger')
                    log_activity(admin_id, 'Inventory Update', f'Failed to update ingredient ID {ingredient_id}: Invalid stock format.')

        except mysql.connector.Error as err:
            flash(f"Database error during POST operation: {err}", 'danger')
            conn.rollback() # Rollback changes on error
            log_activity(admin_id, 'Database Error', f'DB error during POST ({form_type}): {err}.')
        finally:
            cursor.close()
            conn.close()
            return redirect(url_for('admin_dashboard') + anchor_map.get(form_type, ''))

    # --- Handle GET requests (display dashboard data) ---
    # Re-establish connection for GET requests after potential POST redirect
    # (or if it's an initial GET request)
    conn = get_db_connection()
    if not conn:
        flash('Database connection lost for display. Please try again later.', 'danger')
        return render_template('admin_dashboard.html', admin=session.get('admin', {}),
                               waste_trend_labels=[], waste_trend_data=[],
                               savings_labels=[], savings_data=[],
                               waste_items_labels=[], waste_items_data=[],
                               attendance_labels=[], attendance_data=[], waste_data=[],
                               meal_waste_labels=[], meal_waste_data=[],
                               recent_waste_entries=[], activity_logs=[],
                               total_entries=0, current_page=1, total_pages=1, start_entry=0, end_entry=0,
                               total_activity_pages=1, activity_current_page=1,
                               ingredient_inventory=[], daily_summary={})
    cursor = conn.cursor(dictionary=True, buffered=True)

    # Fetch recent attendance records
    cursor.execute('SELECT * FROM attendance ORDER BY date DESC LIMIT 10')
    attendance = cursor.fetchall()
    # Calculate variance for attendance (difference from previous day's count)
    # Note: This variance is a simple day-over-day difference, not statistical variance.
    for i in range(len(attendance)):
        attendance[i]['variance'] = 0 # Default to 0
        if i + 1 < len(attendance):
            attendance[i]['variance'] = attendance[i]['student_count'] - attendance[i+1]['student_count']
    
    # Fetch menu items
    cursor.execute('SELECT * FROM menu ORDER BY id DESC')
    menus = cursor.fetchall()
    
    # Organize menus by meal type for easier access in template
    menus_by_meal = {
        m['meal_type'].lower(): [i.strip() for i in m['items'].split(',')]
        for m in menus
    }

    # --- Calculate Core Metrics & Suggestions ---
    total_students_today = get_today_student_count()
    meals_served         = get_meals_served_count()
    predicted_waste      = get_predicted_waste()
    savings_potential    = calculate_savings_potential()

    pct_students         = calculate_change('students')
    pct_meals            = calculate_change('meals')
    pct_waste            = calculate_change('waste')
    pct_savings          = calculate_change('savings')
    weekly_avg           = calculate_weekly_average(attendance)
    pct_change_week      = calculate_change_from_last_week(attendance)
    monthly_high         = calculate_monthly_high(attendance)
    pct_change_month     = calculate_change_from_peak(attendance)
    
    suggestions, ingredients = suggest_today(total_students_today)

    # --- Check for Waste Alerts ---
    waste_alerts = []
    if predicted_waste > 0: # Only check if there's predicted waste
        total_suggested_quantity = sum(item['suggested_quantity'] for item in suggestions)
        if total_suggested_quantity > 0:
            overall_waste_percentage = (predicted_waste / total_suggested_quantity) * 100
            if overall_waste_percentage > WASTE_PERCENTAGE_THRESHOLD:
                waste_alerts.append(f"Overall predicted waste ({predicted_waste:.2f} kg) is high at {overall_waste_percentage:.2f}% of total suggested quantity.")
        
        for item in suggestions:
            if item['suggested_quantity'] > 0:
                item_waste_percentage = (item['predicted_waste'] / item['suggested_quantity']) * 100
                if item_waste_percentage > WASTE_PERCENTAGE_THRESHOLD:
                    waste_alerts.append(f"High waste for {item['item_name']}: {item['predicted_waste']:.2f} {item['unit']} wasted ({item_waste_percentage:.2f}% of suggested).")


    # --- Pagination for Food History ---
    current_page = int(request.args.get('page', 1))
    offset       = (current_page - 1) * PER_PAGE

    cursor.execute("SELECT COUNT(*) AS cnt FROM food_history")
    count_result = cursor.fetchone()
    total_entries = count_result['cnt'] if count_result and 'cnt' in count_result and count_result['cnt'] is not None else 0
    
    cursor.execute("""
        SELECT id, day, meal_type, item_name, actual_quantity_used, food_waste, student_count,
               CASE WHEN actual_quantity_used > 0 THEN ROUND((food_waste/actual_quantity_used)*100,2) ELSE 0 END AS waste_pct
        FROM food_history
        ORDER BY day DESC
        LIMIT %s OFFSET %s
    """, (PER_PAGE, offset))
    rows = cursor.fetchall()
    
    recent_waste_entries = [
        {
            'id': row['id'],
            'date': row['day'],
            'meal_type': row['meal_type'],
            'item_name': row['item_name'],
            'actual_quantity_used': row['actual_quantity_used'],
            'waste_qty': row['food_waste'],
            'waste_pct': row['waste_pct'],
            'student_count': row['student_count'] # Include student count here
        }
        for row in rows
    ]
    total_pages = (total_entries + PER_PAGE - 1) // PER_PAGE
    start_entry = offset + 1 if total_entries else 0
    end_entry   = min(offset + PER_PAGE, total_entries)

    # --- Analytics Data for Charts (last 30 days) ---
    # Waste trend
    cursor.execute("""
        SELECT DATE(day) AS day, SUM(food_waste) AS total_waste
        FROM food_history
        WHERE day >= CURDATE() - INTERVAL 30 DAY
        GROUP BY DATE(day) ORDER BY day
    """)
    wt = cursor.fetchall()
    waste_trend_labels = [r['day'].strftime('%b %d') for r in wt]
    waste_trend_data   = [float(r['total_waste']) for r in wt]
    
    # Savings trend (derived from waste trend)
    savings_labels = waste_trend_labels
    savings_data   = [round(w * COST_PER_KG, 2) for w in waste_trend_data]
    
    # Top waste items
    cursor.execute("""
        SELECT item_name, SUM(food_waste) AS total_waste
        FROM food_history
        WHERE day >= CURDATE() - INTERVAL 30 DAY
        GROUP BY item_name ORDER BY total_waste DESC LIMIT 5
    """)
    ti = cursor.fetchall()
    waste_items_labels = [r['item_name'] for r in ti]
    waste_items_data   = [float(r['total_waste']) for r in ti]
    
    # Attendance vs Waste comparison
    cursor.execute("""
        SELECT a.date, a.student_count, COALESCE(SUM(f.food_waste),0) AS waste
        FROM attendance a
        LEFT JOIN food_history f ON DATE(f.day)=a.date
        WHERE a.date >= CURDATE() - INTERVAL 30 DAY
        GROUP BY a.date ORDER BY a.date
    """)
    aw = cursor.fetchall()
    attendance_labels = [r['date'].strftime('%b %d') for r in aw]
    attendance_data   = [r['student_count'] for r in aw]
    waste_data        = [float(r['waste']) for r in aw]
    
    # Meal type waste breakdown
    cursor.execute("""
        SELECT meal_type, SUM(food_waste) AS waste
        FROM food_history
        WHERE day >= CURDATE() - INTERVAL 30 DAY
        GROUP BY meal_type
    """)
    mw = cursor.fetchall()
    meal_waste_labels = [r['meal_type'].capitalize() for r in mw]
    meal_waste_data   = [float(r['waste']) for r in mw]
    
    # --- Load Admin Settings (for notification preferences) ---
    cursor.execute(
        "SELECT email, email_alerts, full_name FROM admins WHERE id=%s",
        (session['admin']['id'],)
    )
    admin_settings = cursor.fetchone()
    if admin_settings:
        session['admin'].update({
            'email': admin_settings['email'],
            'email_alerts': bool(admin_settings['email_alerts']),
            'full_name': admin_settings['full_name'] # Ensure full_name is also updated in session if changed
        })
    else:
        session['admin'].update({'email': '', 'email_alerts': False, 'full_name': session['admin'].get('full_name', '')})

    # --- Pagination for Activity Log ---
    activity_current_page = int(request.args.get('activity_page', 1))
    activity_offset = (activity_current_page - 1) * ACTIVITY_PER_PAGE

    cursor.execute("SELECT COUNT(*) AS cnt FROM activity_log WHERE admin_id = %s", (session['admin']['id'],))
    total_acts_row = cursor.fetchone()
    total_acts = total_acts_row['cnt'] if total_acts_row and 'cnt' in total_acts_row and total_acts_row['cnt'] is not None else 0

    cursor.execute("""
    SELECT timestamp, type, details, ip_address
    FROM activity_log
    WHERE admin_id = %s
    ORDER BY timestamp DESC
    LIMIT %s OFFSET %s
    """, (session['admin']['id'], ACTIVITY_PER_PAGE, activity_offset))

    activity_logs = cursor.fetchall()
    total_activity_pages = (total_acts + ACTIVITY_PER_PAGE - 1) // ACTIVITY_PER_PAGE

    # --- Fetch Ingredient Inventory ---
    ingredient_inventory = get_ingredient_inventory()

    # --- Generate Daily Summary ---
    daily_summary = generate_daily_summary()


    cursor.close()
    conn.close()

    # Render the dashboard template with all collected data
    return render_template(
        'admin_dashboard.html',
        admin=session['admin'],
        # Attendance & Menu Data
        attendance=attendance,
        menus=menus,
        menus_by_meal=menus_by_meal,
        item_ingredients=ITEM_INGREDIENTS,
        item_units=ITEM_UNITS, # Pass item units for display
        # Metrics & Suggestions
        total_students_today=total_students_today,
        meals_served=meals_served,
        predicted_waste=predicted_waste,
        savings_potential=savings_potential,
        pct_students=pct_students,
        pct_meals=pct_meals,
        pct_waste=pct_waste,
        pct_savings=pct_savings,
        weekly_avg=weekly_avg,
        pct_change_week=pct_change_week,
        monthly_high=monthly_high,
        pct_change_month=pct_change_month,
        suggestions=suggestions,
        ingredients=ingredients,
        waste_alerts=waste_alerts, # Pass waste alerts to template
        # Waste history pagination data
        recent_waste_entries=recent_waste_entries,
        total_entries=total_entries,
        current_page=current_page,
        total_pages=total_pages,
        start_entry=start_entry,
        end_entry=end_entry,
        # Analytics chart data
        waste_trend_labels=waste_trend_labels,
        waste_trend_data=waste_trend_data,
        savings_labels=savings_labels,
        savings_data=savings_data,
        waste_items_labels=waste_items_labels,
        waste_items_data=waste_items_data,
        attendance_labels=attendance_labels,
        attendance_data=attendance_data,
        waste_data=waste_data,
        meal_waste_labels=meal_waste_labels,
        meal_waste_data=meal_waste_data,
        # Activity log data
        activity_logs=activity_logs,
        total_activity_pages=total_activity_pages,
        activity_current_page=activity_current_page,
        # Ingredient Inventory
        ingredient_inventory=ingredient_inventory,
        # Daily Summary
        daily_summary=daily_summary,
        datetime=datetime # Pass datetime module for template formatting
    )

@app.route('/admin/update-settings', methods=['POST'], endpoint='admin_update_settings')
def admin_update_settings():
    """Handles updating admin profile and notification settings."""
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    conn = get_db_connection()
    if not conn:
        flash('Database connection error. Could not update settings.', 'danger')
        return redirect(url_for('admin_dashboard') + '#settings')

    cursor = conn.cursor()
    admin_id = session['admin']['id']
    form_type = request.form.get('form_type', 'profile') # Default to profile settings

    try:
        if form_type == 'notifications':
            alerts = 1 if request.form.get('emailAlerts') == 'on' else 0
            cursor.execute(
                "UPDATE admins SET email_alerts=%s WHERE id=%s",
                (alerts, admin_id)
            )
            conn.commit() # Commit after update
            session['admin']['email_alerts'] = bool(alerts)
            flash('Notification preferences updated.', 'success')
            log_activity(admin_id, 'Settings Update', f'Notification alerts set to: {bool(alerts)}.')

        elif form_type == 'profile':
            full_name = request.form['full_name'].strip()
            email = request.form['email'].strip()
            new_password = request.form.get('new_password', '')
            confirm_pass = request.form.get('confirm_password', '')

            if not full_name or not email:
                flash('Full name and email cannot be empty.', 'danger')
                return redirect(url_for('admin_dashboard') + '#settings')

            update_sql = "UPDATE admins SET full_name=%s, email=%s WHERE id=%s"
            update_params = [full_name, email, admin_id]
            
            if new_password:
                if new_password != confirm_pass:
                    flash('Passwords do not match. Profile not updated.', 'danger')
                    return redirect(url_for('admin_dashboard') + '#settings')
                if len(new_password) < 6: # Basic password length validation
                    flash('New password must be at least 6 characters long.', 'danger')
                    return redirect(url_for('admin_dashboard') + '#settings')

                hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                update_sql = "UPDATE admins SET full_name=%s, email=%s, password_hash=%s WHERE id=%s"
                update_params = [full_name, email, hashed, admin_id]
            
            cursor.execute(update_sql, tuple(update_params))
            conn.commit() # Commit after update
            
            session['admin']['full_name'] = full_name
            session['admin']['email'] = email # Update email in session
            flash('Profile settings updated.', 'success')
            log_activity(admin_id, 'Settings Update', f'Profile updated (Name: {full_name}, Email: {email}). Password {'changed' if new_password else 'not changed'}.')
        
    except mysql.connector.Error as err:
        flash(f"Database error during settings update: {err}", 'danger')
        conn.rollback()
        log_activity(admin_id, 'Database Error', f'DB error during settings update ({form_type}): {err}.')
    finally:
        cursor.close()
        conn.close()
    
    return redirect(url_for('admin_dashboard') + '#settings')

@app.route('/admin/export_activity_log')
def export_activity_log():
    """Exports a page of the activity log as a CSV file."""
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    page = int(request.args.get('activity_page', 1))
    offset = (page - 1) * ACTIVITY_PER_PAGE
    admin_id = session['admin']['id']

    conn = get_db_connection()
    if not conn:
        flash('Database connection error for export. Cannot export activity log.', 'danger')
        return redirect(url_for('admin_dashboard') + '#activity')

    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT timestamp, type, details, ip_address
            FROM activity_log
            WHERE admin_id = %s
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
        """, (admin_id, ACTIVITY_PER_PAGE, offset))
        rows = cursor.fetchall()
    except mysql.connector.Error as err:
        flash(f"Database error during activity log export: {err}", 'danger')
        rows = []
        log_activity(admin_id, 'Export Error', f'DB error during activity log export: {err}.')
    finally:
        cursor.close()
        conn.close()

    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(['Timestamp', 'Action', 'Details', 'IP Address']) # Added IP Address header
    for ts, t, det, ip in rows: # Unpack IP address
        writer.writerow([ts, t, det, ip])

    log_activity(admin_id, 'Export', f'Exported activity log page {page}.')
    return Response(
        si.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment;filename=activity_log_page_{page}.csv'}
    )

@app.route('/admin/export_food_history')
def export_food_history():
    """Exports all food history data as a CSV file."""
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    admin_id = session['admin']['id']
    conn = get_db_connection()
    if not conn:
        flash('Database connection error for export. Cannot export food history.', 'danger')
        return redirect(url_for('admin_dashboard') + '#recentwaste')

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT day, meal_type, item_name, actual_quantity_used, food_waste, student_count FROM food_history ORDER BY day DESC")
        rows = cursor.fetchall()
    except mysql.connector.Error as err:
        flash(f"Database error during food history export: {err}", 'danger')
        rows = []
        log_activity(admin_id, 'Export Error', f'DB error during food history export: {err}.')
    finally:
        cursor.close()
        conn.close()

    si = io.StringIO()
    writer = csv.DictWriter(si, fieldnames=['day', 'meal_type', 'item_name', 'actual_quantity_used', 'food_waste', 'student_count'])
    writer.writeheader()
    writer.writerows(rows)

    log_activity(admin_id, 'Export', 'Exported all food history data.')
    return Response(
        si.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=food_history.csv'}
    )

@app.route('/admin/export_attendance')
def export_attendance():
    """Exports all attendance data as a CSV file."""
    if 'admin' not in session:
        return redirect(url_for('admin_login') + '#attendance')

    admin_id = session['admin']['id']
    conn = get_db_connection()
    if not conn:
        flash('Database connection error for export. Cannot export attendance.', 'danger')
        return redirect(url_for('admin_dashboard') + '#attendance')

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT date, student_count FROM attendance ORDER BY date DESC")
        rows = cursor.fetchall()
    except mysql.connector.Error as err:
        flash(f"Database error during attendance export: {err}", 'danger')
        rows = []
        log_activity(admin_id, 'Export Error', f'DB error during attendance export: {err}.')
    finally:
        cursor.close()
        conn.close()

    si = io.StringIO()
    writer = csv.DictWriter(si, fieldnames=['date', 'student_count'])
    writer.writeheader()
    writer.writerows(rows)

    log_activity(admin_id, 'Export', 'Exported all attendance data.')
    return Response(
        si.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=attendance_log.csv'}
    )

@app.route('/admin/export_menu')
def export_menu():
    """Exports all menu data as a CSV file."""
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    admin_id = session['admin']['id']
    conn = get_db_connection()
    if not conn:
        flash('Database connection error for export. Cannot export menu.', 'danger')
        return redirect(url_for('admin_dashboard') + '#menus')

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT day, meal_type, items FROM menu ORDER BY day, meal_type")
        rows = cursor.fetchall()
    except mysql.connector.Error as err:
        flash(f"Database error during menu export: {err}", 'danger')
        rows = []
        log_activity(admin_id, 'Export Error', f'DB error during menu export: {err}.')
    finally:
        cursor.close()
        conn.close()

    si = io.StringIO()
    writer = csv.DictWriter(si, fieldnames=['day', 'meal_type', 'items'])
    writer.writeheader()
    writer.writerows(rows)

    log_activity(admin_id, 'Export', 'Exported all menu data.')
    return Response(
        si.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=menu_data.csv'}
    )


@app.route('/edit-menu/<int:menu_id>', methods=['POST'])
def edit_menu(menu_id):
    """Handles editing an existing menu entry."""
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    
    day = request.form['day']
    meal_type = request.form['meal_type']
    items = request.form['items'].strip()
    admin_id = session['admin']['id']
    
    if not items:
        flash('Menu items cannot be empty.', 'danger')
        return redirect(url_for('admin_dashboard') + '#menus')

    conn = get_db_connection()
    if not conn:
        flash('Database connection error. Could not edit menu.', 'danger')
        return redirect(url_for('admin_dashboard') + '#menus')
    
    cursor = conn.cursor()
    try:
        cursor.execute('UPDATE menu SET day=%s, meal_type=%s, items=%s WHERE id=%s',
                       (day, meal_type, items, menu_id))
        conn.commit()
        flash('Menu updated successfully!', 'success')
        log_activity(admin_id, 'Menu Edit', f'Edited menu ID {menu_id} for {day}, {meal_type}: {items}.')
    except mysql.connector.Error as err:
        flash(f"Database error during menu update: {err}", 'danger')
        conn.rollback()
        log_activity(admin_id, 'Database Error', f'DB error editing menu ID {menu_id}: {err}.')
    finally:
        cursor.close()
        conn.close()
    
    return redirect(url_for('admin_dashboard') + '#menus')

@app.route('/delete-menu/<int:menu_id>')
def delete_menu(menu_id):
    """Handles deleting a menu entry."""
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    
    admin_id = session['admin']['id']
    conn = get_db_connection()
    if not conn:
        flash('Database connection error. Could not delete menu.', 'danger')
        return redirect(url_for('admin_dashboard') + '#menus')
    
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM menu WHERE id=%s', (menu_id,))
        conn.commit()
        flash('Menu deleted successfully.', 'success')
        log_activity(admin_id, 'Menu Delete', f'Deleted menu ID {menu_id}.')
    except mysql.connector.Error as err:
        flash(f"Database error during menu deletion: {err}", 'danger')
        conn.rollback()
        log_activity(admin_id, 'Database Error', f'DB error deleting menu ID {menu_id}: {err}.')
    finally:
        cursor.close()
        conn.close()
    
    return redirect(url_for('admin_dashboard') + '#menus')

@app.route('/delete-waste-entry/<int:entry_id>')
def delete_waste_entry(entry_id):
    """Handles deleting a food waste entry."""
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    
    admin_id = session['admin']['id']
    conn = get_db_connection()
    if not conn:
        flash('Database connection error. Could not delete waste entry.', 'danger')
        return redirect(url_for('admin_dashboard') + '#recentwaste')
    
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM food_history WHERE id=%s', (entry_id,))
        conn.commit()
        flash('Waste entry deleted successfully.', 'success')
        log_activity(admin_id, 'Waste Delete', f'Deleted food waste entry ID {entry_id}.')
    except mysql.connector.Error as err:
        flash(f"Database error during waste entry deletion: {err}", 'danger')
        conn.rollback()
        log_activity(admin_id, 'Database Error', f'DB error deleting waste entry ID {entry_id}: {err}.')
    finally:
        cursor.close()
        conn.close()
    
    return redirect(url_for('admin_dashboard') + '#recentwaste')

@app.route('/delete-ingredient/<int:ingredient_id>')
def delete_ingredient_route(ingredient_id):
    """Handles deleting an ingredient from inventory."""
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    
    admin_id = session['admin']['id']
    if delete_ingredient(ingredient_id, admin_id):
        flash('Ingredient deleted successfully.', 'success')
    else:
        flash('Failed to delete ingredient.', 'danger')
    
    return redirect(url_for('admin_dashboard') + '#inventory')


@app.route('/admin/logout')
def admin_logout():
    """Logs out the admin user."""
    admin_id = session['admin']['id'] if 'admin' in session else 'unknown'
    username = session['admin']['username'] if 'admin' in session else 'unknown'
    session.pop('admin', None) # Remove 'admin' from session
    flash('Logged out successfully.', 'success')
    log_activity(admin_id, 'Logout', f'Admin {username} logged out.')
    return redirect(url_for('admin_login'))

if __name__ == '__main__':
    # Perform initial model training if model files do not exist
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(FEATURES_FILENAME):
        print("Model files not found. Attempting initial model training...")
        with app.app_context(): # Ensure app context for initial training
            train_model()
    
    # Run the Flask application with SocketIO
    # Debug mode is controlled by the DEBUG environment variable
    socketio.run(app, host='0.0.0.0', port=5000, debug=(os.getenv('DEBUG', 'False').lower() == 'true'))

