from flask import Flask, render_template, request, redirect, url_for, session, flash
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
from sklearn.metrics import mean_squared_error
from statistics import variance

# Load environment variables
load_dotenv()

# Flask Setup
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or 'change-me'
csrf = CSRFProtect(app)
socketio = SocketIO(app)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Database config
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306))
}

# Model config
MODEL_FILENAME = "food_quantity_model.pkl"
FEATURES_FILENAME = "feature_columns.pkl"
DEFAULT_COMPARE_DAYS = 7
COST_PER_KG = 3.50


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

# DB Connect
def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        print(f"DB connection error: {err}")
        return None
    
def get_today_student_count() -> int | None:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        today = date.today()
        cursor.execute("SELECT student_count FROM attendance WHERE date = %s", (today,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row[0] if row else None
    except:
        return None


# Data fetch & model training

def fetch_history_data():
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT day, meal_type, item_name, actual_quantity_used, food_waste FROM food_history"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)


def prepare_features(df):
    df_enc = pd.get_dummies(df, columns=['day', 'meal_type', 'item_name'], drop_first=True)
    y = df_enc[['actual_quantity_used', 'food_waste']].values
    X = df_enc.drop(columns=['actual_quantity_used', 'food_waste'])
    return X, y


def train_model():
    df = fetch_history_data()
    if df.shape[0] < 10:
        return
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open(MODEL_FILENAME, 'wb') as mf:
        pickle.dump(model, mf)
    with open(FEATURES_FILENAME, 'wb') as ff:
        pickle.dump(X.columns.tolist(), ff)


@scheduler.task('cron', id='retrain_model', hour=2)
def scheduled_training():
    train_model()


def load_model():
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(FEATURES_FILENAME):
        train_model()
    with open(MODEL_FILENAME, 'rb') as mf:
        model = pickle.load(mf)
    with open(FEATURES_FILENAME, 'rb') as ff:
        feature_cols = pickle.load(ff)
    return model, feature_cols

# Suggestion logic
def suggest_today(student_count):
    model, feature_cols = load_model()
    today_day = datetime.now().strftime('%A')
    conn = get_db_connection()
    if not conn:
        return [], []
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT meal_type, items FROM menu WHERE day=%s", (today_day,))
    menu = cursor.fetchall()
    cursor.close()
    conn.close()

    suggestions = []
    total_ingredients = {}
    for row in menu:
        meal = row['meal_type']
        items = [i.strip() for i in row['items'].split(',') if i.strip()]
        for itm in items:
            fv = build_feature_vector(today_day, meal, itm, student_count, feature_cols)
            preds = model.predict(fv)
            pred_qty = float(preds[0][0])
            pred_waste = float(preds[0][1])
            qty = max(0, int(round(pred_qty)))
            waste = max(0, round(pred_waste, 2))
            unit = ITEM_UNITS.get(itm.lower(), 'kg')
            suggestions.append({
                'meal_type': meal,
                'item_name': itm,
                'suggested_quantity': qty,
                'predicted_waste': waste,
                'unit': unit
            })
            for ing, per in ITEM_INGREDIENTS.get(itm.lower(), {}).items():
                total_ingredients[ing] = total_ingredients.get(ing, 0) + per * qty

    ing_list = [{'ingredient': k, 'required_quantity': round(v, 2)} for k, v in total_ingredients.items()]
    return suggestions, ing_list


def get_predicted_waste(for_date=None):
    """
    Predict total waste (in kg) for all menu items on a given date.
    """
    if for_date is None:
        for_date = date.today()
    # Fetch student count for that date
    students = get_student_count_by_date(for_date)
    # Generate daily suggestions which now include 'predicted_waste'
    suggestions, _ = suggest_today(students)
    # Sum up the waste predictions from each suggestion
    total_waste = sum(item['predicted_waste'] for item in suggestions)
    return round(total_waste, 2)


# Metrics

def get_today_student_count():
    conn = get_db_connection()
    cursor = conn.cursor()
    today = date.today()
    cursor.execute('SELECT student_count FROM attendance WHERE date=%s', (today,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else 0


def get_student_count_by_date(target_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT student_count FROM attendance WHERE date=%s', (target_date,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else 0


def get_meals_served_count(for_date=None):
    if for_date is None:
        for_date = date.today()
    conn = get_db_connection()
    cursor = conn.cursor()
    weekday = for_date.strftime('%A')
    cursor.execute('SELECT COUNT(*) FROM menu WHERE day=%s', (weekday,))
    periods = cursor.fetchone()[0] or 0
    cursor.close()
    conn.close()
    return get_student_count_by_date(for_date) * periods


# Build feature vector

def build_feature_vector(day_str, meal, item, student_count, feature_cols):
    vec = {col: 0 for col in feature_cols}
    vec['student_count'] = student_count
    for prefix in (f'day_{day_str}', f'meal_type_{meal}', f'item_name_{item}'):
        if prefix in vec:
            vec[prefix] = 1
    return pd.DataFrame([vec], columns=feature_cols)




def calculate_savings_potential(for_date=None):
    waste_kg = get_predicted_waste(for_date)
    return round(waste_kg * COST_PER_KG, 2)


def calculate_change(metric, days=DEFAULT_COMPARE_DAYS):
    today = date.today()
    prev = today - timedelta(days=days)
    if metric == 'students':
        current = get_today_student_count()
        previous = get_student_count_by_date(prev)
    elif metric == 'meals':
        current = get_meals_served_count(today)
        previous = get_meals_served_count(prev)
    elif metric == 'waste':
        current = get_predicted_waste(today)
        previous = get_predicted_waste(prev)
    elif metric == 'savings':
        current = calculate_savings_potential(today)
        previous = calculate_savings_potential(prev)
    else:
        raise ValueError(f"Unknown metric '{metric}' for change calculation")
    return round(((current - previous) / previous) * 100, 2) if previous else 0.0

def calculate_weekly_average(attendance: list) -> int:
    """Calculates average student count from last 7 days."""
    past_week = []
    today = datetime.today().date()
    for record in attendance:
        record_date = record['date']
        if isinstance(record_date, datetime):
            record_date = record_date.date()
        if 0 <= (today - record_date).days < 7:
            past_week.append(record['student_count'])
    return round(sum(past_week) / len(past_week)) if past_week else 0

def calculate_change_from_last_week(attendance: list) -> float:
    """Calculates percentage change between this week's average and the previous week's."""
    today = datetime.today().date()
    this_week, last_week = [], []
    for record in attendance:
        record_date = record['date']
        if isinstance(record_date, datetime):
            record_date = record_date.date()
        days_ago = (today - record_date).days
        if 0 <= days_ago < 7:
            this_week.append(record['student_count'])
        elif 7 <= days_ago < 14:
            last_week.append(record['student_count'])

    if this_week and last_week:
        avg_this = sum(this_week) / len(this_week)
        avg_last = sum(last_week) / len(last_week)
        if avg_last == 0:
            return 0.0
        return round(((avg_this - avg_last) / avg_last) * 100, 2)
    return 0.0

def calculate_monthly_high(attendance: list) -> int:
    """Finds the highest attendance count in the last 30 days."""
    today = datetime.today().date()
    max_count = 0
    for record in attendance:
        record_date = record['date']
        if isinstance(record_date, datetime):
            record_date = record_date.date()
        if 0 <= (today - record_date).days < 30:
            max_count = max(max_count, record['student_count'])
    return max_count

def calculate_change_from_peak(attendance: list) -> float:
    """Calculates the drop percentage from peak (highest) to today's attendance."""
    today = datetime.today().date()
    today_count = None
    max_count = 0
    for record in attendance:
        record_date = record['date']
        if isinstance(record_date, datetime):
            record_date = record_date.date()
        if record_date == today:
            today_count = record['student_count']
        if 0 <= (today - record_date).days < 30:
            max_count = max(max_count, record['student_count'])

    if today_count is None or max_count == 0:
        return 0.0
    return round(((today_count - max_count) / max_count) * 100, 2)


# Routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if 'admin' in session:
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM admins WHERE username=%s', (username,))
        admin = cursor.fetchone()
        cursor.close()
        conn.close()
        if admin and bcrypt.checkpw(password, admin['password_hash'].encode('utf-8')):
            session['admin'] = {'id': admin['id'], 'username': admin['username'], 'full_name': admin['full_name']}
            flash('Login Successful', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'attendance':
            cursor.execute(
                'INSERT INTO attendance (date, student_count) VALUES (%s, %s)',
                (request.form['date'], request.form['student_count'])
            )

        elif form_type == 'menu':
            cursor.execute(
                'INSERT INTO menu (day, meal_type, items) VALUES (%s, %s, %s)',
                (request.form['day'], request.form['meal_type'], request.form['items'])
            )

        elif form_type == 'waste':
             cursor.execute(
                '''INSERT INTO food_history 
                   (day, meal_type, item_name, actual_quantity_used, food_waste, student_count)
                   VALUES (%s, %s, %s, %s, %s, %s)''',
                (
                    request.form['date'],
                    request.form['meal_type_waste'],
                    request.form['item_name'],
                    request.form['actual_qty'],
                    request.form['waste_qty'],
                    request.form['student_count']  # Ensure student_count is passed from the form
                )
            )
        conn.commit()
        return redirect(url_for('admin_dashboard'))

    # Fetch attendance data
    cursor.execute('SELECT * FROM attendance ORDER BY date DESC LIMIT 10')
    attendance = cursor.fetchall()

    # Compute variance for attendance
    for idx, rec in enumerate(attendance):
        if idx + 1 < len(attendance):
            rec['variance'] = rec['student_count'] - attendance[idx + 1]['student_count']
        else:
            rec['variance'] = 0

    # Fetch menu data
    cursor.execute('SELECT * FROM menu ORDER BY id DESC')
    menus = cursor.fetchall()

    # Prepare dynamic menus data for the form (by meal type)
    menus_by_meal = {}
    for menu in menus:
        items = [item.strip() for item in menu['items'].split(',')]
        menus_by_meal[menu['meal_type'].lower()] = items

    cursor.close()
    conn.close()

    # Dashboard metrics
    total_students_today = get_today_student_count()
    meals_served = get_meals_served_count()
    predicted_waste = get_predicted_waste()
    savings_potential = calculate_savings_potential()

    pct_students = calculate_change('students')
    pct_meals = calculate_change('meals')
    pct_waste = calculate_change('waste')
    pct_savings = calculate_change('savings')

    weekly_avg = calculate_weekly_average(attendance)
    pct_change_week = calculate_change_from_last_week(attendance)
    monthly_high = calculate_monthly_high(attendance)
    pct_change_month = calculate_change_from_peak(attendance)

    # Suggestions & Ingredients
    suggestions, ingredients = suggest_today(total_students_today)

    return render_template('admin_dashboard.html',
                           admin=session['admin'],
                           attendance=attendance,
                           menus=menus,
                           menus_by_meal=menus_by_meal,  # For dynamic meal items
                           total_students_today=total_students_today,
                           meals_served=meals_served,
                           predicted_waste=predicted_waste,
                           savings_potential=savings_potential,
                           pct_students=pct_students,
                           pct_meals=pct_meals,
                           pct_waste=pct_waste,
                           pct_savings=pct_savings,
                           suggestions=suggestions,
                           ingredients=ingredients,
                           item_ingredients=ITEM_INGREDIENTS,
                           weekly_avg=weekly_avg,
                           pct_change_week=pct_change_week,
                           monthly_high=monthly_high,
                           pct_change_month=pct_change_month,
                           datetime=datetime)


@app.route('/edit-menu/<int:menu_id>', methods=['POST'])
def edit_menu(menu_id):
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    day = request.form['day']
    meal_type = request.form['meal_type']
    items = request.form['items']
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE menu SET day=%s, meal_type=%s, items=%s WHERE id=%s',
                   (day, meal_type, items, menu_id))
    conn.commit()
    cursor.close()
    conn.close()
    flash('Menu updated successfully!', 'success')
    return redirect(url_for('admin_dashboard') + '#menus')

@app.route('/delete-menu/<int:menu_id>')
def delete_menu(menu_id):
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM menu WHERE id=%s', (menu_id,))
    conn.commit()
    cursor.close()
    conn.close()
    flash('Menu deleted successfully.', 'success')
    return redirect(url_for('admin_dashboard') + '#menus')

@app.route('/admin/update-settings', methods=['POST'])
def update_admin_settings():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    full_name = request.form['full_name']
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    conn = get_db_connection()
    cursor = conn.cursor()
    if new_password:
        if new_password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('admin_dashboard') + '#settings')
        hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute('UPDATE admins SET full_name=%s, password_hash=%s WHERE id=%s',
                       (full_name, hashed, session['admin']['id']))
    else:
        cursor.execute('UPDATE admins SET full_name=%s WHERE id=%s', (full_name, session['admin']['id']))
    conn.commit()
    cursor.close()
    conn.close()
    session['admin']['full_name'] = full_name
    flash('Admin settings updated successfully!', 'success')
    return redirect(url_for('admin_dashboard') + '#settings')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    flash('Logged out', 'success')
    return redirect(url_for('admin_login'))

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=(os.getenv('DEBUG', 'False')=='True'))
