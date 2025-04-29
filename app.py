
from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import bcrypt
from flask_wtf import CSRFProtect
from flask_socketio import SocketIO
from flask_apscheduler import APScheduler
import os
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load environment variables
load_dotenv()

# Flask Setup
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
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

MODEL_FILENAME = "food_quantity_model.pkl"


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

# AI Functions

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
    
def fetch_history_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT day, meal_type, student_count, item_name, actual_quantity_used FROM food_history')
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except:
        return pd.DataFrame()


def prepare_features(df):
    df = pd.get_dummies(df, columns=['day', 'meal_type', 'item_name'], drop_first=True)
    y = df['actual_quantity_used']
    X = df.drop(columns=['actual_quantity_used'])
    return X, y


def train_model():
    df = fetch_history_data()
    if df.empty:
        return
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump((model, X.columns.tolist()), f)


@scheduler.task('cron', id='retrain_model', hour=2)
def scheduled_training():
    train_model()


def load_model():
    if not os.path.exists(MODEL_FILENAME):
        train_model()
    with open(MODEL_FILENAME, 'rb') as f:
        return pickle.load(f)


# Suggest today

def suggest_today(student_count):
    model, feature_cols = load_model()
    today_day = datetime.now().strftime('%A')
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT meal_type, items FROM menu WHERE day=%s', (today_day,))
    menu = cursor.fetchall()
    cursor.close()
    conn.close()

    suggestions = []
    ingredients_needed = {}

    for row in menu:
        meal = row['meal_type']
        items = row['items'].split(',')
        for item in items:
            item = item.strip()
            if not item:
                continue
            
            row_data = {col: 0 for col in feature_cols}
            row_data['student_count'] = student_count
            if f"day_{today_day}" in row_data:
                row_data[f"day_{today_day}"] = 1
            if f"meal_type_{meal}" in row_data:
                row_data[f"meal_type_{meal}"] = 1
            if f"item_name_{item}" in row_data:
                row_data[f"item_name_{item}"] = 1

            pred_qty = max(0, round(model.predict(pd.DataFrame([row_data]))[0]))
            unit = ITEM_UNITS.get(item.lower(), 'kg')
            suggestions.append({'meal_type': meal, 'item_name': item, 'suggested_quantity': pred_qty, 'unit': unit})

            # Calculate ingredients
            for ing, per in ITEM_INGREDIENTS.get(item.lower(), {}).items():
                ingredients_needed[ing] = ingredients_needed.get(ing, 0) + (per * pred_qty)

    ingredient_list = [{'ingredient': k, 'required_quantity': round(v, 2)} for k, v in ingredients_needed.items()]
    return suggestions, ingredient_list


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
        else:
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
            cursor.execute('INSERT INTO attendance (date, student_count) VALUES (%s, %s)', (request.form['date'], request.form['student_count']))
        elif form_type == 'menu':
            cursor.execute('INSERT INTO menu (day, meal_type, items) VALUES (%s, %s, %s)', (request.form['day'], request.form['meal_type'], request.form['items']))
        conn.commit()
        return redirect(url_for('admin_dashboard'))

    cursor.execute('SELECT * FROM attendance ORDER BY date DESC LIMIT 10')
    attendance = cursor.fetchall()
    cursor.execute('SELECT * FROM menu ORDER BY id DESC')
    menus = cursor.fetchall()

    cursor.close()
    conn.close()

    today_student_count = get_today_student_count()
    suggestions, ingredients = suggest_today(today_student_count) if today_student_count else ([], [])

    return render_template('admin_dashboard.html', admin=session['admin'], attendance=attendance,  item_ingredients=ITEM_INGREDIENTS, menus=menus, suggestions=suggestions, ingredients=ingredients, datetime=datetime)

# Edit Menu
@app.route('/edit-menu/<int:menu_id>', methods=['POST'])
def edit_menu(menu_id):
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    day = request.form.get('day')
    meal_type = request.form.get('meal_type')
    items = request.form.get('items')

    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE menu
            SET day = %s, meal_type = %s, items = %s
            WHERE id = %s
        ''', (day, meal_type, items, menu_id))
        conn.commit()
        cursor.close()
        conn.close()

        flash('Menu updated successfully!', 'success')

    return redirect(url_for('admin_dashboard') + '#menus')

# Delete Menu
@app.route('/delete-menu/<int:menu_id>')
def delete_menu(menu_id):
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM menu WHERE id = %s', (menu_id,))
    conn.commit()
    cursor.close()
    conn.close()

    flash('Menu deleted successfully.', 'success')
    return redirect(url_for('admin_dashboard') + '#menus')
# Update Admin Settings
@app.route('/admin/update-settings', methods=['POST'])
def update_admin_settings():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    full_name = request.form.get('full_name')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        
        if new_password:
            if new_password != confirm_password:
                flash('Passwords do not match.', 'danger')
                return redirect(url_for('admin_dashboard') + '#settings')
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute('''
                UPDATE admins SET full_name = %s, password_hash = %s WHERE id = %s
            ''', (full_name, hashed_password, session['admin']['id']))
        else:
            cursor.execute('''
                UPDATE admins SET full_name = %s WHERE id = %s
            ''', (full_name, session['admin']['id']))

        conn.commit()
        cursor.close()
        conn.close()

        session['admin']['full_name'] = full_name  # Update session
        flash('Admin settings updated successfully!', 'success')

    return redirect(url_for('admin_dashboard') + '#settings')


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    flash('Logged out', 'success')
    return redirect(url_for('admin_login'))

# Start App
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=os.getenv('DEBUG', 'False') == 'True')
