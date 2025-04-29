#!/usr/bin/env python3

import os
import pickle
from datetime import datetime, date
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load environment variables
load_dotenv()
DB_HOST         = os.getenv("DB_HOST", "localhost")
DB_USER         = os.getenv("DB_USER")
DB_PASSWORD     = os.getenv("DB_PASSWORD")
DB_NAME         = os.getenv("DB_NAME")
MODEL_FILENAME  = "food_quantity_model.pkl"

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

def get_today_student_count() -> int | None:
    try:
        conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
        cursor = conn.cursor()
        today = date.today()
        cursor.execute("SELECT student_count FROM attendance WHERE date = %s", (today,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row[0] if row else None
    except Error as e:
        print(f"[ERROR] fetching attendance: {e}")
        return None

def fetch_history_data() -> pd.DataFrame:
    try:
        conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT day, meal_type, student_count, item_name, actual_quantity_used FROM food_history")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Error as e:
        print(f"[ERROR] fetching history: {e}")
        return pd.DataFrame()

def fetch_today_menu() -> dict:
    try:
        today_day = datetime.now().strftime('%A')
        conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT meal_type, items FROM menu WHERE day = %s", (today_day,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        menu_map = {}
        for r in rows:
            menu_map.setdefault(r['meal_type'], []).extend([i.strip() for i in r['items'].split(',') if i.strip()])
        return menu_map
    except Error as e:
        print(f"[ERROR] fetching today's menu: {e}")
        return {}

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.get_dummies(df, columns=['day','meal_type','item_name'], drop_first=True)
    y = df['actual_quantity_used']
    X = df.drop(columns=['actual_quantity_used'])
    return X, y

def train_model():
    df = fetch_history_data()
    if df.empty:
        print("[WARNING] No history data available for training.")
        return
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"[RESULT] Training Completed. MSE: {mse:.4f}")
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump((model, X.columns.tolist()), f)

def load_model():
    with open(MODEL_FILENAME, 'rb') as f:
        return pickle.load(f)

def suggest_today(student_count: int) -> pd.DataFrame:
    model, feature_cols = load_model()
    today_day = datetime.now().strftime('%A')
    menu_map = fetch_today_menu()
    suggestions, total_ingredients = [], {}
    for meal, items in menu_map.items():
        for item in items:
            row = {col:0 for col in feature_cols}
            row['student_count'] = student_count
            if f"day_{today_day}" in row: row[f"day_{today_day}"] = 1
            if f"meal_type_{meal}" in row: row[f"meal_type_{meal}"] = 1
            if f"item_name_{item}" in row: row[f"item_name_{item}"] = 1
            qty = max(0, round(model.predict(pd.DataFrame([row], columns=feature_cols))[0]))
            unit = ITEM_UNITS.get(item.lower(), "kg")
            suggestions.append({'meal_type':meal,'item_name':item,'suggested_quantity':qty,'unit':unit})
            for ing, per in ITEM_INGREDIENTS.get(item.lower(),{}).items():
                total_ingredients[ing] = total_ingredients.get(ing,0) + per*qty
    df_sug = pd.DataFrame(suggestions)
    df_ing = pd.DataFrame([{'ingredient':k,'required_quantity_kg':round(v,2)} for k,v in total_ingredients.items()])
    print("\n[Suggestions]:\n", df_sug.to_string(index=False))
    print("\n[Ingredient Requirements]:\n", df_ing.to_string(index=False))
    return df_sug

if __name__ == "__main__":
    if not os.path.isfile(MODEL_FILENAME):
        train_model()
    count = get_today_student_count()
    if count is None:
        print("Attendance not submitted for today.")
        exit(1)
    suggest_today(count)


