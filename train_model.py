# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306))
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

def train_and_save_model():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM food_history')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        print('No food history available to train.')
        return

    df = pd.DataFrame(rows)

    # Prepare features and labels
    le_day = LabelEncoder()
    le_meal_type = LabelEncoder()
    le_item = LabelEncoder()

    df['day_encoded'] = le_day.fit_transform(df['day'])
    df['meal_type_encoded'] = le_meal_type.fit_transform(df['meal_type'])
    df['item_encoded'] = le_item.fit_transform(df['item_name'])

    X = df[['day_encoded', 'meal_type_encoded', 'student_count', 'item_encoded']]
    y = df['actual_quantity_used']

    model = LinearRegression()
    model.fit(X, y)

    # Save model and encoders
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump((le_day, le_meal_type, le_item), f)

    print('Model trained and saved successfully.')

if __name__ == '__main__':
    train_and_save_model()
