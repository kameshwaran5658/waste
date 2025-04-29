#!/usr/bin/env python3
"""
predict_app.py

A simple Flask application that loads the trained DecisionTreeRegressor model
and provides a web page to auto-predict the actual food quantity used based on user inputs.
"""
import os
import pickle
from dotenv import load_dotenv
from flask import Flask, request, render_template_string
import pandas as pd

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_FILENAME", "food_quantity_model.pkl")

# Load the trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Retrieve feature names from the model (sklearn >=1.0)
feature_cols = getattr(model, 'feature_names_in_', None)

app = Flask(__name__)

# HTML template for the predict page
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Auto Predict Food Quantity</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 flex items-center justify-center h-screen">
  <div class="bg-white p-8 rounded-2xl shadow-lg w-full max-w-md">
    <h1 class="text-2xl font-bold mb-6 text-center">Predict Food Quantity Used</h1>
    <form method="POST" class="space-y-4">
      <div>
        <label class="block text-sm font-medium">Day</label>
        <select name="day" class="mt-1 block w-full border rounded p-2">
          <option>Monday</option><option>Tuesday</option><option>Wednesday</option>
          <option>Thursday</option><option>Friday</option><option>Saturday</option><option>Sunday</option>
        </select>
      </div>
      <div>
        <label class="block text-sm font-medium">Meal Type</label>
        <select name="meal_type" class="mt-1 block w-full border rounded p-2">
          <option>Breakfast</option><option>Lunch</option><option>Dinner</option>
        </select>
      </div>
      <div>
        <label class="block text-sm font-medium">Student Count</label>
        <input type="number" name="student_count" required class="mt-1 block w-full border rounded p-2"/>
      </div>
      <div>
        <label class="block text-sm font-medium">Item Name</label>
        <input type="text" name="item_name" required class="mt-1 block w-full border rounded p-2"/>
      </div>
      <button type="submit" class="w-full py-2 px-4 bg-blue-600 text-white rounded-lg">Predict</button>
    </form>
    {% if result is not none %}
    <div class="mt-6 p-4 bg-green-100 text-green-800 rounded"> 
      <p class="text-lg">Predicted Quantity Used: <span class="font-bold">{{ result }}</span></p>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        # Collect form inputs
        day = request.form['day']
        meal_type = request.form['meal_type']
        student_count = int(request.form['student_count'])
        item_name = request.form['item_name']

        # Build DataFrame for prediction
        df = pd.DataFrame([
            {'student_count': student_count, 'day': day, 'meal_type': meal_type, 'item_name': item_name}
        ])

        # One-hot encode and align with training features
        df = pd.get_dummies(df, columns=['day', 'meal_type', 'item_name'], drop_first=True)
        if feature_cols is not None:
            df = df.reindex(columns=feature_cols, fill_value=0)

        # Predict
        pred = model.predict(df)[0]
        result = round(pred, 2)

    return render_template_string(TEMPLATE, result=result)

if __name__ == '__main__':
    # Run on 0.0.0.0:5000 by default
    app.run(host='0.0.0.0', port=5000, debug=True)
