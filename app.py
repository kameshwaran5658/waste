from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import bcrypt
from flask_wtf.csrf import CSRFProtect
from flask_socketio import SocketIO
from flask_apscheduler import APScheduler
import os
from dotenv import load_dotenv
from datetime import datetime

# Load .env variables
load_dotenv()

# Flask App Setup
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
csrf = CSRFProtect(app)
socketio = SocketIO(app)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Database Configuration
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306))
}

# DB Connection
def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        print(f"DB connection error: {err}")
        return None

# Initialize Database
def initialize_database():
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            # Create tables if not exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS admins (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255),
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS menu (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    day VARCHAR(20) NOT NULL,
                    meal_type VARCHAR(20) NOT NULL,
                    items TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE NOT NULL,
                    student_count INT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS food_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    day VARCHAR(20),
                    meal_type VARCHAR(20),
                    student_count INT,
                    item_name VARCHAR(100),
                    actual_quantity_used FLOAT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE,
                    saved_waste FLOAT,
                    accuracy FLOAT,
                    menu_id INT,
                    attendance_id INT
                )
            ''')
            # Insert default admin if not exist
            cursor.execute('SELECT * FROM admins LIMIT 1')
            if not cursor.fetchone():
                hashed_pw = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
                cursor.execute('''
                    INSERT INTO admins (username, password_hash, full_name)
                    VALUES (%s, %s, %s)
                ''', ('admin', hashed_pw.decode('utf-8'), 'System Admin'))
            conn.commit()
            cursor.close()
            conn.close()
    except Exception as e:
        print(f"DB Initialization error: {e}")

initialize_database()

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Admin Login
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if 'admin' in session:
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password').encode('utf-8')

        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT * FROM admins WHERE username = %s', (username,))
            admin = cursor.fetchone()
            cursor.close()
            conn.close()

            if admin and bcrypt.checkpw(password, admin['password_hash'].encode('utf-8')):
                session['admin'] = {
                    'id': admin['id'],
                    'username': admin['username'],
                    'full_name': admin['full_name']
                }
                flash('Login successful!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid username or password.', 'danger')

    return render_template('admin_login.html')

# Admin Dashboard (Attendance + Menu Management)
@app.route('/admin/dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'danger')
        return redirect(url_for('admin_login'))

    cursor = conn.cursor(dictionary=True)

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'attendance':
            try:
                date = request.form['date']
                student_count = int(request.form['student_count'])
                cursor.execute('''
                    INSERT INTO attendance (date, student_count)
                    VALUES (%s, %s)
                ''', (date, student_count))
                conn.commit()
                flash('Attendance recorded successfully!', 'success')
                return redirect(url_for('admin_dashboard') + '#attendance')  # 🛑 Redirect back to Attendance tab
            except Exception as e:
                flash(f'Error saving attendance: {str(e)}', 'danger')
                return redirect(url_for('admin_dashboard') + '#attendance')

        elif form_type == 'menu':
            try:
                day = request.form['day']
                meal_type = request.form['meal_type']
                items = request.form['items']
                cursor.execute('''
                    INSERT INTO menu (day, meal_type, items)
                    VALUES (%s, %s, %s)
                ''', (day, meal_type, items))
                conn.commit()
                flash('Menu saved successfully!', 'success')
                return redirect(url_for('admin_dashboard') + '#menus')  # 🛑 Redirect back to Menus tab
            except Exception as e:
                flash(f'Error saving menu: {str(e)}', 'danger')
                return redirect(url_for('admin_dashboard') + '#menus')

    # If just GET method, no form submitted yet:
    cursor.execute('SELECT * FROM attendance ORDER BY date DESC LIMIT 10')
    attendance_records = cursor.fetchall()

    cursor.execute('SELECT * FROM menu ORDER BY id DESC')
    menu_records = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('admin_dashboard.html',
                            admin=session['admin'],
                            attendance=attendance_records,
                            menus=menu_records,
                            datetime=datetime)

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

# Admin Logout
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('admin_login'))

# Start App
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=os.getenv('DEBUG', 'False') == 'True')
