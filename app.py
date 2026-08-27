"""
DataSense AI - Main Flask Application
Module 1: Login System
Module 2: Dataset Upload
Module 3: Data Quality Analysis
Module 4: Clean Data Repository
"""

import os
import io
import bcrypt
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session, flash, send_file, abort
from werkzeug.utils import secure_filename

from db import get_connection
from services.data_quality import run_full_analysis, get_issue_report, compute_quality_score
from services.data_cleaner import clean_dataset
from services.dashboard_builder import build_dashboard
from services.excel_loader import load_best_sheet
from services.insights import generate_full_insights
from services.query_engine import get_quick_queries, run_quick_query, run_natural_language_query
from services.dashboard_builder import build_dashboard
from services.rfm_analysis import detect_customer_column, detect_amount_column, detect_date_column, compute_rfm
from services.pii_detector import build_pii_report
from services.feature_engineering import run_feature_engineering
from services.anomaly_detection import detect_anomalies
from services.forecasting import forecast_sales
from services.kmeans_segmentation import run_kmeans_segmentation
from services.report_generator import run_full_pipeline, build_pdf_report
from services.media_analyzer import analyze_image, analyze_video, extract_video_thumbnail
from services.db_connector import test_connection, list_tables, run_query, DBConnectionError
import config

# ------------------------------------------------------------------
# App Setup
# ------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = config.SECRET_KEY  # required for session management

# Upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE_MB = 10

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLEAN_FOLDER = 'cleaned_data'
os.makedirs(CLEAN_FOLDER, exist_ok=True)

# Media (image/video) upload settings
MEDIA_FOLDER = 'media_uploads'
IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
ALLOWED_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
MAX_MEDIA_SIZE_MB = 200

app.config['MEDIA_FOLDER'] = MEDIA_FOLDER
os.makedirs(MEDIA_FOLDER, exist_ok=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Simple in-memory cache so large files aren't re-read from disk on
# every click. Keyed by file path; cleared automatically whenever a
# file is re-cleaned (see clean_data route) so stale data is never served.
_FILE_CACHE = {}

def read_dataframe_cached(filepath):
    ...
    if filepath.lower().endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = load_best_sheet(filepath)          # <-- was pd.read_excel(filepath)

    _FILE_CACHE[filepath] = df
    return df.copy()
def read_dataframe_cached(filepath):
    """Reads a CSV/Excel file once and reuses it from memory on later calls.
    Always returns a fresh .copy() so callers can safely modify it
    without corrupting the cached original."""
    if filepath in _FILE_CACHE:
        return _FILE_CACHE[filepath].copy()

    if filepath.lower().endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    _FILE_CACHE[filepath] = df
    return df.copy()


def allowed_file(filename):
    """Check file extension is csv/xlsx/xls"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_media_file(filename):
    """Check file extension is an accepted image or video type"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MEDIA_EXTENSIONS


def media_type_for(filename):
    """Returns 'image', 'video', or None based on the file extension"""
    if '.' not in filename:
        return None
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    if ext in VIDEO_EXTENSIONS:
        return 'video'
    return None


def login_required(func):
    """Simple decorator to protect routes that require login"""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue')
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return wrapper


# ------------------------------------------------------------------
# ROUTE: Root -> redirect to login or home
# ------------------------------------------------------------------
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))


# ------------------------------------------------------------------
# MODULE 1: Login System
# ------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, skip straight to home
    if 'user_id' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Username and password are required')
            return render_template('login.html')

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT user_id, username, password, role FROM users WHERE username = %s",
            (username,)
        )
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
            return render_template('login.html')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out')
    return redirect(url_for('login'))


# ------------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------------
@app.route('/home')
@login_required
def home():
    return render_template('home.html', username=session.get('username'))


# ------------------------------------------------------------------
# MODULE 2: Dataset Upload
# ------------------------------------------------------------------
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files.get('dataset')

        if not file or file.filename == '':
            flash('No file selected')
            return redirect(url_for('upload'))

        if not allowed_file(file.filename):
            flash('Only .csv and .xlsx files are allowed')
            return redirect(url_for('upload'))

        # File size check
        file.seek(0, os.SEEK_END)
        size_mb = file.tell() / (1024 * 1024)
        file.seek(0)
        if size_mb > MAX_FILE_SIZE_MB:
            flash(f'File too large. Max size is {MAX_FILE_SIZE_MB}MB')
            return redirect(url_for('upload'))

        # Save file into a per-user folder
        filename = secure_filename(file.filename)
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(session['user_id']))
        os.makedirs(user_folder, exist_ok=True)
        filepath = os.path.join(user_folder, filename)
        file.save(filepath)

        # Validate it's readable tabular data
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
        except Exception as e:
            os.remove(filepath)
            flash(f'Could not read file: {str(e)}')
            return redirect(url_for('upload'))

        # Save metadata to MySQL
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO uploaded_files
               (user_id, file_name, file_path, record_count, column_count)
               VALUES (%s, %s, %s, %s, %s)""",
            (session['user_id'], filename, filepath, len(df), len(df.columns))
        )
        conn.commit()
        file_id = cursor.lastrowid
        cursor.close()
        conn.close()

        preview_html = df.head(10).to_html(classes='preview-table', index=False)

        return render_template(
            'upload.html',
            success=True,
            filename=filename,
            record_count=len(df),
            column_count=len(df.columns),
            preview_html=preview_html,
            file_id=file_id
        )

    return render_template('upload.html', success=False)


@app.route('/connect-database', methods=['GET', 'POST'])
@login_required
def connect_database():
    """
    Lets the user pull data from a live external database instead of
    uploading a file. The query result is saved as a CSV in the user's
    upload folder and registered in uploaded_files exactly like a manual
    upload, so every downstream module (quality, cleaning, ML, reports)
    works on it without any changes.
    """
    if request.method == 'POST':
        host = request.form.get('host', '').strip()
        port = request.form.get('port', '3306').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        database = request.form.get('database', '').strip()
        sql_query = request.form.get('sql_query', '').strip()
        action = request.form.get('action')

        if not all([host, port, username, database]):
            flash('Host, port, username, and database name are all required.')
            return render_template('connect_database.html', form=request.form)

        # "Test & List Tables" -- just verifies credentials and shows what's available
        if action == 'list_tables':
            try:
                tables = list_tables(host, port, username, password, database)
                flash(f'Connected successfully. Found {len(tables)} table(s).', 'success')
                return render_template('connect_database.html', form=request.form, tables=tables)
            except DBConnectionError as e:
                flash(str(e))
                return render_template('connect_database.html', form=request.form)

        # "Import" -- runs the query and pulls the result into the pipeline
        if not sql_query:
            flash('Enter a SQL query (e.g. SELECT * FROM customers LIMIT 1000) to import data.')
            return render_template('connect_database.html', form=request.form)

        try:
            df = run_query(host, port, username, password, database, sql_query)
        except DBConnectionError as e:
            flash(str(e))
            return render_template('connect_database.html', form=request.form)

        if df.empty:
            flash('Query ran successfully but returned no rows.')
            return render_template('connect_database.html', form=request.form)

        # From here on, treat it exactly like a file upload
        filename = secure_filename(f"{database}_import.csv")
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(session['user_id']))
        os.makedirs(user_folder, exist_ok=True)
        filepath = os.path.join(user_folder, filename)
        df.to_csv(filepath, index=False)

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO uploaded_files
               (user_id, file_name, file_path, record_count, column_count)
               VALUES (%s, %s, %s, %s, %s)""",
            (session['user_id'], filename, filepath, len(df), len(df.columns))
        )
        conn.commit()
        file_id = cursor.lastrowid
        cursor.close()
        conn.close()

        preview_html = df.head(10).to_html(classes='preview-table', index=False)

        return render_template(
            'upload.html',
            success=True,
            filename=filename,
            record_count=len(df),
            column_count=len(df.columns),
            preview_html=preview_html,
            file_id=file_id
        )

    return render_template('connect_database.html', form={})


@app.route('/upload/history')
@login_required
def upload_history():
    """Shows all previously uploaded files for the logged-in user"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, upload_date, record_count, column_count
           FROM uploaded_files
           WHERE user_id = %s
           ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('upload_history.html', files=files)


@app.route('/delete/<int:file_id>', methods=['POST'])
@login_required
def delete_file(file_id):
    """Deletes a file completely: DB rows (uploaded_files, clean_data, ai_insights),
    the physical files on disk, and any cached in-memory copies."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Confirm ownership and get the original file path
    cursor.execute(
        "SELECT file_path, file_name FROM uploaded_files WHERE file_id = %s AND user_id = %s",
        (file_id, session['user_id'])
    )
    file_record = cursor.fetchone()

    if not file_record:
        cursor.close()
        conn.close()
        flash('File not found')
        return redirect(url_for('upload_history'))

    # Get any cleaned file paths too, so we can remove those from disk as well
    cursor.execute(
        "SELECT clean_file_path FROM clean_data WHERE file_id = %s AND user_id = %s",
        (file_id, session['user_id'])
    )
    clean_records = cursor.fetchall()
    cursor.close()

    # Delete child rows FIRST (foreign key constraints require this order)
    del_cursor = conn.cursor()
    del_cursor.execute("DELETE FROM ai_insights WHERE file_id = %s AND user_id = %s", (file_id, session['user_id']))
    del_cursor.execute("DELETE FROM clean_data WHERE file_id = %s AND user_id = %s", (file_id, session['user_id']))
    del_cursor.execute("DELETE FROM uploaded_files WHERE file_id = %s AND user_id = %s", (file_id, session['user_id']))
    conn.commit()
    del_cursor.close()
    conn.close()

    # Remove the original file from disk + cache
    original_path = file_record['file_path']
    if original_path and os.path.exists(original_path):
        os.remove(original_path)
    _FILE_CACHE.pop(original_path, None)

    # Remove any cleaned versions from disk + cache
    for record in clean_records:
        clean_path = record['clean_file_path']
        if clean_path and os.path.exists(clean_path):
            os.remove(clean_path)
        _FILE_CACHE.pop(clean_path, None)

    flash(f"'{file_record['file_name']}' and all related data have been deleted.")
    return redirect(url_for('upload_history'))


# ------------------------------------------------------------------
# MODULE 3: Data Quality Analysis
# ------------------------------------------------------------------
@app.route('/analyze/<int:file_id>')
@login_required
def analyze(file_id):
    """Runs data quality checks (duplicates, nulls, unknowns, outliers) on a chosen file"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, file_path
           FROM uploaded_files
           WHERE file_id = %s AND user_id = %s""",
        (file_id, session['user_id'])
    )
    file_record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not file_record:
        flash('File not found')
        return redirect(url_for('upload_history'))

    filepath = file_record['file_path']

    if not filepath or not os.path.exists(filepath):
        flash('Original file is missing from the server. Please re-upload it.')
        return redirect(url_for('upload_history'))

    try:
        df = read_dataframe_cached(filepath)
    except Exception as e:
        flash(f'Could not read file: {str(e)}')
        return redirect(url_for('upload_history'))

    results = run_full_analysis(df)
    quality_score = compute_quality_score(results)

    return render_template(
        'analysis.html',
        file_name=file_record['file_name'],
        file_id=file_id,
        results=results,
        quality_score=quality_score
    )


# ------------------------------------------------------------------
# MODULE 4: Clean Data Repository
# ------------------------------------------------------------------
@app.route('/clean')
@login_required
def clean_data_list():
    """Shows a simple file picker with only the Clean Data action, per file"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('clean_data_list.html', files=files)


@app.route('/clean/<int:file_id>')
@login_required
def clean_data(file_id):
    """Cleans the dataset (or loads the already-cleaned version) and displays it"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Check if this file was already cleaned before
    cursor.execute(
        """SELECT * FROM clean_data WHERE file_id = %s AND user_id = %s
           ORDER BY cleaned_date DESC LIMIT 1""",
        (file_id, session['user_id'])
    )
    existing_clean = cursor.fetchone()

    if existing_clean and os.path.exists(existing_clean['clean_file_path']):
        # Already cleaned - just load the existing clean file
        clean_df = read_dataframe_cached(existing_clean['clean_file_path'])
        summary = {
            'original_record_count': existing_clean['original_record_count'],
            'clean_record_count': existing_clean['clean_record_count'],
            'duplicates_removed': existing_clean['duplicates_removed'],
            'nulls_fixed': existing_clean['nulls_fixed'],
            'unknowns_fixed': existing_clean['unknowns_fixed'],
            'outliers_capped': existing_clean['outliers_capped'],
        }
        clean_id = existing_clean['clean_id']

    else:
        # Not cleaned yet - fetch original file and run cleaning pipeline
        cursor.execute(
            """SELECT file_id, file_name, file_path FROM uploaded_files
               WHERE file_id = %s AND user_id = %s""",
            (file_id, session['user_id'])
        )
        file_record = cursor.fetchone()

        if not file_record:
            cursor.close()
            conn.close()
            flash('File not found')
            return redirect(url_for('upload_history'))

        filepath = file_record['file_path']

        if not filepath or not os.path.exists(filepath):
            cursor.close()
            conn.close()
            flash('Original file is missing from the server. Please re-upload it.')
            return redirect(url_for('upload_history'))

        try:
            raw_df = read_dataframe_cached(filepath)
        except Exception as e:
            cursor.close()
            conn.close()
            flash(f'Could not read file: {str(e)}')
            return redirect(url_for('upload_history'))

        result = clean_dataset(raw_df)
        clean_df = result['dataframe']
        summary = result['summary']

        # Save cleaned file to disk
        user_clean_folder = os.path.join(CLEAN_FOLDER, str(session['user_id']))
        os.makedirs(user_clean_folder, exist_ok=True)
        clean_filename = f"clean_{file_record['file_name'].rsplit('.', 1)[0]}.csv"
        clean_filepath = os.path.join(user_clean_folder, clean_filename)
        clean_df.to_csv(clean_filepath, index=False)
        _FILE_CACHE[clean_filepath] = clean_df  # cache immediately so the next click is instant

        # Save metadata to MySQL
        insert_cursor = conn.cursor()
        insert_cursor.execute(
            """INSERT INTO clean_data
               (file_id, user_id, clean_file_path, original_record_count, clean_record_count,
                duplicates_removed, nulls_fixed, unknowns_fixed, outliers_capped)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (file_id, session['user_id'], clean_filepath,
             summary['original_record_count'], summary['clean_record_count'],
             summary['duplicates_removed'], summary['nulls_fixed'],
             summary['unknowns_fixed'], summary['outliers_capped'])
        )
        conn.commit()
        clean_id = insert_cursor.lastrowid
        insert_cursor.close()

    cursor.close()
    conn.close()

    # Pagination (50 records per page, per your module spec)
    page = request.args.get('page', 1, type=int)
    per_page = 50
    total_records = len(clean_df)
    total_pages = max(1, (total_records + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    start = (page - 1) * per_page
    end = start + per_page
    page_df = clean_df.iloc[start:end]

    table_html = page_df.to_html(classes='clean-table', index=False)

    return render_template(
        'clean_data.html',
        file_id=file_id,
        clean_id=clean_id,
        summary=summary,
        table_html=table_html,
        page=page,
        total_pages=total_pages,
        total_records=total_records,
        columns=list(clean_df.columns)
    )


@app.route('/clean/<int:file_id>/download')
@login_required
def download_clean_data(file_id):
    """Lets the user download the cleaned CSV file"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT clean_file_path FROM clean_data
           WHERE file_id = %s AND user_id = %s
           ORDER BY cleaned_date DESC LIMIT 1""",
        (file_id, session['user_id'])
    )
    record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not record or not os.path.exists(record['clean_file_path']):
        flash('Cleaned file not found')
        return redirect(url_for('upload_history'))

    return send_file(record['clean_file_path'], as_attachment=True)


# ------------------------------------------------------------------
# MODULE 5: Data Issues Repository
# ------------------------------------------------------------------
def _load_original_dataframe(file_id):
    """Helper: fetches the uploaded_files record and reads it into a DataFrame.
    Returns (df, file_record) or (None, None) if something's wrong (also flashes a message)."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, file_path FROM uploaded_files
           WHERE file_id = %s AND user_id = %s""",
        (file_id, session['user_id'])
    )
    file_record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not file_record:
        flash('File not found')
        return None, None

    filepath = file_record['file_path']
    if not filepath or not os.path.exists(filepath):
        flash('Original file is missing from the server. Please re-upload it.')
        return None, None

    try:
        df = read_dataframe_cached(filepath)
    except Exception as e:
        flash(f'Could not read file: {str(e)}')
        return None, None

    return df, file_record


@app.route('/issues')
@login_required
def issues_list():
    """Shows a simple file picker with only the Issues action, per file"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('issues_list.html', files=files)


@app.route('/issues/<int:file_id>')
@login_required
def issues(file_id):
    """Shows the actual problem rows in tabs: Duplicates / Nulls / Unknowns / Outliers"""
    df, file_record = _load_original_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    report = get_issue_report(df)

    # Cap each tab's preview to 200 rows so huge datasets don't crash the browser
    tables = {}
    counts = {}
    for issue_type, issue_df in report.items():
        counts[issue_type] = len(issue_df)
        preview_df = issue_df.head(200)
        tables[issue_type] = preview_df.to_html(classes='issue-table', index=False) if not preview_df.empty else None

    return render_template(
        'issues.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        tables=tables,
        counts=counts
    )


@app.route('/issues/<int:file_id>/download')
@login_required
def download_issues(file_id):
    """Exports all 4 issue categories as separate sheets in one Excel file"""
    df, file_record = _load_original_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    report = get_issue_report(df)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for issue_type, issue_df in report.items():
            sheet_name = issue_type.capitalize()[:31]  # Excel sheet names max 31 chars
            if issue_df.empty:
                pd.DataFrame({'Message': ['No issues found']}).to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                issue_df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)

    download_name = f"issues_report_{file_record['file_name'].rsplit('.', 1)[0]}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=download_name,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


# ------------------------------------------------------------------
# MODULE 8: AI Insight Engine
# ------------------------------------------------------------------
def _load_best_available_dataframe(file_id):
    """Prefers the CLEANED version of the file if it exists, otherwise
    falls back to the original uploaded file. Returns (df, file_record) or (None, None)."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        """SELECT clean_file_path FROM clean_data
           WHERE file_id = %s AND user_id = %s
           ORDER BY cleaned_date DESC LIMIT 1""",
        (file_id, session['user_id'])
    )
    clean_record = cursor.fetchone()

    cursor.execute(
        """SELECT file_id, file_name, file_path FROM uploaded_files
           WHERE file_id = %s AND user_id = %s""",
        (file_id, session['user_id'])
    )
    file_record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not file_record:
        flash('File not found')
        return None, None

    # Prefer cleaned data if available
    if clean_record and os.path.exists(clean_record['clean_file_path']):
        try:
            df = read_dataframe_cached(clean_record['clean_file_path'])
            return df, file_record
        except Exception:
            pass  # fall through to original file

    filepath = file_record['file_path']
    if not filepath or not os.path.exists(filepath):
        flash('File is missing from the server. Please re-upload it.')
        return None, None

    try:
        df = read_dataframe_cached(filepath)
    except Exception as e:
        flash(f'Could not read file: {str(e)}')
        return None, None

    return df, file_record


@app.route('/insights')
@login_required
def insights_list():
    """Shows a simple file picker with only the Insights action, per file"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('insights_list.html', files=files)


@app.route('/insights/<int:file_id>')
@login_required
def insights(file_id):
    """Generates rule-based + AI-powered insights for a file (uses cleaned data if available)"""
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    results = generate_full_insights(df)

    # Save the AI recommendation text to MySQL for history
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO ai_insights (file_id, user_id, recommendation)
           VALUES (%s, %s, %s)""",
        (file_id, session['user_id'], results['ai_text'])
    )
    conn.commit()
    cursor.close()
    conn.close()

    return render_template(
        'insights.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        rule_based=results['rule_based'],
        ai_text=results['ai_text']
    )


# ------------------------------------------------------------------
# MODULE 9: Analytics Query Assistant
# ------------------------------------------------------------------
@app.route('/query')
@login_required
def query_list():
    """Shows a simple file picker with only the Query Assistant action, per file"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('query_list.html', files=files)


@app.route('/query/<int:file_id>', methods=['GET', 'POST'])
@login_required
def query(file_id):
    """Lets the user ask questions about their data via quick-query buttons or free text"""
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    quick_queries = get_quick_queries(df)
    result = None
    question_text = ''

    if request.method == 'POST':
        query_type = request.form.get('query_type')

        if query_type == 'quick':
            query_index = int(request.form.get('query_index', -1))
            if 0 <= query_index < len(quick_queries):
                result = run_quick_query(df, quick_queries[query_index])
                question_text = quick_queries[query_index]['label']

        elif query_type == 'natural':
            question_text = request.form.get('question', '').strip()
            if question_text:
                result = run_natural_language_query(df, question_text)
    else:
        query_type = None

    return render_template(
        'query.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        quick_queries=quick_queries,
        result=result,
        question_text=question_text,
        query_type=query_type
    )


# ------------------------------------------------------------------
# MODULE 7: Power BI Dashboard (real, published report)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# MODULE 7: Dashboard (auto-generated from your data, per file)
# ------------------------------------------------------------------
@app.route('/dashboards')
@login_required
def dashboards_list():
    """Shows a simple file picker with only the Dashboard action, per file"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('dashboards_list.html', files=files)


@app.route('/dashboard/<int:file_id>')
@login_required
def dashboard(file_id):
    """Auto-generates KPI cards and charts from the cleaned (or original) data"""
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    dashboard_data = build_dashboard(df)

    return render_template(
        'dashboard.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        kpis=dashboard_data['kpis'],
        charts=dashboard_data['charts']
    )


@app.route('/dashboard/<int:file_id>/download')
@login_required
def download_dashboard(file_id):
    """Exports the dashboard's KPIs and chart data as a single structured CSV"""
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    dashboard_data = build_dashboard(df)

    lines = []
    lines.append(f"Dashboard Summary - {file_record['file_name']}")
    lines.append("")
    lines.append("KPIs")
    lines.append("Label,Value")
    for kpi in dashboard_data['kpis']:
        lines.append(f'"{kpi["label"]}","{kpi["value"]}"')
    lines.append("")

    for chart in dashboard_data['charts']:
        lines.append(f'"{chart["title"]}"')
        if chart.get('type') == 'table':
            lines.append(",".join(f'"{col}"' for col in chart['columns']))
            for row in chart['rows']:
                lines.append(",".join(f'"{val}"' for val in row))
        else:
            lines.append("Label,Value")
            for label, value in zip(chart['labels'], chart['values']):
                lines.append(f'"{label}",{value}')
        lines.append("")

    csv_content = "\n".join(lines)
    output = io.BytesIO(csv_content.encode('utf-8'))

    download_name = f"dashboard_summary_{file_record['file_name'].rsplit('.', 1)[0]}.csv"
    return send_file(output, as_attachment=True, download_name=download_name, mimetype='text/csv')


# ------------------------------------------------------------------
# MODULE 10 (Bonus): Customer Segmentation - RFM Analysis
# ------------------------------------------------------------------
@app.route('/segments')
@login_required
def segments_list():
    """Shows a simple file picker with only the Segments action, per file"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('segments_list.html', files=files)


@app.route('/rfm/<int:file_id>', methods=['GET', 'POST'])
@login_required
def rfm(file_id):
    """Segments customers into Champions / Loyal / At Risk / Lost using RFM scoring"""
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    all_columns = list(df.columns)
    numeric_columns = list(df.select_dtypes(include='number').columns)

    auto_customer_col = detect_customer_column(df)
    auto_amount_col = detect_amount_column(df)
    auto_date_col = detect_date_column(df)

    if not auto_customer_col or not auto_amount_col:
        flash('This dataset needs at least a customer/name column and a numeric amount column for RFM analysis.')
        return redirect(url_for('upload_history'))

    customer_col = request.form.get('customer_col', auto_customer_col)
    amount_col = request.form.get('amount_col', auto_amount_col)
    date_col = request.form.get('date_col', auto_date_col) or None
    if date_col == 'none':
        date_col = None

    result = compute_rfm(df, customer_col, amount_col, date_col)

    if 'error' in result:
        flash(result['error'])
        return redirect(url_for('upload_history'))

    return render_template(
        'rfm.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        result=result,
        all_columns=all_columns,
        numeric_columns=numeric_columns,
        customer_col=customer_col,
        amount_col=amount_col,
        date_col=date_col
    )


@app.route('/rfm/<int:file_id>/download')
@login_required
def download_rfm(file_id):
    """Exports the full RFM table as CSV"""
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    customer_col = detect_customer_column(df)
    amount_col = detect_amount_column(df)
    date_col = detect_date_column(df)

    result = compute_rfm(df, customer_col, amount_col, date_col)
    if 'error' in result:
        flash(result['error'])
        return redirect(url_for('upload_history'))

    output = io.BytesIO()
    result['full_table'].to_csv(output, index=False)
    output.seek(0)

    download_name = f"rfm_segments_{file_record['file_name'].rsplit('.', 1)[0]}.csv"
    return send_file(output, as_attachment=True, download_name=download_name, mimetype='text/csv')


# ------------------------------------------------------------------
# MODULE 11: Image & Video Analysis
# ------------------------------------------------------------------
@app.route('/media')
@login_required
def media_list():
    """Shows all previously uploaded image/video files for the logged-in user"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT media_id, file_name, media_type, thumbnail_path, upload_date
           FROM uploaded_media
           WHERE user_id = %s
           ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('media_list.html', files=files)


@app.route('/media/upload', methods=['GET', 'POST'])
@login_required
def media_upload():
    """Uploads an image or video and analyzes it immediately, same spirit as
    Module 2's dataset upload but the dashboard is shown right away instead
    of on a separate visit, since there's no cleaning step in between."""
    if request.method == 'POST':
        file = request.files.get('media')

        if not file or file.filename == '':
            flash('No file selected')
            return redirect(url_for('media_upload'))

        if not allowed_media_file(file.filename):
            flash('Only image (jpg, png, gif, bmp, webp) and video (mp4, mov, avi, mkv, webm) files are allowed')
            return redirect(url_for('media_upload'))

        # File size check
        file.seek(0, os.SEEK_END)
        size_mb = file.tell() / (1024 * 1024)
        file.seek(0)
        if size_mb > MAX_MEDIA_SIZE_MB:
            flash(f'File too large. Max size is {MAX_MEDIA_SIZE_MB}MB')
            return redirect(url_for('media_upload'))

        # Save file into a per-user folder
        filename = secure_filename(file.filename)
        user_folder = os.path.join(app.config['MEDIA_FOLDER'], str(session['user_id']))
        os.makedirs(user_folder, exist_ok=True)
        filepath = os.path.join(user_folder, filename)
        file.save(filepath)

        media_type = media_type_for(filename)

        # Run analysis now - this also validates the file is actually a
        # readable image/video, the same way Module 2 validates tabular data
        try:
            if media_type == 'image':
                kpis, charts = analyze_image(filepath)
            else:
                kpis, charts = analyze_video(filepath)
        except Exception as e:
            os.remove(filepath)
            flash(f'Could not analyze file: {str(e)}')
            return redirect(url_for('media_upload'))

        # For videos, also grab a poster frame so the library can show a
        # preview instantly without the browser downloading any video data.
        # Non-fatal if this fails - the video just falls back to no poster.
        thumbnail_path = None
        if media_type == 'video':
            thumb_filename = f"{os.path.splitext(filename)[0]}_poster.jpg"
            thumb_filepath = os.path.join(user_folder, thumb_filename)
            try:
                if extract_video_thumbnail(filepath, thumb_filepath):
                    thumbnail_path = thumb_filepath
            except Exception:
                thumbnail_path = None

        # Save metadata to MySQL
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO uploaded_media
               (user_id, file_name, file_path, thumbnail_path, media_type, file_size_kb)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (session['user_id'], filename, filepath, thumbnail_path, media_type, round(os.path.getsize(filepath) / 1024, 1))
        )
        conn.commit()
        media_id = cursor.lastrowid
        cursor.close()
        conn.close()

        return render_template(
            'dashboard.html',
            file_id=media_id,
            file_name=filename,
            kpis=kpis,
            charts=charts
        )

    return render_template('media_upload.html')


@app.route('/media/<int:media_id>')
@login_required
def media_dashboard(media_id):
    """Re-analyzes a previously uploaded image/video and renders it on the
    SAME dashboard.html template the dataset dashboards already use - its
    kpis/charts schema already matches what analyze_image/analyze_video return."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT media_id, file_name, file_path, media_type
           FROM uploaded_media
           WHERE media_id = %s AND user_id = %s""",
        (media_id, session['user_id'])
    )
    media_record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not media_record:
        flash('File not found')
        return redirect(url_for('media_list'))

    filepath = media_record['file_path']
    if not filepath or not os.path.exists(filepath):
        flash('Original file is missing from the server. Please re-upload it.')
        return redirect(url_for('media_list'))

    try:
        if media_record['media_type'] == 'image':
            kpis, charts = analyze_image(filepath)
        else:
            kpis, charts = analyze_video(filepath)
    except Exception as e:
        flash(f'Could not analyze file: {str(e)}')
        return redirect(url_for('media_list'))

    return render_template(
        'dashboard.html',
        file_id=media_id,
        file_name=media_record['file_name'],
        kpis=kpis,
        charts=charts
    )


@app.route('/media/<int:media_id>/download')
@login_required
def download_media_summary(media_id):
    """Exports the media analysis KPIs and chart data as a single structured
    CSV, same format as the dataset dashboard's download."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_name, file_path, media_type
           FROM uploaded_media
           WHERE media_id = %s AND user_id = %s""",
        (media_id, session['user_id'])
    )
    media_record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not media_record or not media_record['file_path'] or not os.path.exists(media_record['file_path']):
        flash('File not found')
        return redirect(url_for('media_list'))

    if media_record['media_type'] == 'image':
        kpis, charts = analyze_image(media_record['file_path'])
    else:
        kpis, charts = analyze_video(media_record['file_path'])

    lines = []
    lines.append(f"Media Analysis Summary - {media_record['file_name']}")
    lines.append("")
    lines.append("KPIs")
    lines.append("Label,Value")
    for kpi in kpis:
        lines.append(f'"{kpi["label"]}","{kpi["value"]}"')
    lines.append("")

    for chart in charts:
        lines.append(f'"{chart["title"]}"')
        if chart['type'] == 'table':
            lines.append(",".join(chart['columns']))
            for row in chart['rows']:
                lines.append(",".join(f'"{v}"' for v in row))
        elif chart['type'] == 'stacked_bar':
            lines.append("Label," + ",".join(s['name'] for s in chart['series']))
            for i, label in enumerate(chart['labels']):
                lines.append(",".join([str(label)] + [str(s['values'][i]) for s in chart['series']]))
        elif 'labels' in chart and 'values' in chart:
            lines.append("Label,Value")
            for label, value in zip(chart['labels'], chart['values']):
                lines.append(f'"{label}",{value}')
        lines.append("")

    csv_content = "\n".join(lines)
    output = io.BytesIO(csv_content.encode('utf-8'))

    download_name = f"media_summary_{media_record['file_name'].rsplit('.', 1)[0]}.csv"
    return send_file(output, as_attachment=True, download_name=download_name, mimetype='text/csv')


@app.route('/media/<int:media_id>/file')
@login_required
def media_file(media_id):
    """Streams the raw image/video bytes so <img>/<video> tags can load them.
    Ownership-checked the same way every other file route in this app is -
    only the uploader can fetch their own file by ID."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT file_path FROM uploaded_media WHERE media_id = %s AND user_id = %s",
        (media_id, session['user_id'])
    )
    media_record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not media_record or not media_record['file_path'] or not os.path.exists(media_record['file_path']):
        flash('File not found')
        return redirect(url_for('media_list'))

    return send_file(media_record['file_path'])


@app.route('/media/<int:media_id>/thumbnail')
@login_required
def media_thumbnail(media_id):
    """Streams the poster frame for a video preview. 404s cleanly if no
    thumbnail was generated (e.g. videos uploaded before this feature
    existed) - the <video> tag just falls back to no poster in that case."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT thumbnail_path FROM uploaded_media WHERE media_id = %s AND user_id = %s",
        (media_id, session['user_id'])
    )
    record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not record or not record['thumbnail_path'] or not os.path.exists(record['thumbnail_path']):
        abort(404)

    return send_file(record['thumbnail_path'])


@app.route('/media/<int:media_id>/delete', methods=['POST'])
@login_required
def delete_media(media_id):
    """Deletes an uploaded media file's DB row and the file(s) on disk"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT file_path, thumbnail_path, file_name FROM uploaded_media WHERE media_id = %s AND user_id = %s",
        (media_id, session['user_id'])
    )
    media_record = cursor.fetchone()

    if not media_record:
        cursor.close()
        conn.close()
        flash('File not found')
        return redirect(url_for('media_list'))

    del_cursor = conn.cursor()
    del_cursor.execute("DELETE FROM uploaded_media WHERE media_id = %s AND user_id = %s", (media_id, session['user_id']))
    conn.commit()
    del_cursor.close()
    cursor.close()
    conn.close()

    if media_record['file_path'] and os.path.exists(media_record['file_path']):
        os.remove(media_record['file_path'])

    if media_record['thumbnail_path'] and os.path.exists(media_record['thumbnail_path']):
        os.remove(media_record['thumbnail_path'])

    flash(f"'{media_record['file_name']}' has been deleted.")
    return redirect(url_for('media_list'))


# ------------------------------------------------------------------
# BONUS: PII Detection & Masking
# ------------------------------------------------------------------
@app.route('/pii-list')
@login_required
def pii_list():
    """Shows a simple file picker with only the PII Scan action, per file"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('pii_list.html', files=files)


@app.route('/pii/<int:file_id>')
@login_required
def pii_scan(file_id):
    """Scans the dataset for sensitive data (emails, phone, PAN, Aadhaar, card numbers)"""
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    report = build_pii_report(df)

    return render_template(
        'pii.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        findings=report['findings'],
        total_pii_columns=report['total_pii_columns'],
        total_pii_cells=report['total_pii_cells'],
        preview_html=report['preview_html']
    )


@app.route('/pii/<int:file_id>/download')
@login_required
def download_pii_masked(file_id):
    """Downloads the full dataset with all detected PII masked"""
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    report = build_pii_report(df)
    output = io.BytesIO()
    report['masked_df'].to_csv(output, index=False)
    output.seek(0)

    download_name = f"masked_{file_record['file_name'].rsplit('.', 1)[0]}.csv"
    return send_file(output, as_attachment=True, download_name=download_name, mimetype='text/csv')


# ------------------------------------------------------------------
# MODULE 5: Feature Engineering
# ------------------------------------------------------------------
@app.route('/features')
@login_required
def features_list():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('features_list.html', files=files)


@app.route('/features/<int:file_id>')
@login_required
def features(file_id):
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    result = run_feature_engineering(df)

    return render_template(
        'features.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        features_added=result['features_added'],
        original_column_count=result['original_column_count'],
        new_column_count=result['new_column_count'],
        preview_html=result['preview_html']
    )


@app.route('/features/<int:file_id>/download')
@login_required
def download_features(file_id):
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    result = run_feature_engineering(df)
    output = io.BytesIO()
    result['dataframe'].to_csv(output, index=False)
    output.seek(0)

    download_name = f"engineered_{file_record['file_name'].rsplit('.', 1)[0]}.csv"
    return send_file(output, as_attachment=True, download_name=download_name, mimetype='text/csv')


# ------------------------------------------------------------------
# MODULE 8: Anomaly Detection (Isolation Forest)
# ------------------------------------------------------------------
@app.route('/anomalies')
@login_required
def anomalies_list():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('anomalies_list.html', files=files)


@app.route('/anomalies/<int:file_id>')
@login_required
def anomalies(file_id):
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    result = detect_anomalies(df)

    if 'error' in result:
        flash(result['error'])
        return redirect(url_for('anomalies_list'))

    return render_template(
        'anomalies.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        result=result
    )


@app.route('/anomalies/<int:file_id>/download')
@login_required
def download_anomalies(file_id):
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    result = detect_anomalies(df)
    if 'error' in result:
        flash(result['error'])
        return redirect(url_for('anomalies_list'))

    output = io.BytesIO()
    result['full_dataframe'].to_csv(output, index=False)
    output.seek(0)

    download_name = f"anomalies_{file_record['file_name'].rsplit('.', 1)[0]}.csv"
    return send_file(output, as_attachment=True, download_name=download_name, mimetype='text/csv')


# ------------------------------------------------------------------
# MODULE 7: Sales Forecasting
# ------------------------------------------------------------------
@app.route('/forecast')
@login_required
def forecast_list():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('forecast_list.html', files=files)


@app.route('/forecast/<int:file_id>')
@login_required
def forecast(file_id):
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    date_col = detect_date_column(df)
    amount_col = detect_amount_column(df)

    if not date_col or not amount_col:
        flash('Forecasting needs both a date column and a numeric amount column - this dataset is missing one.')
        return redirect(url_for('forecast_list'))

    result = forecast_sales(df, date_col, amount_col)

    if 'error' in result:
        flash(result['error'])
        return redirect(url_for('forecast_list'))

    return render_template(
        'forecast.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        result=result
    )


# ------------------------------------------------------------------
# MODULE 6: Customer Segmentation - K-Means (ML-based)
# ------------------------------------------------------------------
@app.route('/kmeans')
@login_required
def kmeans_list():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('kmeans_list.html', files=files)


@app.route('/kmeans/<int:file_id>')
@login_required
def kmeans(file_id):
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    customer_col = detect_customer_column(df)
    amount_col = detect_amount_column(df)
    date_col = detect_date_column(df)

    if not customer_col or not amount_col:
        flash('K-Means segmentation needs a customer/name column and a numeric amount column.')
        return redirect(url_for('kmeans_list'))

    result = run_kmeans_segmentation(df, customer_col, amount_col, date_col)

    if 'error' in result:
        flash(result['error'])
        return redirect(url_for('kmeans_list'))

    return render_template(
        'kmeans.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        result=result
    )


@app.route('/kmeans/<int:file_id>/download')
@login_required
def download_kmeans(file_id):
    df, file_record = _load_best_available_dataframe(file_id)
    if df is None:
        return redirect(url_for('upload_history'))

    customer_col = detect_customer_column(df)
    amount_col = detect_amount_column(df)
    date_col = detect_date_column(df)

    result = run_kmeans_segmentation(df, customer_col, amount_col, date_col)
    if 'error' in result:
        flash(result['error'])
        return redirect(url_for('kmeans_list'))

    output = io.BytesIO()
    result['full_table'].to_csv(output, index=False)
    output.seek(0)

    download_name = f"kmeans_segments_{file_record['file_name'].rsplit('.', 1)[0]}.csv"
    return send_file(output, as_attachment=True, download_name=download_name, mimetype='text/csv')


# ------------------------------------------------------------------
# BONUS: One-Click Full Pipeline (Analyze -> Clean -> Score -> Insights -> PDF Report)
# ------------------------------------------------------------------
@app.route('/pipeline')
@login_required
def pipeline_list():
    """Shows a simple file picker with only the Run Pipeline action, per file"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT file_id, file_name, record_count, column_count
           FROM uploaded_files WHERE user_id = %s ORDER BY upload_date DESC""",
        (session['user_id'],)
    )
    files = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('pipeline_list.html', files=files)


@app.route('/pipeline/<int:file_id>')
@login_required
def pipeline(file_id):
    """Runs the full pipeline (Analyze -> Clean -> Score -> AI Insights) on the
    ORIGINAL uploaded file, then shows an animated step-by-step summary."""
    df, file_record = _load_original_dataframe(file_id)
    if df is None:
        return redirect(url_for('pipeline_list'))

    result = run_full_pipeline(df)

    return render_template(
        'pipeline.html',
        file_id=file_id,
        file_name=file_record['file_name'],
        quality_score=result['quality_score'],
        analysis=result['analysis'],
        clean_summary=result['clean_summary'],
        ai_text=result['ai_text']
    )


@app.route('/pipeline/<int:file_id>/download-report')
@login_required
def download_pipeline_report(file_id):
    """Regenerates the pipeline results and returns them as a downloadable PDF report"""
    df, file_record = _load_original_dataframe(file_id)
    if df is None:
        return redirect(url_for('pipeline_list'))

    result = run_full_pipeline(df)
    pdf_bytes = build_pdf_report(file_record['file_name'], result)

    output = io.BytesIO(pdf_bytes)
    download_name = f"report_{file_record['file_name'].rsplit('.', 1)[0]}.pdf"
    return send_file(output, as_attachment=True, download_name=download_name, mimetype='application/pdf')


# ------------------------------------------------------------------
# Run App
# ------------------------------------------------------------------
if __name__ == '__main__':
    # DEBUG defaults to True for local development. Before deploying
    # anywhere real, set DEBUG=False in .env - Werkzeug's debugger allows
    # arbitrary code execution if it's ever reachable from outside.
    debug_mode = os.environ.get('DEBUG', 'True') == 'True'
    app.run(debug=debug_mode, threaded=True)