import os
import json
import sqlite3
from functools import wraps

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from flask import (Flask, flash, g, jsonify, redirect,
                   render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'catbook-secret-2024')
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}


# ---------- Feature extraction (local, no API) ----------

_extractor = None
_preprocess = None


MODEL_VERSION = 'dinov2-base-v1'
DINO_MODEL_ID = 'facebook/dinov2-base'


def get_extractor():
    global _extractor, _preprocess
    if _extractor is None:
        _preprocess = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
        _extractor = AutoModel.from_pretrained(DINO_MODEL_ID)
        _extractor.eval()
    return _extractor, _preprocess


def _l2_normalize(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


def extract_features(img_input):
    """Extract L2-normalized DINOv2 CLS features with 5-crop TTA."""
    model, processor = get_extractor()
    img = Image.open(img_input).convert('RGB')
    w, h = img.size
    imgs = [
        img,                                                   # original
        img.transpose(Image.FLIP_LEFT_RIGHT),                  # h-flip
        img.crop((w // 10, h // 10, w * 9 // 10, h * 9 // 10)),  # center 80%
        img.crop((0, 0, w * 4 // 5, h)),                       # left 80%
        img.crop((w // 5, 0, w, h)),                           # right 80%
    ]
    inputs = processor(images=imgs, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    feats = outputs.last_hidden_state[:, 0]  # CLS token, shape (5, 768)
    avg = feats.mean(dim=0)
    return _l2_normalize(avg.tolist())


def cosine_sim(a, b):
    # both L2-normalized → dot product == cosine similarity
    return float(np.dot(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)))


def push_notification(db, user_id, ntype, from_user_id=None, cat_id=None, photo=None, location=None, location_precise=None):
    db.execute(
        'INSERT INTO notifications (user_id, type, from_user_id, cat_id, photo, location, location_precise) VALUES (?,?,?,?,?,?,?)',
        (user_id, ntype, from_user_id, cat_id, photo, location, location_precise)
    )
    db.commit()


def find_similar_cat(db, feats, uid, threshold=0.55):
    """Check if any of the given feature vectors match a cat from another user.
    Returns a dict with cat info + photos, or None."""
    other_users = db.execute(
        'SELECT id, username FROM users WHERE id != ?', (uid,)
    ).fetchall()
    best_score = 0
    result = None
    for user in other_users:
        for ocat in db.execute('SELECT id, name FROM cats WHERE user_id = ?', (user['id'],)).fetchall():
            photos = db.execute(
                'SELECT id, filename, features FROM cat_photos WHERE cat_id = ?', (ocat['id'],)
            ).fetchall()
            for pfeat_row in photos:
                pfeat = get_or_compute_features(db, pfeat_row['id'], pfeat_row['filename'])
                if not pfeat:
                    continue
                for feat in feats:
                    s = cosine_sim(feat, pfeat)
                    if s >= threshold and s > best_score:
                        best_score = s
                        photo_filenames = [p['filename'] for p in photos]
                        already_friends = db.execute('''
                            SELECT id FROM friendships
                            WHERE ((requester_id=? AND receiver_id=?) OR (requester_id=? AND receiver_id=?))
                        ''', (uid, user['id'], user['id'], uid)).fetchone() is not None
                        result = {
                            'cat_name': ocat['name'],
                            'cat_id': ocat['id'],
                            'owner': user['username'],
                            'owner_id': user['id'],
                            'photos': photo_filenames[:4],
                            'already_friends': already_friends,
                        }
    return result


# ---------- Database ----------

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('catbook.db')
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()


@app.context_processor
def inject_globals():
    if 'user_id' in session:
        db = get_db()
        uid = session['user_id']
        pending = db.execute(
            "SELECT COUNT(*) as cnt FROM friendships WHERE receiver_id=? AND status='pending'",
            (uid,)
        ).fetchone()['cnt']
        unread = db.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE receiver_id=? AND is_read=0",
            (uid,)
        ).fetchone()['cnt']
        unread_notifs = db.execute(
            "SELECT COUNT(*) as cnt FROM notifications WHERE user_id=? AND is_read=0",
            (uid,)
        ).fetchone()['cnt']
        return {'pending_requests': pending, 'unread_messages': unread, 'unread_notifs': unread_notifs}
    return {'pending_requests': 0, 'unread_messages': 0, 'unread_notifs': 0}


def init_db():
    with app.app_context():
        db = get_db()
        db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS cat_photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cat_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                features TEXT,
                FOREIGN KEY (cat_id) REFERENCES cats(id)
            );
        ''')
        # add features column if missing
        try:
            db.execute('ALTER TABLE cat_photos ADD COLUMN features TEXT')
        except Exception:
            pass
        # add model_version table to detect when features need recomputing
        db.execute('''CREATE TABLE IF NOT EXISTS settings
                      (key TEXT PRIMARY KEY, value TEXT)''')
        stored = db.execute(
            "SELECT value FROM settings WHERE key='model_version'"
        ).fetchone()
        if not stored or stored['value'] != MODEL_VERSION:
            db.execute('UPDATE cat_photos SET features = NULL')
            db.execute("INSERT OR REPLACE INTO settings(key,value) VALUES('model_version',?)",
                       (MODEL_VERSION,))
        db.execute('''CREATE TABLE IF NOT EXISTS friendships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            requester_id INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (requester_id) REFERENCES users(id),
            FOREIGN KEY (receiver_id) REFERENCES users(id),
            UNIQUE(requester_id, receiver_id)
        )''')
        try:
            db.execute('ALTER TABLE friendships ADD COLUMN context_photo TEXT')
        except Exception:
            pass
        try:
            db.execute('ALTER TABLE friendships ADD COLUMN context_cat_id INTEGER')
        except Exception:
            pass
        try:
            db.execute('ALTER TABLE friendships ADD COLUMN context_type TEXT')
        except Exception:
            pass
        db.execute('''CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            from_user_id INTEGER,
            cat_id INTEGER,
            photo TEXT,
            is_read INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        try:
            db.execute('ALTER TABLE notifications ADD COLUMN location TEXT')
        except Exception:
            pass
        try:
            db.execute('ALTER TABLE notifications ADD COLUMN location_precise TEXT')
        except Exception:
            pass
        db.execute('''CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            sent_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_read INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (sender_id) REFERENCES users(id),
            FOREIGN KEY (receiver_id) REFERENCES users(id)
        )''')
        try:
            db.execute('ALTER TABLE messages ADD COLUMN image TEXT')
        except Exception:
            pass
        db.execute('''CREATE TABLE IF NOT EXISTS cat_details (
            cat_id INTEGER PRIMARY KEY,
            gender TEXT,
            birth_date TEXT,
            age TEXT,
            neutered INTEGER DEFAULT 0,
            neuter_date TEXT,
            last_treated TEXT,
            favorite_food TEXT,
            last_fed TEXT,
            presence TEXT,
            tags TEXT,
            FOREIGN KEY (cat_id) REFERENCES cats(id)
        )''')
        try:
            db.execute('ALTER TABLE friendships ADD COLUMN context_own_cat_id INTEGER')
        except Exception:
            pass
        try:
            db.execute('ALTER TABLE friendships ADD COLUMN request_message TEXT')
        except Exception:
            pass
        db.execute('''CREATE TABLE IF NOT EXISTS shared_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cat_id_1 INTEGER NOT NULL,
            cat_id_2 INTEGER NOT NULL,
            gender TEXT,
            birth_date TEXT,
            age TEXT,
            neutered INTEGER DEFAULT 0,
            neuter_date TEXT,
            last_treated TEXT,
            favorite_food TEXT,
            last_fed TEXT,
            presence TEXT,
            tags TEXT,
            FOREIGN KEY (cat_id_1) REFERENCES cats(id),
            FOREIGN KEY (cat_id_2) REFERENCES cats(id),
            UNIQUE(cat_id_1, cat_id_2)
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS details_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cat_id INTEGER,
            shared_details_id INTEGER,
            saved_by INTEGER NOT NULL,
            saved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            data TEXT NOT NULL,
            FOREIGN KEY (saved_by) REFERENCES users(id)
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            cat_id INTEGER,
            photo TEXT,
            caption TEXT,
            visibility TEXT NOT NULL DEFAULT 'friends',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (cat_id) REFERENCES cats(id)
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS post_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES posts(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        try:
            db.execute("ALTER TABLE posts ADD COLUMN purpose TEXT NOT NULL DEFAULT 'share'")
        except Exception:
            pass
        db.execute('''CREATE TABLE IF NOT EXISTS post_saves (
            user_id INTEGER NOT NULL,
            post_id INTEGER NOT NULL,
            saved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, post_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )''')
        db.commit()


# ---------- Auth helpers ----------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ---------- Utility ----------

def save_photo(file_storage, cat_id):
    if not file_storage or not file_storage.filename:
        return None
    ext = os.path.splitext(file_storage.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        print(f"save_photo: extension not allowed: {ext!r}")
        return None
    try:
        filename = f'cat_{cat_id}_{os.urandom(8).hex()}{ext}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = Image.open(file_storage)
        img.thumbnail((900, 900), Image.LANCZOS)
        img.save(filepath)
        return filename
    except Exception as e:
        print(f"save_photo error: {e}")
        return None


def get_or_compute_features(db, photo_id, filename):
    """Return feature vector for a photo, computing and caching if needed."""
    row = db.execute('SELECT features FROM cat_photos WHERE id = ?', (photo_id,)).fetchone()
    if row and row['features']:
        return json.loads(row['features'])
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(path):
        return None
    try:
        feat = extract_features(path)
        db.execute('UPDATE cat_photos SET features = ? WHERE id = ?',
                   (json.dumps(feat), photo_id))
        db.commit()
        return feat
    except Exception as e:
        print(f"feature extraction error: {e}")
        return None


# ---------- Routes ----------

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('cats'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('cats'))
        flash('שם משתמש או סיסמה שגויים')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            flash('נא למלא את כל השדות')
        else:
            db = get_db()
            try:
                db.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                           (username, generate_password_hash(password)))
                db.commit()
                flash('נרשמת בהצלחה! כעת תוכל להתחבר')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('שם המשתמש כבר תפוס')
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/cats')
@login_required
def cats():
    db = get_db()
    uid = session['user_id']
    cat_rows = db.execute(
        'SELECT id, name FROM cats WHERE user_id = ? ORDER BY created_at DESC', (uid,)
    ).fetchall()

    cats_data = []
    for cat in cat_rows:
        photo_rows = db.execute(
            'SELECT id, filename FROM cat_photos WHERE cat_id = ?', (cat['id'],)
        ).fetchall()
        photos = [{'id': p['id'], 'filename': p['filename']} for p in photo_rows]
        cats_data.append({'id': cat['id'], 'name': cat['name'], 'photos': photos})

    # Friends' cats
    friend_rows = db.execute('''
        SELECT u.id, u.username FROM friendships f
        JOIN users u ON (CASE WHEN f.requester_id=? THEN f.receiver_id ELSE f.requester_id END)=u.id
        WHERE (f.requester_id=? OR f.receiver_id=?) AND f.status='accepted'
    ''', (uid, uid, uid)).fetchall()

    friends_cats = []
    for friend in friend_rows:
        fcat_rows = db.execute(
            'SELECT id, name FROM cats WHERE user_id = ? ORDER BY created_at DESC', (friend['id'],)
        ).fetchall()
        fcats = []
        for cat in fcat_rows:
            photo_rows = db.execute(
                'SELECT id, filename FROM cat_photos WHERE cat_id = ?', (cat['id'],)
            ).fetchall()
            photos = [{'id': p['id'], 'filename': p['filename']} for p in photo_rows]
            fcats.append({'id': cat['id'], 'name': cat['name'], 'photos': photos})
        if fcats:
            friends_cats.append({'username': friend['username'], 'cats': fcats})

    similar_notice = session.pop('similar_notice', None)
    return render_template('cats.html', cats=cats_data, friends_cats=friends_cats,
                           similar_notice=similar_notice)


@app.route('/cats/add', methods=['GET', 'POST'])
@login_required
def add_cat():
    if request.method == 'POST':
        name = request.form['name'].strip()
        if not name:
            flash('נא להזין שם לחתול')
            return render_template('add_cat.html')

        db = get_db()
        cursor = db.execute('INSERT INTO cats (user_id, name) VALUES (?, ?)',
                            (session['user_id'], name))
        cat_id = cursor.lastrowid
        db.commit()

        all_feats = []
        for file in request.files.getlist('photos'):
            filename = save_photo(file, cat_id)
            if filename:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    feat = extract_features(filepath)
                    feat_json = json.dumps(feat)
                    all_feats.append(feat)
                except Exception:
                    feat = None
                    feat_json = None
                db.execute('INSERT INTO cat_photos (cat_id, filename, features) VALUES (?, ?, ?)',
                           (cat_id, filename, feat_json))
        db.commit()

        # Check if any uploaded photo resembles another user's cat
        if all_feats:
            similar = find_similar_cat(db, all_feats, session['user_id'])
            if similar:
                similar['my_cat_id'] = cat_id
                session['similar_notice'] = similar
                push_notification(db, similar['owner_id'], 'similar',
                                  from_user_id=session['user_id'], cat_id=similar['cat_id'])
                if similar['already_friends']:
                    c1 = min(cat_id, similar['cat_id'])
                    c2 = max(cat_id, similar['cat_id'])
                    db.execute('INSERT OR IGNORE INTO shared_details (cat_id_1, cat_id_2) VALUES (?,?)', (c1, c2))
                    db.commit()

        flash(f'החתול "{name}" נוסף בהצלחה!')
        return redirect(url_for('cats'))
    return render_template('add_cat.html')


@app.route('/cats/<int:cat_id>/add_photo', methods=['POST'])
@login_required
def add_photo(cat_id):
    db = get_db()
    cat = db.execute('SELECT id FROM cats WHERE id = ? AND user_id = ?',
                     (cat_id, session['user_id'])).fetchone()
    if not cat:
        return jsonify({'error': 'לא נמצא'}), 404

    file = request.files.get('photo')
    filename = save_photo(file, cat_id)
    if filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            feat = extract_features(filepath)
            feat_json = json.dumps(feat)
        except Exception as e:
            app.logger.error(f'extract_features failed: {e}')
            feat = None
            feat_json = None
        db.execute('INSERT INTO cat_photos (cat_id, filename, features) VALUES (?, ?, ?)',
                   (cat_id, filename, feat_json))
        db.commit()

        # Check if the uploaded photo resembles any other user's cat
        if feat:
            similar = find_similar_cat(db, [feat], session['user_id'])
            if similar:
                similar['my_cat_id'] = cat_id
                session['similar_notice'] = similar
                push_notification(db, similar['owner_id'], 'similar',
                                  from_user_id=session['user_id'], cat_id=similar['cat_id'])
                if similar['already_friends']:
                    c1 = min(cat_id, similar['cat_id'])
                    c2 = max(cat_id, similar['cat_id'])
                    db.execute('INSERT OR IGNORE INTO shared_details (cat_id_1, cat_id_2) VALUES (?,?)', (c1, c2))
                    db.commit()

        return jsonify({'success': True, 'filename': filename})
    return jsonify({'error': 'קובץ לא תקין'}), 400


@app.route('/cats/<int:cat_id>/details', methods=['GET', 'POST'])
@login_required
def cat_details_page(cat_id):
    db = get_db()
    uid = session['user_id']
    cat = db.execute('SELECT id, name, user_id FROM cats WHERE id=?', (cat_id,)).fetchone()
    if not cat:
        flash('חתול לא נמצא')
        return redirect(url_for('cats'))

    is_owner = cat['user_id'] == uid
    if not is_owner:
        friendship = db.execute('''
            SELECT id FROM friendships
            WHERE ((requester_id=? AND receiver_id=?) OR (requester_id=? AND receiver_id=?))
              AND status='accepted'
        ''', (uid, cat['user_id'], cat['user_id'], uid)).fetchone()
        if not friendship:
            flash('אין גישה')
            return redirect(url_for('cats'))

    # Check for shared details between this cat and another
    shared_row = db.execute(
        'SELECT * FROM shared_details WHERE cat_id_1=? OR cat_id_2=?', (cat_id, cat_id)
    ).fetchone()

    if shared_row:
        other_cat_id = shared_row['cat_id_2'] if shared_row['cat_id_1'] == cat_id else shared_row['cat_id_1']
        other_cat = db.execute(
            'SELECT c.id, c.name, c.user_id, u.username FROM cats c JOIN users u ON u.id=c.user_id WHERE c.id=?',
            (other_cat_id,)
        ).fetchone()
        # Can edit if owner of either cat
        can_edit = is_owner or (other_cat and other_cat['user_id'] == uid)
        shared_with = {'cat_name': other_cat['name'], 'username': other_cat['username']} if other_cat else None

        if request.method == 'POST' and not can_edit:
            flash('אין הרשאה לערוך')
            return redirect(url_for('cat_details_page', cat_id=cat_id))

        if request.method == 'POST':
            gender = request.form.get('gender', '').strip() or None
            birth_date = request.form.get('birth_date', '').strip() or None
            age = request.form.get('age', '').strip() or None
            neutered = 1 if request.form.get('neutered') else 0
            neuter_date = request.form.get('neuter_date', '').strip() or None
            last_treated = request.form.get('last_treated', '').strip() or None
            favorite_food = request.form.get('favorite_food', '').strip() or None
            last_fed = request.form.get('last_fed', '').strip() or None
            presence = request.form.get('presence', '').strip() or None
            tags = request.form.get('tags', '').strip() or None
            db.execute('''
                UPDATE shared_details SET
                    gender=?, birth_date=?, age=?, neutered=?, neuter_date=?,
                    last_treated=?, favorite_food=?, last_fed=?, presence=?, tags=?
                WHERE id=?
            ''', (gender, birth_date, age, neutered, neuter_date,
                  last_treated, favorite_food, last_fed, presence, tags, shared_row['id']))
            db.commit()
            data_json = json.dumps({'gender': gender, 'birth_date': birth_date, 'age': age,
                'neutered': neutered, 'neuter_date': neuter_date, 'last_treated': last_treated,
                'favorite_food': favorite_food, 'last_fed': last_fed, 'presence': presence, 'tags': tags})
            db.execute('INSERT INTO details_history (shared_details_id, saved_by, data) VALUES (?,?,?)',
                       (shared_row['id'], uid, data_json))
            db.commit()
            flash('הפרטים נשמרו בהצלחה ✓')
            return redirect(url_for('cat_details_page', cat_id=cat_id))

        d = dict(shared_row)
    else:
        # Personal details
        shared_with = None
        can_edit = is_owner

        if request.method == 'POST' and not can_edit:
            flash('אין הרשאה לערוך')
            return redirect(url_for('cat_details_page', cat_id=cat_id))

        if request.method == 'POST':
            gender = request.form.get('gender', '').strip() or None
            birth_date = request.form.get('birth_date', '').strip() or None
            age = request.form.get('age', '').strip() or None
            neutered = 1 if request.form.get('neutered') else 0
            neuter_date = request.form.get('neuter_date', '').strip() or None
            last_treated = request.form.get('last_treated', '').strip() or None
            favorite_food = request.form.get('favorite_food', '').strip() or None
            last_fed = request.form.get('last_fed', '').strip() or None
            presence = request.form.get('presence', '').strip() or None
            tags = request.form.get('tags', '').strip() or None
            db.execute('''
                INSERT INTO cat_details
                    (cat_id, gender, birth_date, age, neutered, neuter_date, last_treated, favorite_food, last_fed, presence, tags)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(cat_id) DO UPDATE SET
                    gender=excluded.gender, birth_date=excluded.birth_date, age=excluded.age,
                    neutered=excluded.neutered, neuter_date=excluded.neuter_date,
                    last_treated=excluded.last_treated, favorite_food=excluded.favorite_food,
                    last_fed=excluded.last_fed, presence=excluded.presence, tags=excluded.tags
            ''', (cat_id, gender, birth_date, age, neutered, neuter_date,
                  last_treated, favorite_food, last_fed, presence, tags))
            db.commit()
            data_json = json.dumps({'gender': gender, 'birth_date': birth_date, 'age': age,
                'neutered': neutered, 'neuter_date': neuter_date, 'last_treated': last_treated,
                'favorite_food': favorite_food, 'last_fed': last_fed, 'presence': presence, 'tags': tags})
            db.execute('INSERT INTO details_history (cat_id, saved_by, data) VALUES (?,?,?)',
                       (cat_id, uid, data_json))
            db.commit()
            flash('הפרטים נשמרו בהצלחה ✓')
            return redirect(url_for('cat_details_page', cat_id=cat_id))

        row = db.execute('SELECT * FROM cat_details WHERE cat_id=?', (cat_id,)).fetchone()
        d = dict(row) if row else {}

    tag_pairs = []
    raw_tags = d.get('tags') or ''
    if raw_tags:
        try:
            tag_pairs = json.loads(raw_tags)
        except (ValueError, TypeError):
            tag_pairs = [{'k': t.strip(), 'v': ''} for t in raw_tags.split(',') if t.strip()]
    # Load history
    if shared_row:
        history_rows = db.execute('''
            SELECT h.id, h.saved_at, u.username FROM details_history h
            JOIN users u ON u.id=h.saved_by
            WHERE h.shared_details_id=? ORDER BY h.saved_at DESC LIMIT 20
        ''', (shared_row['id'],)).fetchall()
    else:
        history_rows = db.execute('''
            SELECT h.id, h.saved_at, u.username FROM details_history h
            JOIN users u ON u.id=h.saved_by
            WHERE h.cat_id=? ORDER BY h.saved_at DESC LIMIT 20
        ''', (cat_id,)).fetchall()
    history = [{'id': h['id'],
                'saved_at': h['saved_at'][:16].replace('T', ' '),
                'username': h['username']} for h in history_rows]
    return render_template('cat_details.html', cat=cat, d=d, tag_pairs=tag_pairs,
                           is_owner=can_edit, shared_with=shared_with, history=history)


@app.route('/cats/<int:cat_id>/details/history/<int:history_id>')
@login_required
def details_history_load(cat_id, history_id):
    db = get_db()
    uid = session['user_id']
    cat = db.execute('SELECT id, name, user_id FROM cats WHERE id=?', (cat_id,)).fetchone()
    if not cat:
        return jsonify({'error': 'not found'}), 404
    is_owner = cat['user_id'] == uid
    if not is_owner:
        friendship = db.execute('''
            SELECT id FROM friendships
            WHERE ((requester_id=? AND receiver_id=?) OR (requester_id=? AND receiver_id=?))
              AND status='accepted'
        ''', (uid, cat['user_id'], cat['user_id'], uid)).fetchone()
        if not friendship:
            return jsonify({'error': 'no access'}), 403
    shared_row = db.execute(
        'SELECT * FROM shared_details WHERE cat_id_1=? OR cat_id_2=?', (cat_id, cat_id)
    ).fetchone()
    if shared_row:
        h = db.execute('SELECT * FROM details_history WHERE id=? AND shared_details_id=?',
                       (history_id, shared_row['id'])).fetchone()
    else:
        h = db.execute('SELECT * FROM details_history WHERE id=? AND cat_id=?',
                       (history_id, cat_id)).fetchone()
    if not h:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'success': True, 'data': json.loads(h['data'])})


@app.route('/photos/<int:photo_id>/delete', methods=['POST'])
@login_required
def delete_photo(photo_id):
    db = get_db()
    # verify photo belongs to the logged-in user via the cat
    photo = db.execute('''
        SELECT p.id, p.filename, p.cat_id FROM cat_photos p
        JOIN cats c ON c.id = p.cat_id
        WHERE p.id = ? AND c.user_id = ?
    ''', (photo_id, session['user_id'])).fetchone()
    if photo:
        path = os.path.join(app.config['UPLOAD_FOLDER'], photo['filename'])
        if os.path.exists(path):
            os.remove(path)
        db.execute('DELETE FROM cat_photos WHERE id = ?', (photo_id,))
        db.commit()
    return redirect(url_for('cats'))


@app.route('/cats/<int:cat_id>/delete', methods=['POST'])
@login_required
def delete_cat(cat_id):
    db = get_db()
    cat = db.execute('SELECT id FROM cats WHERE id = ? AND user_id = ?',
                     (cat_id, session['user_id'])).fetchone()
    if cat:
        photos = db.execute('SELECT filename FROM cat_photos WHERE cat_id = ?',
                            (cat_id,)).fetchall()
        for p in photos:
            path = os.path.join(app.config['UPLOAD_FOLDER'], p['filename'])
            if os.path.exists(path):
                os.remove(path)
        db.execute('DELETE FROM cat_photos WHERE cat_id = ?', (cat_id,))
        db.execute('DELETE FROM cat_details WHERE cat_id = ?', (cat_id,))
        # Clean up shared details and their history
        shared = db.execute(
            'SELECT id FROM shared_details WHERE cat_id_1=? OR cat_id_2=?', (cat_id, cat_id)
        ).fetchone()
        if shared:
            db.execute('DELETE FROM details_history WHERE shared_details_id=?', (shared['id'],))
            db.execute('DELETE FROM shared_details WHERE id=?', (shared['id'],))
        db.execute('DELETE FROM details_history WHERE cat_id=?', (cat_id,))
        db.execute('DELETE FROM cats WHERE id = ?', (cat_id,))
        db.commit()
        flash('החתול נמחק')
    return redirect(url_for('cats'))


@app.route('/identify', methods=['GET', 'POST'])
@login_required
def identify():
    result = None
    if request.method == 'POST':
        file = request.files.get('photo')
        location = request.form.get('location', '').strip() or None
        location_precise = request.form.get('location_precise', '').strip() or None
        if not file or not file.filename:
            flash('נא לבחור תמונה')
            return render_template('identify.html', result=None)

        # Save temp copy so user can add it to the identified cat later
        ext = os.path.splitext(file.filename)[1].lower()
        temp_filename = f'temp_{session["user_id"]}_{os.urandom(6).hex()}{ext}'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        img = Image.open(file).convert('RGB')
        img.thumbnail((900, 900), Image.LANCZOS)
        img.save(temp_path)

        try:
            query_feat = extract_features(temp_path)
        except Exception as e:
            os.remove(temp_path)
            flash(f'שגיאה בעיבוד התמונה: {e}')
            return render_template('identify.html', result=None)

        db = get_db()
        uid = session['user_id']

        # All cats from all users
        cat_rows = db.execute(
            'SELECT c.id, c.name, c.user_id, u.username FROM cats c JOIN users u ON u.id=c.user_id'
        ).fetchall()

        if not cat_rows:
            os.remove(temp_path)
            flash('אין חתולים רשומים באתר')
            return render_template('identify.html', result=None)

        MATCH_THRESHOLD = 0.55
        few_photos_warning = False
        cat_scores = []

        for cat in cat_rows:
            photos = db.execute(
                'SELECT id, filename, features FROM cat_photos WHERE cat_id = ?',
                (cat['id'],)
            ).fetchall()
            if not photos:
                continue
            if len(photos) < 3:
                few_photos_warning = True

            scores = []
            for photo in photos:
                feat = get_or_compute_features(db, photo['id'], photo['filename'])
                if feat is not None:
                    scores.append(cosine_sim(query_feat, feat))

            if scores:
                score = 0.5 * (sum(scores) / len(scores)) + 0.5 * max(scores)
                is_mine = cat['user_id'] == uid
                already_friends = False
                if not is_mine:
                    already_friends = db.execute('''
                        SELECT id FROM friendships
                        WHERE ((requester_id=? AND receiver_id=?) OR (requester_id=? AND receiver_id=?))
                    ''', (uid, cat['user_id'], cat['user_id'], uid)).fetchone() is not None
                cat_scores.append({
                    'name': cat['name'],
                    'cat_id': cat['id'],
                    'score': score,
                    'is_mine': is_mine,
                    'owner': None if is_mine else cat['username'],
                    'owner_id': None if is_mine else cat['user_id'],
                    'already_friends': already_friends,
                })

        cat_scores.sort(key=lambda x: x['score'], reverse=True)
        margin = (cat_scores[0]['score'] - cat_scores[1]['score']) if len(cat_scores) >= 2 else 0.15
        matched = [c for c in cat_scores if c['score'] >= MATCH_THRESHOLD]

        if matched:
            def calc_confidence(c):
                abs_conf = (c['score'] - 0.55) / 0.45  # 0 at 0.55, 1 at 1.0
                margin_conf = min(1.0, margin / 0.08)
                return min(100, max(0, int((0.6 * abs_conf + 0.4 * margin_conf) * 100)))

            identified = [
                {
                    'name': c['name'],
                    'cat_id': c['cat_id'],
                    'confidence': calc_confidence(c),
                    'is_mine': c['is_mine'],
                    'owner': c['owner'],
                    'owner_id': c['owner_id'],
                    'already_friends': c['already_friends'],
                }
                for c in matched
            ]
            result = {'identified': identified, 'few_photos': few_photos_warning,
                      'temp_filename': temp_filename}
            # Notify owners of identified cats
            notified = set()
            for c in matched:
                if not c['is_mine'] and c['owner_id'] not in notified:
                    push_notification(db, c['owner_id'], 'identified',
                                      from_user_id=uid, cat_id=c['cat_id'],
                                      photo=temp_filename, location=location,
                                      location_precise=location_precise)
                    notified.add(c['owner_id'])
        else:
            os.remove(temp_path)
            result = {'identified': [], 'few_photos': False, 'temp_filename': None}

    return render_template('identify.html', result=result)


@app.route('/identify/save_photo', methods=['POST'])
@login_required
def identify_save_photo():
    cat_id = request.form.get('cat_id', type=int)
    temp_filename = request.form.get('temp_filename', '')

    # Validate cat belongs to user
    db = get_db()
    cat = db.execute('SELECT id, name FROM cats WHERE id = ? AND user_id = ?',
                     (cat_id, session['user_id'])).fetchone()
    if not cat:
        flash('שגיאה: חתול לא נמצא')
        return redirect(url_for('identify'))

    # Validate temp file exists and belongs to this user (filename contains user_id)
    expected_prefix = f'temp_{session["user_id"]}_'
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    if not temp_filename.startswith(expected_prefix) or not os.path.exists(temp_path):
        flash('שגיאה: קובץ זמני לא נמצא')
        return redirect(url_for('identify'))

    # Rename to permanent filename
    ext = os.path.splitext(temp_filename)[1]
    new_filename = f'cat_{cat_id}_{os.urandom(8).hex()}{ext}'
    new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    os.rename(temp_path, new_path)

    # Compute features and store
    try:
        feat = extract_features(new_path)
        feat_json = json.dumps(feat)
    except Exception:
        feat_json = None

    db.execute('INSERT INTO cat_photos (cat_id, filename, features) VALUES (?, ?, ?)',
               (cat_id, new_filename, feat_json))
    db.commit()

    flash(f'התמונה נוספה לחתול "{cat["name"]}" בהצלחה!')
    return redirect(url_for('cats'))


# ---------- Friends ----------

@app.route('/friends')
@login_required
def friends():
    db = get_db()
    uid = session['user_id']

    friends_rows = db.execute('''
        SELECT u.id, u.username, f.id as fid FROM friendships f
        JOIN users u ON (CASE WHEN f.requester_id=? THEN f.receiver_id ELSE f.requester_id END)=u.id
        WHERE (f.requester_id=? OR f.receiver_id=?) AND f.status='accepted'
        ORDER BY u.username
    ''', (uid, uid, uid)).fetchall()
    friends_list = []
    for fr in friends_rows:
        unread = db.execute(
            'SELECT COUNT(*) as cnt FROM messages WHERE sender_id=? AND receiver_id=? AND is_read=0',
            (fr['id'], uid)
        ).fetchone()['cnt']
        friends_list.append({'id': fr['id'], 'username': fr['username'], 'fid': fr['fid'], 'unread': unread})

    received_rows = db.execute('''
        SELECT f.id, u.username, f.context_photo, f.context_type, f.context_cat_id, f.request_message FROM friendships f
        JOIN users u ON f.requester_id=u.id
        WHERE f.receiver_id=? AND f.status='pending'
    ''', (uid,)).fetchall()
    received = []
    for r in received_rows:
        cat_photos = []
        cat_name = None
        if r['context_cat_id']:
            cat_row = db.execute('SELECT name FROM cats WHERE id=?', (r['context_cat_id'],)).fetchone()
            if cat_row:
                cat_name = cat_row['name']
            cat_photos = [p['filename'] for p in db.execute(
                'SELECT filename FROM cat_photos WHERE cat_id=? LIMIT 4', (r['context_cat_id'],)
            ).fetchall()]
        received.append({
            'id': r['id'],
            'username': r['username'],
            'context_photo': r['context_photo'],
            'context_type': r['context_type'],
            'cat_photos': cat_photos,
            'cat_name': cat_name,
            'request_message': r['request_message'],
        })

    sent = db.execute('''
        SELECT f.id, u.username FROM friendships f
        JOIN users u ON f.receiver_id=u.id
        WHERE f.requester_id=? AND f.status='pending'
    ''', (uid,)).fetchall()

    q = request.args.get('q', '').strip()
    search_results = []
    if q:
        rows = db.execute(
            "SELECT id, username FROM users WHERE username LIKE ? AND id != ? LIMIT 20",
            (f'%{q}%', uid)
        ).fetchall()
        friend_ids = {r['id'] for r in friends_list}
        pending_sent_ids = {r['id'] for r in db.execute(
            "SELECT receiver_id as id FROM friendships WHERE requester_id=? AND status='pending'", (uid,)
        ).fetchall()}
        pending_recv_ids = {r['id'] for r in db.execute(
            "SELECT requester_id as id FROM friendships WHERE receiver_id=? AND status='pending'", (uid,)
        ).fetchall()}
        for row in rows:
            if row['id'] in friend_ids:
                status = 'friend'
            elif row['id'] in pending_sent_ids:
                status = 'sent'
            elif row['id'] in pending_recv_ids:
                status = 'received'
            else:
                status = 'none'
            search_results.append({'id': row['id'], 'username': row['username'], 'status': status})

    found_cat_notice = session.pop('found_cat_notice', None)
    return render_template('friends.html',
                           friends=friends_list, received=received,
                           sent=sent, search_results=search_results, q=q,
                           found_cat_notice=found_cat_notice)


@app.route('/friends/add/<int:user_id>', methods=['POST'])
@login_required
def friend_add(user_id):
    uid = session['user_id']
    if user_id == uid:
        flash('לא ניתן להוסיף את עצמך')
        return redirect(url_for('friends'))
    db = get_db()
    context_photo = request.form.get('context_photo', '').strip()
    if context_photo and not context_photo.startswith(f'temp_{uid}_'):
        context_photo = ''
    context_cat_id = request.form.get('context_cat_id', type=int)
    context_own_cat_id = request.form.get('context_own_cat_id', type=int)
    context_type = request.form.get('context_type', '').strip() or None
    request_message = request.form.get('request_message', '').strip()[:300] or None
    try:
        db.execute(
            'INSERT INTO friendships (requester_id, receiver_id, context_photo, context_cat_id, context_own_cat_id, context_type, request_message) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (uid, user_id, context_photo or None, context_cat_id, context_own_cat_id, context_type, request_message)
        )
        db.commit()
        u = db.execute('SELECT username FROM users WHERE id=?', (user_id,)).fetchone()
        flash(f'בקשת חברות נשלחה ל-{u["username"]}')
    except sqlite3.IntegrityError:
        flash('בקשת חברות כבר קיימת')
    return redirect(url_for('friends', q=request.form.get('q', '')))


@app.route('/friends/accept/<int:fid>', methods=['POST'])
@login_required
def friend_accept(fid):
    db = get_db()
    f = db.execute(
        "SELECT * FROM friendships WHERE id=? AND receiver_id=? AND status='pending'",
        (fid, session['user_id'])
    ).fetchone()
    if f:
        db.execute("UPDATE friendships SET status='accepted' WHERE id=?", (fid,))
        db.commit()
        flash('בקשת החברות אושרה!')
        if f['context_type'] == 'similar' and f['context_cat_id'] and f['context_own_cat_id']:
            # Create shared details table (empty) for the two similar cats
            c1 = min(f['context_own_cat_id'], f['context_cat_id'])
            c2 = max(f['context_own_cat_id'], f['context_cat_id'])
            try:
                db.execute('INSERT OR IGNORE INTO shared_details (cat_id_1, cat_id_2) VALUES (?, ?)', (c1, c2))
                db.commit()
            except Exception:
                pass
        if f['context_photo'] or f['context_type'] == 'similar':
            requester = db.execute('SELECT username FROM users WHERE id=?', (f['requester_id'],)).fetchone()
            session['found_cat_notice'] = {
                'photo': f['context_photo'],
                'requester': requester['username'],
                'cat_id': f['context_cat_id'],
                'context_type': f['context_type'] or 'identified',
            }
    return redirect(url_for('friends'))


@app.route('/friends/reject/<int:fid>', methods=['POST'])
@login_required
def friend_reject(fid):
    db = get_db()
    db.execute(
        'DELETE FROM friendships WHERE id=? AND (receiver_id=? OR requester_id=?)',
        (fid, session['user_id'], session['user_id'])
    )
    db.commit()
    return redirect(url_for('friends'))


@app.route('/friends/save_found_photo', methods=['POST'])
@login_required
def save_found_photo():
    cat_id = request.form.get('cat_id', type=int)
    photo_filename = request.form.get('photo_filename', '').strip()
    db = get_db()

    # Verify cat belongs to current user
    cat = db.execute('SELECT id, name FROM cats WHERE id=? AND user_id=?',
                     (cat_id, session['user_id'])).fetchone()
    if not cat:
        flash('לא נמצא חתול')
        return redirect(url_for('friends'))

    # Verify file exists
    src = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
    if not os.path.exists(src):
        flash('הקובץ לא נמצא')
        return redirect(url_for('friends'))

    # Save as permanent photo
    ext = os.path.splitext(photo_filename)[1].lower()
    new_filename = f'cat_{cat_id}_{os.urandom(6).hex()}{ext}'
    dst = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    import shutil
    shutil.copy2(src, dst)

    try:
        feat = extract_features(dst)
        feat_json = json.dumps(feat)
    except Exception:
        feat_json = None

    db.execute('INSERT INTO cat_photos (cat_id, filename, features) VALUES (?, ?, ?)',
               (cat_id, new_filename, feat_json))
    db.commit()
    flash(f'התמונה נוספה ל"{cat["name"]}" בהצלחה!')
    return redirect(url_for('cats'))


@app.route('/friends/remove/<int:fid>', methods=['POST'])
@login_required
def friend_remove(fid):
    db = get_db()
    db.execute(
        'DELETE FROM friendships WHERE id=? AND (requester_id=? OR receiver_id=?)',
        (fid, session['user_id'], session['user_id'])
    )
    db.commit()
    flash('החבר הוסר')
    return redirect(url_for('friends'))


@app.route('/notifications')
@login_required
def notifications():
    db = get_db()
    uid = session['user_id']

    filter_cat_id = request.args.get('cat_id', type=int)
    filter_from = request.args.get('from_date', '').strip()
    filter_to = request.args.get('to_date', '').strip()

    query = '''
        SELECT n.id, n.type, n.photo, n.cat_id, n.is_read, n.created_at, n.location, n.location_precise,
               u.username as from_username, c.name as cat_name
        FROM notifications n
        LEFT JOIN users u ON u.id = n.from_user_id
        LEFT JOIN cats c ON c.id = n.cat_id
        WHERE n.user_id = ?
    '''
    params = [uid]
    if filter_cat_id:
        query += ' AND n.cat_id = ?'
        params.append(filter_cat_id)
    if filter_from:
        query += ' AND date(n.created_at) >= ?'
        params.append(filter_from)
    if filter_to:
        query += ' AND date(n.created_at) <= ?'
        params.append(filter_to)
    query += ' ORDER BY n.created_at DESC'

    notifs_rows = db.execute(query, params).fetchall()

    # Build list with cat photos
    notifs = []
    for n in notifs_rows:
        cat_photos = []
        if n['cat_id']:
            cat_photos = [p['filename'] for p in db.execute(
                'SELECT filename FROM cat_photos WHERE cat_id=? LIMIT 3', (n['cat_id'],)
            ).fetchall()]
        notifs.append({
            'id': n['id'],
            'type': n['type'],
            'photo': n['photo'],
            'cat_name': n['cat_name'],
            'from_username': n['from_username'],
            'is_read': n['is_read'],
            'created_at': n['created_at'][:16].replace('T', ' '),
            'cat_photos': cat_photos,
            'location': n['location'],
            'location_precise': n['location_precise'],
        })
    # Mark all as read
    db.execute('UPDATE notifications SET is_read=1 WHERE user_id=?', (uid,))
    db.commit()

    my_cats = db.execute('SELECT id, name FROM cats WHERE user_id=? ORDER BY name', (uid,)).fetchall()
    return render_template('notifications.html', notifs=notifs, my_cats=my_cats,
                           filter_cat_id=filter_cat_id, filter_from=filter_from, filter_to=filter_to)


@app.route('/chat/<int:friend_id>', methods=['GET', 'POST'])
@login_required
def chat(friend_id):
    db = get_db()
    uid = session['user_id']

    # Verify they are friends
    friendship = db.execute('''
        SELECT id FROM friendships
        WHERE ((requester_id=? AND receiver_id=?) OR (requester_id=? AND receiver_id=?))
          AND status='accepted'
    ''', (uid, friend_id, friend_id, uid)).fetchone()
    if not friendship:
        flash('אינך חבר עם משתמש זה')
        return redirect(url_for('friends'))

    friend = db.execute('SELECT id, username FROM users WHERE id=?', (friend_id,)).fetchone()
    if not friend:
        return redirect(url_for('friends'))

    if request.method == 'POST':
        content = request.form.get('content', '').strip()
        image_file = request.files.get('image')
        image_filename = None

        if image_file and image_file.filename:
            ext = os.path.splitext(image_file.filename)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                image_filename = f'msg_{uid}_{os.urandom(8).hex()}{ext}'
                img = Image.open(image_file).convert('RGB')
                img.thumbnail((1200, 1200), Image.LANCZOS)
                img.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))

        if content or image_filename:
            db.execute(
                'INSERT INTO messages (sender_id, receiver_id, content, image) VALUES (?, ?, ?, ?)',
                (uid, friend_id, content, image_filename)
            )
            db.commit()

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True})
        return redirect(url_for('chat', friend_id=friend_id))

    # Mark incoming messages as read
    db.execute(
        'UPDATE messages SET is_read=1 WHERE sender_id=? AND receiver_id=? AND is_read=0',
        (friend_id, uid)
    )
    db.commit()

    messages = db.execute('''
        SELECT sender_id, content, image, sent_at FROM messages
        WHERE (sender_id=? AND receiver_id=?) OR (sender_id=? AND receiver_id=?)
        ORDER BY sent_at ASC
    ''', (uid, friend_id, friend_id, uid)).fetchall()

    return render_template('chat.html', friend=friend, messages=messages, uid=uid)


# ---------- Posts ----------

@app.route('/posts')
@login_required
def posts():
    db = get_db()
    uid = session['user_id']
    friend_ids = {r['id'] for r in db.execute('''
        SELECT u.id FROM friendships f
        JOIN users u ON (CASE WHEN f.requester_id=? THEN f.receiver_id ELSE f.requester_id END)=u.id
        WHERE (f.requester_id=? OR f.receiver_id=?) AND f.status='accepted'
    ''', (uid, uid, uid)).fetchall()}
    visible_user_ids = friend_ids | {uid}

    filter_purpose = request.args.get('purpose', '').strip()
    show_saved = request.args.get('saved') == '1'

    saved_ids = {r['post_id'] for r in db.execute(
        'SELECT post_id FROM post_saves WHERE user_id=?', (uid,)
    ).fetchall()}

    base_query = '''
        SELECT p.id, p.user_id, p.cat_id, p.photo, p.caption, p.visibility, p.purpose, p.created_at,
               u.username, c.name as cat_name
        FROM posts p
        JOIN users u ON u.id=p.user_id
        LEFT JOIN cats c ON c.id=p.cat_id
        WHERE (p.visibility='everyone' OR p.user_id IN ({}))
    '''.format(','.join('?' * len(visible_user_ids)))
    params = list(visible_user_ids)
    if filter_purpose in ('share', 'adoption'):
        base_query += ' AND p.purpose = ?'
        params.append(filter_purpose)
    if show_saved:
        if saved_ids:
            base_query += ' AND p.id IN ({})'.format(','.join('?' * len(saved_ids)))
            params.extend(list(saved_ids))
        else:
            base_query += ' AND 0'  # no saved posts
    base_query += ' ORDER BY p.created_at DESC'
    all_posts = db.execute(base_query, params).fetchall()

    posts_data = []
    for p in all_posts:
        comments = db.execute('''
            SELECT pc.id, pc.content, pc.created_at, pc.user_id, u.username
            FROM post_comments pc JOIN users u ON u.id=pc.user_id
            WHERE pc.post_id=? ORDER BY pc.created_at ASC
        ''', (p['id'],)).fetchall()
        posts_data.append({
            'id': p['id'], 'user_id': p['user_id'], 'cat_id': p['cat_id'],
            'photo': p['photo'], 'caption': p['caption'], 'visibility': p['visibility'],
            'purpose': p['purpose'],
            'created_at': p['created_at'][:16].replace('T', ' '),
            'username': p['username'], 'cat_name': p['cat_name'],
            'comments': [{'id': c['id'], 'content': c['content'], 'username': c['username'],
                          'user_id': c['user_id'],
                          'created_at': c['created_at'][:16].replace('T', ' ')} for c in comments],
        })
    my_cats = db.execute('SELECT id, name FROM cats WHERE user_id=? ORDER BY name', (uid,)).fetchall()
    return render_template('posts.html', posts=posts_data, my_cats=my_cats, uid=uid,
                           filter_purpose=filter_purpose, saved_ids=saved_ids, show_saved=show_saved)


@app.route('/posts/create', methods=['POST'])
@login_required
def post_create():
    db = get_db()
    uid = session['user_id']
    cat_id = request.form.get('cat_id', type=int) or None
    caption = request.form.get('caption', '').strip() or None
    visibility = request.form.get('visibility', 'friends')
    if visibility not in ('friends', 'everyone'):
        visibility = 'friends'
    purpose = request.form.get('purpose', 'share')
    if purpose not in ('share', 'adoption'):
        purpose = 'share'
    file = request.files.get('photo')
    photo = None
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            photo = f'post_{uid}_{os.urandom(8).hex()}{ext}'
            img = Image.open(file).convert('RGB')
            img.thumbnail((1200, 1200), Image.LANCZOS)
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], photo))
    if not photo and not caption:
        flash('נא להוסיף תמונה או כיתוב')
        return redirect(url_for('posts'))
    db.execute('INSERT INTO posts (user_id, cat_id, photo, caption, visibility, purpose) VALUES (?,?,?,?,?,?)',
               (uid, cat_id, photo, caption, visibility, purpose))
    db.commit()
    return redirect(url_for('posts'))


@app.route('/posts/<int:post_id>/delete', methods=['POST'])
@login_required
def post_delete(post_id):
    db = get_db()
    uid = session['user_id']
    post = db.execute('SELECT id, photo, user_id FROM posts WHERE id=?', (post_id,)).fetchone()
    if post and post['user_id'] == uid:
        if post['photo']:
            path = os.path.join(app.config['UPLOAD_FOLDER'], post['photo'])
            if os.path.exists(path):
                os.remove(path)
        db.execute('DELETE FROM post_comments WHERE post_id=?', (post_id,))
        db.execute('DELETE FROM posts WHERE id=?', (post_id,))
        db.commit()
    return redirect(url_for('posts'))


@app.route('/posts/<int:post_id>/comment', methods=['POST'])
@login_required
def post_comment(post_id):
    db = get_db()
    uid = session['user_id']
    content = request.form.get('content', '').strip()
    if content:
        db.execute('INSERT INTO post_comments (post_id, user_id, content) VALUES (?,?,?)',
                   (post_id, uid, content))
        db.commit()
    return redirect(url_for('posts') + f'#post-{post_id}')


@app.route('/posts/<int:post_id>/comments/<int:comment_id>/delete', methods=['POST'])
@login_required
def post_comment_delete(post_id, comment_id):
    db = get_db()
    uid = session['user_id']
    comment = db.execute('SELECT id, user_id FROM post_comments WHERE id=? AND post_id=?',
                         (comment_id, post_id)).fetchone()
    post = db.execute('SELECT user_id FROM posts WHERE id=?', (post_id,)).fetchone()
    if comment and (comment['user_id'] == uid or (post and post['user_id'] == uid)):
        db.execute('DELETE FROM post_comments WHERE id=?', (comment_id,))
        db.commit()
    return redirect(url_for('posts') + f'#post-{post_id}')


@app.route('/posts/<int:post_id>/save', methods=['POST'])
@login_required
def post_save(post_id):
    db = get_db()
    uid = session['user_id']
    existing = db.execute('SELECT 1 FROM post_saves WHERE user_id=? AND post_id=?', (uid, post_id)).fetchone()
    if existing:
        db.execute('DELETE FROM post_saves WHERE user_id=? AND post_id=?', (uid, post_id))
    else:
        db.execute('INSERT OR IGNORE INTO post_saves (user_id, post_id) VALUES (?,?)', (uid, post_id))
    db.commit()
    return redirect(request.referrer or url_for('posts') + f'#post-{post_id}')


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5001)
