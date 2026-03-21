import os
import json
import sqlite3
import uuid
import threading
import hashlib
import time
import urllib.request
from datetime import datetime
import urllib.parse
from functools import wraps

import numpy as np
from flask import (Flask, flash, g, jsonify, redirect,
                   render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
from dotenv import load_dotenv

load_dotenv()

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', '')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '')
ADMIN_SECRET   = os.environ.get('ADMIN_SECRET', '')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'catbook-secret-2024')
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME', 'demo')
CLOUDINARY_UPLOAD_PRESET = os.environ.get('CLOUDINARY_UPLOAD_PRESET', 'ml_default')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY', '')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET', '')


def upload_to_cloudinary(file_path, folder='catbook/street_cats'):
    """Upload a local file to Cloudinary using signed upload. Returns secure_url or None."""
    if not CLOUDINARY_API_KEY or not CLOUDINARY_API_SECRET or not CLOUDINARY_CLOUD_NAME:
        return None
    try:
        ts = str(int(time.time()))
        params_to_sign = f'folder={folder}&timestamp={ts}'
        signature = hashlib.sha256((params_to_sign + CLOUDINARY_API_SECRET).encode()).hexdigest()
        boundary = uuid.uuid4().hex
        with open(file_path, 'rb') as f:
            file_data = f.read()
        ext = os.path.splitext(file_path)[1] or '.jpg'
        mime = 'image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/png' if ext == '.png' else 'application/octet-stream'

        def field(name, value):
            return (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode()

        body = (
            field('api_key', CLOUDINARY_API_KEY) +
            field('timestamp', ts) +
            field('folder', folder) +
            field('signature', signature) +
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="photo{ext}"\r\nContent-Type: {mime}\r\n\r\n'.encode() +
            file_data + b'\r\n' +
            f'--{boundary}--\r\n'.encode()
        )
        req = urllib.request.Request(
            f'https://api.cloudinary.com/v1_1/{CLOUDINARY_CLOUD_NAME}/image/upload',
            data=body,
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        return data.get('secure_url')
    except Exception as e:
        print(f'upload_to_cloudinary error: {e}')
        return None


SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY', '')
MAIL_SENDER = os.environ.get('MAIL_SENDER', '')


def send_email(to, subject, body):
    """Send email via SendGrid HTTP API in a background thread."""
    if not SENDGRID_API_KEY or not MAIL_SENDER or not to:
        return

    def _send():
        try:
            payload = json.dumps({
                'personalizations': [{'to': [{'email': to}]}],
                'from': {'email': MAIL_SENDER, 'name': 'CatBook'},
                'subject': subject,
                'content': [{'type': 'text/html', 'value': body}]
            }).encode('utf-8')
            req = urllib.request.Request(
                'https://api.sendgrid.com/v3/mail/send',
                data=payload,
                headers={
                    'Authorization': f'Bearer {SENDGRID_API_KEY}',
                    'Content-Type': 'application/json'
                }
            )
            resp = urllib.request.urlopen(req)
            print(f'send_email OK: {resp.status} → {to}')
        except urllib.error.HTTPError as e:
            print(f'send_email HTTP error: {e.code} {e.read().decode()}')
        except Exception as e:
            print(f'send_email error: {e}')

    threading.Thread(target=_send, daemon=True).start()


@app.template_global()
def photo_url(filename):
    """Return displayable URL — handles Cloudinary URLs and local filenames."""
    if not filename:
        return ''
    if filename.startswith('http'):
        return filename
    return url_for('static', filename='uploads/' + filename)

def local_save(pil_img, public_id):
    """Save a PIL image locally, return filename."""
    filename = f'{public_id}.jpg'
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pil_img.convert('RGB').save(path, format='JPEG', quality=85)
    return filename


def local_delete(filename):
    """Delete a local image file."""
    if not filename or filename.startswith('http'):
        return
    try:
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"local_delete error: {e}")


# ---------- Feature extraction ----------

MODEL_VERSION = 'dinov2-small-onnx-v1'
_onnx_session = None

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_onnx_session():
    global _onnx_session
    if _onnx_session is None:
        import onnxruntime as ort
        model_path = os.path.join(app.root_path, 'dinov2_small.onnx')
        _onnx_session = ort.InferenceSession(model_path,
                                             providers=['CPUExecutionProvider'])
    return _onnx_session


def _preprocess_img(img):
    arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)  # HWC -> CHW


def extract_features(img_input):
    """Extract DINOv2-small CLS features with 5-crop TTA via ONNX."""
    session = get_onnx_session()
    if isinstance(img_input, Image.Image):
        img = img_input.convert('RGB')
    else:
        img = Image.open(img_input).convert('RGB')
    w, h = img.size
    crops = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.crop((w // 10, h // 10, w * 9 // 10, h * 9 // 10)),
        img.crop((0, 0, w * 4 // 5, h)),
        img.crop((w // 5, 0, w, h)),
    ]
    batch = np.stack([_preprocess_img(c) for c in crops])
    out = session.run(None, {'pixel_values': batch})[0]  # (5, seq, 384)
    cls_tokens = out[:, 0, :]  # CLS token per crop
    avg = cls_tokens.mean(axis=0)
    norm = np.linalg.norm(avg)
    return (avg / norm).tolist() if norm > 0 else avg.tolist()


def cosine_sim(a, b):
    return float(np.dot(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)))


def push_notification(db, user_id, ntype, from_user_id=None, cat_id=None, photo=None, location=None, location_precise=None, tree_id=None):
    db.execute(
        'INSERT INTO notifications (user_id, type, from_user_id, cat_id, photo, location, location_precise, tree_id) VALUES (?,?,?,?,?,?,?,?)',
        (user_id, ntype, from_user_id, cat_id, photo, location, location_precise, tree_id)
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
        db_path = os.path.join(app.root_path, 'catbook.db')
        g.db = sqlite3.connect(db_path)
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
        return {'pending_requests': pending, 'unread_messages': unread, 'unread_notifs': unread_notifs,
                'CLOUDINARY_CLOUD_NAME': CLOUDINARY_CLOUD_NAME, 'CLOUDINARY_UPLOAD_PRESET': CLOUDINARY_UPLOAD_PRESET}
    return {'pending_requests': 0, 'unread_messages': 0, 'unread_notifs': 0,
            'CLOUDINARY_CLOUD_NAME': CLOUDINARY_CLOUD_NAME, 'CLOUDINARY_UPLOAD_PRESET': CLOUDINARY_UPLOAD_PRESET}


def init_db():
    with app.app_context():
        db = get_db()
        db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT
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
        # add email column if missing
        try:
            db.execute('ALTER TABLE users ADD COLUMN email TEXT')
        except Exception:
            pass
        # add home_bg column if missing
        try:
            db.execute('ALTER TABLE users ADD COLUMN home_bg TEXT')
        except Exception:
            pass
        # add features column if missing
        try:
            db.execute('ALTER TABLE cat_photos ADD COLUMN features TEXT')
        except Exception:
            pass
        # feature_tokens: temp store for features computed before Cloudinary upload
        db.execute('''CREATE TABLE IF NOT EXISTS feature_tokens (
            token TEXT PRIMARY KEY,
            features TEXT NOT NULL,
            created_at TEXT NOT NULL
        )''')
        # clean old tokens
        db.execute("DELETE FROM feature_tokens WHERE created_at < datetime('now', '-1 day')")
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
        try:
            db.execute('ALTER TABLE notifications ADD COLUMN tree_id INTEGER')
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
        db.execute('''CREATE TABLE IF NOT EXISTS cat_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cat_id INTEGER NOT NULL,
            related_cat_id INTEGER NOT NULL,
            relation TEXT NOT NULL,
            created_by INTEGER NOT NULL,
            FOREIGN KEY (cat_id) REFERENCES cats(id),
            FOREIGN KEY (related_cat_id) REFERENCES cats(id),
            UNIQUE(cat_id, related_cat_id, relation)
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS family_trees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            owner_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users(id)
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            user_id INTEGER,
            success INTEGER NOT NULL DEFAULT 0,
            ip TEXT,
            logged_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        try:
            db.execute('ALTER TABLE cat_relations ADD COLUMN tree_id INTEGER')
        except Exception:
            pass
        db.execute('''CREATE TABLE IF NOT EXISTS tree_shares (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tree_id INTEGER NOT NULL,
            shared_with_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (tree_id) REFERENCES family_trees(id),
            FOREIGN KEY (shared_with_id) REFERENCES users(id),
            UNIQUE(tree_id, shared_with_id)
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS street_cats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nickname TEXT,
            auto_number INTEGER NOT NULL,
            created_by INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            adopted_by_cat_id INTEGER,
            FOREIGN KEY (created_by) REFERENCES users(id),
            FOREIGN KEY (adopted_by_cat_id) REFERENCES cats(id)
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS street_cat_sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            street_cat_id INTEGER NOT NULL,
            post_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            location_text TEXT,
            fed INTEGER NOT NULL DEFAULT 0,
            health_status TEXT NOT NULL DEFAULT 'בריא',
            sighted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            features TEXT,
            FOREIGN KEY (street_cat_id) REFERENCES street_cats(id),
            FOREIGN KEY (post_id) REFERENCES posts(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
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
    """Save photo locally. Returns (filename, pil_img) or (None, None)."""
    if not file_storage or not file_storage.filename:
        return None, None
    ext = os.path.splitext(file_storage.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        print(f"save_photo: extension not allowed: {ext!r}")
        return None, None
    try:
        img = Image.open(file_storage).convert('RGB')
        img.thumbnail((900, 900), Image.LANCZOS)
        public_id = f'cat_{cat_id}_{os.urandom(8).hex()}'
        filename = local_save(img, public_id)
        return filename, img
    except Exception as e:
        print(f"save_photo error: {e}")
        return None, None


def get_or_compute_features(db, photo_id, filename):
    """Return feature vector for a photo, computing and caching if needed."""
    row = db.execute('SELECT features FROM cat_photos WHERE id = ?', (photo_id,)).fetchone()
    if row and row['features']:
        return json.loads(row['features'])
    try:
        if filename.startswith('http'):
            img_input = filename
        else:
            img_input = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        feat = extract_features(img_input)
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
    if 'user_id' not in session:
        return redirect(url_for('login'))
    db = get_db()
    uid = session['user_id']

    notifs = db.execute('''
        SELECT n.*, u.username as from_username
        FROM notifications n
        LEFT JOIN users u ON u.id = n.from_user_id
        WHERE n.user_id = ? ORDER BY n.created_at DESC LIMIT 5
    ''', (uid,)).fetchall()

    conversations = db.execute('''
        SELECT u.id, u.username,
               m.content, m.sent_at,
               SUM(CASE WHEN m.receiver_id=? AND m.is_read=0 THEN 1 ELSE 0 END) as unread
        FROM messages m
        JOIN users u ON u.id = CASE WHEN m.sender_id=? THEN m.receiver_id ELSE m.sender_id END
        WHERE m.sender_id=? OR m.receiver_id=?
        GROUP BY u.id
        ORDER BY MAX(m.sent_at) DESC LIMIT 5
    ''', (uid, uid, uid, uid)).fetchall()

    posts = db.execute('''
        SELECT p.*, u.username, c.name as cat_name
        FROM posts p
        JOIN users u ON u.id = p.user_id
        LEFT JOIN cats c ON c.id = p.cat_id
        JOIN friendships f ON (f.requester_id=? AND f.receiver_id=p.user_id)
                           OR (f.receiver_id=? AND f.requester_id=p.user_id)
        WHERE f.status='accepted'
        ORDER BY p.created_at DESC LIMIT 5
    ''', (uid, uid)).fetchall()

    user = db.execute('SELECT home_bg FROM users WHERE id=?', (uid,)).fetchone()
    home_bg = photo_url(user['home_bg']) if user and user['home_bg'] else None
    return render_template('home.html', notifs=notifs, conversations=conversations, posts=posts, home_bg=home_bg)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        ip = request.remote_addr
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            db.execute('INSERT INTO login_logs (username, user_id, success, ip) VALUES (?,?,1,?)',
                       (username, user['id'], ip))
            db.commit()
            return redirect(url_for('index'))
        db.execute('INSERT INTO login_logs (username, user_id, success, ip) VALUES (?,?,0,?)',
                   (username, user['id'] if user else None, ip))
        db.commit()
        flash('שם משתמש או סיסמה שגויים')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        email = request.form.get('email', '').strip() or None
        if not username or not password:
            flash('נא למלא את כל השדות')
        else:
            db = get_db()
            try:
                db.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                           (username, generate_password_hash(password), email))
                db.commit()
                flash('נרשמת בהצלחה! כעת תוכל להתחבר')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('שם המשתמש כבר תפוס')
    return render_template('register.html')


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    db = get_db()
    uid = session['user_id']
    user = db.execute('SELECT username, email, home_bg FROM users WHERE id=?', (uid,)).fetchone()
    if request.method == 'POST':
        new_username = request.form.get('username', '').strip()
        new_email = request.form.get('email', '').strip() or None
        new_home_bg = request.form.get('home_bg', '').strip() or None
        if not new_username:
            flash('שם משתמש לא יכול להיות ריק', 'warning')
        else:
            try:
                db.execute('UPDATE users SET username=?, email=?, home_bg=? WHERE id=?',
                           (new_username, new_email, new_home_bg, uid))
                db.commit()
                session['username'] = new_username
                flash('הפרטים עודכנו בהצלחה')
                return redirect(url_for('settings'))
            except sqlite3.IntegrityError:
                flash('שם המשתמש כבר תפוס', 'warning')
    return render_template('settings.html', user=user)


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


@app.route('/api/nav-counts')
@login_required
def api_nav_counts():
    db = get_db()
    uid = session['user_id']
    pending = db.execute(
        "SELECT COUNT(*) as cnt FROM friendships WHERE receiver_id=? AND status='pending'", (uid,)
    ).fetchone()['cnt']
    unread = db.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE receiver_id=? AND is_read=0", (uid,)
    ).fetchone()['cnt']
    unread_notifs = db.execute(
        "SELECT COUNT(*) as cnt FROM notifications WHERE user_id=? AND is_read=0", (uid,)
    ).fetchone()['cnt']
    return jsonify({'pending_requests': pending, 'unread_messages': unread, 'unread_notifs': unread_notifs})


@app.route('/api/extract-features', methods=['POST'])
@login_required
def extract_features_api():
    """Receive image, compute DINOv2 features, return token. Used before Cloudinary upload."""
    file = request.files.get('photo')
    if not file:
        return jsonify({'error': 'no file'}), 400
    ext = os.path.splitext(secure_filename(file.filename or 'photo.jpg'))[1] or '.jpg'
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"feat_{uuid.uuid4().hex}{ext}")
    file.save(temp_path)
    try:
        feat = extract_features(temp_path)
        token = uuid.uuid4().hex
        db = get_db()
        db.execute('INSERT INTO feature_tokens (token, features, created_at) VALUES (?, ?, datetime("now"))',
                   (token, json.dumps(feat if isinstance(feat, list) else feat.tolist())))
        db.commit()
        return jsonify({'token': token})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


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
        # Accept Cloudinary URLs with optional feature tokens
        photo_urls = request.form.getlist('photo_urls')
        feature_tokens = request.form.getlist('feature_tokens')
        for photo_url_val, token in zip(photo_urls, feature_tokens + [''] * len(photo_urls)):
            if not photo_url_val:
                continue
            feat_json = None
            if token:
                tok_row = db.execute('SELECT features FROM feature_tokens WHERE token=?', (token,)).fetchone()
                if tok_row:
                    feat_json = tok_row['features']
                    all_feats.append(json.loads(feat_json))
                    db.execute('DELETE FROM feature_tokens WHERE token=?', (token,))
            db.execute('INSERT INTO cat_photos (cat_id, filename, features) VALUES (?, ?, ?)',
                       (cat_id, photo_url_val, feat_json))
        # Also accept legacy file uploads (for backward compatibility)
        for file in request.files.getlist('photos'):
            url, pil_img = save_photo(file, cat_id)
            if url:
                try:
                    feat = extract_features(pil_img)
                    feat_json = json.dumps(feat)
                    all_feats.append(feat)
                except Exception:
                    feat_json = None
                db.execute('INSERT INTO cat_photos (cat_id, filename, features) VALUES (?, ?, ?)',
                           (cat_id, url, feat_json))
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

    # Accept Cloudinary URL (direct browser upload)
    cloudinary_url = request.form.get('photo_url', '')
    if cloudinary_url and cloudinary_url.startswith('http'):
        feat_json = None
        token = request.form.get('feature_token', '')
        if token:
            tok_row = db.execute('SELECT features FROM feature_tokens WHERE token=?', (token,)).fetchone()
            if tok_row:
                feat_json = tok_row['features']
                db.execute('DELETE FROM feature_tokens WHERE token=?', (token,))
        db.execute('INSERT INTO cat_photos (cat_id, filename, features) VALUES (?, ?, ?)',
                   (cat_id, cloudinary_url, feat_json))
        db.commit()
        return jsonify({'success': True, 'filename': cloudinary_url})

    # Legacy file upload
    file = request.files.get('photo')
    url, pil_img = save_photo(file, cat_id)
    if url:
        try:
            feat = extract_features(pil_img)
            feat_json = json.dumps(feat)
        except Exception as e:
            app.logger.error(f'extract_features failed: {e}')
            feat = None
            feat_json = None
        db.execute('INSERT INTO cat_photos (cat_id, filename, features) VALUES (?, ?, ?)',
                   (cat_id, url, feat_json))
        db.commit()

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

        return jsonify({'success': True, 'filename': url})
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
        local_delete(photo['filename'])
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
            local_delete(p['filename'])
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
    db = get_db()
    uid = session['user_id']
    result = None

    # Fetch user's cat photos for "existing photo" selection
    my_photos = db.execute('''
        SELECT cp.id, cp.filename, c.name as cat_name
        FROM cat_photos cp JOIN cats c ON c.id = cp.cat_id
        WHERE c.user_id = ? AND cp.features IS NOT NULL ORDER BY c.name, cp.id
    ''', (uid,)).fetchall()

    if request.method == 'POST':
        location = request.form.get('location', '').strip() or None
        location_precise = request.form.get('location_precise', '').strip() or None
        existing_photo_id = request.form.get('existing_photo_id', '').strip()

        if existing_photo_id:
            # Use features already stored in DB
            photo_row = db.execute(
                'SELECT id, filename, features FROM cat_photos WHERE id=? AND EXISTS (SELECT 1 FROM cats WHERE id=cat_photos.cat_id AND user_id=?)',
                (existing_photo_id, uid)
            ).fetchone()
            if not photo_row or not photo_row['features']:
                flash('לא ניתן להשתמש בתמונה זו — אין features מאוחסנים')
                return render_template('identify.html', result=None, my_photos=my_photos)
            query_feat = np.array(json.loads(photo_row['features']), dtype=np.float32)
            temp_filename = photo_row['filename']
        else:
            file = request.files.get('photo')
            if not file or not file.filename:
                flash('נא לבחור תמונה')
                return render_template('identify.html', result=None, my_photos=my_photos)

            # Save original (uncropped) as temp — used for display in posts/street cats
            ext = os.path.splitext(file.filename)[1].lower()
            temp_filename = f'temp_{session["user_id"]}_{os.urandom(6).hex()}{ext}'
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            img = Image.open(file).convert('RGB')
            img.thumbnail((900, 900), Image.LANCZOS)
            img.save(temp_path)

            # Use pre-extracted features from cropped image if provided
            features_token = request.form.get('features_token', '').strip()
            tok_row = db.execute('SELECT features FROM feature_tokens WHERE token=?', (features_token,)).fetchone() if features_token else None
            if tok_row and tok_row['features']:
                query_feat = np.array(json.loads(tok_row['features']), dtype=np.float32)
                db.execute('DELETE FROM feature_tokens WHERE token=?', (features_token,))
                db.commit()
            else:
                try:
                    query_feat = extract_features(temp_path)
                except Exception as e:
                    os.remove(temp_path)
                    flash(f'שגיאה בעיבוד התמונה: {e}')
                    return render_template('identify.html', result=None, my_photos=my_photos)

        # All cats from all users
        cat_rows = db.execute(
            'SELECT c.id, c.name, c.user_id, u.username FROM cats c JOIN users u ON u.id=c.user_id'
        ).fetchall()

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
                k = min(3, len(scores))
                top_k = sorted(scores, reverse=True)[:k]
                score = sum(top_k) / k
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
            session['identify_temp_url'] = temp_filename
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
                    owner = db.execute('SELECT username, email FROM users WHERE id=?', (c['owner_id'],)).fetchone()
                    if owner and owner['email']:
                        loc_text = f'<p>📍 מיקום: {location}</p>' if location else ''
                        send_email(
                            to=owner['email'],
                            subject=f'החתול שלך {c["name"]} זוהה! 🐱',
                            body=f'''
                            <div dir="rtl" style="font-family:Arial,sans-serif;font-size:15px">
                              <h2>שלום {owner["username"]}!</h2>
                              <p>החתול שלך <strong>{c["name"]}</strong> זוהה על ידי <strong>{session["username"]}</strong>.</p>
                              {loc_text}
                              <p><a href="https://catbook.pythonanywhere.com/notifications" style="background:#6a0dad;color:#fff;padding:10px 20px;border-radius:8px;text-decoration:none">צפה בהתראות</a></p>
                            </div>'''
                        )
                    notified.add(c['owner_id'])
        else:
            session['identify_temp_url'] = temp_filename
            result = {'identified': [], 'few_photos': False, 'temp_filename': temp_filename}

        # ── Check against street cats ──
        street_sightings = db.execute('''
            SELECT s.id, s.street_cat_id, s.features, s.location_text, s.sighted_at,
                   sc.nickname, sc.auto_number,
                   p.photo
            FROM street_cat_sightings s
            JOIN street_cats sc ON sc.id = s.street_cat_id
            JOIN posts p ON p.id = s.post_id
            WHERE s.features IS NOT NULL
        ''').fetchall()

        sc_scores = {}  # sc_id → best score + info
        for row in street_sightings:
            try:
                feat = np.array(json.loads(row['features']), dtype=np.float32)
                score = cosine_sim(query_feat, feat)
            except Exception:
                continue
            sc_id = row['street_cat_id']
            if sc_id not in sc_scores or score > sc_scores[sc_id]['score']:
                sc_scores[sc_id] = {
                    'sc_id': sc_id,
                    'score': score,
                    'nickname': row['nickname'],
                    'auto_number': row['auto_number'],
                    'photo': row['photo'],
                    'location_text': row['location_text'],
                    'sighted_at': row['sighted_at'],
                    'auto_link': score >= 0.85,
                }

        street_matched = sorted(
            [v for v in sc_scores.values() if v['score'] >= 0.55],
            key=lambda x: x['score'], reverse=True
        )
        result['street_matched'] = street_matched
        qf = query_feat if isinstance(query_feat, list) else query_feat.tolist()
        result['query_features'] = json.dumps(qf) if not existing_photo_id else None

    return render_template('identify.html', result=result, my_photos=my_photos)


@app.route('/identify/post', methods=['POST'])
@login_required
def identify_post():
    temp_filename = request.form.get('temp_filename', '')
    session_url = session.get('identify_temp_url', '')
    if not temp_filename or temp_filename != session_url:
        flash('קובץ זמני לא נמצא')
        return redirect(url_for('identify'))
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    if not os.path.exists(temp_path):
        flash('קובץ זמני לא נמצא')
        return redirect(url_for('identify'))
    visibility = request.form.get('visibility', 'friends')
    if visibility not in ('friends', 'everyone'):
        visibility = 'friends'
    purpose = request.form.get('purpose', 'share')
    if purpose not in ('share', 'adoption'):
        purpose = 'share'
    caption = request.form.get('caption', '').strip() or None
    db = get_db()
    db.execute('INSERT INTO posts (user_id, photo, caption, visibility, purpose) VALUES (?,?,?,?,?)',
               (session['user_id'], temp_filename, caption, visibility, purpose))
    db.commit()
    flash('הפוסט פורסם בהצלחה')
    return redirect(url_for('posts'))


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

    # Validate temp URL matches what we stored in session
    session_url = session.get('identify_temp_url', '')
    if not temp_filename or temp_filename != session_url:
        flash('שגיאה: קובץ זמני לא נמצא')
        return redirect(url_for('identify'))
    session.pop('identify_temp_url', None)

    # Move temp file to permanent cat photo
    try:
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        if not os.path.exists(temp_path):
            flash('קובץ זמני לא נמצא')
            return redirect(url_for('identify'))
        img = Image.open(temp_path).convert('RGB')
        img.thumbnail((900, 900), Image.LANCZOS)
        public_id = f'cat_{cat_id}_{os.urandom(8).hex()}'
        new_filename = local_save(img, public_id)
        os.remove(temp_path)
        feat = extract_features(img)
        feat_json = json.dumps(feat)
    except Exception as e:
        flash(f'שגיאה בשמירת התמונה: {e}')
        return redirect(url_for('identify'))

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
    if context_photo and context_photo != session.get('identify_temp_url', ''):
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
        u = db.execute('SELECT username, email FROM users WHERE id=?', (user_id,)).fetchone()
        flash(f'בקשת חברות נשלחה ל-{u["username"]}')
        if u['email']:
            send_email(
                to=u['email'],
                subject='קיבלת בקשת חברות ב-CatBook 🐱',
                body=f'''
                <div dir="rtl" style="font-family:Arial,sans-serif;font-size:15px">
                  <h2>שלום {u["username"]}!</h2>
                  <p><strong>{session["username"]}</strong> שלח/ה לך בקשת חברות ב-CatBook.</p>
                  <p><a href="https://catbook.pythonanywhere.com/friends" style="background:#6a0dad;color:#fff;padding:10px 20px;border-radius:8px;text-decoration:none">צפה בבקשה</a></p>
                </div>'''
            )
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

    if not photo_filename:
        flash('הקובץ לא נמצא')
        return redirect(url_for('friends'))

    # Copy local photo to permanent cat photo
    try:
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
        if not os.path.exists(src_path):
            flash('קובץ לא נמצא')
            return redirect(url_for('friends'))
        img = Image.open(src_path).convert('RGB')
        img.thumbnail((900, 900), Image.LANCZOS)
        public_id = f'cat_{cat_id}_{os.urandom(6).hex()}'
        new_filename = local_save(img, public_id)
        feat = extract_features(img)
        feat_json = json.dumps(feat)
    except Exception as e:
        flash(f'שגיאה בשמירת התמונה: {e}')
        return redirect(url_for('friends'))

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
               n.tree_id, u.username as from_username, c.name as cat_name, ft.name as tree_name
        FROM notifications n
        LEFT JOIN users u ON u.id = n.from_user_id
        LEFT JOIN cats c ON c.id = n.cat_id
        LEFT JOIN family_trees ft ON ft.id = n.tree_id
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
            'tree_id': n['tree_id'],
            'tree_name': n['tree_name'],
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
        # Accept Cloudinary URL or legacy file upload
        image_filename = request.form.get('image_url', '') or None
        if image_filename and not image_filename.startswith('http'):
            image_filename = None
        if not image_filename:
            image_file = request.files.get('image')
            if image_file and image_file.filename:
                ext = os.path.splitext(image_file.filename)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    try:
                        img = Image.open(image_file).convert('RGB')
                        img.thumbnail((1200, 1200), Image.LANCZOS)
                        public_id = f'msg_{uid}_{os.urandom(8).hex()}'
                        image_filename = local_save(img, public_id)
                    except Exception as e:
                        print(f"chat image upload error: {e}")

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

@app.route('/posts/new')
@login_required
def post_new():
    db = get_db()
    uid = session['user_id']
    my_cats = db.execute('SELECT id, name FROM cats WHERE user_id=? ORDER BY name', (uid,)).fetchall()
    my_cat_photos = db.execute('''
        SELECT cp.id, cp.filename, c.name as cat_name
        FROM cat_photos cp JOIN cats c ON c.id=cp.cat_id
        WHERE c.user_id=? ORDER BY c.name, cp.id
    ''', (uid,)).fetchall()
    return render_template('post_new.html', my_cats=my_cats, my_cat_photos=my_cat_photos)


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
    show_friends = request.args.get('friends') == '1'

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
    if filter_purpose in ('share', 'adoption', 'street_cat'):
        base_query += ' AND p.purpose = ?'
        params.append(filter_purpose)
    if show_friends:
        if friend_ids:
            base_query += ' AND p.user_id IN ({})'.format(','.join('?' * len(friend_ids)))
            params.extend(list(friend_ids))
        else:
            base_query += ' AND 0'
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
    friends = db.execute('''
        SELECT u.id, u.username, u.email FROM friendships f
        JOIN users u ON u.id = CASE WHEN f.requester_id=? THEN f.receiver_id ELSE f.requester_id END
        WHERE (f.requester_id=? OR f.receiver_id=?) AND f.status='accepted' ORDER BY u.username
    ''', (uid, uid, uid)).fetchall()
    return render_template('posts.html', posts=posts_data, uid=uid,
                           filter_purpose=filter_purpose, saved_ids=saved_ids,
                           show_saved=show_saved, show_friends=show_friends,
                           friends=friends)


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
    # Accept Cloudinary URL or legacy file upload
    photo = request.form.get('photo_url', '') or None
    if photo and not photo.startswith('http'):
        photo = None
    if not photo:
        file = request.files.get('photo')
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                try:
                    img = Image.open(file).convert('RGB')
                    img.thumbnail((1200, 1200), Image.LANCZOS)
                    public_id = f'post_{uid}_{os.urandom(8).hex()}'
                    photo = local_save(img, public_id)
                except Exception as e:
                    print(f"post upload error: {e}")
    if not photo and not caption:
        flash('נא להוסיף תמונה או כיתוב')
        return redirect(url_for('posts'))
    db.execute('INSERT INTO posts (user_id, cat_id, photo, caption, visibility, purpose) VALUES (?,?,?,?,?,?)',
               (uid, cat_id, photo, caption, visibility, purpose))
    db.commit()
    return redirect(url_for('posts'))


@app.route('/posts/<int:post_id>/send-chat', methods=['POST'])
@login_required
def post_send_chat(post_id):
    db = get_db()
    uid = session['user_id']
    post = db.execute('SELECT * FROM posts WHERE id=?', (post_id,)).fetchone()
    if not post:
        return redirect(url_for('posts'))
    friend_id_raw = request.form.get('friend_id', '')
    # Build message content
    parts = []
    if post['caption']:
        parts.append(post['caption'])
    content = '\n'.join(parts) if parts else '📸 שיתף פוסט'
    image = post['photo'] if post['photo'] else None

    def _send(fid):
        db.execute(
            'INSERT INTO messages (sender_id, receiver_id, content, image) VALUES (?, ?, ?, ?)',
            (uid, fid, content, image)
        )

    if friend_id_raw == 'all':
        friends = db.execute('''
            SELECT u.id FROM users u
            JOIN friendships f ON (f.requester_id=? AND f.receiver_id=u.id) OR (f.receiver_id=? AND f.requester_id=u.id)
            WHERE f.status='accepted'
        ''', (uid, uid)).fetchall()
        for f in friends:
            _send(f['id'])
        db.commit()
        flash(f'הפוסט נשלח בצ\'אט לכל החברים ({len(friends)})')
        return redirect(url_for('posts') + f'#post-{post_id}')
    else:
        friend = db.execute('''
            SELECT u.id FROM users u
            JOIN friendships f ON (f.requester_id=? AND f.receiver_id=u.id) OR (f.receiver_id=? AND f.requester_id=u.id)
            WHERE u.id=? AND f.status='accepted'
        ''', (uid, uid, friend_id_raw)).fetchone()
        if not friend:
            flash('חבר לא נמצא')
            return redirect(url_for('posts') + f'#post-{post_id}')
        _send(int(friend_id_raw))
        db.commit()
        return redirect(url_for('chat', friend_id=friend_id_raw))


@app.route('/posts/<int:post_id>/send-email', methods=['POST'])
@login_required
def post_send_email(post_id):
    db = get_db()
    uid = session['user_id']
    post = db.execute('''
        SELECT p.*, u.username FROM posts p JOIN users u ON u.id=p.user_id WHERE p.id=?
    ''', (post_id,)).fetchone()
    if not post:
        return redirect(url_for('posts'))
    friend_id_raw = request.form.get('friend_id', '')
    caption_text = f'<p style="font-style:italic">"{post["caption"]}"</p>' if post['caption'] else ''
    raw_url = photo_url(post['photo']) if post['photo'] else ''
    if raw_url and not raw_url.startswith('http'):
        raw_url = 'https://catbook.pythonanywhere.com' + raw_url
    photo_text = f'<p><img src="{raw_url}" style="max-width:400px;border-radius:8px"></p>' if raw_url else ''

    def _email_body(username):
        return f'''
        <div dir="rtl" style="font-family:Arial,sans-serif;font-size:15px">
          <h2>שלום {username}!</h2>
          <p><strong>{session["username"]}</strong> שיתף איתך פוסט:</p>
          {caption_text}
          {photo_text}
          <p><a href="https://catbook.pythonanywhere.com/posts" style="background:#6a0dad;color:#fff;padding:10px 20px;border-radius:8px;text-decoration:none">צפה בפוסטים</a></p>
        </div>'''

    if friend_id_raw == 'all':
        recipients = db.execute('''
            SELECT u.username, u.email FROM users u
            JOIN friendships f ON (f.requester_id=? AND f.receiver_id=u.id) OR (f.receiver_id=? AND f.requester_id=u.id)
            WHERE f.status='accepted' AND u.email IS NOT NULL AND u.email != ''
        ''', (uid, uid)).fetchall()
        for r in recipients:
            send_email(to=r['email'],
                       subject=f'{session["username"]} שיתף איתך פוסט מ-CatBook 🐱',
                       body=_email_body(r['username']))
        flash(f'הפוסט נשלח במייל לכל החברים ({len(recipients)})')
    else:
        friend = db.execute('''
            SELECT u.username, u.email FROM users u
            JOIN friendships f ON (f.requester_id=? AND f.receiver_id=u.id) OR (f.receiver_id=? AND f.requester_id=u.id)
            WHERE u.id=? AND f.status='accepted'
        ''', (uid, uid, friend_id_raw)).fetchone()
        if not friend or not friend['email']:
            flash('לא ניתן לשלוח — לחבר זה אין אימייל רשום')
            return redirect(url_for('posts') + f'#post-{post_id}')
        send_email(to=friend['email'],
                   subject=f'{session["username"]} שיתף איתך פוסט מ-CatBook 🐱',
                   body=_email_body(friend['username']))
        flash(f'הפוסט נשלח במייל ל-{friend["username"]}')
    return redirect(url_for('posts') + f'#post-{post_id}')


@app.route('/posts/<int:post_id>/edit', methods=['POST'])
@login_required
def post_edit(post_id):
    db = get_db()
    uid = session['user_id']
    post = db.execute('SELECT id, user_id FROM posts WHERE id=?', (post_id,)).fetchone()
    if not post or post['user_id'] != uid:
        return redirect(url_for('posts'))
    caption = request.form.get('caption', '').strip() or None
    visibility = request.form.get('visibility', 'friends')
    if visibility not in ('friends', 'everyone'):
        visibility = 'friends'
    purpose = request.form.get('purpose', 'share')
    if purpose not in ('share', 'adoption'):
        purpose = 'share'
    db.execute('UPDATE posts SET caption=?, visibility=?, purpose=? WHERE id=?',
               (caption, visibility, purpose, post_id))
    db.commit()
    return redirect(url_for('posts') + f'#post-{post_id}')


@app.route('/posts/<int:post_id>/delete', methods=['POST'])
@login_required
def post_delete(post_id):
    db = get_db()
    uid = session['user_id']
    post = db.execute('SELECT id, photo, user_id FROM posts WHERE id=?', (post_id,)).fetchone()
    if post and post['user_id'] == uid:
        if post['photo']:
            local_delete(post['photo'])
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



def _get_accessible_cats(db, uid):
    """Return list of cat dicts accessible to uid (own + friends')."""
    my_cats = db.execute(
        """SELECT c.id, c.name, MIN(cp.filename) as photo, ? as owner, ? as owner_id, 1 as is_mine
           FROM cats c LEFT JOIN cat_photos cp ON cp.cat_id=c.id
           WHERE c.user_id=? GROUP BY c.id""", (session['username'], uid, uid)
    ).fetchall()
    friends = db.execute(
        """SELECT u.id, u.username FROM friendships f
           JOIN users u ON u.id=CASE WHEN f.requester_id=? THEN f.receiver_id ELSE f.requester_id END
           WHERE (f.requester_id=? OR f.receiver_id=?) AND f.status='accepted'""",
        (uid, uid, uid)
    ).fetchall()
    friend_cats = []
    for fr in friends:
        for cat in db.execute(
            """SELECT c.id, c.name, MIN(cp.filename) as photo
               FROM cats c LEFT JOIN cat_photos cp ON cp.cat_id=c.id
               WHERE c.user_id=? GROUP BY c.id""", (fr['id'],)
        ).fetchall():
            friend_cats.append({'id': cat['id'], 'name': cat['name'], 'photo': cat['photo'],
                                 'owner': fr['username'], 'owner_id': fr['id'], 'is_mine': False})
    return [dict(c) for c in my_cats], friend_cats


@app.route('/family-tree')
@login_required
def family_tree():
    db = get_db()
    uid = session['user_id']
    my_trees = db.execute(
        'SELECT * FROM family_trees WHERE owner_id=? ORDER BY created_at DESC', (uid,)
    ).fetchall()
    shared_trees = db.execute(
        '''SELECT ft.*, u.username as owner_name FROM tree_shares ts
           JOIN family_trees ft ON ft.id = ts.tree_id
           JOIN users u ON u.id = ft.owner_id
           WHERE ts.shared_with_id=? ORDER BY ft.created_at DESC''', (uid,)
    ).fetchall()
    return render_template('family_trees.html', trees=my_trees, shared_trees=shared_trees)


@app.route('/family-tree/create', methods=['POST'])
@login_required
def family_tree_create():
    name = request.form.get('name', '').strip() or 'עץ משפחה חדש'
    db = get_db()
    uid = session['user_id']
    cur = db.execute('INSERT INTO family_trees (name, owner_id) VALUES (?,?)', (name, uid))
    db.commit()
    return redirect(url_for('family_tree_view', tree_id=cur.lastrowid))


@app.route('/family-tree/<int:tree_id>/delete', methods=['POST'])
@login_required
def family_tree_delete(tree_id):
    db = get_db()
    uid = session['user_id']
    db.execute('DELETE FROM family_trees WHERE id=? AND owner_id=?', (tree_id, uid))
    db.execute('DELETE FROM cat_relations WHERE tree_id=? AND created_by=?', (tree_id, uid))
    db.commit()
    return redirect(url_for('family_tree'))


@app.route('/family-tree/<int:tree_id>/rename', methods=['POST'])
@login_required
def family_tree_rename(tree_id):
    name = request.form.get('name', '').strip()
    if name:
        db = get_db()
        db.execute('UPDATE family_trees SET name=? WHERE id=? AND owner_id=?',
                   (name, tree_id, session['user_id']))
        db.commit()
    return redirect(url_for('family_tree_view', tree_id=tree_id))


@app.route('/family-tree/<int:tree_id>')
@login_required
def family_tree_view(tree_id):
    db = get_db()
    uid = session['user_id']
    tree = db.execute('SELECT * FROM family_trees WHERE id=?', (tree_id,)).fetchone()

    if not tree:
        flash('עץ לא נמצא')
        return redirect(url_for('family_tree'))

    is_owner = (tree['owner_id'] == uid)
    is_shared = db.execute(
        'SELECT 1 FROM tree_shares WHERE tree_id=? AND shared_with_id=?', (tree_id, uid)
    ).fetchone() is not None

    if not is_owner and not is_shared:
        flash('אין לך גישה לעץ זה')
        return redirect(url_for('family_tree'))

    my_cats, friend_cats = _get_accessible_cats(db, uid)
    all_ids = [c['id'] for c in my_cats] + [c['id'] for c in friend_cats]

    relations = []
    if all_ids:
        ph = ','.join('?' * len(all_ids))
        relations = [dict(r) for r in db.execute(
            f"SELECT id, cat_id, related_cat_id, relation, created_by FROM cat_relations "
            f"WHERE tree_id=? AND cat_id IN ({ph}) AND related_cat_id IN ({ph})",
            [tree_id] + all_ids + all_ids
        ).fetchall()]

    cat_names = {c['id']: c['name'] for c in my_cats}
    cat_names.update({c['id']: c['name'] for c in friend_cats})
    rel_labels = {'father': '← האבא שלו/שלה:', 'mother': '← האמא שלו/שלה:',
                  'child': '← ילד/ה שלו/שלה:', 'sibling': 'אח/אחות של'}

    # Friends to potentially share with (only for owner)
    share_friends = []
    shared_with = []
    if is_owner:
        friends_rows = db.execute(
            """SELECT u.id, u.username FROM friendships f
               JOIN users u ON u.id=CASE WHEN f.requester_id=? THEN f.receiver_id ELSE f.requester_id END
               WHERE (f.requester_id=? OR f.receiver_id=?) AND f.status='accepted'
               ORDER BY u.username""",
            (uid, uid, uid)
        ).fetchall()
        already_shared_ids = {
            r['shared_with_id'] for r in
            db.execute('SELECT shared_with_id FROM tree_shares WHERE tree_id=?', (tree_id,)).fetchall()
        }
        share_friends = [dict(f) for f in friends_rows if f['id'] not in already_shared_ids]
        shared_with = [dict(r) for r in db.execute(
            '''SELECT u.id, u.username FROM tree_shares ts
               JOIN users u ON u.id=ts.shared_with_id
               WHERE ts.tree_id=? ORDER BY u.username''', (tree_id,)
        ).fetchall()]

    return render_template('family_tree.html',
                           tree=dict(tree),
                           my_cats=my_cats,
                           friend_cats=friend_cats,
                           relations=relations,
                           cat_names=cat_names,
                           rel_labels=rel_labels,
                           current_user_id=uid,
                           is_owner=is_owner,
                           share_friends=share_friends,
                           shared_with=shared_with)


@app.route('/family-tree/<int:tree_id>/share', methods=['POST'])
@login_required
def family_tree_share(tree_id):
    db = get_db()
    uid = session['user_id']
    tree = db.execute('SELECT 1 FROM family_trees WHERE id=? AND owner_id=?', (tree_id, uid)).fetchone()
    if not tree:
        return redirect(url_for('family_tree'))
    friend_id = request.form.get('friend_id', type=int)
    if friend_id:
        # Verify they are friends
        is_friend = db.execute(
            """SELECT 1 FROM friendships
               WHERE ((requester_id=? AND receiver_id=?) OR (requester_id=? AND receiver_id=?))
               AND status='accepted'""",
            (uid, friend_id, friend_id, uid)
        ).fetchone()
        if is_friend:
            db.execute('INSERT OR IGNORE INTO tree_shares (tree_id, shared_with_id) VALUES (?,?)',
                       (tree_id, friend_id))
            db.commit()
            push_notification(db, friend_id, 'tree_share', from_user_id=uid, tree_id=tree_id)
    return redirect(url_for('family_tree_view', tree_id=tree_id) + '#share')


@app.route('/family-tree/<int:tree_id>/unshare/<int:friend_id>', methods=['POST'])
@login_required
def family_tree_unshare(tree_id, friend_id):
    db = get_db()
    uid = session['user_id']
    tree = db.execute('SELECT 1 FROM family_trees WHERE id=? AND owner_id=?', (tree_id, uid)).fetchone()
    if tree:
        db.execute('DELETE FROM tree_shares WHERE tree_id=? AND shared_with_id=?', (tree_id, friend_id))
        db.commit()
    return redirect(url_for('family_tree_view', tree_id=tree_id) + '#share')


@app.route('/relation/add', methods=['POST'])
@login_required
def relation_add():
    cat_id    = request.form.get('cat_id', type=int)
    related_id = request.form.get('related_cat_id', type=int)
    relation  = request.form.get('relation', '').strip()
    tree_id   = request.form.get('tree_id', type=int)
    uid = session['user_id']
    db = get_db()

    if not cat_id or not related_id or cat_id == related_id or not tree_id:
        return jsonify({'error': 'invalid'}), 400
    if relation not in ('father', 'mother', 'sibling', 'child'):
        return jsonify({'error': 'invalid relation'}), 400
    if not db.execute('SELECT 1 FROM family_trees WHERE id=? AND owner_id=?',
                      (tree_id, uid)).fetchone():
        return jsonify({'error': 'access denied'}), 403

    accessible = {r['id'] for r in db.execute('SELECT id FROM cats WHERE user_id=?', (uid,)).fetchall()}
    for fr in db.execute(
        """SELECT u.id FROM friendships f
           JOIN users u ON u.id=CASE WHEN f.requester_id=? THEN f.receiver_id ELSE f.requester_id END
           WHERE (f.requester_id=? OR f.receiver_id=?) AND f.status='accepted'""",
        (uid, uid, uid)
    ).fetchall():
        for r in db.execute('SELECT id FROM cats WHERE user_id=?', (fr['id'],)).fetchall():
            accessible.add(r['id'])

    if cat_id not in accessible or related_id not in accessible:
        return jsonify({'error': 'access denied'}), 403

    try:
        db.execute(
            'INSERT OR IGNORE INTO cat_relations (cat_id, related_cat_id, relation, created_by, tree_id) VALUES (?,?,?,?,?)',
            (cat_id, related_id, relation, uid, tree_id))
        if relation == 'sibling':
            db.execute(
                'INSERT OR IGNORE INTO cat_relations (cat_id, related_cat_id, relation, created_by, tree_id) VALUES (?,?,?,?,?)',
                (related_id, cat_id, 'sibling', uid, tree_id))
        db.commit()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/relation/<int:rel_id>/delete', methods=['POST'])
@login_required
def relation_delete(rel_id):
    db = get_db()
    uid = session['user_id']
    row = db.execute('SELECT * FROM cat_relations WHERE id=?', (rel_id,)).fetchone()
    tree_id = row['tree_id'] if row else None
    if row and row['created_by'] == uid:
        db.execute('DELETE FROM cat_relations WHERE id=?', (rel_id,))
        if row['relation'] == 'sibling':
            db.execute(
                "DELETE FROM cat_relations WHERE cat_id=? AND related_cat_id=? AND relation='sibling' AND tree_id=?",
                (row['related_cat_id'], row['cat_id'], tree_id))
        db.commit()
    return redirect(url_for('family_tree_view', tree_id=tree_id) if tree_id else url_for('family_tree'))



# ═══════════════════════════════════════════════════════════════
#  ADMIN
# ═══════════════════════════════════════════════════════════════

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('is_admin'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if session.get('is_admin'):
        return redirect(url_for('admin_dashboard'))
    error = None
    if request.method == 'POST':
        u = request.form.get('username', '')
        p = request.form.get('password', '')
        s = request.form.get('secret', '')
        if ADMIN_SECRET and u == ADMIN_USERNAME and p == ADMIN_PASSWORD and s == ADMIN_SECRET:
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        error = 'פרטים שגויים'
    return render_template('admin_login.html', error=error)


@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('admin_login'))


@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    db = get_db()
    users = db.execute('''
        SELECT u.id, u.username, u.email,
               COUNT(DISTINCT c.id)  as cat_count,
               COUNT(DISTINCT po.id) as post_count,
               COUNT(DISTINCT f.id)  as friend_count
        FROM users u
        LEFT JOIN cats c  ON c.user_id  = u.id
        LEFT JOIN posts po ON po.user_id = u.id
        LEFT JOIN friendships f ON (f.requester_id=u.id OR f.receiver_id=u.id) AND f.status='accepted'
        GROUP BY u.id ORDER BY u.id
    ''').fetchall()

    friendships = db.execute('''
        SELECT f.id, u1.username as user1, u2.username as user2, f.status, f.created_at
        FROM friendships f
        JOIN users u1 ON u1.id = f.requester_id
        JOIN users u2 ON u2.id = f.receiver_id
        ORDER BY f.status, u1.username
    ''').fetchall()

    posts = db.execute('''
        SELECT p.id, p.caption, p.photo, p.created_at, u.username,
               COUNT(pc.id) as comment_count
        FROM posts p
        JOIN users u ON u.id = p.user_id
        LEFT JOIN post_comments pc ON pc.post_id = p.id
        GROUP BY p.id ORDER BY p.created_at DESC LIMIT 100
    ''').fetchall()

    logs = db.execute('''
        SELECT username, user_id, success, ip, logged_at
        FROM login_logs ORDER BY logged_at DESC LIMIT 200
    ''').fetchall()

    return render_template('admin_dashboard.html',
                           users=users,
                           friendships=friendships,
                           posts=posts,
                           logs=logs)


@app.route('/admin/user/<int:uid>/reset', methods=['POST'])
@admin_required
def admin_reset_password(uid):
    import secrets
    temp_pw = secrets.token_urlsafe(8)
    db = get_db()
    user = db.execute('SELECT username FROM users WHERE id=?', (uid,)).fetchone()
    db.execute('UPDATE users SET password=? WHERE id=?',
               (generate_password_hash(temp_pw), uid))
    db.commit()
    flash(f'סיסמה זמנית למשתמש "{user["username"]}": {temp_pw}', 'admin_info')
    return redirect(url_for('admin_dashboard') + '#users')


@app.route('/admin/user/<int:uid>/delete', methods=['POST'])
@admin_required
def admin_delete_user(uid):
    db = get_db()
    user = db.execute('SELECT username FROM users WHERE id=?', (uid,)).fetchone()
    if not user:
        return redirect(url_for('admin_dashboard'))
    # Delete files
    for row in db.execute(
            'SELECT cp.filename FROM cat_photos cp JOIN cats c ON c.id=cp.cat_id WHERE c.user_id=?', (uid,)):
        local_delete(row['filename'])
    for row in db.execute('SELECT photo FROM posts WHERE user_id=? AND photo IS NOT NULL', (uid,)):
        local_delete(row['photo'])
    # Delete DB rows
    db.execute('DELETE FROM post_comments WHERE user_id=?', (uid,))
    db.execute('DELETE FROM post_saves WHERE user_id=?', (uid,))
    db.execute('DELETE FROM posts WHERE user_id=?', (uid,))
    db.execute('DELETE FROM cat_relations WHERE created_by=?', (uid,))
    db.execute('DELETE FROM family_trees WHERE owner_id=?', (uid,))
    db.execute('DELETE FROM notifications WHERE user_id=? OR from_user_id=?', (uid, uid))
    db.execute('DELETE FROM messages WHERE sender_id=? OR receiver_id=?', (uid, uid))
    db.execute('DELETE FROM friendships WHERE requester_id=? OR receiver_id=?', (uid, uid))
    db.execute('DELETE FROM cat_photos WHERE cat_id IN (SELECT id FROM cats WHERE user_id=?)', (uid,))
    db.execute('DELETE FROM cat_details WHERE cat_id IN (SELECT id FROM cats WHERE user_id=?)', (uid,))
    db.execute('DELETE FROM cats WHERE user_id=?', (uid,))
    db.execute('DELETE FROM users WHERE id=?', (uid,))
    db.commit()
    flash(f'משתמש "{user["username"]}" נמחק', 'admin_info')
    return redirect(url_for('admin_dashboard') + '#users')


@app.route('/admin/post/<int:post_id>/delete', methods=['POST'])
@admin_required
def admin_delete_post(post_id):
    db = get_db()
    post = db.execute('SELECT * FROM posts WHERE id=?', (post_id,)).fetchone()
    if post:
        if post['photo']:
            local_delete(post['photo'])
        db.execute('DELETE FROM post_comments WHERE post_id=?', (post_id,))
        db.execute('DELETE FROM post_saves WHERE post_id=?', (post_id,))
        db.execute('DELETE FROM posts WHERE id=?', (post_id,))
        db.commit()
    return redirect(url_for('admin_dashboard') + '#posts')


@app.route('/admin/friendship/<int:fid>/delete', methods=['POST'])
@admin_required
def admin_delete_friendship(fid):
    db = get_db()
    db.execute('DELETE FROM friendships WHERE id=?', (fid,))
    db.commit()
    return redirect(url_for('admin_dashboard') + '#friends')


# ───────────────────────────── Street Cats ──────────────────────────────

def _next_street_cat_number(db):
    row = db.execute('SELECT MAX(auto_number) FROM street_cats').fetchone()
    return (row[0] or 0) + 1


@app.route('/street-cats/create', methods=['POST'])
@login_required
def street_cat_create():
    """Create a new street cat + first sighting linked to a post."""
    db = get_db()
    uid = session['user_id']
    nickname = request.form.get('nickname', '').strip() or None
    location = request.form.get('location', '').strip()
    fed = 1 if request.form.get('fed') else 0
    health = request.form.get('health_status', 'בריא')
    sighted_at = request.form.get('sighted_at') or None
    if not sighted_at:
        sighted_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    caption = request.form.get('caption', '').strip()
    photo = request.form.get('photo', '').strip() or None
    features_json = request.form.get('features_json', '').strip() or None

    # Upload temp photo to Cloudinary for permanent storage
    if photo and not photo.startswith('http'):
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], photo)
        cloud_url = upload_to_cloudinary(temp_path)
        if cloud_url:
            photo = cloud_url
            try:
                os.remove(temp_path)
            except Exception:
                pass

    auto_num = _next_street_cat_number(db)
    db.execute(
        'INSERT INTO street_cats (nickname, auto_number, created_by) VALUES (?,?,?)',
        (nickname, auto_num, uid)
    )
    sc_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]

    db.execute(
        "INSERT INTO posts (user_id, photo, caption, visibility, purpose) VALUES (?,?,?,?,?)",
        (uid, photo, caption, 'friends', 'street_cat')
    )
    post_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]

    db.execute(
        '''INSERT INTO street_cat_sightings
           (street_cat_id, post_id, user_id, location_text, fed, health_status, sighted_at, features)
           VALUES (?,?,?,?,?,?,?,?)''',
        (sc_id, post_id, uid, location, fed, health, sighted_at, features_json)
    )
    db.commit()
    flash(f'חתול הרחוב {nickname or "#" + str(auto_num)} נוסף!')
    return redirect(url_for('street_cat_profile', sc_id=sc_id))


@app.route('/street-cats/<int:sc_id>')
@login_required
def street_cat_profile(sc_id):
    db = get_db()
    sc = db.execute('SELECT * FROM street_cats WHERE id=?', (sc_id,)).fetchone()
    if not sc:
        return redirect(url_for('posts'))
    sightings = db.execute('''
        SELECT s.*, u.username, p.photo, p.caption
        FROM street_cat_sightings s
        JOIN users u ON u.id = s.user_id
        JOIN posts p ON p.id = s.post_id
        WHERE s.street_cat_id = ?
        ORDER BY s.sighted_at DESC
    ''', (sc_id,)).fetchall()
    adopted_cat = None
    if sc['adopted_by_cat_id']:
        adopted_cat = db.execute('SELECT * FROM cats WHERE id=?', (sc['adopted_by_cat_id'],)).fetchone()
    my_cats = db.execute('SELECT id, name FROM cats WHERE user_id=?', (session['user_id'],)).fetchall()
    return render_template('street_cat_profile.html', sc=sc, sightings=sightings, adopted_cat=adopted_cat, my_cats=my_cats)


@app.route('/street-cats/<int:sc_id>/add-sighting', methods=['POST'])
@login_required
def street_cat_add_sighting(sc_id):
    db = get_db()
    uid = session['user_id']
    sc = db.execute('SELECT id FROM street_cats WHERE id=?', (sc_id,)).fetchone()
    if not sc:
        return redirect(url_for('posts'))
    location = request.form.get('location', '').strip()
    fed = 1 if request.form.get('fed') else 0
    health = request.form.get('health_status', 'בריא')
    sighted_at = request.form.get('sighted_at') or None
    if not sighted_at:
        sighted_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    caption = request.form.get('caption', '').strip()
    photo = request.form.get('photo', '').strip() or None
    features_json = request.form.get('features_json', '').strip() or None

    # Handle direct file upload from the add-sighting form
    uploaded_file = request.files.get('photo_file')
    if uploaded_file and uploaded_file.filename:
        ext = uploaded_file.filename.rsplit('.', 1)[-1].lower() if '.' in uploaded_file.filename else 'jpg'
        temp_name = f'sighting_{uid}_{uuid.uuid4().hex[:8]}.{ext}'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
        uploaded_file.save(temp_path)
        cloud_url = upload_to_cloudinary(temp_path, folder='catbook/street_cats')
        if cloud_url:
            photo = cloud_url
        else:
            flash('העלאת התמונה נכשלה — ההופעה נשמרה ללא תמונה', 'warning')
        try:
            os.remove(temp_path)
        except Exception:
            pass
    # Upload temp photo to Cloudinary for permanent storage
    elif photo and not photo.startswith('http'):
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], photo)
        cloud_url = upload_to_cloudinary(temp_path)
        if cloud_url:
            photo = cloud_url
            try:
                os.remove(temp_path)
            except Exception:
                pass

    db.execute(
        "INSERT INTO posts (user_id, photo, caption, visibility, purpose) VALUES (?,?,?,?,?)",
        (uid, photo, caption, 'friends', 'street_cat')
    )
    post_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
    db.execute(
        '''INSERT INTO street_cat_sightings
           (street_cat_id, post_id, user_id, location_text, fed, health_status, sighted_at, features)
           VALUES (?,?,?,?,?,?,?,?)''',
        (sc_id, post_id, uid, location, fed, health, sighted_at, features_json)
    )
    # notify all previous reporters (excluding current user)
    sc_row = db.execute('SELECT created_by, nickname, auto_number FROM street_cats WHERE id=?', (sc_id,)).fetchone()
    if sc_row:
        name = sc_row['nickname'] or f'חתול רחוב #{sc_row["auto_number"]}'
        prev_reporters = db.execute('''
            SELECT DISTINCT user_id FROM street_cat_sightings
            WHERE street_cat_id=? AND user_id != ?
        ''', (sc_id, uid)).fetchall()
        notified = set()
        for r in prev_reporters:
            if r['user_id'] not in notified:
                db.execute(
                    "INSERT INTO notifications (user_id, type, message, related_id) VALUES (?,?,?,?)",
                    (r['user_id'], 'street_cat_sighting', f'{session["username"]} ראה את {name}', sc_id)
                )
                notified.add(r['user_id'])
    db.commit()
    flash('ההופעה נוספה!')
    return redirect(url_for('street_cat_profile', sc_id=sc_id))


@app.route('/street-cats/sightings/<int:sighting_id>/unlink', methods=['POST'])
@login_required
def street_cat_unlink_sighting(sighting_id):
    """Original reporter of the street cat can unlink a wrong sighting."""
    db = get_db()
    uid = session['user_id']
    sighting = db.execute('''
        SELECT s.*, sc.created_by, sc.id as sc_id
        FROM street_cat_sightings s
        JOIN street_cats sc ON sc.id = s.street_cat_id
        WHERE s.id=?
    ''', (sighting_id,)).fetchone()
    if not sighting or sighting['created_by'] != uid:
        flash('אין הרשאה')
        return redirect(url_for('posts'))
    # move sighting to a new street cat
    auto_num = _next_street_cat_number(db)
    db.execute('INSERT INTO street_cats (auto_number, created_by) VALUES (?,?)', (auto_num, sighting['user_id']))
    new_sc_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
    db.execute('UPDATE street_cat_sightings SET street_cat_id=? WHERE id=?', (new_sc_id, sighting_id))
    db.commit()
    flash('ההופעה נותקה ונוצר חתול רחוב חדש')
    return redirect(url_for('street_cat_profile', sc_id=sighting['sc_id']))


@app.route('/street-cats/<int:sc_id>/adopt', methods=['POST'])
@login_required
def street_cat_adopt(sc_id):
    """Link a street cat to an owned cat."""
    db = get_db()
    uid = session['user_id']
    cat_id = request.form.get('cat_id')
    cat = db.execute('SELECT id FROM cats WHERE id=? AND user_id=?', (cat_id, uid)).fetchone()
    if not cat:
        flash('חתול לא נמצא')
        return redirect(url_for('street_cat_profile', sc_id=sc_id))
    db.execute('UPDATE street_cats SET adopted_by_cat_id=? WHERE id=?', (cat_id, sc_id))
    db.commit()
    flash('החתול קושר לפרופיל שלך!')
    return redirect(url_for('street_cat_profile', sc_id=sc_id))


@app.route('/street-cats/<int:sc_id>/delete', methods=['POST'])
@login_required
def street_cat_delete(sc_id):
    db = get_db()
    uid = session['user_id']
    sc = db.execute('SELECT created_by FROM street_cats WHERE id=?', (sc_id,)).fetchone()
    if not sc:
        return redirect(url_for('street_cats_list'))
    is_admin = session.get('is_admin')
    if sc['created_by'] != uid and not is_admin:
        flash('אין הרשאה למחוק חתול רחוב זה')
        return redirect(url_for('street_cat_profile', sc_id=sc_id))
    # Delete sightings and their posts
    sightings = db.execute('SELECT post_id FROM street_cat_sightings WHERE street_cat_id=?', (sc_id,)).fetchall()
    for s in sightings:
        db.execute('DELETE FROM post_comments WHERE post_id=?', (s['post_id'],))
        db.execute('DELETE FROM post_saves WHERE post_id=?', (s['post_id'],))
        db.execute('DELETE FROM posts WHERE id=?', (s['post_id'],))
    db.execute('DELETE FROM street_cat_sightings WHERE street_cat_id=?', (sc_id,))
    db.execute('DELETE FROM street_cats WHERE id=?', (sc_id,))
    db.commit()
    flash('חתול הרחוב נמחק')
    return redirect(url_for('street_cats_list'))


@app.route('/street-cats')
@login_required
def street_cats_list():
    db = get_db()
    cats = db.execute('''
        SELECT sc.*, COUNT(s.id) as sighting_count,
               MAX(s.sighted_at) as last_seen,
               (SELECT p.photo FROM street_cat_sightings s2
                JOIN posts p ON p.id=s2.post_id
                WHERE s2.street_cat_id=sc.id ORDER BY s2.sighted_at DESC LIMIT 1) as last_photo
        FROM street_cats sc
        LEFT JOIN street_cat_sightings s ON s.street_cat_id=sc.id
        GROUP BY sc.id
        ORDER BY last_seen DESC
    ''').fetchall()
    return render_template('street_cats_list.html', cats=cats)


init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
