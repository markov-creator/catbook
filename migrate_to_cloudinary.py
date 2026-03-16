"""
Migrate local images to Cloudinary.
Run from: c:/claude workspace/catbook/
Requires: pip install requests
"""
import sqlite3
import os
import requests

CLOUD_NAME = 'ddo0urbwv'
UPLOAD_PRESET = 'catbook_upload'
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
DB_PATH = os.path.join(os.path.dirname(__file__), 'catbook.db')
UPLOAD_URL = f'https://api.cloudinary.com/v1_1/{CLOUD_NAME}/image/upload'


def upload_to_cloudinary(filepath):
    """Upload a file to Cloudinary, return secure_url or None."""
    if not os.path.exists(filepath):
        print(f'  FILE NOT FOUND: {filepath}')
        return None
    with open(filepath, 'rb') as f:
        res = requests.post(UPLOAD_URL, data={'upload_preset': UPLOAD_PRESET}, files={'file': f})
    if res.status_code == 200:
        return res.json()['secure_url']
    else:
        print(f'  ERROR {res.status_code}: {res.text[:200]}')
        return None


def migrate():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    updated = 0

    # --- cat_photos ---
    rows = conn.execute("SELECT id, filename FROM cat_photos WHERE filename NOT LIKE 'http%'").fetchall()
    print(f'\ncat_photos: {len(rows)} to migrate')
    for row in rows:
        filepath = os.path.join(UPLOADS_DIR, row['filename'])
        print(f'  Uploading {row["filename"]}...', end=' ')
        url = upload_to_cloudinary(filepath)
        if url:
            conn.execute('UPDATE cat_photos SET filename=? WHERE id=?', (url, row['id']))
            print(f'OK -> {url[:60]}...')
            updated += 1
        else:
            print('SKIPPED')

    # --- posts ---
    rows = conn.execute("SELECT id, photo FROM posts WHERE photo IS NOT NULL AND photo NOT LIKE 'http%'").fetchall()
    print(f'\nposts: {len(rows)} to migrate')
    for row in rows:
        filepath = os.path.join(UPLOADS_DIR, row['photo'])
        print(f'  Uploading {row["photo"]}...', end=' ')
        url = upload_to_cloudinary(filepath)
        if url:
            conn.execute('UPDATE posts SET photo=? WHERE id=?', (url, row['id']))
            print(f'OK -> {url[:60]}...')
            updated += 1
        else:
            print('SKIPPED')

    # --- messages ---
    rows = conn.execute("SELECT id, image FROM messages WHERE image IS NOT NULL AND image NOT LIKE 'http%'").fetchall()
    print(f'\nmessages: {len(rows)} to migrate')
    for row in rows:
        filepath = os.path.join(UPLOADS_DIR, row['image'])
        print(f'  Uploading {row["image"]}...', end=' ')
        url = upload_to_cloudinary(filepath)
        if url:
            conn.execute('UPDATE messages SET image=? WHERE id=?', (url, row['id']))
            print(f'OK -> {url[:60]}...')
            updated += 1
        else:
            print('SKIPPED')

    conn.commit()
    conn.close()
    print(f'\nDone! Updated {updated} records.')


if __name__ == '__main__':
    migrate()
