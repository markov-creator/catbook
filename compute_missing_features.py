"""
compute_missing_features.py
----------------------------
Computes DINOv2 features for cat photos that have features=NULL in the DB.
Run locally (requires dinov2_small.onnx in the same folder as app.py).

Usage:
    python compute_missing_features.py
    python compute_missing_features.py --dry-run   # just show how many photos
"""

import argparse
import io
import json
import os
import sqlite3
import urllib.request

import numpy as np
from PIL import Image

DB_PATH = os.path.join(os.path.dirname(__file__), 'catbook.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dinov2_small.onnx')

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_session():
    import onnxruntime as ort
    return ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])


def preprocess(img):
    arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)


def extract_features(session, img: Image.Image):
    img = img.convert('RGB')
    w, h = img.size
    crops = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.crop((w // 10, h // 10, w * 9 // 10, h * 9 // 10)),
        img.crop((0, 0, w * 4 // 5, h)),
        img.crop((w // 5, 0, w, h)),
    ]
    batch = np.stack([preprocess(c) for c in crops])
    out = session.run(None, {'pixel_values': batch})[0]
    cls_tokens = out[:, 0, :]
    avg = cls_tokens.mean(axis=0)
    norm = np.linalg.norm(avg)
    return (avg / norm).tolist() if norm > 0 else avg.tolist()


def fetch_image(url_or_path: str) -> Image.Image:
    if url_or_path.startswith('http'):
        with urllib.request.urlopen(url_or_path, timeout=15) as resp:
            return Image.open(io.BytesIO(resp.read()))
    return Image.open(url_or_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Only count, do not update')
    parser.add_argument('--db', default=DB_PATH, help='Path to SQLite DB file')
    args = parser.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        print(f'DB not found: {db_path}')
        return
    if not os.path.exists(MODEL_PATH):
        print(f'Model not found: {MODEL_PATH}')
        return

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        'SELECT id, filename FROM cat_photos WHERE features IS NULL OR features = ""'
    ).fetchall()

    print(f'Photos without features: {len(rows)}')
    if args.dry_run or not rows:
        con.close()
        return

    session = get_session()
    ok = 0
    fail = 0

    for row in rows:
        photo_id = row['id']
        filename = row['filename']
        try:
            img = fetch_image(filename)
            feat = extract_features(session, img)
            con.execute(
                'UPDATE cat_photos SET features = ? WHERE id = ?',
                (json.dumps(feat), photo_id)
            )
            con.commit()
            ok += 1
            print(f'  [{ok}/{len(rows)}] ✓ photo {photo_id}')
        except Exception as e:
            fail += 1
            print(f'  ✗ photo {photo_id} ({filename[:60]}): {e}')

    con.close()
    print(f'\nDone: {ok} updated, {fail} failed')


if __name__ == '__main__':
    main()
