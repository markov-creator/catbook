# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
CatBook — Hebrew RTL social network for cat owners.
- Live: https://catbook.pythonanywhere.com
- Host: PythonAnywhere free tier (512MB quota, Python 3.10)
- GitHub: https://github.com/markov-creator/catbook

## Running Locally
```bash
cd "c:/claude workspace/catbook"
pip install -r requirements.txt
python app.py
```
App runs on http://localhost:5000. DB and uploads are created automatically on first run.

## Deploying to PythonAnywhere
**Local (Git Bash):**
```bash
git add <files>
git commit -m "description"
git push
```
**PythonAnywhere Bash Console:**
```bash
cd ~/catbook && git fetch origin && git checkout origin/master -- app.py static/main.js templates/
```
Then click **Reload** in the Web tab.

> ⚠️ Do NOT use `git pull` — it fails when `dinov2_small.onnx` is present (untracked, blocks merge). Use `git checkout origin/master -- <files>` instead.

## Architecture
Single-file Flask app (`app.py`, ~2600 lines) with SQLite. All routes, DB init, and helpers are in `app.py`.

**Request flow:**
1. `inject_globals()` context processor injects `pending_requests`, `unread_messages`, `unread_notifs` into every template
2. `@login_required` decorator guards user routes; `@admin_required` guards `/admin/*`
3. DB connection via `get_db()` (stored on `g`, closed after request)

**Cat identification pipeline:**
- Identify page: user uploads photo (full/original) → server `extract_features()` → cosine similarity via `get_or_compute_features()` → top-3 average score → threshold 0.55
- Add photo flow: browser sends **cropped** blob to `/api/extract-features` (AI features) + **original** blob to Cloudinary (display) in parallel → form submits Cloudinary URL + feature token → server links them via `feature_tokens` table
- `get_or_compute_features(db, photo_id, filename)` — returns cached features or downloads + computes on demand
- Scoring: `score = mean(top_3_scores)`; confidence = 60% absolute + 40% margin
- `find_similar_cat()` — checks new upload against other users' cats; notifies owner if similarity ≥ 0.55

**Family tree layout (JS in `templates/family_tree.html`):**
- BFS assigns generations (parents above children)
- Union-Find merges sibling groups
- T-junction SVG connectors drawn per family group
- Relation semantics: `(cat_id=A, related_cat_id=B, relation='father')` = "A's father is B"

## Image Upload Flow (Cropper.js)
All cat photo uploads go through a crop modal before uploading:
1. User selects file → `showCropModal(file, callback)` in `main.js`
2. Crop confirmed → callback receives `croppedBlob`
3. Two parallel requests: `croppedBlob` → `/api/extract-features` (returns token) + `originalFile` → Cloudinary (returns URL)
4. Form submits `photo_url` + `feature_token` → server looks up features from `feature_tokens` table and saves to `cat_photos.features`

**Identify page:** also sends the **cropped** image to the server — consistent with stored features which are also computed from cropped images.

## Image Storage (Cloudinary)
- Cloud name: `ddo0urbwv` (`CLOUDINARY_CLOUD_NAME` env var)
- Upload preset: `catbook_upload` (`CLOUDINARY_UPLOAD_PRESET` env var) — unsigned
- **`photo_url(filename)`** — Jinja2 global: if value starts with `http` → Cloudinary URL as-is; otherwise → `/static/uploads/<filename>` (legacy)
- `static/uploads/` holds temp files from identify sessions (named `temp_<uid>_<hash>.<ext>`)
- **`upload_to_cloudinary(file_path, folder)`** — server-side signed upload (used for street cat sighting photos uploaded directly from the profile form). Requires `CLOUDINARY_API_KEY` + `CLOUDINARY_API_SECRET`.

## Email
Uses SendGrid HTTP API (via `urllib.request`). `send_email(to, subject, body)` runs in a background thread — silently skips if `SENDGRID_API_KEY`/`MAIL_SENDER` not set. Sender must be verified as a Single Sender in SendGrid dashboard.

## Street Cats
Community tracking system for unowned street cats (`/street-cats`).
- `street_cats` table: one row per cat, with `nickname`, `auto_number`, `created_by`, `adopted_by_cat_id`
- `street_cat_sightings` table: links to a `posts` row (for photo/caption) + sighting metadata (location, health, fed, sighted_at)
- On identify, results are checked against street cat sightings (threshold 0.55). Similarity ≥ 0.85 auto-links the identified cat.
- On new sighting, notifications sent to all previous reporters (not just creator).
- Delete permission: first reporter (`created_by`) or admin only.
- Sighting photos uploaded directly from profile page go through server-side `upload_to_cloudinary()`.

## Real-time Nav Badges
`/api/nav-counts` returns `{pending_requests, unread_messages, unread_notifs}` as JSON.
`main.js` polls this endpoint every 20 seconds and updates the navbar badges without page reload.

## Key Conventions
- All UI is Hebrew RTL; font is Rubik (Google Fonts)
- Flash categories: `'warning'` → yellow style; `'admin_*'` → admin dashboard only
- Admin session: `session['is_admin']` (separate from `session['user_id']`)
- Notification types: `identified`, `similar`, `tree_share`, `street_cat_sighting`
- Username inputs use `dir="auto"`; password inputs use `dir="ltr"`
- After login → redirects to `index` (home page)
- Cloudinary uploads use `folder: 'catbook/{{ session.user_id }}'`

## Database Tables
`users` (with `email`, `home_bg`), `cats`, `cat_photos` (with `features` JSON), `friendships`, `notifications` (with `message`, `related_id`), `messages`, `cat_details`, `shared_details`, `details_history`, `posts`, `post_comments`, `post_saves`, `cat_relations`, `family_trees`, `tree_shares`, `settings`, `login_logs`, `feature_tokens`, `street_cats`, `street_cat_sightings`

Schema initialized in `init_db()`. Migrations run automatically on startup.

## Utility Scripts
- `compute_missing_features.py` — backfills `cat_photos.features` for photos with NULL. Supports `--db <path>` and `--dry-run`. Run locally (requires `dinov2_small.onnx`).

## Files NOT in Git (`.gitignore`)
| File | Why |
|------|-----|
| `catbook.db` | SQLite DB |
| `static/uploads/` | Temp identify images + legacy local photos |
| `*.onnx` | 84MB model — copy manually to server |
| `.env` | Secrets |

## PythonAnywhere Config
- Project path: `/home/catbook/catbook/`
- WSGI: `/var/www/catbook_pythonanywhere_com_wsgi.py` — `project_home = '/home/catbook/catbook'`
- Python version: **3.10**
- **Do NOT add `torch` or `transformers` to requirements.txt** — too large (915MB). Use `onnxruntime`.

## .env Variables
```
SECRET_KEY=...
ADMIN_USERNAME=admin
ADMIN_PASSWORD=...
ADMIN_SECRET=...
CLOUDINARY_CLOUD_NAME=ddo0urbwv
CLOUDINARY_UPLOAD_PRESET=catbook_upload
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
SENDGRID_API_KEY=...
MAIL_SENDER=...@gmail.com
```
