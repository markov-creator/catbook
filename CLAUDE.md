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
git add .
git commit -m "description"
git push
```
**PythonAnywhere Bash Console:**
```bash
cd ~/catbook && git fetch origin && git checkout origin/master -- app.py templates/
```
Then click **Reload** in the Web tab.

> ⚠️ Do NOT use `git pull` — it fails when `dinov2_small.onnx` is present (untracked, blocks merge). Use `git checkout origin/master -- <files>` instead.

## Architecture
Single-file Flask app (`app.py`, ~2000 lines) with SQLite. All routes, DB init, and helpers are in `app.py`.

**Request flow:**
1. `before_request` → `inject_nav_counts()` injects `pending_requests`, `unread_messages`, `unread_notifs` into every template via `g`
2. `@login_required` decorator guards user routes; `@admin_required` guards `/admin/*`
3. DB connection via `get_db()` (stored on `g`, closed after request)

**Cat identification pipeline:**
- Upload → `extract_features()` → DINOv2-small ONNX (`dinov2_small.onnx`, 84MB, NOT in git) → cosine similarity against stored features → `find_similar_cat()` notifies owner if match found
- Similarity threshold: 0.55; confidence = 60% absolute score + 40% margin

**Family tree layout (JS in `templates/family_tree.html`):**
- BFS assigns generations (parents above children)
- Union-Find merges sibling groups
- T-junction SVG connectors drawn per family group
- Relation semantics: `(cat_id=A, related_cat_id=B, relation='father')` = "A's father is B"

## Image Storage (Cloudinary)
New uploads go directly from the browser to Cloudinary (Upload Widget), bypassing the server — required because PythonAnywhere free tier blocks outbound HTTP.

- Cloud name: `ddo0urbwv` (`CLOUDINARY_CLOUD_NAME` env var)
- Upload preset: `catbook_upload` (`CLOUDINARY_UPLOAD_PRESET` env var) — unsigned
- **`photo_url(filename)`** — Jinja2 global: if value starts with `http` → Cloudinary URL as-is; otherwise → `/static/uploads/<filename>` (legacy)

## Email
Uses `smtplib` (built-in). `send_email(to, subject, body)` runs in a background thread — silently skips if `MAIL_USER`/`MAIL_PASSWORD` not set.

Triggered on: friend request received, cat identified (owner notified), post sent to specific friend(s).

> ⚠️ PythonAnywhere free tier blocks outbound SMTP (ports 465 and 587 to smtp.gmail.com are unreachable). Email works locally but **not on the server**. Planned fix: switch to SendGrid or Mailgun HTTP API (whitelisted by PythonAnywhere).

## Key Conventions
- All UI is Hebrew RTL; font is Rubik (Google Fonts)
- Flash categories: `'warning'` gets yellow style; `'admin_*'` shown only in admin dashboard
- Admin session: `session['is_admin']` (separate from `session['user_id']`)
- Notifications: types are `identified`, `similar`, `tree_share`
- Username inputs use `dir="auto"`; password inputs use `dir="ltr"`

## Database Tables
`users` (with `email`), `cats`, `cat_photos` (with `features` JSON), `friendships`, `notifications`, `messages`, `cat_details`, `shared_details`, `details_history`, `posts`, `post_comments`, `post_saves`, `cat_relations`, `family_trees`, `tree_shares`, `settings`, `login_logs`

Schema initialized in `init_db()`. Migrations run automatically on startup.

## Files NOT in Git (`.gitignore`)
| File | Why |
|------|-----|
| `catbook.db` | SQLite DB |
| `static/uploads/` | Legacy local images |
| `*.onnx` | 84MB model — copy manually to server |
| `static/uploads.zip` | Backup archive |
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
MAIL_USER=...@gmail.com
MAIL_PASSWORD=...   # Gmail App Password (16 chars)
```
