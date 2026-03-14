# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
CatBook — Hebrew RTL social network for cat owners.
- Live: https://catbook.pythonanywhere.com
- Host: PythonAnywhere free tier (512MB quota, Python 3.13)
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
cd ~/catbook && git pull
```
Then click **Reload** in the Web tab.

## Architecture
Single-file Flask app (`app.py`) with SQLite. All routes, DB init, and helpers are in `app.py`.

**Request flow:**
1. `before_request` → `inject_nav_counts()` injects `pending_requests`, `unread_messages`, `unread_notifs` into every template via `g`
2. `@login_required` decorator guards user routes; `@admin_required` guards `/admin/*`
3. DB connection via `get_db()` (stored on `g`, closed after request)

**Cat identification pipeline:**
- Upload → `extract_features()` → DINOv2-small ONNX (`dinov2_small.onnx`, 84MB, NOT in git) → cosine similarity against stored features → `find_similar_cat()` notifies owner if match found

**Family tree layout (JS in `templates/family_tree.html`):**
- BFS assigns generations (parents above children)
- Union-Find merges sibling groups
- T-junction SVG connectors drawn per family group
- Relation semantics: `(cat_id=A, related_cat_id=B, relation='father')` = "A's father is B"

## Key Conventions
- All UI is Hebrew RTL; font is Rubik (Google Fonts)
- Flash categories: `'warning'` gets yellow style; `'admin_*'` shown only in admin dashboard
- Admin session: `session['is_admin']` (separate from `session['user_id']`)
- Notifications: types are `identified`, `similar`, `tree_share`
- Images saved locally to `static/uploads/` as JPEG (quality 85)

## Files NOT in Git
| File | Why |
|------|-----|
| `catbook.db` | SQLite DB with all user data |
| `static/uploads/` | User-uploaded images |
| `dinov2_small.onnx` | 84MB ONNX model |
| `.env` | Secrets |

## PythonAnywhere Config
- Project path: `/home/catbook/catbook/`
- WSGI: `/var/www/catbook_pythonanywhere_com_wsgi.py`
- **Do NOT add `torch` or `transformers` to requirements.txt** — too large for free tier (915MB). Use `onnxruntime` instead.

## .env Variables
```
SECRET_KEY=catbook-secret-2024
ADMIN_USERNAME=admin
ADMIN_PASSWORD=CatBook2024!
ADMIN_SECRET=secret77
```
