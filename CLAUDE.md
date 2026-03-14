# CatBook — Project Overview

## What is this?
CatBook is a Hebrew RTL social network for cat owners. Built with Flask + SQLite.
Live at: https://catbook.pythonanywhere.com
Hosted on: PythonAnywhere (free tier, 512MB quota)
GitHub: https://github.com/markov-creator/catbook

## Stack
- Backend: Python/Flask
- DB: SQLite (`catbook.db`) — NOT in git
- Templates: Jinja2 (Hebrew, RTL, Rubik font)
- CSS: `static/style.css` (responsive, media queries for mobile)
- Cat identification: ONNX Runtime + `dinov2_small.onnx` (NOT in git, 84MB)
- Image uploads: `static/uploads/` — NOT in git

## Files NOT in Git (must upload manually to PythonAnywhere)
- `catbook.db` — SQLite database
- `static/uploads/` — user-uploaded images
- `dinov2_small.onnx` — DINOv2 ONNX model for cat identification
- `.env` — environment variables

## Environment Variables (.env)
```
SECRET_KEY=catbook-secret-2024
ADMIN_USERNAME=admin
ADMIN_PASSWORD=CatBook2024!
ADMIN_SECRET=secret77
```

## PythonAnywhere Setup
- Username: catbook
- Project path: `/home/catbook/catbook/`
- WSGI file: `/var/www/catbook_pythonanywhere_com_wsgi.py`
- Python version: 3.13

## Deploying Updates
**Local machine (Git Bash):**
```bash
cd "c:/claude workspace/catbook"
git add .
git commit -m "description"
git push
```
**PythonAnywhere Bash Console:**
```bash
cd ~/catbook
git pull
```
Then click **Reload** in the Web tab.

## Key Features
- User registration/login with login activity log
- Cat profiles with photos (multi-photo, grid display)
- Cat identification using DINOv2 ONNX (cosine similarity)
- Similar cat detection (notifies owner when similar cat uploaded)
- Friends system (send/accept requests, chat, shared cat viewing)
- Posts feed (photos + captions + comments, visibility settings)
- Family trees (multiple trees per user, sharing between friends, SVG visualization)
- Notifications system (identified, similar, tree_share types)
- Admin panel at `/admin` (hidden, 3-factor auth: username + password + secret token)

## Database Tables
users, cats, cat_photos, friendships, messages, notifications, posts, comments,
cat_relations, family_trees, tree_shares, login_logs, settings

## Important Notes
- All templates are Hebrew RTL
- Font: Rubik (Google Fonts)
- Cat identification: `(cat_id=A, related_cat_id=B, relation='father')` means "A's father is B"
- Admin session uses `session['is_admin']` separate from user session
- `static/uploads/` and `catbook.db` are in .gitignore — never overwrite on deploy
- Free PythonAnywhere: do NOT add torch/transformers to requirements.txt (too large)
- onnxruntime (~17MB) is used instead of torch
