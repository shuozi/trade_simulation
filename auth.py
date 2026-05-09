"""
auth.py — Shared authentication layer for Dash apps
=====================================================
Adds a login page to any Dash/Flask app.

Usage in exercise.py / report.py:
    from auth import add_auth
    app = create_app()
    add_auth(app, password="your_password")
    app.run(...)

Or set via environment variable (preferred):
    export DASHBOARD_PASSWORD=your_password
    python exercise.py

If no password is set, the app runs without auth (localhost-only default).
"""

import os
import hashlib
import secrets
from functools import wraps
from flask import request, session, redirect, url_for, Response


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login — Trading Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #12121f;
    color: white;
    font-family: Inter, system-ui, sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
  }}
  .card {{
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 40px 48px;
    width: 360px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }}
  h2 {{
    font-size: 1.4rem;
    margin-bottom: 8px;
    color: #4fc3f7;
  }}
  p.sub {{
    color: #888;
    font-size: 0.85rem;
    margin-bottom: 28px;
  }}
  label {{
    display: block;
    font-size: 0.8rem;
    color: #aaa;
    margin-bottom: 6px;
  }}
  input[type=password] {{
    width: 100%;
    padding: 10px 14px;
    background: #2c2c3e;
    border: 1px solid #444;
    border-radius: 6px;
    color: white;
    font-size: 1rem;
    margin-bottom: 20px;
    outline: none;
    transition: border-color 0.2s;
  }}
  input[type=password]:focus {{ border-color: #4fc3f7; }}
  button {{
    width: 100%;
    padding: 11px;
    background: #27ae60;
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 1rem;
    font-weight: bold;
    cursor: pointer;
    transition: background 0.2s;
  }}
  button:hover {{ background: #219a52; }}
  .error {{
    background: rgba(231,76,60,0.15);
    border: 1px solid #e74c3c;
    border-radius: 6px;
    padding: 10px 14px;
    color: #e74c3c;
    font-size: 0.85rem;
    margin-bottom: 16px;
  }}
  .icon {{ font-size: 2.5rem; margin-bottom: 16px; }}
</style>
</head>
<body>
<div class="card">
  <div class="icon">📒</div>
  <h2>{title}</h2>
  <p class="sub">Personal access only. Enter your password to continue.</p>
  {error}
  <form method="POST" action="/login">
    <label>Password</label>
    <input type="password" name="password" autofocus placeholder="Enter password…">
    <input type="hidden" name="next" value="{next}">
    <button type="submit">Sign In →</button>
  </form>
</div>
</body>
</html>"""


def add_auth(app, password: str = None, title: str = "Trading Dashboard"):
    """
    Attach authentication to a Dash app.

    Parameters
    ----------
    app      : dash.Dash instance
    password : plaintext password (or set DASHBOARD_PASSWORD env var)
    title    : shown on the login page
    """
    pw = password or os.environ.get("DASHBOARD_PASSWORD", "")
    if not pw:
        print("  ⚠️  No password set — running without authentication.")
        print("     Set DASHBOARD_PASSWORD env var or pass --password to enable auth.")
        return

    pw_hash = _hash(pw)
    secret  = secrets.token_hex(32)

    server = app.server          # underlying Flask app
    server.secret_key = secret   # needed for sessions

    # ── Login route ──────────────────────────────────────────────────────────
    @server.route("/login", methods=["GET", "POST"])
    def login():
        error_html = ""
        next_url   = request.args.get("next", "/")

        if request.method == "POST":
            pw_input = request.form.get("password", "")
            next_url = request.form.get("next", "/")
            if _hash(pw_input) == pw_hash:
                session["authenticated"] = True
                session.permanent = True
                return redirect(next_url or "/")
            error_html = '<div class="error">Incorrect password. Try again.</div>'

        html = LOGIN_HTML.format(title=title, error=error_html,
                                  next=next_url)
        return Response(html, mimetype="text/html")

    # ── Logout route ─────────────────────────────────────────────────────────
    @server.route("/logout")
    def logout():
        session.clear()
        return redirect("/login")

    # ── Guard all routes ─────────────────────────────────────────────────────
    @server.before_request
    def require_login():
        public = {"/login", "/logout", "/favicon.ico"}
        # Allow Dash's internal asset/health endpoints
        if (request.path in public
                or request.path.startswith("/_dash")
                or request.path.startswith("/assets")):
            return
        if not session.get("authenticated"):
            return redirect(f"/login?next={request.path}")

    print(f"  🔒  Auth enabled — password protected.")
