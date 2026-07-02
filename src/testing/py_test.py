from flask import Flask, request, render_template_string
import sqlite3, os

app = Flask(__name__)

@app.route("/search")
def search():
    query = request.args.get("q")          # taint source
    conn  = sqlite3.connect("users.db")
    cur   = conn.cursor()
    # VULNERABLE: string concatenation into SQL sink
    cur.execute("SELECT * FROM users WHERE name = '" + query + "'")
    rows = cur.fetchall()
    return render_template_string("<ul>{% for r in rows %}<li>{{r}}</li>{% endfor %}</ul>",
                                   rows=rows)

@app.route("/safe_search")
def safe_search():
    query = request.args.get("q")          # taint source
    conn  = sqlite3.connect("users.db")
    cur   = conn.cursor()
    # SAFE: parameterized query (sanitizer)
    cur.execute("SELECT * FROM users WHERE name = ?", (query,))
    rows = cur.fetchall()
    return str(rows)