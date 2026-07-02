"""
python_feature_extractor.py
Extracts source-code-level features from a Python web application file
(Flask / Django) for use in a vulnerable/non-vulnerable ML classifier.

Mirrors the feature schema of feature_extractor.py (Java version) so that
both languages feed the same classifier columns.
"""

import re
import ast
from collections import defaultdict

# ---------------------------------------------------------------------------
# Keyword / pattern definitions  (Python equivalents of Java patterns)
# ---------------------------------------------------------------------------

TAINT_SOURCES = [
    # Flask
    "request.args", "request.form", "request.json",
    "request.data", "request.files", "request.cookies",
    "request.values", "request.get_json", "request.stream",
    # Django
    "request.GET", "request.POST", "request.FILES",
    "request.COOKIES", "request.body", "request.META",
    # Generic
    "input(", "sys.argv", "os.environ",
]

SQL_SINKS    = ["execute", "executemany", "raw(", "cursor.execute",
                "engine.execute", "session.execute", "text("]
CMD_SINKS    = ["os.system", "subprocess.run", "subprocess.Popen",
                "subprocess.call", "subprocess.check_output",
                "os.popen", "os.execv", "os.execve", "commands.getoutput"]
XSS_SINKS    = ["render_template_string", "Markup(", "make_response",
                "Response(", "jsonify", "send_file"]
PATH_SINKS   = ["open(", "os.path.join", "pathlib.Path",
                "os.listdir", "os.remove", "shutil.copy",
                "io.open", "os.rename"]
LDAP_SINKS   = ["ldap.search", "connection.search", "ldap3.search"]
XXE_SINKS    = ["etree.parse", "ElementTree.parse", "minidom.parse",
                "xml.dom.parse", "lxml.etree", "BeautifulSoup"]
WEAK_CRYPTO_SINKS = ["md5(", "sha1(", "DES", "hashlib.md5",
                     "hashlib.sha1", "Crypto.Cipher.DES",
                     "ECB", "ARC4", "RC4"]

SQL_SANITIZERS  = ["?", "%s", ":param", "bindparams", "text(",
                   "sqlalchemy.text", "django.db.connection.cursor"]
XSS_SANITIZERS  = ["escape(", "bleach.clean", "markupsafe.escape",
                   "html.escape", "cgi.escape", "django.utils.html.escape"]
PATH_SANITIZERS = ["os.path.abspath", "os.path.realpath",
                   "pathlib.Path.resolve", "secure_filename"]
CMD_SANITIZERS  = ["shlex.quote", "shlex.split"]
GENERIC_SANITIZERS = ["escape", "sanitize", "validate", "clean",
                      "encode", "bleach", "markupsafe"]


# ---------------------------------------------------------------------------
# Helper: flatten source for fast substring scans
# ---------------------------------------------------------------------------

def _flatten_source(source_code: str) -> str:
    return re.sub(r"\s+", " ", source_code)


# ---------------------------------------------------------------------------
# AST visitor: count functions and try/except blocks
# ---------------------------------------------------------------------------

class _StructureVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_count = 0
        self.try_count = 0

    def visit_FunctionDef(self, node):
        self.function_count += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.function_count += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.try_count += 1
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# AST-based taint tracer (Python)
# ---------------------------------------------------------------------------

class _TaintVisitor(ast.NodeVisitor):
    """
    Lightweight single-pass taint tracker.
    Tracks variable assignments that originate from taint sources,
    then checks whether those tainted variables reach a sink,
    and whether a sanitizer appears between source and sink.
    """

    def __init__(self, source_lines):
        self.source_lines = source_lines
        # var_name -> line number where taint was assigned
        self.tainted_vars: dict[str, int] = {}
        self.sink_hits: list[dict] = []         # {sink_type, line, tainted, sanitized}
        self.sanitizer_lines: list[int] = []

        # Quick lookup sets
        self._source_kws  = set(TAINT_SOURCES)
        self._sink_map = {
            "sql":  set(SQL_SINKS),
            "cmd":  set(CMD_SINKS),
            "xss":  set(XSS_SINKS),
            "path": set(PATH_SINKS),
            "ldap": set(LDAP_SINKS),
            "xxe":  set(XXE_SINKS),
        }
        self._sanitizer_kws = set(
            SQL_SANITIZERS + XSS_SANITIZERS + PATH_SANITIZERS +
            CMD_SANITIZERS + GENERIC_SANITIZERS
        )

    # ------------------------------------------------------------------
    # Utility: get source text of a node
    # ------------------------------------------------------------------
    def _node_src(self, node) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Utility: check if a string contains any pattern from a set
    # ------------------------------------------------------------------
    @staticmethod
    def _matches(text: str, patterns) -> bool:
        return any(p in text for p in patterns)

    # ------------------------------------------------------------------
    # Assignments: x = request.args.get("q")  →  x is tainted
    # ------------------------------------------------------------------
    def visit_Assign(self, node):
        val_src = self._node_src(node.value)
        if self._matches(val_src, self._source_kws):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.tainted_vars[target.id] = node.lineno
        # Also propagate taint through aliases:  y = x  (if x is tainted)
        if isinstance(node.value, ast.Name) and node.value.id in self.tainted_vars:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.tainted_vars[target.id] = node.lineno
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value:
            val_src = self._node_src(node.value)
            if self._matches(val_src, self._source_kws):
                if isinstance(node.target, ast.Name):
                    self.tainted_vars[node.target.id] = node.lineno
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Function calls: cursor.execute(query)  →  check for sink + taint
    # ------------------------------------------------------------------
    def visit_Call(self, node):
        call_src = self._node_src(node)
        lineno   = node.lineno

        # Check sanitizer
        if self._matches(call_src, self._sanitizer_kws):
            self.sanitizer_lines.append(lineno)

        # Check each sink family
        for sink_type, patterns in self._sink_map.items():
            if self._matches(call_src, patterns):
                # Is any argument tainted?
                tainted = False
                for arg in ast.walk(node):
                    if isinstance(arg, ast.Name) and arg.id in self.tainted_vars:
                        tainted = True
                        break
                    if isinstance(arg, (ast.Constant, ast.JoinedStr)):
                        arg_src = self._node_src(arg)
                        if self._matches(arg_src, self._source_kws):
                            tainted = True
                            break

                # Is there a sanitizer before this sink?
                sanitized = any(sl < lineno for sl in self.sanitizer_lines)

                self.hit_sink(sink_type, lineno, tainted, sanitized)

        self.generic_visit(node)

    def hit_sink(self, sink_type, line, tainted, sanitized):
        self.sink_hits.append({
            "sink_type": sink_type,
            "line": line,
            "tainted": tainted,
            "sanitized": sanitized,
        })


def _run_taint_visitor(source_code: str, lines: list[str]) -> dict:
    # Column names must exactly match what the Java taint_tracer.py produces
    # so the same trained classifier works for both languages.
    features = {
        # Per-category: did a tainted variable reach this sink?
        "sqli_tainted_sink_reached":      0,
        "sqli_sanitized_before_sink":     0,
        "cmdi_tainted_sink_reached":      0,
        "cmdi_sanitized_before_sink":     0,
        "pathtraver_tainted_sink_reached":0,
        "pathtraver_sanitized_before_sink":0,
        "ldapi_tainted_sink_reached":     0,
        "ldapi_sanitized_before_sink":    0,
        "xpathi_tainted_sink_reached":    0,   # XPath injection (no Python sink yet, kept for schema)
        "xpathi_sanitized_before_sink":   0,
        # Aggregate helpers
        "taint_reached_sink":             0,
        "taint_reached_sink_unsanitized": 0,
        "sanitizer_before_sink_count":    0,
        "tainted_var_count":              0,
        # Required by classifier — 0 means parse succeeded
        "taint_trace_parse_error":        0,
    }

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        features["taint_trace_parse_error"] = 1
        return features

    visitor = _TaintVisitor(lines)
    visitor.visit(tree)

    features["tainted_var_count"] = len(visitor.tainted_vars)
    features["sanitizer_before_sink_count"] = sum(
        1 for h in visitor.sink_hits if h["sanitized"]
    )

    # Map internal sink_type names → classifier column prefix
    SINK_TYPE_TO_PREFIX = {
        "sql":  "sqli",
        "cmd":  "cmdi",
        "path": "pathtraver",
        "ldap": "ldapi",
        "xss":  None,   # XSS taint columns not in classifier schema; skip
        "xxe":  None,   # XXE taint columns not in classifier schema; skip
    }

    for hit in visitor.sink_hits:
        prefix = SINK_TYPE_TO_PREFIX.get(hit["sink_type"])
        if prefix is None:
            continue
        if hit["tainted"]:
            features["taint_reached_sink"] = 1
            features[f"{prefix}_tainted_sink_reached"] = 1
            if hit["sanitized"]:
                features[f"{prefix}_sanitized_before_sink"] = 1
            else:
                features["taint_reached_sink_unsanitized"] = 1

    return features


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_python_features(source_code: str, file_name: str = "uploaded.py") -> dict:
    """
    Returns a flat feature dict with the same column names as the Java extractor
    so both languages can feed the same trained classifier.
    """
    features = {"file_name": file_name, "language": "python"}

    flat  = _flatten_source(source_code)
    lines = source_code.splitlines()

    # ------------------------------------------------------------------
    # 1. AST parse
    # ------------------------------------------------------------------
    tree   = None
    ast_ok = True
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        ast_ok = False
        features["ast_parse_error"] = str(e)
    features["ast_parse_success"] = int(ast_ok)

    # ------------------------------------------------------------------
    # 2. Taint source / sink / sanitizer keyword counts
    # ------------------------------------------------------------------
    def count_any(patterns):
        return sum(flat.count(p) for p in patterns)

    features["taint_source_count"]      = count_any(TAINT_SOURCES)
    features["sql_sink_count"]          = count_any(SQL_SINKS)
    features["cmd_sink_count"]          = count_any(CMD_SINKS)
    features["xss_sink_count"]          = count_any(XSS_SINKS)
    features["path_sink_count"]         = count_any(PATH_SINKS)
    features["ldap_sink_count"]         = count_any(LDAP_SINKS)
    features["xxe_sink_count"]          = count_any(XXE_SINKS)
    features["weak_crypto_sink_count"]  = count_any(WEAK_CRYPTO_SINKS)

    features["sql_sanitizer_count"]     = count_any(SQL_SANITIZERS)
    features["xss_sanitizer_count"]     = count_any(XSS_SANITIZERS)
    features["path_sanitizer_count"]    = count_any(PATH_SANITIZERS)
    features["generic_sanitizer_count"] = count_any(GENERIC_SANITIZERS)

    total_sinks = sum([
        features["sql_sink_count"], features["cmd_sink_count"],
        features["xss_sink_count"], features["path_sink_count"],
        features["ldap_sink_count"], features["xxe_sink_count"],
    ])
    total_sanitizers = sum([
        features["sql_sanitizer_count"], features["xss_sanitizer_count"],
        features["path_sanitizer_count"], features["generic_sanitizer_count"],
    ])

    features["total_sink_count"]       = total_sinks
    features["total_sanitizer_count"]  = total_sanitizers
    features["sanitizer_to_sink_ratio"] = round(total_sanitizers / total_sinks, 3) if total_sinks else 0
    features["has_any_sink"]           = int(total_sinks > 0)
    features["has_any_sanitizer"]      = int(total_sanitizers > 0)

    # ------------------------------------------------------------------
    # 3. String concatenation / f-string near a sink
    #    Python uses f-strings and % formatting too, not just +
    # ------------------------------------------------------------------
    concat_near_sink = 0
    fstring_near_sink = 0
    percent_fmt_near_sink = 0

    for i, line in enumerate(lines):
        window = " ".join(lines[max(0, i - 1): i + 2])
        has_sink_nearby = any(s in window for s in SQL_SINKS + CMD_SINKS)

        if has_sink_nearby:
            if "+" in line:
                concat_near_sink += 1
            if "f\"" in line or "f'" in line:
                fstring_near_sink += 1
            if "%" in line and ("(" in line):
                percent_fmt_near_sink += 1

    features["string_concat_near_sink_count"]    = concat_near_sink
    features["fstring_near_sink_count"]          = fstring_near_sink
    features["percent_format_near_sink_count"]   = percent_fmt_near_sink
    # Combined injection smell score (any unsafe formatting near a sink)
    features["unsafe_format_near_sink_count"]    = (
        concat_near_sink + fstring_near_sink + percent_fmt_near_sink
    )

    # ------------------------------------------------------------------
    # 4. Distance between first taint source and first sink (in lines)
    # ------------------------------------------------------------------
    source_line = -1
    sink_line   = -1
    for i, line in enumerate(lines):
        if source_line == -1 and any(s in line for s in TAINT_SOURCES):
            source_line = i
        if sink_line == -1 and any(s in line for s in SQL_SINKS + CMD_SINKS + PATH_SINKS):
            sink_line = i

    if source_line != -1 and sink_line != -1:
        features["source_to_sink_line_distance"] = abs(sink_line - source_line)
        features["source_before_sink"]           = int(source_line < sink_line)
    else:
        features["source_to_sink_line_distance"] = -1
        features["source_before_sink"]           = 0

    # ------------------------------------------------------------------
    # 5. Variable aliasing depth
    #    Python assignments don't end in ";" so we adjust the regex
    # ------------------------------------------------------------------
    # Matches:  x = y   (simple name-to-name assignment)
    assignment_pattern = re.compile(r"\b(\w+)\s*=\s*(\w+)\b")
    alias_hops = len(assignment_pattern.findall(flat))
    features["variable_alias_hops"] = alias_hops

    # ------------------------------------------------------------------
    # 6. AST structural features
    # ------------------------------------------------------------------
    if ast_ok and tree is not None:
        visitor = _StructureVisitor()
        visitor.visit(tree)
        features["method_count"]    = visitor.function_count
        features["try_catch_count"] = visitor.try_count
    else:
        features["method_count"]    = -1
        features["try_catch_count"] = -1

    # ------------------------------------------------------------------
    # 7. General code stats
    # ------------------------------------------------------------------
    features["total_lines"]          = len(lines)
    features["total_chars"]          = len(source_code)
    features["import_count"]         = flat.count("import ")
    features["string_literal_count"] = len(re.findall(r'"[^"]*"|\'[^\']*\'', source_code))

    # Python-specific extras that are still useful for the classifier
    features["decorator_count"]      = source_code.count("@")          # @app.route, @login_required
    features["assert_count"]         = flat.count("assert ")           # validation smell
    features["eval_count"]           = flat.count("eval(")             # dangerous eval
    features["exec_count"]           = flat.count("exec(")             # dangerous exec

    # ------------------------------------------------------------------
    # 8. AST-based taint tracing (source → alias → sink, sanitizer-aware)
    # ------------------------------------------------------------------
    taint_features = _run_taint_visitor(source_code, lines)
    features.update(taint_features)

    return features


# ---------------------------------------------------------------------------
# Category hint (same interface as Java extractor)
# ---------------------------------------------------------------------------

def extract_features_for_category_hints(features: dict) -> dict:
    sink_scores = {
        "sqli":       features.get("sql_sink_count", 0),
        "cmdi":       features.get("cmd_sink_count", 0),
        "xss":        features.get("xss_sink_count", 0),
        "pathtraver": features.get("path_sink_count", 0),
        "ldapi":      features.get("ldap_sink_count", 0),
        "xxe":        features.get("xxe_sink_count", 0),
        "weak_crypto":features.get("weak_crypto_sink_count", 0),
    }
    likely = max(sink_scores, key=sink_scores.get) if any(sink_scores.values()) else "none_detected"
    return {"likely_category_hint": likely, "sink_score_breakdown": sink_scores}


# ---------------------------------------------------------------------------
# Universal dispatcher  (drop-in replacement in your pipeline)
# ---------------------------------------------------------------------------

# def extract_features_universal(source_code: str, file_name: str) -> dict:
#     """
#     Auto-detects language by file extension and routes to the correct extractor.
#     Both extractors return the same feature column names so one classifier works
#     for both languages.
#     """
#     if file_name.endswith(".py"):
#         return extract_python_features(source_code, file_name)
#     elif file_name.endswith(".java"):
#         # Import the Java extractor only when needed
#         from feature_extractor import extract_features as java_extract
#         return java_extract(source_code, file_name)
#     else:
#         raise ValueError(f"Unsupported file type: {file_name}")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = '''
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
'''
    import json
    feats = extract_python_features(SAMPLE, "sample_flask.py")
    hints = extract_features_for_category_hints(feats)
    print(json.dumps({**feats, **hints}, indent=2))