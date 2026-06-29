"""
feature_extractor.py
Extracts source-code-level features from a single Java servlet file
(matching OWASP Benchmark's one-test-case-per-file structure) for use
in a vulnerable/non-vulnerable ML classifier.

Approach: AST-based taint tracking (lightweight, not full dataflow).
We track simple variable aliasing within a method body, identify
TAINT SOURCES (request.getParameter/getHeader/getCookies/etc.),
SINKS (sql execute, Runtime.exec, response writer, file ops, etc.),
and SANITIZERS (PreparedStatement, ESAPI encoder, etc.) along the way.
"""

import re
import javalang
from collections import Counter

from src.scanners.taint_tracer import trace_taint_to_sinks

# ---------------------------------------------------------------------------
# Keyword / pattern definitions
# ---------------------------------------------------------------------------

TAINT_SOURCES = [
    "getParameter", "getHeader", "getHeaders", "getCookies", "getQueryString",
    "getInputStream", "getReader", "getRequestURI", "getRequestURL",
    "getParameterValues", "getParameterMap", "getPathInfo",
]

SQL_SINKS = ["executeQuery", "executeUpdate", "execute", "addBatch", "prepareStatement"]
CMD_SINKS = ["exec", "ProcessBuilder", "Runtime.getRuntime"]
XSS_SINKS = ["println", "write", "print", "getWriter"]
PATH_SINKS = ["FileInputStream", "FileOutputStream", "File(", "Files.read", "Paths.get"]
LDAP_SINKS = ["search", "DirContext", "InitialDirContext"]
XXE_SINKS = ["parse", "DocumentBuilder", "SAXParser", "XMLReader"]
WEAK_CRYPTO_SINKS = ["DES", "MD5", "SHA1", "ECB", "getInstance(\"DES", "getInstance(\"MD5", "getInstance(\"SHA1"]

SQL_SANITIZERS = ["PreparedStatement", "prepareStatement", "setEscapeProcessing", "ESAPI"]
XSS_SANITIZERS = ["ESAPI.encoder", "escapeHtml", "encodeForHTML", "StringEscapeUtils"]
PATH_SANITIZERS = ["getCanonicalPath", "normalize", "FilenameUtils"]
GENERIC_SANITIZERS = ["ESAPI", "Encode.", "escape", "sanitize", "validate"]


def _flatten_source(source_code: str) -> str:
    """Collapse whitespace for fast substring scans (used for keyword features)."""
    return re.sub(r"\s+", " ", source_code)


def _line_index_of(source_lines, needle: str):
    for i, line in enumerate(source_lines):
        if needle in line:
            return i
    return -1


def extract_features(source_code: str, file_name: str = "uploaded.java") -> dict:
    """
    Main entry point. Returns a flat feature dict suitable for a row
    in a pandas DataFrame / model input.
    """
    features = {"file_name": file_name}
    flat = _flatten_source(source_code)
    lines = source_code.splitlines()

    # ------------------------------------------------------------------
    # 1. Try AST parse; fall back to regex-only features if parse fails
    # ------------------------------------------------------------------
    ast_ok = True
    tree = None
    try:
        tree = javalang.parse.parse(source_code)
    except Exception as e:
        ast_ok = False
        features["ast_parse_error"] = str(e)

    features["ast_parse_success"] = int(ast_ok)

    # ------------------------------------------------------------------
    # 2. Taint source / sink / sanitizer keyword counts (always computed)
    # ------------------------------------------------------------------
    def count_any(patterns):
        return sum(flat.count(p) for p in patterns)

    features["taint_source_count"] = count_any(TAINT_SOURCES)
    features["sql_sink_count"] = count_any(SQL_SINKS)
    features["cmd_sink_count"] = count_any(CMD_SINKS)
    features["xss_sink_count"] = count_any(XSS_SINKS)
    features["path_sink_count"] = count_any(PATH_SINKS)
    features["ldap_sink_count"] = count_any(LDAP_SINKS)
    features["xxe_sink_count"] = count_any(XXE_SINKS)
    features["weak_crypto_sink_count"] = count_any(WEAK_CRYPTO_SINKS)

    features["sql_sanitizer_count"] = count_any(SQL_SANITIZERS)
    features["xss_sanitizer_count"] = count_any(XSS_SANITIZERS)
    features["path_sanitizer_count"] = count_any(PATH_SANITIZERS)
    features["generic_sanitizer_count"] = count_any(GENERIC_SANITIZERS)

    total_sinks = (features["sql_sink_count"] + features["cmd_sink_count"] +
                   features["xss_sink_count"] + features["path_sink_count"] +
                   features["ldap_sink_count"] + features["xxe_sink_count"])
    total_sanitizers = (features["sql_sanitizer_count"] + features["xss_sanitizer_count"] +
                         features["path_sanitizer_count"] + features["generic_sanitizer_count"])

    features["total_sink_count"] = total_sinks
    features["total_sanitizer_count"] = total_sanitizers
    features["sanitizer_to_sink_ratio"] = round(total_sanitizers / total_sinks, 3) if total_sinks else 0
    features["has_any_sink"] = int(total_sinks > 0)
    features["has_any_sanitizer"] = int(total_sanitizers > 0)

    # ------------------------------------------------------------------
    # 3. String concatenation near a sink (classic injection smell)
    # ------------------------------------------------------------------
    concat_near_sink = 0
    for i, line in enumerate(lines):
        if "+" in line and any(s in line for s in SQL_SINKS + CMD_SINKS):
            concat_near_sink += 1
        # also check next/prev line for multi-line statements
        elif "+" in line:
            window = " ".join(lines[max(0, i - 1):i + 2])
            if any(s in window for s in SQL_SINKS + CMD_SINKS):
                concat_near_sink += 1
    features["string_concat_near_sink_count"] = concat_near_sink

    # ------------------------------------------------------------------
    # 4. Distance (in lines) between first taint source and first sink
    #    Smaller distance = more direct, suspicious flow
    # ------------------------------------------------------------------
    source_line = -1
    sink_line = -1
    for i, line in enumerate(lines):
        if source_line == -1 and any(s in line for s in TAINT_SOURCES):
            source_line = i
        if sink_line == -1 and any(s in line for s in SQL_SINKS + CMD_SINKS + PATH_SINKS):
            sink_line = i
    if source_line != -1 and sink_line != -1:
        features["source_to_sink_line_distance"] = abs(sink_line - source_line)
        features["source_before_sink"] = int(source_line < sink_line)
    else:
        features["source_to_sink_line_distance"] = -1
        features["source_before_sink"] = 0

    # ------------------------------------------------------------------
    # 5. Variable-aliasing depth (simple heuristic, no full dataflow):
    #    count how many "X = Y;" assignment hops occur between source and sink
    # ------------------------------------------------------------------
    assignment_pattern = re.compile(r"\b(\w+)\s*=\s*(\w+)\s*;")
    alias_hops = len(assignment_pattern.findall(flat))
    features["variable_alias_hops"] = alias_hops

    # ------------------------------------------------------------------
    # 6. AST-derived structural features (only if parse succeeded)
    # ------------------------------------------------------------------
    if ast_ok:
        method_count = 0
        try_catch_count = 0
        max_method_lines = 0
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            method_count += 1
        for path, node in tree.filter(javalang.tree.TryStatement):
            try_catch_count += 1
        features["method_count"] = method_count
        features["try_catch_count"] = try_catch_count
    else:
        features["method_count"] = -1
        features["try_catch_count"] = -1

    # ------------------------------------------------------------------
    # 7. General code stats
    # ------------------------------------------------------------------
    features["total_lines"] = len(lines)
    features["total_chars"] = len(source_code)
    features["import_count"] = flat.count("import ")
    features["string_literal_count"] = len(re.findall(r'"[^"]*"', source_code))

    # ------------------------------------------------------------------
    # 8. Real AST-based taint tracing (source -> sink, with sanitizer awareness)
    # ------------------------------------------------------------------
    taint_features = trace_taint_to_sinks(source_code)
    features.update(taint_features)

    return features


def extract_features_for_category_hints(features: dict) -> dict:
    """
    Lightweight heuristic to suggest a likely vulnerability category
    based on which sink type dominates -- useful for the report UI,
    NOT a replacement for the trained classifier's prediction.
    """
    sink_scores = {
        "sqli": features["sql_sink_count"],
        "cmdi": features["cmd_sink_count"],
        "xss": features["xss_sink_count"],
        "pathtraver": features["path_sink_count"],
        "ldapi": features["ldap_sink_count"],
        "xxe": features["xxe_sink_count"],
        "weak_crypto": features["weak_crypto_sink_count"],
    }
    likely_category = max(sink_scores, key=sink_scores.get) if any(sink_scores.values()) else "none_detected"
    return {"likely_category_hint": likely_category, "sink_score_breakdown": sink_scores}


if __name__ == "__main__":
    with open("sample_test.java") as f:
        code = f.read()
    feats = extract_features(code, "sample_test.java")
    hints = extract_features_for_category_hints(feats)
    import json
    print(json.dumps({**feats, **hints}, indent=2))