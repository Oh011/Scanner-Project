"""
taint_tracer.py
Lightweight intraprocedural taint tracing over a javalang AST.

For each method body, walks statements in source order, maintaining a
set of currently-tainted variable names. Tracks:
  - taint introduced by source calls (request.getParameter, etc.)
  - taint propagated through reassignment (bar = foo;)
  - taint propagated from a tainted object's method call (cookie.getValue())
  - taint propagated through enhanced-for loop variables (for Cookie c : cookies)
  - taint cleared by sanitizer calls (bar = ESAPI.encoder().encodeForSQL(foo);)
  - taint reaching a sink call's arguments (statement.executeQuery(sql))
  - taint reaching a sink via the qualifier object (ps.execute() where ps was built
    from tainted prepareStatement(sql))

This is a heuristic, single-method, intraprocedural tracer -- it does
NOT follow taint across method calls, does not do full points-to
analysis, and treats control flow approximately (if/try/for bodies are
walked in document order, not modeled for actual branching/loops).
It is meant to be a stronger signal than line-distance/keyword-proximity
features, not a sound, complete dataflow analysis.

Fixes over original version
---------------------------
1. Enhanced-for loop variable taint propagation:
       for (Cookie cookie : cookies)  -- if 'cookies' is tainted, 'cookie' is tainted.
2. Method-on-tainted-object propagation:
       String v = cookie.getValue()   -- if 'cookie' is tainted, 'v' is tainted.
3. Sink qualifier taint check:
       ps.execute()                   -- if 'ps' is tainted (e.g. built from
       prepareStatement(tainted_sql)) the sink fires even with no tainted arguments.
4. Basic for-loop init walked for taint sources.
5. While/do-while condition walked for sinks.
6. Qualified ClassCreator name resolution:
       new java.io.FileInputStream(f) -- type.name = "java"; walk sub_type chain to
       get leaf class name "FileInputStream" for sink matching.
7. Sink check in LocalVariableDeclaration initializer:
       Process p = r.exec(args, env)  -- exec is a sink even when the result is
       assigned to a variable; previously only checked in StatementExpression path.
8. Added missing sink methods: prepareCall (sqli), command (cmdi).
9. Collection taint via .add(): argList.add(param) marks argList as tainted so
       pb.command(argList) and pb.start() fire correctly.
"""

import javalang


TAINT_SOURCE_METHODS = {
    "getParameter", "getHeader", "getHeaders", "getCookies", "getQueryString",
    "getInputStream", "getReader", "getRequestURI", "getRequestURL",
    "getParameterValues", "getParameterMap", "getPathInfo",
}

SINK_METHODS_BY_CATEGORY = {
    "sqli":      {"executeQuery", "executeUpdate", "execute", "addBatch",
                  "prepareCall"},                        # FIX 8
    "cmdi":      {"exec", "start", "command"},           # FIX 8
    "pathtraver":{"FileInputStream", "FileOutputStream", "File",
                  "newInputStream", "newOutputStream"},
    "ldapi":     {"search"},
    "xpathi":    {"evaluate", "compile"},
}

SANITIZER_METHODS = {
    "encodeForSQL", "encodeForOS", "encodeForLDAP", "encodeForXPath",
    "getCanonicalPath", "normalize", "escapeHtml", "encodeForHTML",
    "setEscapeProcessing", "escape", "sanitize",
}

# Methods that add a value into a collection object, tainting the collection
# e.g. argList.add(param) → argList becomes tainted
COLLECTION_ADD_METHODS = {"add", "put", "offer", "push", "addAll"}


# ---------------------------------------------------------------------------
# Generic AST helpers
# ---------------------------------------------------------------------------

def _iter_child_nodes(node):
    """Yield all child Node/list-of-Node attributes of a javalang AST node."""
    if not hasattr(node, "attrs"):
        return
    for attr in node.attrs:
        try:
            value = getattr(node, attr)
        except Exception:
            continue
        if isinstance(value, javalang.ast.Node):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, javalang.ast.Node):
                    yield item


def _node_identifiers(node, found=None):
    """Recursively collect variable-like identifier names referenced in an expression."""
    if found is None:
        found = set()
    if node is None:
        return found
    if isinstance(node, javalang.tree.MemberReference):
        if node.qualifier:
            found.add(node.qualifier)
        found.add(node.member)
    elif isinstance(node, javalang.tree.MethodInvocation):
        if node.qualifier:
            found.add(node.qualifier)
        for arg in (node.arguments or []):
            _node_identifiers(arg, found)
    for child in _iter_child_nodes(node):
        _node_identifiers(child, found)
    return found


def _resolve_class_name(ref_type):
    """
    FIX 6: Walk a javalang ReferenceType sub_type chain to get the leaf class name.
    new java.io.FileInputStream(f)
      → ReferenceType(name='java', sub_type=ReferenceType(name='io',
           sub_type=ReferenceType(name='FileInputStream', sub_type=None)))
    Returns 'FileInputStream'.
    """
    if ref_type is None:
        return None
    current = ref_type
    while current.sub_type is not None:
        current = current.sub_type
    return current.name


def _is_taint_source_call(node):
    return isinstance(node, javalang.tree.MethodInvocation) and node.member in TAINT_SOURCE_METHODS


def _is_sanitizer_call(node):
    return isinstance(node, javalang.tree.MethodInvocation) and node.member in SANITIZER_METHODS


def _contains_call_matching(node, method_names):
    """Search anywhere within an expression subtree for a call to one of method_names."""
    if node is None:
        return False
    if isinstance(node, javalang.tree.MethodInvocation) and node.member in method_names:
        return True
    for child in _iter_child_nodes(node):
        if _contains_call_matching(child, method_names):
            return True
    return False


def _contains_taint_source(node):
    return _contains_call_matching(node, TAINT_SOURCE_METHODS)


def _contains_sanitizer(node):
    return _contains_call_matching(node, SANITIZER_METHODS)


def _find_method_invocations(node, found=None):
    """Find every MethodInvocation/ClassCreator anywhere within a statement subtree."""
    if found is None:
        found = []
    if node is None:
        return found
    if isinstance(node, javalang.tree.MethodInvocation):
        found.append(node)
    if isinstance(node, javalang.tree.ClassCreator):
        found.append(node)
    for child in _iter_child_nodes(node):
        _find_method_invocations(child, found)
    return found


def _qualifier_is_tainted(call_node, tainted_vars):
    """
    FIX 3: Return True if the object on which the method is called is itself tainted.
    e.g.  ps.execute()  where 'ps' is in tainted_vars.
    javalang stores the qualifier as a plain string on MethodInvocation.
    """
    qualifier = getattr(call_node, "qualifier", None)
    return bool(qualifier and qualifier in tainted_vars)


# ---------------------------------------------------------------------------
# Core taint-tracking walk
# ---------------------------------------------------------------------------

def _check_call_for_sink(call_node, tainted_vars, sanitized_vars, results):
    """
    Given a MethodInvocation or ClassCreator node, check if it matches a
    known sink for any category, and if:
      (a) any of its arguments reference a currently-tainted variable, OR
      (b) the qualifier object itself is tainted (ps.execute() case)

    FIX 6: For ClassCreator, resolve the leaf class name through sub_type chain.
    """
    member_name = getattr(call_node, "member", None)
    if member_name is None and isinstance(call_node, javalang.tree.ClassCreator):
        # FIX 6: walk sub_type chain instead of just .type.name
        member_name = _resolve_class_name(call_node.type) if call_node.type else None

    if member_name is None:
        return

    args = getattr(call_node, "arguments", None) or []
    arg_identifiers = set()
    for arg in args:
        arg_identifiers |= _node_identifiers(arg)

    tainted_args = arg_identifiers & tainted_vars
    sanitized_args = arg_identifiers & sanitized_vars

    # FIX 3: also fire if the qualifier itself is tainted (ps.execute() case)
    qualifier_tainted = _qualifier_is_tainted(call_node, tainted_vars)

    for category, sink_methods in SINK_METHODS_BY_CATEGORY.items():
        if member_name in sink_methods:
            if tainted_args or qualifier_tainted:
                results[f"{category}_tainted_sink_reached"] = 1
            elif sanitized_args:
                results[f"{category}_sanitized_before_sink"] = 1


def _process_declarator(var_name, initializer, tainted_vars, sanitized_vars, results=None):
    """
    Update taint state for a single variable assignment/declaration.
    Also handles:
      FIX 2: method called on tainted qualifier propagates taint to result variable.
      FIX 7: if initializer is a sink call, fire the sink check too.
    """
    if initializer is None:
        return

    # FIX 7: always check if the initializer itself is a sink call
    # e.g.  Process p = r.exec(args, env);
    if results is not None:
        for call in _find_method_invocations(initializer):
            _check_call_for_sink(call, tainted_vars, sanitized_vars, results)

    # Direct taint source call: param = request.getParameter(...)
    if _contains_taint_source(initializer):
        tainted_vars.add(var_name)
        sanitized_vars.discard(var_name)
        return

    # Sanitizer wraps the value: safe = ESAPI.encoder().encodeForSQL(tainted)
    if _contains_sanitizer(initializer):
        sanitized_vars.add(var_name)
        tainted_vars.discard(var_name)
        return

    # FIX 2: method called on a tainted object propagates taint to result
    #   String v = cookie.getValue()   -- cookie is tainted → v is tainted
    if isinstance(initializer, javalang.tree.MethodInvocation):
        qualifier = initializer.qualifier or ""
        if qualifier in tainted_vars:
            tainted_vars.add(var_name)
            sanitized_vars.discard(var_name)
            return

    # Standard propagation: result = tainted_var (possibly via concat / expression)
    ids = _node_identifiers(initializer)
    if ids & tainted_vars:
        tainted_vars.add(var_name)
        sanitized_vars.discard(var_name)
    elif ids & sanitized_vars:
        sanitized_vars.add(var_name)


def _handle_collection_add(call_node, tainted_vars, sanitized_vars):
    """
    FIX 9: argList.add(param) — if any argument is tainted, mark the collection
    (qualifier) as tainted so pb.command(argList) and pb.start() fire.
    """
    if not isinstance(call_node, javalang.tree.MethodInvocation):
        return
    if call_node.member not in COLLECTION_ADD_METHODS:
        return
    qualifier = call_node.qualifier or ""
    if not qualifier:
        return
    args = call_node.arguments or []
    arg_ids = set()
    for a in args:
        arg_ids |= _node_identifiers(a)
    if arg_ids & tainted_vars:
        tainted_vars.add(qualifier)
        sanitized_vars.discard(qualifier)


def _walk_statement(stmt, tainted_vars, sanitized_vars, results):
    """Recursively walk a statement (or list of statements), updating taint
    state and recording any sink reaches found along the way."""
    if stmt is None:
        return
    if isinstance(stmt, (list, tuple)):
        for s in stmt:
            _walk_statement(s, tainted_vars, sanitized_vars, results)
        return

    if isinstance(stmt, javalang.tree.LocalVariableDeclaration):
        for declarator in stmt.declarators:
            # FIX 7: pass results so sink calls inside initializers are checked
            _process_declarator(declarator.name, declarator.initializer,
                                 tainted_vars, sanitized_vars, results)
        return

    if isinstance(stmt, javalang.tree.StatementExpression):
        expr = stmt.expression

        if isinstance(expr, javalang.tree.Assignment):
            lhs_name = None
            if isinstance(expr.expressionl, javalang.tree.MemberReference):
                lhs_name = expr.expressionl.member
            rhs = expr.value
            if lhs_name is not None:
                _process_declarator(lhs_name, rhs, tainted_vars, sanitized_vars, results)
            else:
                # Even if we can't name the LHS, check RHS for sinks
                for call in _find_method_invocations(rhs):
                    _check_call_for_sink(call, tainted_vars, sanitized_vars, results)
            return

        # FIX 9: check collection adds before general sink check
        if isinstance(expr, javalang.tree.MethodInvocation):
            _handle_collection_add(expr, tainted_vars, sanitized_vars)

        for call in _find_method_invocations(expr):
            _check_call_for_sink(call, tainted_vars, sanitized_vars, results)
        return

    if isinstance(stmt, javalang.tree.IfStatement):
        for call in _find_method_invocations(stmt.condition):
            _check_call_for_sink(call, tainted_vars, sanitized_vars, results)
        _walk_statement(stmt.then_statement, tainted_vars, sanitized_vars, results)
        _walk_statement(stmt.else_statement, tainted_vars, sanitized_vars, results)
        return

    if isinstance(stmt, javalang.tree.TryStatement):
        _walk_statement(stmt.block, tainted_vars, sanitized_vars, results)
        for catch in (stmt.catches or []):
            _walk_statement(catch.block, tainted_vars, sanitized_vars, results)
        if stmt.finally_block:
            _walk_statement(stmt.finally_block, tainted_vars, sanitized_vars, results)
        return

    # FIX 1 + 4 + 5: for/while/do — walk init, control, and body properly
    if isinstance(stmt, javalang.tree.ForStatement):
        ctrl = stmt.control

        if isinstance(ctrl, javalang.tree.EnhancedForControl):
            # FIX 1: for (Cookie cookie : cookies)
            # If the iterable is tainted, mark the loop variable as tainted too.
            iterable = ctrl.iterable
            iterable_ids = _node_identifiers(iterable)
            loop_var_name = None
            if ctrl.var and ctrl.var.declarators:
                loop_var_name = ctrl.var.declarators[0].name

            if loop_var_name:
                if _contains_taint_source(iterable):
                    # e.g. for (String v : request.getParameterValues("p"))
                    tainted_vars.add(loop_var_name)
                    sanitized_vars.discard(loop_var_name)
                elif iterable_ids & tainted_vars:
                    # e.g. cookies is already tainted → cookie is tainted
                    tainted_vars.add(loop_var_name)
                    sanitized_vars.discard(loop_var_name)

        elif isinstance(ctrl, javalang.tree.ForControl):
            # FIX 4: basic for — walk the init declarations for taint sources
            if ctrl.init:
                inits = ctrl.init if isinstance(ctrl.init, list) else [ctrl.init]
                for init_stmt in inits:
                    _walk_statement(init_stmt, tainted_vars, sanitized_vars, results)

        # FIX 5: check condition for sinks
        if hasattr(ctrl, "condition") and ctrl.condition is not None:
            for call in _find_method_invocations(ctrl.condition):
                _check_call_for_sink(call, tainted_vars, sanitized_vars, results)

        _walk_statement(stmt.body, tainted_vars, sanitized_vars, results)
        return

    if isinstance(stmt, (javalang.tree.WhileStatement, javalang.tree.DoStatement)):
        condition = getattr(stmt, "condition", None)
        if condition is not None:
            for call in _find_method_invocations(condition):
                _check_call_for_sink(call, tainted_vars, sanitized_vars, results)
        _walk_statement(stmt.body, tainted_vars, sanitized_vars, results)
        return

    if isinstance(stmt, javalang.tree.BlockStatement):
        _walk_statement(stmt.statements, tainted_vars, sanitized_vars, results)
        return

    for call in _find_method_invocations(stmt):
        _check_call_for_sink(call, tainted_vars, sanitized_vars, results)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def trace_taint_to_sinks(source_code: str) -> dict:
    """
    Returns a flat dict of per-category taint signals, safe to merge
    directly into the feature dict produced by feature_extractor.py:

        {category}_tainted_sink_reached   -> 1 if an unsanitized tainted
                                              variable reaches a sink for
                                              this category
        {category}_sanitized_before_sink  -> 1 if a *sanitized* variable
                                              (derived from taint but
                                              passed through a sanitizer)
                                              reaches a sink for this
                                              category
    """
    categories = list(SINK_METHODS_BY_CATEGORY.keys())
    results = {}
    for cat in categories:
        results[f"{cat}_tainted_sink_reached"] = 0
        results[f"{cat}_sanitized_before_sink"] = 0

    try:
        tree = javalang.parse.parse(source_code)
    except Exception:
        results["taint_trace_parse_error"] = 1
        return results

    results["taint_trace_parse_error"] = 0

    for _, method in tree.filter(javalang.tree.MethodDeclaration):
        if method.body is None:
            continue
        tainted_vars = set()
        sanitized_vars = set()
        _walk_statement(method.body, tainted_vars, sanitized_vars, results)

    return results


if __name__ == "__main__":
    import json
    with open("sample_test.java") as f:
        code = f.read()
    print(json.dumps(trace_taint_to_sinks(code), indent=2))