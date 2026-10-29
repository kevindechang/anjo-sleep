"""Static inventory support for Sleep's injected environment mappings.

No Sleep runtime code is imported. Extend the shared scanner's AST index for mapping
reads, aliases of imported constants, and finite comprehension inputs.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def prepare(module: Any, tree: ast.Module) -> None:
    """Expose constant aliases and normalize explicitly named environment mappings."""
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) == 1 and isinstance(targets[0], ast.Name) and isinstance(node.value, ast.Attribute):
                dotted = ast.unparse(node.value)
                head, _, tail = dotted.partition(".")
                if head in module.aliases:
                    module.aliases[targets[0].id] = f"{module.aliases[head]}.{tail}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            receiver = node.func.value
            # Sleep consistently calls injected environment mappings env/environ. A
            # conditional receiver chooses os.environ or the injected environ.
            named = isinstance(receiver, ast.Name) and receiver.id in {"env", "environ"}
            conditional = isinstance(receiver, ast.IfExp) and any(
                isinstance(part, ast.Attribute) and ast.unparse(part) == "os.environ"
                for part in ast.walk(receiver)
            )
            if named or conditional:
                node.func.value = ast.Attribute(value=ast.Name(id="os", ctx=ast.Load()), attr="environ", ctx=ast.Load())
                module.aliases.setdefault("os", "os")


def declared_capability_reads(root: Any) -> list[tuple[str, dict[str, Any]]]:
    """Inventory the host-only capability declaration without mutating the manifest."""
    rel = "anjo/core/capabilities.py"
    tree = ast.parse((root / rel).read_text(encoding="utf-8"))
    declaration = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "declare_sleep_capability")
    default_on = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == "_DEFAULT_ON_CAPABILITIES"
                          for target in node.targets))
    defaults = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "get_capability_defaults")
    fallback = next(node.args[1] for node in ast.walk(defaults) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "_env_enabled")
    switch_default = ast.literal_eval(fallback.body if "sleep" in default_on else fallback.orelse)
    cohort = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cohort_place")
    cohort_defaults = {node.args[0].attr: ast.literal_eval(node.args[1]) for node in ast.walk(cohort)
                       if isinstance(node, ast.Call) and ast.unparse(node.func) == "os.environ.get"
                       and isinstance(node.args[0], ast.Attribute)}
    reads = []
    for node in ast.walk(declaration):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(isinstance(target, ast.Name) and target.id == "switch" for target in node.targets):
                reads.append((node.value.value, {"where": f"{rel}:get_capability_defaults",
                                                "default": switch_default, "via": "capability:sleep"}))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "CapabilityCohort":
            for attr, arg in zip(("percent_env", "forced_ids_env"), node.args):
                reads.append((ast.literal_eval(arg), {"where": f"{rel}:cohort_place", "default": cohort_defaults[attr],
                                                     "via": "cohort:sleep"}))
    return reads


def expand_collections(idx: Any, sites: list[Any]) -> None:
    """Resolve finite nested tuples used to validate sets of environment variables."""
    for module in idx.modules.values():
        if not module.name.startswith("anjo.sleep."):
            continue
        tree = ast.parse((Path(__file__).resolve().parents[3] / module.rel).read_text())
        assignments = {}
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    assignments[targets[0].id] = node.value

        def value(expr: ast.expr, depth: int = 0) -> Any:
            if depth > 16:
                return None
            if isinstance(expr, (ast.Tuple, ast.List)):
                return tuple(value(item, depth + 1) for item in expr.elts)
            kind, found = idx.resolve(module, frozenset(), expr)
            if kind == "value":
                return found
            if isinstance(expr, ast.Name) and expr.id in assignments:
                return value(assignments[expr.id], depth + 1)
            return None

        def bind(target: ast.expr, found: Any, bindings: dict[str, list[Any]]) -> None:
            if isinstance(target, ast.Name):
                bindings.setdefault(target.id, []).append(found)
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(found, tuple):
                for part, item in zip(target.elts, found):
                    bind(part, item, bindings)

        def walk(node: ast.AST, bindings: dict[str, list[Any]]) -> None:
            if isinstance(node, (ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                bindings = {key: list(items) for key, items in bindings.items()}
                for generator in node.generators:
                    sources = (bindings.get(generator.iter.id, []) if isinstance(generator.iter, ast.Name)
                               and generator.iter.id in bindings else [value(generator.iter)])
                    for source in sources:
                        if isinstance(source, tuple):
                            for item in source:
                                bind(generator.target, item, bindings)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                    and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in bindings):
                names = bindings[node.args[0].id]
                if names and all(isinstance(name, str) for name in names):
                    for site in sites:
                        if site.module is module and isinstance(site.node, ast.Call) and site.node.lineno == node.lineno:
                            if ast.unparse(site.node.func) == "os.environ.get" and site.node.args:
                                synthetic = f"_inventory_collection_{node.lineno}"
                                module.name_maps[synthetic] = tuple(enumerate(names))
                                site.node.args[0] = ast.Subscript(value=ast.Name(id=synthetic, ctx=ast.Load()),
                                                                slice=ast.Constant(value=None), ctx=ast.Load())
            for child in ast.iter_child_nodes(node):
                walk(child, bindings)

        walk(tree, {})
