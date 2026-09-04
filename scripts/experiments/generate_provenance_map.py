"""Generate the machine-checkable PROVENANCE map for the shipped dataset.

REVIEW_FINDINGS C4 / P4 ("the release inventory is the image of the
generator inventory"): every file published under
``experiments/validation/`` and shipped under
``dataset/mission_reported/annotations/`` is mapped to the registered
generator command(s)
that write it.  The mapping is derived mechanically, not hand-maintained:

1. the registered commands of BOTH dispatchers (``run_experiments.py`` and
   ``scripts/data_pipeline/process.py``) are enumerated;
2. for each command's module the write call-sites are extracted from the
   AST (``to_csv`` / ``write_text`` / ``shutil.copy`` whose argument
   resolves to a filename literal directly, through an assignment, or
   through a module-level constant);
3. a shipped file maps to every registered command whose module claims a
   write site with that filename; files with NO claiming command are
   reported as ``orphan`` (the C4 audit artifact) -- reported, never
   silently ignored.

Output: ``experiments/validation/PROVENANCE.json``.

Example:
    python scripts/experiments/run_experiments.py provenance-map
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import bootstrap (same as the run_experiments dispatcher): make the flat
# data_pipeline packages and this directory importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data_pipeline"))

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path  # noqa: E402

# Write-side call attributes/imports whose destination argument identifies an
# output file.  read_csv / read_parquet are deliberately absent.
METHOD_WRITERS = {"to_csv", "to_parquet", "to_json", "to_excel"}  # destination = args[0]
RECEIVER_WRITERS = {"write_text", "to_text"}  # destination = the receiver object
COPY_FUNCS = {"copy", "copy2"}  # destination = args[1]

# Shipped QC artifacts whose content is produced inside the release gate /
# staging pipeline rather than by a registered generator command (their
# writers are internal code paths, so the mechanical source analysis cannot
# attribute them).  value = human-readable origin note recorded as the
# generator label.
CURATED_ARTIFACTS = {
    # per-file SLR QC ledger (rejected rows, source-zero fields, sigma>1m
    # counts) frozen from the WP4 staging re-parse; content is final unless
    # the SLR parsers change.
    "rejection_ledger.csv": "staging-qc (alignment/stage_evidence_slr_tle SLR re-parse, frozen artifact)",
}


def _looks_like_filename(value: str) -> bool:
    return isinstance(value, str) and value.endswith((".csv", ".json", ".json.gz", ".csv.gz"))


def _const_strings(node: ast.AST | None, depth: int = 0) -> set[str]:
    """Possible constant string values of an expression (constants and
    conditional expressions of constants -- enough to expand simple
    f-string output names)."""
    if node is None or depth > 6:
        return set()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _const_strings(node.body, depth + 1) | _const_strings(node.orelse, depth + 1)
    return set()


def _collect(
    node: ast.AST | None,
    consts: dict[tuple[str, str], list[ast.AST]],
    depth: int = 0,
    _memo: dict[int, set[str]] | None = None,
) -> set[str]:
    """Recursively collect filename literals referenced by an expression.

    ``Name`` references resolve through module/function assignments, argparse
    ``add_argument(default=...)`` defaults (the shipped output location of a
    CLI writer), function parameter defaults, and ``for``-loop variables
    bound to constant containers (``for name in FILES: df.to_csv(dest /
    name)``).  f-string destinations expand over their constant
    substitutions (``f"table_{clean|all}.csv"``).  Results are memoized per
    AST node (``consts`` is fixed within one extraction pass) -- the union
    over multi-binding names otherwise explodes exponentially.
    """
    if node is None or depth > 8:
        return set()
    if _memo is None:
        _memo = {}
    key = id(node)
    if key in _memo:
        return _memo[key]
    _memo[key] = set()  # cycle guard: in-progress nodes resolve to nothing
    result = _collect_uncached(node, consts, depth, _memo)
    _memo[key] = result
    return result


def _collect_uncached(
    node: ast.AST,
    consts: dict[tuple[str, str], list[ast.AST]],
    depth: int,
    memo: dict[int, set[str]],
) -> set[str]:
    if isinstance(node, ast.Constant):
        return {node.value} if _looks_like_filename(node.value) else set()
    if isinstance(node, ast.JoinedStr):
        candidates = [""]
        for value in node.values:
            if isinstance(value, ast.Constant):
                additions = [str(value.value)]
            else:
                additions = sorted(
                    _const_strings(getattr(value, "value", None))
                )  # FormattedValue -> .value expression
            if not additions:
                return set()
            candidates = [prefix + addition for prefix in candidates for addition in additions]
        return {c for c in candidates if _looks_like_filename(c)}
    if isinstance(node, ast.BinOp):
        return _collect(node.left, consts, depth + 1, memo) | _collect(node.right, consts, depth + 1, memo)
    if isinstance(node, ast.Name):
        found: set[str] = set()
        for kind in ("assign", "argparse", "param", "loop"):
            for source in consts.get((kind, node.id), []):
                found |= _collect(source, consts, depth + 1, memo)
        return found
    if isinstance(node, ast.Attribute):
        found = _collect_name(node.attr, "argparse", consts, depth, memo)
        if found:
            return found
        return _collect(node.value, consts, depth + 1, memo)
    if isinstance(node, ast.Call):
        found = set()
        for arg in node.args:
            found |= _collect(arg, consts, depth + 1, memo)
        return found
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        found = set()
        for element in node.elts:
            found |= _collect(element, consts, depth + 1, memo)
        return found
    if isinstance(node, ast.Dict):
        found = set()
        for key in node.keys:
            found |= _collect(key, consts, depth + 1, memo)
        for value in node.values:
            found |= _collect(value, consts, depth + 1, memo)
        return found
    if isinstance(node, ast.IfExp):
        return _collect(node.body, consts, depth + 1, memo) | _collect(node.orelse, consts, depth + 1, memo)
    return set()


def _collect_name(identifier: str, kind: str, consts: dict, depth: int, memo: dict[int, set[str]]) -> set[str]:
    found: set[str] = set()
    for source in consts.get((kind, identifier), []):
        found |= _collect(source, consts, depth + 1, memo)
    return found


def extract_write_filenames(source: str, import_resolver=None, depth: int = 0) -> set[str]:
    """Filename literals written by one module (AST write-site extraction).

    A literal counts only when it REACHES a write call (``.to_csv`` /
    ``.write_text`` / ``shutil.copy`` destination) directly, through an
    assignment-bound name, an argparse default, a function-parameter
    default, or a loop variable over a constant container.  A module that
    only READS tables (its input tables never reach a write destination)
    therefore claims nothing.

    Writer helpers imported from sibling modules are followed one level
    deep: the imported function's body is re-analyzed with the call-site
    arguments bound (``write_alignment_audit(audit, args.alignment_output)``
    carries the caller's argparse default into the helper's destination).
    """
    tree = ast.parse(source)
    consts: dict[tuple[str, str], list[ast.AST]] = {}

    def _bind(kind: str, identifier: str, value: ast.AST) -> None:
        consts.setdefault((kind, identifier), []).append(value)

    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    call_sites: list[ast.Call] = []
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
            for index, arg in enumerate(node.args.args):
                offset = len(node.args.args) - len(node.args.defaults)
                if 0 <= index - offset < len(node.args.defaults):
                    _bind("param", arg.arg, node.args.defaults[index - offset])
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if default is not None:
                    _bind("param", arg.arg, default)
        elif isinstance(node, ast.Call):
            call_sites.append(node)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            _bind("assign", node.targets[0].id, node.value)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            _bind("loop", node.target.id, node.iter)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            flag = str(node.args[0].value).lstrip("-").replace("-", "_")
            for keyword in node.keywords:
                if keyword.arg == "default":
                    _bind("argparse", flag, keyword.value)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports[alias.name] = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = alias.name

    # Bind local-function parameters from their CALL-SITE arguments (a writer
    # helper's destination usually arrives from main()'s argparse default).
    for call in call_sites:
        if isinstance(call.func, ast.Name) and call.func.id in functions:
            params = functions[call.func.id].args.args
            for param, argument in zip(params, call.args):
                _bind("param", param.arg, argument)
            for keyword in call.keywords:
                _bind("param", keyword.arg, keyword.value)

    written: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        destinations: list[ast.AST] = []
        if isinstance(func, ast.Attribute):
            if func.attr in METHOD_WRITERS and node.args:
                # method-style writers: first positional arg is the destination
                destinations.append(node.args[0])
            elif func.attr in RECEIVER_WRITERS:
                # (path / "x").write_text(data): the RECEIVER is the destination
                destinations.append(func.value)
            elif func.attr in COPY_FUNCS and len(node.args) >= 2:
                # shutil.copy(src, dest): second positional arg is the destination
                destinations.append(node.args[1])
        elif isinstance(func, ast.Name) and func.id in COPY_FUNCS and len(node.args) >= 2:
            destinations.append(node.args[1])
        for destination in destinations:
            written |= _collect(destination, consts)

    # One-level follow of imported writer helpers.
    if import_resolver is not None and depth == 0:
        for call in call_sites:
            fname = call.func.id if isinstance(call.func, ast.Name) else None
            if not fname or fname in functions or fname not in imports:
                continue
            module_source = import_resolver(imports[fname])
            if not module_source:
                continue
            try:
                helper_tree = ast.parse(module_source)
            except SyntaxError:
                continue
            target = next(
                (
                    node
                    for node in helper_tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fname
                ),
                None,
            )
            if target is None:
                continue
            segment = ast.get_source_segment(module_source, target)
            if not segment:
                continue
            binding_lines: list[str] = []
            for param, argument in zip(target.args.args, call.args):
                resolved = _collect(argument, consts)
                values = [repr(value) for value in sorted(resolved)] or [ast.unparse(argument)]
                for value in values:
                    binding_lines.append(f"{param.arg} = {value}")
            synthetic = "\n".join(binding_lines) + "\n\n" + segment
            written |= extract_write_filenames(synthetic, import_resolver, depth + 1)
    return written


def _module_source_resolver(module_name: str) -> str | None:
    """Source text of an importable module, or None (no import side effects kept)."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    if spec is None or not spec.origin or spec.origin.endswith("__init__.py"):
        return None
    try:
        return Path(spec.origin).read_text(encoding="utf-8")
    except OSError:
        return None


def collect_registered_writers() -> dict[str, set[str]]:
    """Command -> written-filenames for every command in both dispatchers."""
    import run_experiments  # noqa: F401  (bootstraps sys.path for experiments)

    import process

    writers: dict[str, set[str]] = {}
    for label, module_name in {**run_experiments.COMMANDS, **process.COMMANDS}.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - registry sanity
            raise RuntimeError(f"registered command {label!r} -> {module_name!r} failed to import: {exc}") from exc
        source = Path(module.__file__).read_text(encoding="utf-8")
        names = extract_write_filenames(source, import_resolver=_module_source_resolver)
        if names:
            # normalize to basenames: shipped files are matched by name
            writers[label] = {Path(name).name for name in names}
    return writers


def build_provenance_map(
    dataset_root: Path | None = None,
    experiments_root: Path | None = None,
) -> tuple[list[dict], list[str]]:
    """Map every published validation/annotation file to its generator command(s).

    Returns ``(entries, orphans)``; ``orphans`` lists files no registered
    command claims (the C4 audit artifact).
    """
    root = Path(dataset_root) if dataset_root is not None else REPO_ROOT / "dataset"
    exp_root = Path(experiments_root) if experiments_root is not None else REPO_ROOT / "experiments"
    writers = collect_registered_writers()
    entries: list[dict] = []
    for directory, prefix in (
        (exp_root / "validation", "experiments/validation"),
        (root / "mission_reported" / "annotations", "dataset/mission_reported/annotations"),
    ):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            generators = sorted(label for label, names in writers.items() if path.name in names)
            if not generators and path.name in CURATED_ARTIFACTS:
                generators = [CURATED_ARTIFACTS[path.name]]
            entries.append(
                {
                    "file": f"{prefix}/{path.name}",
                    "generators": generators,
                    "status": "mapped" if generators else "orphan",
                }
            )
    orphans = [entry["file"] for entry in entries if entry["status"] == "orphan"]
    return entries, orphans


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the dataset PROVENANCE map")
    parser.add_argument("--dataset-root", default=None, help="Dataset root (default: <repo>/dataset)")
    parser.add_argument("--experiments-root", default=None, help="Experiments root (default: <repo>/experiments)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: experiments/validation/PROVENANCE.json)")
    args = parser.parse_args()

    from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path  # noqa: F401

    entries, orphans = build_provenance_map(
        Path(resolve_repo_path(str(args.dataset_root)) or (REPO_ROOT / "dataset")) if args.dataset_root else None,
        Path(resolve_repo_path(str(args.experiments_root)) or (REPO_ROOT / "experiments")) if args.experiments_root else None,
    )
    import process
    import run_experiments

    provenance = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "method": (
            "AST write-site extraction (to_csv/write_text/shutil.copy destinations "
            "resolved through literals, assignments, and module constants) over the "
            "modules registered in the run_experiments and process dispatchers; "
            "files claimed by no registered command are listed as orphan"
        ),
        "dispatchers": {
            "scripts/experiments/run_experiments.py": dict(sorted(run_experiments.COMMANDS.items())),
            "scripts/data_pipeline/process.py": dict(sorted(process.COMMANDS.items())),
        },
        "scopes": ["experiments/validation/", "dataset/mission_reported/annotations/"],
        "file_count": len(entries),
        "mapped_count": sum(1 for entry in entries if entry["status"] == "mapped"),
        "orphan_count": len(orphans),
        "orphans": orphans,
        "files": entries,
    }
    if args.output:
        output = Path(resolve_repo_path(args.output) or (REPO_ROOT / args.output))
    else:
        output = REPO_ROOT / "experiments" / "validation" / "PROVENANCE.json"
    ensure_directory(output.parent)
    output.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "file_count": provenance["file_count"],
                "mapped_count": provenance["mapped_count"],
                "orphan_count": provenance["orphan_count"],
                "orphans": orphans,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
