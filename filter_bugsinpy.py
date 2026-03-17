#!/usr/bin/env python3
"""
filter_bugsinpy.py
------------------
Process all BugsInPy bugs -> filter single-function intraprocedural bugs
-> build IR4/OR2 dataset for evaluating Snake-RepairLLaMA.

Filtering criteria:
  1. Exactly 1 Python (.py) file modified
  2. All hunks fall inside exactly 1 function (single-function bug)
  3. No signature / parameter change (intraprocedural only)
  4. Function body change only (not decorator / class-level / module-level)
  5. combined IR4+OR2 token count <= MAX_TOKENS
  6. Buggy function source must parse without SyntaxError

IR4 format (matches training data exactly):
  - Full buggy function
  - Buggy lines commented out with '# ' prefix, preceded by '# Buggy code:' marker
  - For multi-hunk: comment EVERYTHING from first removed line to last removed line
  - '<FILL_ME>' token on its own line right after the commented block

OR2 format:
  - The fixed lines that replace <FILL_ME>
  - For multi-hunk: fixed version of the entire commented range
    (unchanged lines between hunks stay as-is, removed lines replaced by added)

Usage:
    python filter_bugsinpy.py
    python filter_bugsinpy.py --bugsinpy-dir D:/BugsInPy --repos-dir D:/BugsInPy/repos
                              --output bugsinpy_eval.jsonl
"""

import os
import re
import ast
import json
import subprocess
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─────────────────────────────── constants ────────────────────────────────────

DEFAULT_BUGSINPY_DIR = Path("D:/BugsInPy")
DEFAULT_REPOS_DIR    = Path("D:/BugsInPy/repos")
DEFAULT_OUTPUT       = Path("bugsinpy_eval.jsonl")
TOKENIZER_NAME       = "codellama/CodeLlama-7b-Python-hf"
MAX_TOKENS           = 1024
FILL_ME              = "<FILL_ME>"

# ─────────────────────────────── data classes ─────────────────────────────────

@dataclass
class Hunk:
    old_start: int          # 1-indexed line number in old (buggy) file
    old_count: int
    new_start: int
    new_count: int
    removed: List[str]      # lines removed from old file (without leading '-')
    added:   List[str]      # lines added to new file (without leading '+')
    # removed_line_nos: absolute 1-indexed line numbers in old file that were removed
    removed_line_nos: List[int] = field(default_factory=list)
    # context lines between removed sections (for reconstructing fixed range)
    raw_lines: List[Tuple[str, str]] = field(default_factory=list)
    # raw_lines: list of (kind, content) where kind in {' ', '-', '+'}


@dataclass
class FileDiff:
    old_path: str
    new_path: str
    hunks: List[Hunk]


# ─────────────────────────────── diff parsing ─────────────────────────────────

def parse_patch(patch_text: str) -> List[FileDiff]:
    """Parse unified diff text into FileDiff objects."""
    file_diffs = []
    lines = patch_text.splitlines(keepends=True)
    i = 0

    while i < len(lines):
        line = lines[i]

        # Start of a new file diff
        if line.startswith("diff --git "):
            old_path = new_path = None
            i += 1
            # Read file header lines (index, ---, +++)
            while i < len(lines) and not lines[i].startswith("diff --git ") and not lines[i].startswith("@@"):
                if lines[i].startswith("--- "):
                    p = lines[i][4:].strip()
                    old_path = p[2:] if p.startswith("a/") else p  # strip a/
                    if old_path == "/dev/null":
                        old_path = None
                elif lines[i].startswith("+++ "):
                    p = lines[i][4:].strip()
                    new_path = p[2:] if p.startswith("b/") else p  # strip b/
                    if new_path == "/dev/null":
                        new_path = None
                i += 1

            hunks = []
            # Parse hunks for this file
            while i < len(lines) and lines[i].startswith("@@"):
                hunk, i = parse_hunk(lines, i)
                hunks.append(hunk)

            file_diffs.append(FileDiff(
                old_path=old_path or new_path,
                new_path=new_path or old_path,
                hunks=hunks,
            ))
        else:
            i += 1

    return file_diffs


def parse_hunk(lines: List[str], i: int) -> Tuple[Hunk, int]:
    """Parse one @@ hunk starting at lines[i]. Returns (Hunk, next_i)."""
    header = lines[i]
    i += 1

    # @@ -old_start[,old_count] +new_start[,new_count] @@
    m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
    old_start  = int(m.group(1))
    old_count  = int(m.group(2)) if m.group(2) is not None else 1
    new_start  = int(m.group(3))
    new_count  = int(m.group(4)) if m.group(4) is not None else 1

    removed          = []
    added            = []
    removed_line_nos = []
    raw_lines        = []   # (kind, content)
    current_old_line = old_start

    while i < len(lines):
        line = lines[i]
        if line.startswith("@@") or line.startswith("diff --git "):
            break
        if line.startswith("\\ No newline"):
            i += 1
            continue

        kind    = line[0] if line else " "
        content = line[1:] if len(line) > 1 else "\n"

        if kind == " ":
            raw_lines.append((" ", content))
            current_old_line += 1
        elif kind == "-":
            raw_lines.append(("-", content))
            removed.append(content)
            removed_line_nos.append(current_old_line)
            current_old_line += 1
        elif kind == "+":
            raw_lines.append(("+", content))
            added.append(content)
        else:
            # Treat unknown as context
            raw_lines.append((" ", content))
            current_old_line += 1

        i += 1

    return Hunk(
        old_start=old_start,
        old_count=old_count,
        new_start=new_start,
        new_count=new_count,
        removed=removed,
        added=added,
        removed_line_nos=removed_line_nos,
        raw_lines=raw_lines,
    ), i


# ─────────────────────────────── git helpers ──────────────────────────────────

def clone_repo(github_url: str, dest: Path, verbose: bool = True) -> bool:
    """Clone repo (blobless for speed). Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists():
        if verbose:
            print(f"  [git] {dest.name}: already cloned, skipping")
        return True
    if verbose:
        print(f"  [git] Cloning {github_url} -> {dest} ...")
    result = subprocess.run(
        ["git", "clone", "--filter=blob:none", github_url, str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [ERROR] Clone failed: {result.stderr.strip()}")
        return False
    return True


def get_file_at_commit(repo_dir: Path, commit: str, filepath: str) -> Optional[str]:
    """Return file content at a specific commit using git show."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{filepath}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(repo_dir)
    )
    if result.returncode != 0:
        return None
    return result.stdout


# ─────────────────────────────── AST analysis ─────────────────────────────────

def get_all_functions(tree: ast.AST) -> List[ast.FunctionDef]:
    """Return all FunctionDef/AsyncFunctionDef nodes, sorted by start line."""
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
    return sorted(funcs, key=lambda n: n.lineno)


def find_enclosing_function(
    tree: ast.AST,
    changed_lines: List[int]   # 1-indexed absolute file line numbers
) -> Optional[ast.FunctionDef]:
    """
    Find the innermost function that contains ALL changed_lines.
    Returns None if no single function covers all lines.
    """
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None)
            if end is None:
                continue
            if all(node.lineno <= ln <= end for ln in changed_lines):
                candidates.append(node)

    if not candidates:
        return None

    # Innermost = smallest span
    return min(candidates, key=lambda n: n.end_lineno - n.lineno)


def is_signature_change(func_node: ast.FunctionDef, removed_line_nos: List[int]) -> bool:
    """
    Return True if any removed line falls on the 'def' signature
    (from 'def' keyword line up to first line of body).
    Also catches decorator lines.
    """
    # Decorator lines
    decorator_lines = set()
    for dec in func_node.decorator_list:
        decorator_lines.add(dec.lineno)

    # Signature lines: from func_node.lineno to body[0].lineno - 1
    body_start = func_node.body[0].lineno if func_node.body else func_node.lineno
    sig_lines = set(range(func_node.lineno, body_start))

    for ln in removed_line_nos:
        if ln in sig_lines or ln in decorator_lines:
            return True
    return False


def extract_function_source_lines(source: str, func_node: ast.FunctionDef) -> List[str]:
    """Extract the function's source as a list of lines (with newlines)."""
    all_lines = source.splitlines(keepends=True)
    start = func_node.lineno - 1       # 0-indexed
    end   = func_node.end_lineno       # exclusive (end_lineno is inclusive 1-indexed)
    return all_lines[start:end]


# ─────────────────────────────── IR4 / OR2 builder ────────────────────────────

def build_ir4_or2(
    func_lines: List[str],            # function source lines (with newlines)
    func_start_line: int,             # 1-indexed absolute file line of func start
    all_removed_line_nos: List[int],  # 1-indexed absolute file lines that were removed
    hunks: List[Hunk],                # all hunks that touch this function (sorted)
) -> Tuple[str, str]:
    """
    Build IR4 (input) and OR2 (output) strings.

    IR4: buggy function with the range [first_removed, last_removed] commented out,
         preceded by '# Buggy code:' and followed by '<FILL_ME>'.

    OR2: fixed version of the commented range.
         Processes each hunk's raw_lines to correctly interleave context lines
         between change blocks (handles the case where context lines exist
         between separate removed-line runs within a single hunk).
    """
    def abs_to_rel(abs_ln: int) -> int:
        return abs_ln - func_start_line   # 0-indexed

    removed_rel = sorted(set(abs_to_rel(ln) for ln in all_removed_line_nos))
    first_rel   = removed_rel[0]
    last_rel    = removed_rel[-1]

    # ── Build IR4 ──────────────────────────────────────────────────────────────
    ir4_parts = []
    ir4_parts.extend(func_lines[:first_rel])
    ir4_parts.append("# Buggy code:\n")

    for rel_idx in range(first_rel, last_rel + 1):
        if rel_idx < len(func_lines):
            line    = func_lines[rel_idx]
            content = line.rstrip("\n").rstrip("\r")
            ir4_parts.append(("# \n" if content == "" else "# " + content + "\n"))
        else:
            ir4_parts.append("# \n")

    ir4_parts.append(FILL_ME + "\n")
    ir4_parts.extend(func_lines[last_rel + 1:])
    ir4 = "".join(ir4_parts)

    # ── Build OR2 ──────────────────────────────────────────────────────────────
    # Strategy: replay each hunk's raw_lines line-by-line.
    # Group consecutive '-'/'+' sequences as "change blocks" (no context between).
    # Context (' ') lines and added ('+') lines within the range are kept.
    # Removed ('-') lines within the range are dropped (their '+' replacements kept).
    #
    # abs_first / abs_last: absolute file line numbers bounding the range.
    abs_first = func_start_line + first_rel
    abs_last  = func_start_line + last_rel

    or2_parts = []

    for hunk in sorted(hunks, key=lambda h: h.old_start):
        old_cursor = hunk.old_start   # current 1-indexed absolute line in old file

        # Segment the raw_lines into change-blocks and context
        # A change-block is a maximal run of '-' and '+' lines (possibly mixed).
        i = 0
        raw = hunk.raw_lines
        while i < len(raw):
            kind, content = raw[i]

            if kind == " ":
                # Context line
                if abs_first <= old_cursor <= abs_last:
                    or2_parts.append(content)
                old_cursor += 1
                i += 1

            else:
                # Start of a change block: consume all consecutive '-' and '+' lines
                block_old_start = old_cursor
                removed_in_block = []
                added_in_block   = []

                while i < len(raw) and raw[i][0] in ("-", "+"):
                    k, c = raw[i]
                    if k == "-":
                        removed_in_block.append(c)
                        old_cursor += 1
                    else:
                        added_in_block.append(c)
                    i += 1

                # Block covers old lines [block_old_start, block_old_start + len(removed) - 1]
                block_old_end = old_cursor - 1  # inclusive, last old line consumed

                # Include this block's added lines if it overlaps with [abs_first, abs_last]
                if block_old_start <= abs_last and block_old_end >= abs_first:
                    or2_parts.extend(added_in_block)

    or2 = "".join(or2_parts)
    return ir4, or2


# ─────────────────────────────── tokenizer ────────────────────────────────────

def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {TOKENIZER_NAME} ...")
        tok = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            trust_remote_code=True,
        )
        print("Tokenizer loaded.")
        return tok
    except Exception as e:
        print(f"[WARN] Could not load tokenizer: {e}")
        print("       Token-length filter will be SKIPPED.")
        return None


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# ─────────────────────────────── syntax check ─────────────────────────────────

def has_syntax_error(source: str) -> bool:
    try:
        ast.parse(source)
        return False
    except SyntaxError:
        return True


# ─────────────────────────────── per-bug processing ───────────────────────────

def process_bug(
    project: str,
    bug_id: int,
    bugsinpy_dir: Path,
    repos_dir: Path,
    repo_dir: Path,
    tokenizer,
    stats: dict,
) -> Optional[dict]:
    """
    Process a single bug. Returns a result dict or None (filtered out).
    Updates stats dict with reason for filtering.
    """
    bug_dir  = bugsinpy_dir / "projects" / project / "bugs" / str(bug_id)
    patch_f  = bug_dir / "bug_patch.txt"
    info_f   = bug_dir / "bug.info"

    if not patch_f.exists() or not info_f.exists():
        stats["missing_files"] += 1
        return None

    patch_text = patch_f.read_text(encoding="utf-8", errors="replace")
    info_text  = info_f.read_text(encoding="utf-8", errors="replace")

    # Parse bug.info (allow optional spaces around '=' for robustness)
    buggy_commit = re.search(r'buggy_commit_id\s*=\s*"([^"]+)"', info_text)
    fixed_commit = re.search(r'fixed_commit_id\s*=\s*"([^"]+)"', info_text)
    test_file    = re.search(r'test_file\s*=\s*"([^"]+)"', info_text)
    if not buggy_commit or not fixed_commit:
        stats["missing_commits"] += 1
        return None
    buggy_commit = buggy_commit.group(1)
    fixed_commit = fixed_commit.group(1)
    test_command = test_file.group(1) if test_file else ""

    # ── Filter 1: exactly one Python file changed ──────────────────────────────
    file_diffs = parse_patch(patch_text)

    py_diffs = [fd for fd in file_diffs
                if fd.old_path and fd.old_path.endswith(".py")]

    if len(py_diffs) != 1:
        stats["multi_file"] += 1
        return None
    if len(file_diffs) != 1:
        # Non-python files also changed
        stats["multi_file"] += 1
        return None

    fd = py_diffs[0]
    filepath = fd.old_path

    # Collect all removed line numbers (absolute, 1-indexed) across all hunks
    all_removed = []
    for hunk in fd.hunks:
        all_removed.extend(hunk.removed_line_nos)

    if not all_removed:
        # No lines removed → could be pure addition (global/config add), skip
        stats["no_removed_lines"] += 1
        return None

    # ── Get buggy file source from git ────────────────────────────────────────
    source = get_file_at_commit(repo_dir, buggy_commit, filepath)
    if source is None:
        stats["git_error"] += 1
        return None

    # Normalize line endings
    source = source.replace("\r\n", "\n").replace("\r", "\n")

    # ── Filter 2+3+4: AST-based ────────────────────────────────────────────────
    # Check syntax of buggy source
    if has_syntax_error(source):
        stats["syntax_error_source"] += 1
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        stats["syntax_error_source"] += 1
        return None

    # Ensure Python 3.8+ (need end_lineno)
    # Find enclosing function for ALL removed lines
    enclosing = find_enclosing_function(tree, all_removed)
    if enclosing is None:
        stats["multi_function"] += 1
        return None

    # Also check that pure-addition hunks (no removed lines) are within the
    # enclosing function. For hunks with removed lines, we've already verified
    # them above. For pure-addition hunks we use the hunk's old_start as a
    # position proxy: if old_start is far outside [func_start, func_end], it's
    # in a different function. We allow a small tolerance of 3 lines for the
    # standard unified-diff context window.
    func_start_abs = enclosing.lineno
    func_end_abs   = enclosing.end_lineno
    CONTEXT_TOLERANCE = 3
    for hunk in fd.hunks:
        if hunk.removed_line_nos:
            continue  # already verified removed lines are in the function
        # Pure-addition hunk: check its location via old_start
        hunk_start = hunk.old_start
        hunk_end   = hunk.old_start + hunk.old_count - 1
        # If the entire hunk range is outside the function (with tolerance), reject
        if hunk_end < func_start_abs - CONTEXT_TOLERANCE or \
           hunk_start > func_end_abs + CONTEXT_TOLERANCE:
            stats["multi_function"] += 1
            return None

    # Verify signature is not changed
    if is_signature_change(enclosing, all_removed):
        stats["signature_change"] += 1
        return None

    # Verify change is inside body (body[0].lineno exists)
    if not enclosing.body:
        stats["empty_function"] += 1
        return None

    # ── Extract function source ────────────────────────────────────────────────
    func_start  = enclosing.lineno      # 1-indexed
    func_lines  = extract_function_source_lines(source, enclosing)
    func_source = "".join(func_lines)

    # Double-check: all removed lines are inside function body (not signature)
    body_start = enclosing.body[0].lineno
    func_end   = enclosing.end_lineno
    for ln in all_removed:
        if ln < body_start or ln > func_end:
            stats["signature_change"] += 1
            return None

    # ── Build IR4 / OR2 ────────────────────────────────────────────────────────
    try:
        ir4, or2 = build_ir4_or2(func_lines, func_start, all_removed, fd.hunks)
    except Exception as e:
        stats["build_error"] += 1
        return None

    # ── Syntax check on OR2 ────────────────────────────────────────────────────
    # OR2 on its own isn't valid Python (it's a snippet), so we reconstruct
    # the full fixed function and check that.
    fixed_func = ir4.replace(
        "# Buggy code:\n", ""
    )
    # Actually just check IR4 without the comment block = can't easily do this.
    # Instead, check that OR2 doesn't contain obvious syntax issues by wrapping.
    or2_check = "def _check():\n" + "".join("    " + l for l in or2.splitlines(keepends=True))
    # This is a rough check - skip if it blows up badly
    # Just check or2 doesn't have the original buggy content repeated weirdly
    # We'll skip this and do a simpler check: tokenizer will handle it.

    # ── Token length filter ────────────────────────────────────────────────────
    if tokenizer is not None:
        n_input  = count_tokens(tokenizer, ir4)
        n_output = count_tokens(tokenizer, or2)
        if n_input + n_output > MAX_TOKENS:
            stats["too_long"] += 1
            return None
    else:
        # Rough character-based fallback (4 chars ~= 1 token)
        if (len(ir4) + len(or2)) // 4 > MAX_TOKENS:
            stats["too_long"] += 1
            return None

    # ── Build result ───────────────────────────────────────────────────────────
    stats["kept"] += 1
    return {
        "bug_id"          : f"{project}/{bug_id}",
        "project"         : project,
        "bug_num"         : bug_id,
        "file_path"       : filepath,
        "buggy_commit"    : buggy_commit,
        "fixed_commit"    : fixed_commit,
        "test_command"    : test_command,
        "function_name"   : enclosing.name,
        "func_start_line" : func_start,
        "func_end_line"   : enclosing.end_lineno,
        "num_hunks"       : len(fd.hunks),
        "num_removed"     : len(all_removed),
        "input"           : ir4,
        "output"          : or2,
        "buggy_function"  : func_source,
    }


# ─────────────────────────────── main ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter BugsInPy bugs and build IR4/OR2 dataset.")
    parser.add_argument("--bugsinpy-dir", default=str(DEFAULT_BUGSINPY_DIR))
    parser.add_argument("--repos-dir",    default=str(DEFAULT_REPOS_DIR))
    parser.add_argument("--output",       default=str(DEFAULT_OUTPUT))
    parser.add_argument("--project",      default=None,
                        help="Process only this project (e.g. thefuck). Default: all.")
    parser.add_argument("--no-tokenizer", action="store_true",
                        help="Skip tokenizer check (use char-based fallback).")
    args = parser.parse_args()

    bugsinpy_dir = Path(args.bugsinpy_dir)
    repos_dir    = Path(args.repos_dir)
    output_path  = Path(args.output)

    repos_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = None if args.no_tokenizer else load_tokenizer()

    # Discover projects
    projects_dir = bugsinpy_dir / "projects"
    if args.project:
        projects = [args.project]
    else:
        projects = sorted(d.name for d in projects_dir.iterdir() if d.is_dir())

    stats = {
        "total"              : 0,
        "kept"               : 0,
        "missing_files"      : 0,
        "missing_commits"    : 0,
        "multi_file"         : 0,
        "no_removed_lines"   : 0,
        "git_error"          : 0,
        "syntax_error_source": 0,
        "multi_function"     : 0,
        "signature_change"   : 0,
        "empty_function"     : 0,
        "build_error"        : 0,
        "too_long"           : 0,
        "clone_failed"       : 0,
    }

    results = []

    for project in projects:
        proj_dir  = projects_dir / project
        info_file = proj_dir / "project.info"
        if not info_file.exists():
            continue

        # Get GitHub URL
        info_text = info_file.read_text(encoding="utf-8", errors="replace")
        url_m = re.search(r'github_url="([^"]+)"', info_text)
        if not url_m:
            print(f"[SKIP] {project}: no github_url in project.info")
            continue
        github_url = url_m.group(1).rstrip("/")

        # Clone repo
        repo_dir = repos_dir / project
        ok = clone_repo(github_url, repo_dir)
        if not ok:
            # Count all bugs as clone_failed
            bugs_dir = proj_dir / "bugs"
            n = sum(1 for _ in bugs_dir.iterdir() if _.is_dir()) if bugs_dir.exists() else 0
            stats["clone_failed"] += n
            stats["total"] += n
            continue

        # Get bug list
        bugs_dir = proj_dir / "bugs"
        if not bugs_dir.exists():
            continue

        bug_ids = sorted(
            int(d.name) for d in bugs_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

        print(f"\n[{project}] Processing {len(bug_ids)} bugs ...")

        for bug_id in bug_ids:
            stats["total"] += 1
            result = process_bug(
                project, bug_id, bugsinpy_dir, repos_dir, repo_dir, tokenizer, stats
            )
            if result:
                results.append(result)
                print(f"  [KEPT] {project}/{bug_id}: {result['function_name']} "
                      f"(hunks={result['num_hunks']}, removed={result['num_removed']})")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)
    print(f"Total bugs processed  : {stats['total']}")
    print(f"KEPT                  : {stats['kept']}")
    print(f"-- Filtered reasons --")
    print(f"  Multi-file change   : {stats['multi_file']}")
    print(f"  Multi-function      : {stats['multi_function']}")
    print(f"  Signature change    : {stats['signature_change']}")
    print(f"  No removed lines    : {stats['no_removed_lines']}")
    print(f"  Syntax error        : {stats['syntax_error_source']}")
    print(f"  Too long (tokens)   : {stats['too_long']}")
    print(f"  Git errors          : {stats['git_error']}")
    print(f"  Clone failed        : {stats['clone_failed']}")
    print(f"  Build error         : {stats['build_error']}")
    print(f"  Other               : {stats['missing_files'] + stats['missing_commits'] + stats['empty_function']}")
    print(f"-----------------------------")
    print(f"Output written to     : {output_path}")
    print(f"Samples in output     : {len(results)}")


if __name__ == "__main__":
    main()
