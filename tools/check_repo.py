#!/usr/bin/env python3
"""Portable CI checks; no GPU or Python dependencies required."""
from __future__ import annotations

import ast
import csv
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", ".uv-cache", ".swiftcache", ".pytest_cache",
            "__pycache__", "build", "dist", "out"}


def source_files() -> list[Path]:
    # Once initialized, validate exactly the files that can reach the public repo.
    result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others",
                             "--exclude-standard"], cwd=ROOT, capture_output=True)
    if result.returncode == 0:
        return [ROOT / p.decode() for p in result.stdout.split(b"\0") if p]
    return [p for p in ROOT.rglob("*") if p.is_file()
            and not EXCLUDED.intersection(p.relative_to(ROOT).parts)]


def check_file(path: Path) -> list[str]:
    errors = []
    label = path.relative_to(ROOT)
    if path.suffix == ".py":
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(label))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{label}: {exc}")
    elif path.suffix == ".md":
        # Check inline local file links, skipping fenced code and external URLs.
        content = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", path.read_text(),
                         flags=re.MULTILINE | re.DOTALL)
        for match in re.finditer(r"!?\[[^\]\n]*\]\((<[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)", content):
            target = match.group(1).strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            local_path = (ROOT if url.path.startswith("/") else path.parent) / unquote(url.path).lstrip("/")
            if not local_path.exists():
                errors.append(f"{label}: missing link target {target}")
    elif path.suffix == ".csv" and "benchmarks" in path.parts:
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        if not rows or not all(rows[0]) or len(rows[0]) != len(set(rows[0])):
            errors.append(f"{label}: CSV header must have unique, nonempty fields")
        elif any(len(row) != len(rows[0]) for row in rows[1:]):
            errors.append(f"{label}: CSV rows must match the header width")
    return errors


def main() -> int:
    files = source_files()
    errors = [error for path in files for error in check_file(path)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"PASS: repository checks ({len(files)} source files); GPU execution checked separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
