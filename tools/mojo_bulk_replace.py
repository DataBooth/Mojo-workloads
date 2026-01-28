#!/usr/bin/env python3
"""Bulk Mojo search/replace helper driven by a TOML config.

Usage examples (run from repo root):

  python3 tools/mojo_bulk_replace.py --config mojo_bulk_replacements.toml --dry-run
  python3 tools/mojo_bulk_replace.py --config mojo_bulk_replacements.toml --apply

Config schema (TOML):

  [settings]
  include_extensions = ["mojo"]
  directories = ["7-point-stencil/Mojo", ...]

  [[replacements]]
  name = "alias-to-comptime"
  search = "alias "
  replace = "comptime "
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    raise SystemExit("Python 3.11+ with tomllib is required")


@dataclass
class Replacement:
    name: str
    search: str
    replace: str


@dataclass
class Settings:
    include_extensions: List[str]
    directories: List[str]


def load_config(path: Path) -> tuple[Settings, List[Replacement]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    settings_tbl = data.get("settings") or {}
    settings = Settings(
        include_extensions=list(settings_tbl.get("include_extensions", ["mojo"])),
        directories=list(settings_tbl.get("directories", [])),
    )

    replacements: List[Replacement] = []
    for item in data.get("replacements", []):
        replacements.append(
            Replacement(
                name=str(item.get("name", "unnamed")),
                search=str(item.get("search", "")),
                replace=str(item.get("replace", "")),
            )
        )

    return settings, replacements


def process_file(path: Path, replacements: List[Replacement]) -> tuple[str, dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    new_text = text
    for repl in replacements:
        before = new_text.count(repl.search)
        if before:
            new_text = new_text.replace(repl.search, repl.replace)
        counts[repl.name] = before
    return new_text, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk Mojo search/replace helper")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument("--apply", action="store_true", help="Apply changes to files")

    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        parser.error("Specify either --dry-run or --apply")

    settings, replacements = load_config(args.config)

    if not settings.directories:
        raise SystemExit("No directories specified in [settings.directories]")

    repo_root = Path.cwd()

    total_counts: dict[str, int] = {r.name: 0 for r in replacements}

    for rel_dir in settings.directories:
        base = repo_root / rel_dir
        if not base.is_dir():
            continue
        for root, _dirs, files in os.walk(base):
            for fname in files:
                ext = Path(fname).suffix.lstrip(".")
                if ext not in settings.include_extensions:
                    continue
                fpath = Path(root) / fname
                new_text, counts = process_file(fpath, replacements)
                if any(counts.values()):
                    rel = fpath.relative_to(repo_root)
                    print(f"\n[FILE] {rel}")
                    for name, c in counts.items():
                        if c:
                            print(f"  {name}: {c} occurrence(s)")
                    if args.apply and new_text != fpath.read_text(encoding="utf-8"):
                        fpath.write_text(new_text, encoding="utf-8")
                for name, c in counts.items():
                    total_counts[name] += c

    print("\nSummary:")
    for name, c in total_counts.items():
        print(f"  {name}: {c} total occurrence(s)")


if __name__ == "__main__":  # pragma: no cover
    main()
