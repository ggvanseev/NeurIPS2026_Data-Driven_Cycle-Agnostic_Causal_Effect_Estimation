#!/usr/bin/env python3
"""Rename LSAS result CSV files to compact metadata-based names.

Run from the repository root:

    python NeurIPS_2026/analysis/rename_result_csvs.py FOLDER

This prints a dry-run. To actually rename files, add ``--apply``:

    python NeurIPS_2026/analysis/rename_result_csvs.py FOLDER --apply

For subfolders too, add ``--recursive``:

    python NeurIPS_2026/analysis/rename_result_csvs.py FOLDER --recursive --apply

The script reads CSV filenames in one input folder and builds names of the form:

    linear_8n_gau_acyclic_1Ks_100g.csv
    linear_8n_gau_acyclic_no_pre_1Ks_100g.csv

It preserves these fields from the existing filename:

- SCM type: ``linear`` or ``nonlinear``
- node count: ``nodes8`` becomes ``8n``
- noise type: ``gaussian`` -> ``gau``, ``non_gaussian`` -> ``non_gau``,
  ``random`` -> ``ran``
- cyclicity: ``acyclic`` or ``cyclic``
- sample count: parsed from generated-style ``_n1000_``, not from an earlier
  ``1000s`` token
- no-pretreatment flag: ``no_pre`` or ``dno_pre`` becomes ``no_pre``
- graph count: ``graphs100`` becomes ``100g``

Ambiguous or incomplete filenames are left unchanged. By default this is a
dry-run. Add ``--apply`` to actually rename files.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

GENERATED_TAG_RE = re.compile(
    r"(?P<cyclicity>acyclic|cyclic)"
    r"(?:_(?P<no_pre>d?no_pre))?"
    r"_(?P<scm_type>linear|nonlinear)"
    r"_graphs(?P<n_graphs>\d+)"
    r"_n(?P<n_samples>\d+)"
    r"_nodes(?P<n_nodes>\d+)"
    r"_edges(?P<n_edges>\d+)",
    flags=re.IGNORECASE,
)
COMPACT_NAME_RE = re.compile(
    r"^(?:linear|nonlinear)_\d+n_(?:gau|non_gau|ran)_(?:acyclic|cyclic)"
    r"(?:_no_pre)?_\d+(?:\.\d+)?Ks_\d+g$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RenamePlan:
    source: Path
    target: Path
    reason: str = ""

    @property
    def can_rename(self) -> bool:
        return not self.reason and self.source != self.target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder containing CSV files.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Rename CSV files recursively below folder.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, only print a dry-run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing target file.",
    )
    parser.add_argument(
        "--list-found",
        action="store_true",
        help="Print every CSV file discovered before planning renames.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "After the summary, print one example for each skip reason except "
            "'already named'. Duplicate examples include the shared target and "
            "the source files that map to it."
        ),
    )
    return parser.parse_args()


def token_present(text: str, token: str) -> bool:
    return re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", text) is not None


def parse_noise(stem: str) -> str | None:
    """Return the compact noise code, or None if noise type is unclear."""
    protected_stem = re.sub(r"(?:^|_)non_gaussian(?:_|$)", "_", stem)
    matches = []
    if token_present(stem, "non_gaussian"):
        matches.append("non_gau")
    if token_present(protected_stem, "gaussian"):
        matches.append("gau")
    if token_present(stem, "random"):
        matches.append("ran")
    if len(matches) != 1:
        return None
    return matches[0]


def sample_count_tag(n_samples: int) -> str:
    """Format sample count as a compact K-samples tag."""
    if n_samples % 1000 == 0:
        value = str(n_samples // 1000)
    else:
        value = f"{n_samples / 1000:g}"
    return f"{value}Ks"


def compact_name(path: Path) -> str | None:
    """Build the compact CSV filename, or None when parsing is ambiguous."""
    stem = path.stem
    if COMPACT_NAME_RE.fullmatch(stem):
        return path.name

    if stem.endswith("_lsas"):
        stem = stem[: -len("_lsas")]

    matches = list(GENERATED_TAG_RE.finditer(stem))
    if len(matches) != 1:
        return None

    match = matches[0]
    path_context = "_".join((*path.parent.parts, stem)).lower()
    noise = parse_noise(path_context)
    if noise is None:
        return None

    scm_type = match.group("scm_type").lower()
    cyclicity = match.group("cyclicity").lower()
    n_nodes = int(match.group("n_nodes"))
    n_graphs = int(match.group("n_graphs"))
    n_samples = int(match.group("n_samples"))
    no_pre = (
        match.group("no_pre") is not None
        or token_present(stem, "no_pre")
        or token_present(stem, "dno_pre")
    )

    parts = [
        scm_type,
        f"{n_nodes}n",
        noise,
        cyclicity,
    ]
    if no_pre:
        parts.append("no_pre")
    parts.extend(
        [
            sample_count_tag(n_samples),
            f"{n_graphs}g",
        ]
    )
    return "_".join(parts) + path.suffix


def build_plan(path: Path, overwrite: bool) -> RenamePlan:
    new_name = compact_name(path)
    if new_name is None:
        return RenamePlan(path, path, "ambiguous or incomplete metadata")

    target = path.with_name(new_name)
    if target == path:
        return RenamePlan(path, target, "already named")
    if path_exists(target) and not overwrite:
        return RenamePlan(path, target, "target exists")
    return RenamePlan(path, target)


def discover_csvs(folder: Path, recursive: bool) -> list[Path]:
    if not recursive:
        return sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".csv"
        )

    csv_paths: list[Path] = []
    for root, _, filenames in os.walk(folder):
        root_path = Path(root)
        for filename in filenames:
            path = root_path / filename
            if path.suffix.lower() == ".csv":
                csv_paths.append(path)
    return sorted(csv_paths)


def filesystem_path(path: Path) -> str:
    """Return a path string suitable for Windows long-path filesystem calls."""
    path_text = os.path.abspath(os.fspath(path))
    if os.name != "nt" or path_text.startswith("\\\\?\\"):
        return path_text
    if path_text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_text.lstrip("\\")
    return "\\\\?\\" + path_text


def path_exists(path: Path) -> bool:
    return os.path.exists(filesystem_path(path))


def unlink_path(path: Path) -> None:
    os.unlink(filesystem_path(path))


def rename_path(source: Path, target: Path) -> None:
    os.rename(filesystem_path(source), filesystem_path(target))


def remember_example(
    examples: dict[str, object],
    reason: str,
    value: object,
) -> None:
    if reason != "already named" and reason not in examples:
        examples[reason] = value


def rename_file(plan: RenamePlan, overwrite: bool) -> str | None:
    """Rename one file, returning None on success or a skip reason on failure."""
    if not path_exists(plan.source):
        return "source missing"
    if path_exists(plan.target) and not overwrite:
        return "target exists"

    if overwrite and path_exists(plan.target):
        unlink_path(plan.target)

    try:
        rename_path(plan.source, plan.target)
    except FileNotFoundError:
        return "source missing"
    except FileExistsError:
        return "target exists"
    except OSError as error:
        return f"rename failed: {error}"

    return None


def main() -> None:
    args = parse_args()
    if not args.folder.is_dir():
        raise SystemExit(f"Folder does not exist: {args.folder}")

    folder = args.folder.resolve()
    csv_files = discover_csvs(folder, args.recursive)
    plans = [
        build_plan(path, overwrite=args.overwrite)
        for path in csv_files
    ]
    if not plans:
        print(f"No CSV files found in {folder}.")
        if not args.recursive and any(folder.glob("*/*.csv")):
            print("CSV files exist in subfolders. Add --recursive to include them.")
        return

    search_mode = "recursively" if args.recursive else "in this folder"
    print(f"Scanning folder: {folder}")
    print(f"Found {len(csv_files)} CSV file(s) {search_mode}.")
    if args.list_found:
        for path in csv_files:
            print(f"found: {path.relative_to(folder)}")

    target_counts = Counter(
        plan.target for plan in plans if not plan.reason and plan.source != plan.target
    )
    duplicate_sources: dict[Path, list[Path]] = defaultdict(list)
    for plan in plans:
        if not plan.reason and plan.source != plan.target and target_counts[plan.target] > 1:
            duplicate_sources[plan.target].append(plan.source)

    renamed = 0
    skipped = 0
    skipped_by_reason: dict[str, int] = defaultdict(int)
    debug_examples: dict[str, object] = {}
    for plan in plans:
        relative_source = plan.source.relative_to(folder)
        relative_target = plan.target.relative_to(folder)
        if target_counts[plan.target] > 1:
            skipped += 1
            reason = f"duplicate target {relative_target}"
            skipped_by_reason["duplicate target"] += 1
            remember_example(
                debug_examples,
                "duplicate target",
                (plan.target, duplicate_sources[plan.target]),
            )
            print(f"skip: {relative_source} ({reason})")
            continue
        if not plan.can_rename:
            skipped += 1
            skipped_by_reason[plan.reason] += 1
            remember_example(
                debug_examples,
                plan.reason,
                (plan.source, plan.target),
            )
            print(f"skip: {relative_source} ({plan.reason})")
            continue

        action = "rename" if args.apply else "would rename"
        print(f"{action}: {relative_source} -> {relative_target}")
        if args.apply:
            reason = rename_file(plan, overwrite=args.overwrite)
            if reason is not None:
                skipped += 1
                skipped_by_reason[reason] += 1
                remember_example(
                    debug_examples,
                    reason,
                    (plan.source, plan.target),
                )
                print(f"skip: {relative_source} ({reason})")
                continue
        renamed += 1

    mode = "Renamed" if args.apply else "Would rename"
    print(
        f"Scanned {len(plans)} file(s). "
        f"{mode} {renamed} file(s); skipped {skipped} file(s)."
    )
    for reason, count in sorted(skipped_by_reason.items()):
        print(f"  skipped {count}: {reason}")
    if args.debug and debug_examples:
        print("Debug skip examples:")
        for reason, example in sorted(debug_examples.items()):
            if reason == "duplicate target":
                target, sources = example
                print(f"  duplicate target: {target.relative_to(folder)}")
                for source in sources:
                    print(f"    source: {source.relative_to(folder)}")
                continue

            source, target = example
            print(f"  {reason}:")
            print(f"    source: {source.relative_to(folder)}")
            print(f"    target: {target.relative_to(folder)}")


if __name__ == "__main__":
    main()
