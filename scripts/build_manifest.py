"""Build a validated, portable ESP-Fi HAR manifest without reading sample arrays.

Run with the project's Python interpreter: python scripts/build_manifest.py
Paths in the CSV are POSIX paths relative to --data-root, not the project root.
Existing output is kept if identical; different output is never overwritten.
"""

import argparse
import csv
import io
import itertools
import re
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "sample_id", "mat_path", "csv_path", "environment_id",
    "participant_id", "activity_id", "activity_name", "repetition_id",
)
ACTIVITIES = {
    1: "running", 2: "falling", 3: "walking", 4: "turning",
    5: "jumping", 6: "squatting", 7: "waving",
}
EXPECTED_IDS = set(itertools.product(range(1, 5), range(1, 9), range(1, 8), range(1, 11)))


def parse_sample_id(stem):
    """Require canonical X-Y-Z-M IDs with the documented field ranges."""
    if re.fullmatch(r"[1-9][0-9]*-[1-9][0-9]*-[1-9][0-9]*-[1-9][0-9]*", stem) is None:
        raise ValueError(f"Unparseable sample filename: {stem!r}")
    ids = tuple(map(int, stem.split("-")))
    if ids not in EXPECTED_IDS:
        raise ValueError(f"Out-of-range sample ID: {stem!r}")
    return ids


def relative_file_path(path, root):
    if not path.is_file():
        raise ValueError(f"Missing sample file: {path}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Sample path resolves outside the data root: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty sample file: {path}")
    return path.relative_to(root).as_posix()


def build_rows(root):
    """Validate the complete filename grid and exact MAT/CSV pairing."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Data root is not a directory: {root}")
    rows_by_id = {}
    paired_csv_paths = set()
    mat_files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mat")
    if not mat_files:
        raise ValueError(f"No MAT files found under {root}")
    for mat in mat_files:
        environment, participant, activity, repetition = ids = parse_sample_id(mat.stem)
        if ids in rows_by_id:
            raise ValueError(f"Duplicate sample ID {mat.stem!r}: {mat}")
        parts = mat.relative_to(root).parts
        if len(parts) != 3 or parts[1] != "mat" or mat.suffix != ".mat":
            raise ValueError(f"Expected <environment>/mat/X-Y-Z-M.mat: {mat}")
        env_match = re.fullmatch(r"EnvironmentNo\.(\d+).+", parts[0])
        if env_match is None or int(env_match.group(1)) != environment:
            raise ValueError(f"Environment directory/filename mismatch: {mat}")
        csv_path = mat.parent.parent / "csv" / f"{mat.stem}.csv"
        row = {
            "sample_id": mat.stem,
            "mat_path": relative_file_path(mat, root),
            "csv_path": relative_file_path(csv_path, root),
            "environment_id": environment,
            "participant_id": participant,
            "activity_id": activity,
            "activity_name": ACTIVITIES[activity],
            "repetition_id": repetition,
        }
        if any(value is None or value == "" for value in row.values()):
            raise ValueError(f"Missing manifest field for {mat}")
        rows_by_id[ids] = row
        paired_csv_paths.add(csv_path.relative_to(root).as_posix())

    actual_ids = set(rows_by_id)
    if actual_ids != EXPECTED_IDS:
        missing = sorted(EXPECTED_IDS - actual_ids)
        raise ValueError(f"Incomplete sample grid: {len(actual_ids)}/2240; missing IDs (first 10): {missing[:10]}")
    actual_csv_paths = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"
    }
    if actual_csv_paths != paired_csv_paths:
        extras = sorted(actual_csv_paths - paired_csv_paths)
        raise ValueError(f"CSV pairing mismatch; unpaired CSV paths (first 10): {extras[:10]}")
    return [rows_by_id[ids] for ids in sorted(rows_by_id)]


def write_manifest(rows, output):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    if output.exists():
        if not output.is_file() or output.read_bytes() != content:
            raise ValueError(f"Output already exists with different content; refusing to overwrite: {output}")
        return "unchanged (identical output already exists)"
    if not output.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {output.parent}")
    with output.open("xb") as stream:
        stream.write(content)
    return "created"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/ESP-Fi_dataset")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/dataset_manifest.csv")
    args = parser.parse_args()
    try:
        rows = build_rows(args.data_root)
        output = args.output.resolve()
        # A manifest must never be written into the source data tree.
        if output.is_relative_to(args.data_root.resolve()):
            raise ValueError("Output must be outside the raw data root")
        result = write_manifest(rows, output)
        print(f"Manifest: {output} ({result})")
        print(f"Rows: {len(rows)}; unique sample IDs: {len({r['sample_id'] for r in rows})}")
        print("Fields: " + ", ".join(FIELDS))
        for field in ("environment_id", "participant_id", "activity_id"):
            counts = Counter(row[field] for row in rows)
            print(f"{field}: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
        print("Validation: complete grid, unique IDs, required fields, and all MAT/CSV paths PASS")
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
