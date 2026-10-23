"""Create the fixed cross-person split from the manifest, without opening MAT files.

Usage: python scripts/create_split.py
All original columns and values are preserved; only the split column is added.
Rows are sorted numerically by environment, participant, activity, and repetition.
An identical existing output is left unchanged; different output is not overwritten.
"""

import argparse
import csv
import io
import itertools
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = {
    "sample_id", "mat_path", "csv_path", "environment_id", "participant_id",
    "activity_id", "activity_name", "repetition_id",
}
ID_FIELDS = ("environment_id", "participant_id", "activity_id", "repetition_id")
PARTICIPANTS = {"train": {1, 2, 3, 4, 5}, "val": {6}, "test": {7, 8}}
EXPECTED_COUNTS = {"train": 1400, "val": 280, "test": 560}
EXPECTED_IDS = set(itertools.product(range(1, 5), range(1, 9), range(1, 8), range(1, 11)))


def split_rows(fields, rows):
    """Validate all samples and return a deterministic, exhaustive split."""
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Missing or duplicate column names")
    if not REQUIRED_FIELDS.issubset(fields) or "split" in fields:
        raise ValueError("Input must contain the manifest fields and no split column")
    if len(rows) != 2240:
        raise ValueError(f"Expected 2240 input rows, got {len(rows)}")
    by_ids = {}
    sample_ids = set()
    for line, row in enumerate(rows, start=2):
        if set(row) != set(fields) or any(value is None or value.strip() == "" for value in row.values()):
            raise ValueError(f"Missing fields or malformed CSV at line {line}")
        try:
            ids = tuple(int(row[field]) for field in ID_FIELDS)
        except ValueError as error:
            raise ValueError(f"Non-integer sample metadata at line {line}") from error
        if ids not in EXPECTED_IDS or row["sample_id"] != "-".join(map(str, ids)):
            raise ValueError(f"Invalid or inconsistent sample metadata at line {line}")
        if ids in by_ids or row["sample_id"] in sample_ids:
            raise ValueError(f"Duplicate sample: {row['sample_id']}")
        participant = ids[1]
        split = next(name for name, people in PARTICIPANTS.items() if participant in people)
        by_ids[ids] = {**row, "split": split}
        sample_ids.add(row["sample_id"])
    if set(by_ids) != EXPECTED_IDS:
        raise ValueError("Manifest does not cover the complete sample grid")
    result = [by_ids[ids] for ids in sorted(by_ids)]
    counts = Counter(row["split"] for row in result)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected split counts: {dict(counts)}")
    observed_people = {}
    for name, expected_people in PARTICIPANTS.items():
        subset = [row for row in result if row["split"] == name]
        observed_people[name] = {int(row["participant_id"]) for row in subset}
        if observed_people[name] != expected_people:
            raise ValueError(f"Unexpected participants in {name}")
        if {int(row["activity_id"]) for row in subset} != set(range(1, 8)):
            raise ValueError(f"Missing activity classes in {name}")
        if {int(row["environment_id"]) for row in subset} != set(range(1, 5)):
            raise ValueError(f"Missing environments in {name}")
    for left, right in itertools.combinations(observed_people.values(), 2):
        if left & right:
            raise ValueError("Participant overlap across splits")
    if {row["sample_id"] for row in result} != sample_ids:
        raise ValueError("Split sample coverage differs from the input")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results/dataset_manifest.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/dataset_split.csv")
    args = parser.parse_args()
    try:
        manifest = args.manifest.resolve(strict=True)
        output = args.output.resolve()
        if output == manifest:
            raise ValueError("Output must not overwrite the source manifest")
        if output.is_relative_to((PROJECT_ROOT / "data/raw").resolve()):
            raise ValueError("Output must not be inside the raw data directory")
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        result = split_rows(fields, rows)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=[*fields, "split"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
        content = buffer.getvalue().encode("utf-8")
        if output.exists():
            if not output.is_file() or output.read_bytes() != content:
                raise ValueError(f"Existing output differs; refusing to overwrite: {output}")
            action = "unchanged (identical)"
        else:
            with output.open("xb") as stream:
                stream.write(content)
            action = "created"
        print(f"Output: {output} ({action})")
        print(f"Samples: {len(result)}; unique IDs: {len({r['sample_id'] for r in result})}")
        for name in PARTICIPANTS:
            subset = [row for row in result if row["split"] == name]
            people = sorted({int(row["participant_id"]) for row in subset})
            activities = dict(sorted(Counter(int(row["activity_id"]) for row in subset).items()))
            environments = dict(sorted(Counter(int(row["environment_id"]) for row in subset).items()))
            print(f"{name}: n={len(subset)}, participants={people}, activities={activities}, environments={environments}")
        print("PASS: exact counts; disjoint participants; complete samples; seven activities and four environments per split")
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
