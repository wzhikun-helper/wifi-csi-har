"""Read-only release check: standard library only; no MAT, inference or writes."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def read_csv(relative):
    with (PROJECT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    lock = json.loads((HERE / "release_integrity.json").read_text(encoding="utf-8"))
    for relative, expected in lock["current_file_sha256"].items():
        path = PROJECT / relative
        require(path.is_file(), f"Missing archived file: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"SHA256 mismatch: {relative}")
    # Historical hashes identify private originals, not the public redactions.
    redaction = json.loads((HERE / "provenance/public_redaction.json").read_text(encoding="utf-8"))
    require(redaction["private_originals_in_public_repository"] is False, "Private originals must stay outside release")
    require(len(redaction["documents"]) == 2, "Expected two public redactions")
    for document in redaction["documents"]:
        relative = document["public_path"]
        public_hash = hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest()
        require(public_hash == document["public_sha256"], f"Public redaction hash: {relative}")
        require(public_hash != document["original_sha256"], f"Redaction must differ from original: {relative}")
        reference = document["historical_hash_reference"]
        historical = json.loads((PROJECT / reference["path"]).read_text(encoding="utf-8"))
        for key in reference["json_keys"]:
            historical = historical[key]
        require(historical == document["original_sha256"], f"Historical original hash: {relative}")
    for relative in lock["removed_report_roots"]:
        require(not (PROJECT / relative).exists(), f"Unexpected obsolete report: {relative}")
    manifest = read_csv("results/dataset_manifest.csv")
    split = read_csv("results/dataset_split.csv")
    require(len(manifest) == len(split) == 2240, "Manifest/split row count")
    require(len({row["sample_id"] for row in manifest}) == 2240, "Duplicate sample IDs")
    require(all({k: v for k, v in row.items() if k != "split"} == manifest[i]
                for i, row in enumerate(split)), "Split differs from manifest")
    require(Counter(row["split"] for row in split) == {"train": 1400, "val": 280, "test": 560}, "Split counts")
    for name, people in {"train": {1, 2, 3, 4, 5}, "val": {6}, "test": {7, 8}}.items():
        rows = [row for row in split if row["split"] == name]
        require({int(row["participant_id"]) for row in rows} == people, f"Participants: {name}")
        require({int(row["environment_id"]) for row in rows} == set(range(1, 5)), f"Environments: {name}")
        require({int(row["activity_id"]) for row in rows} == set(range(1, 8)), f"Activities: {name}")
    base = PROJECT / "results/formal_seed42"
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((base / "test/test_metrics.json").read_text(encoding="utf-8"))
    history = read_csv("results/formal_seed42/metrics.csv")
    require([int(row["epoch"]) for row in history] == list(range(1, 27)), "Epoch history")
    require(summary["best_epoch"] == 16 and summary["epochs_completed"] == 26, "Frozen epoch")
    require(round(summary["best_val_metrics"]["accuracy"] * 100, 4) == 27.1429, "Validation accuracy")
    require(round(metrics["accuracy"] * 100, 4) == 26.4286, "Test accuracy")
    require(round(metrics["macro_f1"], 6) == 0.257181, "Test Macro-F1")
    require((PROJECT / "configs/formal_seed42.yaml").read_bytes() == (base / "config_snapshot.yaml").read_bytes(), "Config snapshot")
    predictions = read_csv("results/formal_seed42/test/predictions.csv")
    require(len(predictions) == len({row["sample_id"] for row in predictions}) == 560, "Predictions")
    require(Counter(row["participant_id"] for row in predictions) == {"7": 280, "8": 280}, "Test participants")
    matrix = read_csv("results/formal_seed42/test/confusion_matrix.csv")
    cells = [[int(row[name]) for name in metrics["class_names"]] for row in matrix]
    require(cells == metrics["confusion_matrix"] and sum(map(sum, cells)) == 560, "Confusion matrix")
    raw_present = (PROJECT / "data/raw/ESP-Fi_dataset").is_dir()
    if raw_present:
        require(all((PROJECT / "data/raw/ESP-Fi_dataset" / row[key]).is_file()
                    for row in manifest for key in ("mat_path", "csv_path")), "Manifest file paths")
    print(json.dumps({"status": "PASS", "archived_files_checked": len(lock["current_file_sha256"]),
                      "manifest_samples": 2240, "split_counts": [1400, 280, 560], "best_epoch": 16,
                      "val_accuracy_percent": 27.1429, "test_accuracy_percent": 26.4286,
                      "test_macro_f1": 0.257181, "raw_paths_checked": raw_present,
                      "public_redactions_checked": len(redaction["documents"]),
                      "mat_arrays_read": False, "model_inference_executed": False,
                      "files_written": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
