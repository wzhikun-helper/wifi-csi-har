"""只读核对排版输入、冻结 CSV/JSON 和编译输出；不导入训练或评价代码。"""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    before = json.loads((HERE / "before_sha256.json").read_text(encoding="utf-8"))
    changed = [name for name, expected in before.items() if sha(PROJECT / name) != expected]
    compiler_intermediates = {
        "reports/wifi_csi_baseline_report_academic/build/main.log",
        "reports/wifi_csi_baseline_report_academic/build/main.pdf",
    }
    assert set(changed) == compiler_intermediates, changed
    for name, expected in before.items():
        if name not in compiler_intermediates:
            assert sha(PROJECT / name) == expected, f"原文件变化：{name}"
    assert sha(PROJECT / "reports/wifi_csi_baseline_report_academic.pdf") == "2db3324deed05023ec073e1daf85ec61fe6ab11ff70b4a09e24c79c5dc26c201"
    original = PROJECT / "reports/wifi_csi_baseline_report"
    provenance = json.loads((original / "report_provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["source_sha256"].items():
        assert sha(PROJECT / name) == expected, f"冻结来源变化：{name}"
    assert (HERE / "report_data.tex").read_bytes() == (original / "report_data.tex").read_bytes()
    base = PROJECT / "results/formal_seed42"
    history = rows(base / "metrics.csv")
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((base / "test/test_metrics.json").read_text(encoding="utf-8"))
    assert [int(row["epoch"]) for row in history] == list(range(1, 27))
    assert summary["best_epoch"] == 16 and summary["epochs_completed"] == 26
    assert round(float(history[15]["val_accuracy"]) * 100, 4) == 27.1429
    assert round(metrics["accuracy"] * 100, 4) == 26.4286
    assert round(metrics["macro_f1"], 6) == 0.257181
    matrix_rows = rows(base / "test/confusion_matrix.csv")
    matrix = [[int(row[name]) for name in metrics["class_names"]] for row in matrix_rows]
    assert matrix == metrics["confusion_matrix"] and sum(map(sum, matrix)) == 560
    assert sum(matrix[i][i] for i in range(7)) == 148
    splits = rows(PROJECT / "results/dataset_split.csv")
    assert Counter(r["split"] for r in splits) == {"train": 1400, "val": 280, "test": 560}
    for split, people in {"train": {1, 2, 3, 4, 5}, "val": {6}, "test": {7, 8}}.items():
        assert {int(r["participant_id"]) for r in splits if r["split"] == split} == people
    data = (HERE / "report_data.tex").read_text(encoding="utf-8")
    for row in rows(base / "test/classification_report.csv"):
        for key in ("precision", "recall", "f1"):
            assert f"{float(row[key]) * 100:.2f}" in data
    assert 10048 + 24704 + 903 == 35655
    source = (HERE / "main.tex").read_text(encoding="utf-8")
    assert not re.search(r"\\(?:newpage|clearpage)\b", source)
    assert not re.search(r"\\includegraphics\[[^\]]*height=", source)
    log = (HERE / "build/main.log").read_text(encoding="utf-8", errors="replace")
    for bad in ("Overfull", "Missing character", "There were undefined references", "undefined on input"):
        assert bad not in log, bad
    pdf = HERE / "build/main.pdf"
    page_checks = []
    with pdfplumber.open(pdf) as document:
        assert len(document.pages) == 6
        text = "\n".join(page.extract_text() or "" for page in document.pages)
        for repeated in ("Training improves while validation loss rises", "The training-validation accuracy gap widens", "All 26 recorded epochs; raw epoch points"):
            assert repeated not in text
        for expected in ("27.1429", "26.4286", "0.257181", "35,655", "23.9286", "28.9286"):
            assert expected in text, expected
        for i, page in enumerate(document.pages, 1):
            content = page.extract_text() or ""
            assert content.startswith("ESP-Fi"), f"页眉不一致：{i}"
            outside = [c for c in page.chars if c["x0"] < 15 or c["x1"] > page.width - 15
                       or c["top"] < 10 or c["bottom"] > page.height - 10]
            assert not outside, f"字符越界：{i}"
            page_checks.append({"page": i, "characters": len(page.chars), "outside_page_safety_area": 0})
        assert "各类动作表现" in (document.pages[4].extract_text() or "")
        assert "来源审计与参考资料" in (document.pages[5].extract_text() or "")
        assert "参考文献" in (document.pages[5].extract_text() or "")
    reader = PdfReader(pdf)
    uris = set()
    fonts = {}
    for page in reader.pages:
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A")
            if action and action.get("/URI"):
                uris.add(str(action["/URI"]))
        resources = page["/Resources"].get_object()
        for item in resources.get("/Font", {}).values():
            font = item.get_object()
            descendants = font.get("/DescendantFonts")
            actual = descendants[0].get_object() if descendants else font
            descriptor = actual.get("/FontDescriptor")
            if descriptor:
                descriptor = descriptor.get_object()
                embedded = any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3"))
                fonts[str(font.get("/BaseFont"))] = embedded
                assert embedded, "未嵌入字体"
    assert len(uris) == 5, uris
    result = {
        "status": "PASS",
        "original_files_byte_unchanged": len(before) - len(changed),
        "prior_build_intermediates_refreshed": changed,
        "frozen_provenance_inputs_byte_unchanged": len(provenance["source_sha256"]),
        "original_pdf_sha256": sha(PROJECT / "reports/wifi_csi_baseline_report.pdf"),
        "new_pdf_sha256": sha(pdf), "new_pdf_bytes": pdf.stat().st_size,
        "pages": page_checks, "fonts_embedded": fonts, "pdf_reference_uri_targets": sorted(uris),
        "epochs": len(history), "best_epoch": 16, "val_accuracy_percent": 27.1429,
        "test_accuracy_percent": metrics["accuracy"] * 100, "test_macro_f1": metrics["macro_f1"],
        "confusion_matrix_total": 560, "parameters": 35655,
        "report_data_byte_identical": True, "overfull_boxes": 0, "missing_glyphs": 0,
        "undefined_references": 0, "hard_pagebreak_commands": 0,
        "visual_qa": "人工逐页检查记录见 REVISION_LOG.md；自动检查不替代视觉检查",
    }
    (HERE / "layout_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "original_files_byte_unchanged", "epochs",
                     "best_epoch", "test_accuracy_percent", "test_macro_f1", "new_pdf_bytes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
