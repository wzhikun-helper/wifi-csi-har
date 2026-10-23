"""从冻结矢量 PDF 制作排版副本，只移除三条重复说明的文字绘制指令。"""
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "figures/formal_seed42"
TITLES = {
    "loss_curve": "Training improves while validation loss rises",
    "accuracy_curve": "The training-validation accuracy gap widens",
}
FOOTERS = {
    "Train n=1400; validation n=280 | Seed 42; one run; no confidence intervals.",
    "All 26 recorded epochs; raw epoch points, without smoothing or resampling.",
}


def main():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    record = {}
    for name, title in TITLES.items():
        source = SOURCE / f"{name}.pdf"
        reader = PdfReader(source)
        page = reader.pages[0]
        stream = page.get_contents()
        original = list(stream.operations)
        remaining, removed = [], []
        for operands, operator in original:
            text = ""
            if operator == b"TJ":
                text = "".join(value for value in operands[0] if isinstance(value, str))
            elif operator == b"Tj":
                text = str(operands[0])
            if text in FOOTERS | {title}:
                removed.append(text)
            else:
                remaining.append((operands, operator))
        assert set(removed) == FOOTERS | {title} and len(removed) == 3
        # All non-text drawing operations (curves, markers, axes) remain byte-for-byte operands.
        original_graphics = [item for item in original if item[1] not in (b"TJ", b"Tj")]
        remaining_graphics = [item for item in remaining if item[1] not in (b"TJ", b"Tj")]
        assert original_graphics == remaining_graphics
        stream.operations = remaining
        page.replace_contents(stream)
        writer = PdfWriter()
        writer.add_page(page)
        target = assets / source.name
        with target.open("wb") as output:
            writer.write(output)
        extracted = PdfReader(target).pages[0].extract_text()
        assert all(text not in extracted for text in FOOTERS | {title})
        assert "Selected checkpoint: epoch 16" in extracted
        record[name] = {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "layout_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "removed_text_only": removed,
            "all_non_text_drawing_operations_unchanged": True,
        }
    (HERE / "curve_layout_provenance.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Two vector layout copies prepared; axes, legends, markers and curves preserved.")


if __name__ == "__main__":
    main()
