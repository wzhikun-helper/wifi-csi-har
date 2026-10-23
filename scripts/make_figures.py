"""从冻结 CSV 生成四张正式图和分析；不读取 MAT、不运行网络。

python -B scripts/make_figures.py
重新生成到新目录：python -B scripts/make_figures.py --output-dir figures/formal_seed42_reproduced --analysis-path results/formal_seed42/analysis_reproduced.md
默认拒绝覆盖已有图表或分析，保存 PNG (300 dpi)、PDF、SVG。
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wifi_csi_har_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "results/formal_seed42"
TEST = BASE / "test"
NAMES = ("running", "falling", "walking", "turning", "jumping", "squatting", "waving")
COLORS = ("#0072B2", "#D55E00", "#009E73")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def recompute(rows):
    matrix = np.zeros((7, 7), dtype=np.int64)
    for row in rows:
        true, predicted = int(row["true_label"]), int(row["predicted_label"])
        require(true in range(7) and predicted in range(7), "标签越界")
        matrix[true, predicted] += 1
    tp = matrix.diagonal().astype(float)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(7), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(7), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(7), where=(support + predicted) != 0)
    return matrix, precision, recall, f1, float(tp.sum() / len(rows)), float(f1.mean())


def load_sources():
    paths = [BASE / "metrics.csv", TEST / "predictions.csv", TEST / "confusion_matrix.csv", TEST / "classification_report.csv",
             BASE / "config_snapshot.yaml", BASE / "summary.json", BASE / "status.json", TEST / "test_metrics.json", TEST / "evaluation_metadata.json"]
    hashes = {p.relative_to(PROJECT).as_posix(): sha256(p) for p in paths}
    history, predictions, matrix_rows, report = (read_csv(p) for p in paths[:4])
    summary = json.loads((BASE / "summary.json").read_text(encoding="utf-8"))
    frozen_metrics = json.loads((TEST / "test_metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((TEST / "evaluation_metadata.json").read_text(encoding="utf-8"))
    require(len(history) == 26 and [int(r["epoch"]) for r in history] == list(range(1, 27)), "必须包含连续完整的 26 Epoch")
    require(all(r["run_type"] == "formal" for r in history), "不得混入 Smoke Test")
    require(summary["run_status"] == "completed" and summary["best_epoch"] == 16 and metadata["status"] == "completed", "正式运行/测试状态错误")
    best = max(history, key=lambda r: (float(r["val_accuracy"]), -float(r["val_loss"])))
    require(int(best["epoch"]) == summary["best_epoch"] and float(best["val_accuracy"]) == summary["best_val_metrics"]["accuracy"], "最佳 Epoch 与冻结记录不一致")
    require(len(predictions) == len({r["sample_id"] for r in predictions}) == 560, "测试预测不完整或重复")
    require({int(r["participant_id"]) for r in predictions} == {7, 8}, "测试人员错误")
    matrix, precision, recall, f1, accuracy, macro_f1 = recompute(predictions)
    require(len(matrix_rows) == len(report) == 7, "类别数错误")
    require(tuple(r["true_activity"] for r in matrix_rows) == NAMES, "混淆矩阵行序错误")
    csv_matrix = np.array([[int(r[name]) for name in NAMES] for r in matrix_rows])
    require(np.array_equal(matrix, csv_matrix) and matrix.sum() == 560, "混淆矩阵与预测不一致")
    require(np.array_equal(matrix.sum(axis=1), np.full(7, 80)), "各类 Support 不等于 80")
    for label, row in enumerate(report):
        require(int(row["label"]) == label and row["activity_name"] == NAMES[label], "报告类别映射错误")
        for key, vector in (("precision", precision), ("recall", recall), ("f1", f1)):
            require(math.isclose(float(row[key]), vector[label], abs_tol=1e-12), f"{key} 与预测不一致")
        require(int(row["support"]) == 80, "报告 Support 错误")
    require(math.isclose(accuracy, frozen_metrics["accuracy"], abs_tol=1e-12) and math.isclose(macro_f1, frozen_metrics["macro_f1"], abs_tol=1e-12), "测试指标不一致")
    for name, expected in metadata["output_sha256"].items():
        require(sha256(TEST / name) == expected, f"冻结测试输出哈希变化: {name}")
    for name, expected in summary["provenance"]["protected_file_sha256"].items():
        require(sha256(PROJECT / name) == expected, f"训练来源哈希变化: {name}")
    participants = {}
    for person in (7, 8):
        rows = [r for r in predictions if int(r["participant_id"]) == person]
        _, _, _, _, pa, pf = recompute(rows)
        require(len(rows) == 280, "测试人员样本数错误")
        participants[person] = (pa, pf)
        require(math.isclose(pa, frozen_metrics["per_participant"][str(person)]["accuracy"], abs_tol=1e-12) and math.isclose(pf, frozen_metrics["per_participant"][str(person)]["macro_f1"], abs_tol=1e-12), "按人员指标不一致")
    for row in history:
        for field in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
            require(math.isfinite(float(row[field])), "曲线数据含非有限值")
    return history, report, csv_matrix, participants, accuracy, macro_f1, summary, hashes


def paper_style():
    """统一一次设置：可编辑矢量字体、白底、清晰字号与冗余编码。"""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, output, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    clipped = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1):
            clipped.append(text.get_text())
    require(not clipped, f"文字超出画布: {clipped}")
    files = []
    for extension in (".png", ".pdf", ".svg"):
        path = output / f"{name}{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
        files.append(path.relative_to(PROJECT).as_posix())
    plt.close(fig)
    return files


def curve(history, best_epoch, metric, output):
    epochs = np.array([int(r["epoch"]) for r in history])
    scale = 100 if metric == "accuracy" else 1
    train = np.array([float(r[f"train_{metric}"]) for r in history]) * scale
    val = np.array([float(r[f"val_{metric}"]) for r in history]) * scale
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.23, top=0.84)
    title = "Training improves while validation loss rises" if metric == "loss" else "The training-validation accuracy gap widens"
    fig.suptitle(title, x=0.12, y=0.96, ha="left", fontsize=12, fontweight="bold")
    for values, label, color, marker, style in ((train, "Train (P1-P5)", COLORS[0], "o", "-"), (val, "Validation (P6)", COLORS[1], "s", "--")):
        line, = ax.plot(epochs, values, label=label, color=color, marker=marker, markersize=3.4, linewidth=1.5, linestyle=style)
        require(np.array_equal(line.get_xdata(), epochs) and np.array_equal(line.get_ydata(), values), "曲线 artist 数值不一致")
    ax.axvline(best_epoch, color="#555555", linestyle=":", linewidth=1.2, label=f"Selected checkpoint: epoch {best_epoch}")
    ax.scatter([best_epoch], [val[best_epoch - 1]], s=65, facecolors="none", edgecolors="#222222", linewidths=1.2, zorder=5)
    ax.set(xlabel="Epoch", ylabel="Accuracy (%)" if metric == "accuracy" else "Cross-entropy loss (dimensionless)", xlim=(0.5, len(history) + 0.5))
    ax.set_ylim(0, 100 if metric == "accuracy" else max(train.max(), val.max()) * 1.12)
    ax.set_xticks([1, 5, 10, best_epoch, 20, len(history)])
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.text(0.12, 0.105, "Train n=1400; validation n=280 | Seed 42; one run; no confidence intervals.", fontsize=8.5)
    fig.text(0.12, 0.055, "All 26 recorded epochs; raw epoch points, without smoothing or resampling.", fontsize=8.5)
    return save(fig, output, f"{metric}_curve")


def confusion(matrix, accuracy, macro_f1, output):
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.28, top=0.85)
    fig.suptitle("Substantial cross-person activity confusion remains", x=0.09, y=0.96, ha="left", fontsize=12, fontweight="bold")
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=int(matrix.sum(axis=1).max()), interpolation="nearest")
    require(np.array_equal(image.get_array(), matrix), "热力图 artist 数值不一致")
    for row in range(7):
        for col in range(7):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=10, color="white" if matrix[row, col] > 44 else "#222222")
    ax.set_xticks(range(7), NAMES, rotation=35, ha="right")
    ax.set_yticks(range(7), NAMES)
    ax.set(xlabel="Predicted activity", ylabel="True activity")
    bar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.035, ticks=[0, 20, 40, 60, 80])
    bar.set_label("Samples (count)")
    fig.text(0.09, 0.105, f"Test n=560 (P7/P8); n=80 per class | Accuracy={accuracy:.2%}; Macro-F1={macro_f1:.4f}", fontsize=8.3)
    fig.text(0.09, 0.055, "Rows=true; columns=predicted. Raw counts; one frozen checkpoint and one test run.", fontsize=8.3)
    return save(fig, output, "confusion_matrix")


def class_metrics(report, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.28, top=0.80)
    fig.suptitle("Falling has the highest F1; walking recall is limited", x=0.12, y=0.96, ha="left", fontsize=12, fontweight="bold")
    positions = np.arange(7)
    for offset, metric, color, hatch in zip((-1, 0, 1), ("precision", "recall", "f1"), COLORS, ("//", "xx", "..")):
        values = np.array([float(r[metric]) for r in report]) * 100
        bars = ax.bar(positions + offset * 0.24, values, width=0.24, color=color, label=metric.capitalize(), edgecolor="#333333", linewidth=0.45, hatch=hatch)
        require(np.array_equal(np.array([bar.get_height() for bar in bars]), values), "柱状图 artist 数值不一致")
    ax.set_xticks(positions, NAMES, rotation=25, ha="right")
    ax.set(xlabel="Activity", ylabel="Metric (%)", ylim=(0, 105))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.55, 0.91), ncol=3)
    walking = report[2]
    predicted_walking = round(int(walking["support"]) * float(walking["recall"]) / float(walking["precision"]))
    fig.text(0.12, 0.105, "Test n=560; support n=80 per activity | Single seed/split; no uncertainty intervals.", fontsize=8.3)
    fig.text(0.12, 0.055, f"Walking precision={float(walking['precision']):.0%} uses only {predicted_walking} predictions; recall={float(walking['recall']):.2%}.", fontsize=8.3)
    return save(fig, output, "per_class_metrics")


def analysis_text(history, report, matrix, participants, accuracy, macro_f1, summary, output):
    best, last = history[summary["best_epoch"] - 1], history[-1]
    errors = sorted(((int(matrix[i, j]), NAMES[i], NAMES[j]) for i in range(7) for j in range(7) if i != j), reverse=True)
    error_lines = "\n".join(f"| {true} → {pred} | {count} | {count / 80:.2%} |" for count, true, pred in errors[:8])
    class_lines = "\n".join(f"| {r['activity_name']} | {float(r['precision']):.2%} | {float(r['recall']):.2%} | {float(r['f1']):.6f} | {r['support']} |" for r in report)
    return f"""# ESP-Fi HAR：Seed 42 正式实验分析

## 实验协议与结果来源

固定跨人员划分：P1–P5 训练（1400 样本），P6 验证（280），P7–P8 测试（560）。四个环境均包含在各集合中；这不是留一环境评估。仅使用训练集拟合的 52 通道均值和总体标准差，所有集合共用 `models/csi_normalizer.npz`。

唯一模型为 35,655 参数 LightweightCNN。Seed=42、Batch Size=32、Adam、学习率 0.001、weight decay=0、CrossEntropyLoss、上限 40 Epoch、patience=10；GPU 为 RTX 3060 Laptop GPU。以 Val Accuracy 优先、相同时 Val Loss 更低选择 checkpoint；完全相同时不算改善。正式配置保留于 `results/formal_seed42/config_snapshot.yaml`。

实际完成 {len(history)} Epoch，第 16 轮后连续 10 轮无改善而早停。冻结模型为 `{summary['best_checkpoint']}`。测试只执行一次，未根据测试结果重选 Epoch 或调整预处理。本分析和图表只读取已有 CSV/元数据，没有运行模型或读取 MAT。

## 观察事实

最佳 Epoch={summary['best_epoch']}：Train Loss={float(best['train_loss']):.6f}、Train Accuracy={float(best['train_accuracy']):.4%}；Val Loss={float(best['val_loss']):.6f}、Val Accuracy={float(best['val_accuracy']):.4%}。

正式 Test Accuracy={accuracy:.4%}（{int(matrix.trace())}/560），Macro-F1={macro_f1:.6f}，固定包含全部七类，零分母按 0 处理。

| 测试人员 | 样本数 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| P7 | 280 | {participants[7][0]:.4%} | {participants[7][1]:.6f} |
| P8 | 280 | {participants[8][0]:.4%} | {participants[8][1]:.6f} |

P8 Accuracy 比 P7 高 {(participants[8][0] - participants[7][0]) * 100:.4f} 个百分点。两人的类别级错误结构可能不同；合并 Macro-F1 不等于两位人员 Macro-F1 的简单平均。这些差异是描述性观察，不代表具有统计显著性。

### 动作差异

| 动作 | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
{class_lines}

按 F1，falling 最容易识别（{float(report[1]['f1']):.6f}），walking 最难（{float(report[2]['f1']):.6f}）。walking 的 Precision=100% 仅来自 3 个预测、3 个正确，真实 80 个 walking 中仅识别 3 个，Recall=3.75%，不能据此声称行走识别优秀。

### 主要错误分类

混淆矩阵行为真实类别、列为预测类别；以下比例以相应真实类别的 80 个样本为分母。

| 真实 → 预测 | 数量 | 真实类别内比例 |
| --- | ---: | ---: |
{error_lines}

### 训练与验证走势

Train Loss 从 {float(history[0]['train_loss']):.6f} 降至 {float(last['train_loss']):.6f}，Train Accuracy 从 {float(history[0]['train_accuracy']):.4%} 升至 {float(last['train_accuracy']):.4%}。Val Loss 从 {float(history[0]['val_loss']):.6f} 升至 {float(last['val_loss']):.6f}，末轮 Val Accuracy={float(last['val_accuracy']):.4%}。训练优化正常，但验证损失总体上升、训练/验证差距扩大，存在过拟合或跨人员泛化不足的迹象。当前结果不支持良好的跨人员识别能力。

## 可能原因（尚未验证）

人员动作执行差异和传播条件差异可能引起输入分布偏移；有限训练人员与轻量模型的表示能力也可能影响泛化。平均时间池化可能削弱动作时序信息。这些都是待验证假设，不能仅凭当前混淆矩阵确定因果，也不能直接把特定混淆归因于动作相似性或采集质量。本阶段不为验证这些假设重跑测试或调整实验。

## 局限与解释边界

只有一个 seed、一次固定划分和两名测试参与者，没有重复实验的方差或置信区间。样本共享参与者和环境，不能将 560 个样本当作 560 名独立人员来估计泛化不确定性。P6 参与模型选择，其最优验证表现不属于独立测试证据。

各类别支持数均为 80，类别样本数平衡不代表各类同样容易识别。此实验未做跨环境留出评估，也未验证实时部署、在线流数据、硬件迁移或真实场景可靠性。不能将结果解释为跨环境泛化或真实部署能力。

## 图表及图注

图中准确率与 Precision/Recall/F1 均使用百分比；混淆矩阵使用原始计数。曲线完整保留 26 个 Epoch 的记录点，仅连接实际点，不平滑、不重采样、不补点。没有跨 seed/划分的重复实验，故不添加误差条或置信区间。

1. `loss_curve`：训练损失下降而验证损失总体升高，提示泛化差距。竖虚线标记验证规则选择的 Epoch 16；训练 n=1400、验证 n=280，Seed 42 单次运行。
2. `accuracy_curve`：训练准确率提升幅度大于验证集，最佳验证点为 Epoch 16；准确率单位为百分比。训练指标为该 Epoch 各训练 Batch 更新过程中的累计指标。
3. `confusion_matrix`：未见人员上的错误分布不均；行=真实、列=预测，每格为样本数。测试 n=560、每类 n=80，Epoch 16 单次测试。
4. `per_class_metrics`：falling 的 F1 最高；walking 的高 Precision 与低 Recall 并存。每类 n=80，全部来自同一冻结测试预测。

每张图保存 PNG（300 dpi）、PDF、SVG，目录 `{output.relative_to(PROJECT).as_posix()}/`；PNG 用于预览，PDF/SVG 保留可编辑文字。图形使用英文动作标签，避免中文字体缺失；颜色与线型/标记或填充纹理共同编码。

## 来源、数值核验与重新生成

绘图来源：`results/formal_seed42/metrics.csv`、`test/predictions.csv`、`test/confusion_matrix.csv`、`test/classification_report.csv`（后三者相对于正式结果目录）。从预测 CSV 独立复算总体、按人员和按类指标，并与冻结结果、报告和混淆矩阵核对；图形 artist 的曲线点、矩阵和柱高均与 CSV 数值核对。

生成脚本：`scripts/make_figures.py`。来源文件 SHA256、配置路径、图表清单、输出 SHA256 与自动版面检查记录于图表目录的 `figure_provenance.json` 和 `figure_manifest.csv`。自动几何检查不能替代实际图像检查。

本次命令（在项目根目录运行）：

```powershell
& 'D:\\miniconda\\envs\\wifi-csi\\python.exe' -B scripts/make_figures.py
```

保留现有产物并重新生成到新位置：

```powershell
python -B scripts/make_figures.py --output-dir figures/formal_seed42_reproduced --analysis-path results/formal_seed42/analysis_reproduced.md
```

脚本不覆盖已存在的输出，不修改冻结 CSV、checkpoint、标准化参数或实验配置；重新绘图不等于重新测试。当前图表结论仅适用于本次 ESP-Fi HAR 固定跨人员基线。
"""


def run(output, analysis):
    output, analysis = output.resolve(), analysis.resolve()
    require(output.is_relative_to(PROJECT / "figures") and analysis.is_relative_to(PROJECT / "results"), "输出必须位于项目 figures/ 和 results/ 内")
    require(not output.exists() and not analysis.exists(), "已有图表/分析需要保留；请指定新的输出位置")
    history, report, matrix, participants, accuracy, macro_f1, summary, hashes = load_sources()
    paper_style()
    output.mkdir(parents=True, exist_ok=False)
    contracts = [
        ("loss_curve", "Train loss declines while validation loss rises", ["results/formal_seed42/metrics.csv"]),
        ("accuracy_curve", "A training-validation accuracy gap develops", ["results/formal_seed42/metrics.csv"]),
        ("confusion_matrix", "Cross-person errors differ by activity", ["results/formal_seed42/test/confusion_matrix.csv", "results/formal_seed42/test/predictions.csv"]),
        ("per_class_metrics", "Falling leads F1; walking has low recall despite high precision", ["results/formal_seed42/test/classification_report.csv", "results/formal_seed42/test/predictions.csv"]),
    ]
    files_by_id = [curve(history, summary["best_epoch"], "loss", output), curve(history, summary["best_epoch"], "accuracy", output), confusion(matrix, accuracy, macro_f1, output), class_metrics(report, output)]
    manifest = [{"figure_id": name, "core_claim": claim, "source_csv": ";".join(sources), "outputs": ";".join(paths), "script": "scripts/make_figures.py"} for (name, claim, sources), paths in zip(contracts, files_by_id)]
    with (output / "figure_manifest.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(manifest)
    for name, expected in hashes.items():
        require(sha256(PROJECT / name) == expected, f"冻结来源发生变化: {name}")
    with analysis.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(analysis_text(history, report, matrix, participants, accuracy, macro_f1, summary, output))
    provenance = {
        "backend": "python/matplotlib", "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
        "figure_count": 4, "image_file_count": 12, "formats": ["png", "pdf", "svg"], "png_dpi": 300,
        "source_sha256": hashes, "script_sha256": sha256(Path(__file__)), "configuration": "results/formal_seed42/config_snapshot.yaml",
        "generation_command": "python -B scripts/make_figures.py", "contracts": manifest,
        "statistics": {"train_samples": 1400, "val_samples": 280, "test_samples": 560, "epochs": len(history), "selected_epoch": summary["best_epoch"], "seeds": 1, "intervals": "none; single seed and fixed split; samples are clustered by participant/environment"},
        "checks": {"prediction_report_matrix_consistency": True, "all_26_epoch_points_used": True, "artist_values_match_csv": True, "text_within_canvas": True, "source_files_unchanged": True},
        "visual_review": "pending manual inspection of all four PNG exports",
        "output_sha256": {name: sha256(PROJECT / name) for paths in files_by_id for name in paths},
        "analysis_path": analysis.relative_to(PROJECT).as_posix(), "analysis_sha256": sha256(analysis),
    }
    with (output / "figure_provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"figure_count": 4, "image_files": 12, "output": str(output), "analysis": str(analysis), "checks": provenance["checks"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "figures/formal_seed42")
    parser.add_argument("--analysis-path", type=Path, default=BASE / "analysis.md")
    args = parser.parse_args()
    run(args.output_dir, args.analysis_path)
