"""在独立新目录评价已验证的最佳权重；冻结正式测试目录永不写入。

仅预检示例：python -B scripts/evaluate.py --output-dir results/reproduction_eval --device cpu --check-only
只遍历 P7/P8 test，按 Dataset 顺序记录全部七类 softmax 分数。
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import argparse
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CSIDataset
from src.model import LightweightCNN
from src.dataset import load_normalizer
from src.runtime import reproduction_output, select_device

CLASS_NAMES = ("running", "falling", "walking", "turning", "jumping", "squatting", "waving")
LABELS = list(range(7))
CHECKPOINT = "models/formal_seed42/epoch_016.pt"
RESULTS = "results/formal_seed42/test"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, mode="x"):
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_path(value):
    path = (PROJECT_ROOT / value).resolve()
    require(path.is_relative_to(PROJECT_ROOT), "Source must be inside the project")
    return path


def verify_sources(checkpoint_path=None, training_summary=None):
    summary_path = source_path(training_summary or "results/formal_seed42/summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path = summary_path.parent / "config_snapshot.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(summary["run_status"] == "completed", "训练未完成")
    if training_summary is None:
        require(summary["formal_experiment"], "正式训练未完成")
        require(summary["best_epoch"] == 16 and summary["best_checkpoint"] == CHECKPOINT, "冻结 checkpoint 不匹配；停止而非重新选择")
    else:
        require(summary.get("reproduction_run") is True and not summary["formal_experiment"], "需要独立复现的训练记录")
        require(summary.get("test_used") is False, "复现训练不应使用test")
        with (summary_path.parent / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        require(bool(history), "缺少复现训练历史")
        chosen = max(history, key=lambda row: (float(row["val_accuracy"]), -float(row["val_loss"])))
        require(int(chosen["epoch"]) == summary["best_epoch"], "最佳权重必须由验证记录选择")
    require(summary["config"] == config, "正式配置快照不匹配")
    require((PROJECT_ROOT / "configs/formal_seed42.yaml").read_bytes() == config_path.read_bytes(), "当前配置与冻结快照不一致")
    require(config["run_type"] == "formal" and config["seed"] == 42 and config["batch_size"] == 32, "正式协议不匹配")
    require(config["checkpoint_selection"] == "val_accuracy_then_lower_val_loss", "选择规则不匹配")
    require(config["normalizer_path"] == "models/csi_normalizer.npz", "标准化来源路径不匹配")
    require(summary["provenance"]["config_sha256"] == digest(config_path), "配置哈希不匹配")
    selected_path = source_path(summary["best_checkpoint"])
    actual_path = source_path(checkpoint_path) if checkpoint_path is not None else selected_path
    require(digest(actual_path) == digest(selected_path), "Checkpoint必须是训练summary按验证规则选出的最佳权重，不能任意替换")
    if training_summary is None:
        frozen = json.loads((PROJECT_ROOT / "results/formal_seed42/test/evaluation_metadata.json").read_text(encoding="utf-8"))
        require(digest(actual_path) == frozen["checkpoint_sha256"], "原Epoch16 checkpoint SHA256不匹配")
    checkpoint = torch.load(actual_path, map_location="cpu", weights_only=True)
    require(checkpoint["epoch"] == summary["best_epoch"] and checkpoint["parameter_count"] == 35655, "checkpoint Epoch/模型不匹配")
    require(checkpoint["config"] == config and checkpoint["provenance"] == summary["provenance"], "checkpoint 配置/来源不匹配")
    require(checkpoint["val_metrics"] == summary["best_val_metrics"], "checkpoint 验证指标不匹配")
    require(checkpoint["selection"] == {"primary": "val_accuracy", "tie_break": "lower_val_loss", "selected_as_best_at_save": True}, "checkpoint 不是已选定的最佳模型")
    for name, expected in summary["provenance"]["protected_file_sha256"].items():
        require(digest(PROJECT_ROOT / name) == expected, f"训练来源文件发生变化: {name}")
    normalizer_path = PROJECT_ROOT / config["normalizer_path"]
    load_normalizer(normalizer_path, PROJECT_ROOT / config["split_csv"])
    with np.load(normalizer_path, allow_pickle=False) as saved:
        require(saved["source_split"].item() == "train" and saved["participant_ids"].tolist() == [1, 2, 3, 4, 5], "标准化不是来自 P1–P5")
        require(saved["n_samples"].item() == 1400 and saved["split_sha256"].item() == summary["provenance"]["split_sha256"], "标准化来源计数/划分不一致")
    protected = dict(summary["provenance"]["protected_file_sha256"])
    context = {"checkpoint": actual_path.relative_to(PROJECT_ROOT).as_posix(),
               "training_summary": summary_path.relative_to(PROJECT_ROOT).as_posix(),
               "checkpoint_epoch": summary["best_epoch"], "frozen_baseline": training_summary is None}
    names = [context["checkpoint"], "configs/formal_seed42.yaml", context["training_summary"],
             config_path.relative_to(PROJECT_ROOT).as_posix(), "scripts/run_experiment.py", "src/runtime.py", "scripts/evaluate.py"]
    if training_summary is None:
        names += ["results/formal_seed42/status.json", "results/formal_seed42/metrics.csv",
                  "results/formal_seed42/test/evaluation_metadata.json"]
    for name in names:
        protected[name] = digest(PROJECT_ROOT / name)
    return config, checkpoint, protected, context


def infer(model, loader, metadata_by_id, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y in loader:
            require(not model.training and not torch.is_grad_enabled(), "未处于只读推理模式")
            require(torch.isfinite(x).all().item(), "测试输入含 NaN/Inf")
            logits = model(x.to(device, non_blocking=True))
            require(logits.shape == (y.numel(), 7) and torch.isfinite(logits).all().item(), "logits 形状/数值异常")
            scores = logits.softmax(dim=1).cpu()
            require(torch.isfinite(scores).all().item(), "预测分数非有限")
            predictions = scores.argmax(dim=1)
            offset = len(rows)
            for index, sample in enumerate(loader.dataset.samples[offset:offset + y.numel()]):
                meta = metadata_by_id[sample.sample_id]
                actual, predicted = int(y[index]), int(predictions[index])
                require(actual == sample.activity_id - 1 == int(meta["activity_id"]) - 1, "标签映射错误")
                require(meta["activity_name"] == CLASS_NAMES[actual], "动作名称映射错误")
                row = {
                    "sample_id": sample.sample_id, "participant_id": sample.participant_id,
                    "environment_id": int(meta["environment_id"]), "mat_path": meta["mat_path"],
                    "true_label": actual, "predicted_label": predicted,
                    "true_activity_id": actual + 1, "predicted_activity_id": predicted + 1,
                    "true_activity_name": CLASS_NAMES[actual], "predicted_activity_name": CLASS_NAMES[predicted],
                    "predicted_score": float(scores[index, predicted]),
                }
                row.update({f"probability_{name}": float(scores[index, label]) for label, name in enumerate(CLASS_NAMES)})
                rows.append(row)
            print(f"test inference: {len(rows)}/560", flush=True)
    require(len(rows) == 560 and len({row['sample_id'] for row in rows}) == 560, "预测样本数量/唯一性不符")
    require({row["sample_id"] for row in rows} == set(metadata_by_id), "未完整覆盖固定 test")
    return rows


def calculate_metrics(rows):
    actual = [row["true_label"] for row in rows]
    predicted = [row["predicted_label"] for row in rows]
    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    require(matrix.shape == (7, 7) and matrix.sum() == 560, "混淆矩阵形状/计数不符")
    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=LABELS, zero_division=0)
    report = [
        {"label": label, "activity_id": label + 1, "activity_name": name,
         "precision": float(precision[label]), "recall": float(recall[label]),
         "f1": float(f1[label]), "support": int(support[label])}
        for label, name in enumerate(CLASS_NAMES)
    ]
    participants = {}
    for participant in (7, 8):
        subset = [row for row in rows if row["participant_id"] == participant]
        require(len(subset) == 280, "每位测试参与者必须为 280 个样本")
        participants[str(participant)] = {
            "samples": len(subset),
            "accuracy": accuracy_score([r["true_label"] for r in subset], [r["predicted_label"] for r in subset]),
            "macro_f1": f1_score([r["true_label"] for r in subset], [r["predicted_label"] for r in subset], labels=LABELS, average="macro", zero_division=0),
        }
    return {
        "samples": 560, "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0),
        "labels": LABELS, "class_names": list(CLASS_NAMES), "zero_division": 0,
        "confusion_matrix_orientation": "rows=true, columns=predicted",
        "confusion_matrix": matrix.tolist(), "per_class": report, "per_participant": participants,
    }


def independently_verify_predictions(path, expected, metadata_by_id):
    # 独立读取已保存 CSV，以整数计数和公式复算，不再次运行模型。
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 560 and len({r["sample_id"] for r in rows}) == 560, "预测 CSV 行数/唯一性异常")
    require({r["sample_id"] for r in rows} == set(metadata_by_id), "预测 CSV 样本覆盖异常")
    matrix = [[0] * 7 for _ in range(7)]
    for row in rows:
        actual, predicted = int(row["true_label"]), int(row["predicted_label"])
        meta = metadata_by_id[row["sample_id"]]
        require(actual in LABELS and predicted in LABELS, "预测 CSV 标签越界")
        require(int(row["true_activity_id"]) == actual + 1 == int(meta["activity_id"]), "CSV 真实标签错误")
        require(int(row["predicted_activity_id"]) == predicted + 1, "CSV 预测标签错误")
        require(row["true_activity_name"] == CLASS_NAMES[actual] and row["predicted_activity_name"] == CLASS_NAMES[predicted], "CSV 类别映射错误")
        require(row["participant_id"] == meta["participant_id"] and row["environment_id"] == meta["environment_id"], "CSV 元数据错误")
        probabilities = [float(row[f"probability_{name}"]) for name in CLASS_NAMES]
        require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities) and abs(sum(probabilities) - 1) < 1e-6, "CSV 概率异常")
        require(max(range(7), key=probabilities.__getitem__) == predicted and float(row["predicted_score"]) == probabilities[predicted], "CSV 预测分数不匹配")
        matrix[actual][predicted] += 1
    accuracy = sum(matrix[i][i] for i in LABELS) / len(rows)
    f1s = []
    for i in LABELS:
        denominator = sum(matrix[i]) + sum(matrix[j][i] for j in LABELS)
        f1s.append(2 * matrix[i][i] / denominator if denominator else 0.0)
    macro_f1 = sum(f1s) / 7
    require(matrix == expected["confusion_matrix"], "CSV 复算混淆矩阵不一致")
    require(abs(accuracy - expected["accuracy"]) < 1e-12 and abs(macro_f1 - expected["macro_f1"]) < 1e-12, "CSV 复算指标不一致")
    return {"samples": len(rows), "unique_samples": len(rows), "accuracy": accuracy, "macro_f1": macro_f1, "confusion_matrix_sum": sum(map(sum, matrix)), "passed": True}


def run(output_dir=None, checkpoint_path=None, training_summary=None,
        requested_device="cuda", allow_cpu_fallback=False, check_only=False):
    require(output_dir is not None, "请显式指定新的 --output-dir；冻结正式测试目录永久禁止写入")
    output = reproduction_output(output_dir, PROJECT_ROOT)
    config, saved, protected, context = verify_sources(checkpoint_path, training_summary)
    device, runtime = select_device(requested_device, allow_cpu_fallback)
    print("RUNTIME=" + json.dumps(runtime, ensure_ascii=False), flush=True)
    if check_only:
        print("CHECK_ONLY_PASS: checkpoint/config/split/normalizer verified; no dataset, inference, metrics or writes")
        return {**runtime, **context, "output_dir": output.relative_to(PROJECT_ROOT).as_posix()}
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    dataset = CSIDataset("test", split_csv=PROJECT_ROOT / config["split_csv"], data_root=PROJECT_ROOT / config["data_root"], normalizer_path=PROJECT_ROOT / config["normalizer_path"])
    require(len(dataset) == 560 and dataset.participant_ids == frozenset({7, 8}), "测试人员/数量不符")
    with (PROJECT_ROOT / config["split_csv"]).open(encoding="utf-8", newline="") as stream:
        test_rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    metadata_by_id = {row["sample_id"]: row for row in test_rows}
    require(len(test_rows) == len(metadata_by_id) == 560, "测试清单有重复或数量不符")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda", drop_last=False)
    model = LightweightCNN().to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.requires_grad_(False)
    before = {name: p.detach().cpu().clone() for name, p in model.state_dict().items()}
    require(all(torch.isfinite(p).all().item() for p in before.values()), "checkpoint 含非有限权重")
    metadata = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": context["checkpoint"], "checkpoint_sha256": protected[context["checkpoint"]], "checkpoint_epoch": context["checkpoint_epoch"],
        "training_summary": context["training_summary"], "runtime": runtime, "reproduction_run": True,
        "normalizer_path": config["normalizer_path"], "normalizer_sha256": protected[config["normalizer_path"]],
        "normalizer_source": {"split": "train", "participants": [1, 2, 3, 4, 5], "samples": 1400},
        "config": config, "protected_file_sha256": protected, "evaluation_script_sha256": digest(__file__),
        "test_participants": [7, 8], "expected_samples": 560, "batch_size": 32, "shuffle": False,
        "gpu": runtime["gpu"], "pytorch": str(torch.__version__),
        "cuda_build": torch.version.cuda, "scipy": version("scipy"), "scikit_learn": version("scikit-learn"),
        "class_mapping": {str(i): {"activity_id": i + 1, "activity_name": name} for i, name in enumerate(CLASS_NAMES)},
        "prediction_score": "softmax probability; all seven scores saved; argmax ties choose first label",
        "zero_division": 0, "macro_f1_labels": LABELS, "model_selection_uses_test": False,
        "test_inference_passes": 1,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "evaluation_metadata.json", metadata)
    print(f"SOURCES_VERIFIED: selected epoch={context['checkpoint_epoch']}; train-fitted normalizer; test=P7/P8 only; independent reproduction output", flush=True)
    try:
        rows = infer(model, loader, metadata_by_id, device)
        metrics = calculate_metrics(rows)
        require(all(torch.equal(before[name], p.detach().cpu()) for name, p in model.state_dict().items()), "推理修改了模型参数")
        require(all(digest(PROJECT_ROOT / name) == value for name, value in protected.items()), "受保护文件发生变化")
        write_csv(output / "predictions.csv", rows, list(rows[0]))
        write_csv(output / "classification_report.csv", metrics["per_class"], list(metrics["per_class"][0]))
        matrix_rows = [{"true_activity": name, **dict(zip(CLASS_NAMES, metrics["confusion_matrix"][label]))} for label, name in enumerate(CLASS_NAMES)]
        write_csv(output / "confusion_matrix.csv", matrix_rows, ["true_activity", *CLASS_NAMES])
        write_json(output / "test_metrics.json", metrics)
        checks = independently_verify_predictions(output / "predictions.csv", metrics, metadata_by_id)
        metadata.update(
            status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
            independent_csv_recalculation=checks, model_parameters_unchanged=True, protected_files_unchanged=True,
            sample_ids_sha256=hashlib.sha256("".join(row["sample_id"] + "\n" for row in rows).encode()).hexdigest(),
            output_sha256={name: digest(output / name) for name in ("test_metrics.json", "predictions.csv", "confusion_matrix.csv", "classification_report.csv")},
        )
        write_json(output / "evaluation_metadata.json", metadata, mode="w")
        print("TEST_METRICS=" + json.dumps(metrics, ensure_ascii=False, allow_nan=False), flush=True)
        print("PASS: 独立复现评价结束；原正式测试结果未变。", flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(output / "evaluation_metadata.json", metadata, mode="w")
        raise


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Required new results/<name>; original formal/smoke trees are forbidden")
    parser.add_argument("--checkpoint", type=Path, help="Must match the selected checkpoint from the associated summary")
    parser.add_argument("--training-summary", type=Path, help="Completed reproduction summary; default is the frozen Epoch16 summary")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N or explicit cpu")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate provenance and device only; no test arrays, inference, metrics or files")
    args = parser.parse_args()
    run(args.output_dir, args.checkpoint, args.training_summary, args.device, args.allow_cpu_fallback, args.check_only)
