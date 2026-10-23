"""固定跨人员划分的 GPU 训练入口；默认配置仅执行两轮 Smoke Test。

运行：python -B scripts/run_experiment.py --config configs/baseline.yaml
只创建 train/val Dataset；绝不加载 test。已有输出目录会导致明确报错。
"""

import os

# CUDA 确定性矩阵运算的进程级设置，不修改系统环境配置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CSIDataset
from src.model import LightweightCNN


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_path(value):
    path = (PROJECT_ROOT / value).resolve()
    require(path.is_relative_to(PROJECT_ROOT), f"路径必须位于项目目录内: {value}")
    return path


def load_config(path):
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    require(isinstance(config, dict), "配置必须为映射")
    require(config["device"] == "cuda", "本入口要求显式 CUDA，不回退到 CPU")
    require(config["model"] == "LightweightCNN", "仅允许已有 LightweightCNN")
    require(config["optimizer"] == "Adam" and config["loss"] == "CrossEntropyLoss", "仅支持 Adam/CrossEntropyLoss")
    require(config["checkpoint_selection"] == "val_accuracy_then_lower_val_loss", "checkpoint 选择规则不符")
    require(type(config["batch_size"]) is int and config["batch_size"] > 0, "batch_size 必须为正整数")
    require(type(config["num_workers"]) is int and config["num_workers"] >= 0, "num_workers 无效")
    require(math.isfinite(config["learning_rate"]) and config["learning_rate"] > 0, "学习率无效")
    require(math.isfinite(config["weight_decay"]) and config["weight_decay"] >= 0, "weight_decay 无效")
    require(config["run_type"] in ("smoke_test", "formal"), "未知 run_type")
    if config["run_type"] == "smoke_test":
        require(config["epochs"] == 2, "Smoke Test 必须恰好为 2 Epoch")
        require(config["results_dir"] == "results/smoke_test" and config["checkpoints_dir"] == "models/smoke_test", "Smoke Test 必须隔离输出")
    else:
        frozen = {
            "seed": 42, "batch_size": 32, "optimizer": "Adam", "learning_rate": 0.001,
            "weight_decay": 0.0, "loss": "CrossEntropyLoss", "max_epochs": 40,
            "early_stopping_patience": 10, "results_dir": "results/formal_seed42",
            "checkpoints_dir": "models/formal_seed42", "normalizer_path": "models/csi_normalizer.npz",
        }
        require(all(config.get(key) == value for key, value in frozen.items()), "正式配置与预先冻结的 seed42 协议不符")
        require("epochs" not in config, "正式实验使用 max_epochs，避免轮数配置歧义")
    for key in ("split_csv", "data_root", "normalizer_path", "results_dir", "checkpoints_dir"):
        project_path(config[key])
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_loaders(config):
    datasets = {
        split: CSIDataset(
            split, split_csv=project_path(config["split_csv"]),
            data_root=project_path(config["data_root"]),
            normalizer_path=project_path(config["normalizer_path"]),
        )
        for split in ("train", "val")
    }
    require(datasets["train"].participant_ids == frozenset(range(1, 6)), "训练集人员错误")
    require(datasets["val"].participant_ids == frozenset({6}), "验证集人员错误")
    require(np.array_equal(datasets["train"].channel_mean, datasets["val"].channel_mean), "Train/Val 均值不同")
    require(np.array_equal(datasets["train"].channel_std, datasets["val"].channel_std), "Train/Val 标准差不同")
    return {
        split: DataLoader(
            data, batch_size=config["batch_size"], shuffle=(split == "train"),
            num_workers=config["num_workers"], pin_memory=True, drop_last=False,
            generator=torch.Generator().manual_seed(config["seed"] + offset),
        )
        for offset, (split, data) in enumerate(datasets.items())
    }


def finite(tensor, name):
    require(torch.isfinite(tensor).all().item(), f"NaN/Inf detected in {name}")


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, count, steps, max_gradient_norm = 0.0, 0, 0, 0, 0.0
    probe = None
    for x, y in loader:
        if probe is None:
            probe = x[:2].clone()  # 仅留在内存，用于 checkpoint 重载一致性检查。
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        finite(x, "train input")
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        finite(logits, "train logits")
        loss = criterion(logits, y)
        finite(loss, "train loss")
        loss.backward()
        squared_norm = 0.0
        for name, parameter in model.named_parameters():
            require(parameter.grad is not None, f"Missing gradient: {name}")
            finite(parameter.grad, f"gradient {name}")
            squared_norm += parameter.grad.detach().double().square().sum().item()
        gradient_norm = math.sqrt(squared_norm)
        require(math.isfinite(gradient_norm), "Non-finite overall gradient norm")
        max_gradient_norm = max(max_gradient_norm, gradient_norm)
        optimizer.step()
        for name, parameter in model.named_parameters():
            finite(parameter, f"updated weight {name}")
        batch_count = y.numel()
        loss_sum += loss.item() * batch_count
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += batch_count
        steps += 1
    require(count == len(loader.dataset), "训练未覆盖全部样本")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count, "steps": steps, "max_gradient_l2": max_gradient_norm}, probe


def validate_epoch(model, loader, criterion, device):
    before = cpu_state(model)
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            finite(x, "val input")
            logits = model(x)
            finite(logits, "val logits")
            loss = criterion(logits, y)
            finite(loss, "val loss")
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += y.numel()
    require(count == len(loader.dataset), "验证未覆盖全部样本")
    require(all(torch.equal(before[name], value) for name, value in cpu_state(model).items()), "验证阶段意外修改权重")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count, "parameters_unchanged": True}


def is_better(val, best):
    return best is None or (val["accuracy"], -val["loss"]) > (best["accuracy"], -best["loss"])


def save_checkpoint(path, model, optimizer, epoch, config, train, val, provenance, selected):
    payload = {
        "model_state_dict": cpu_state(model), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "config": config, "run_type": config["run_type"],
        "train_metrics": train, "val_metrics": val, "provenance": provenance,
        "selection": {"primary": "val_accuracy", "tie_break": "lower_val_loss", "selected_as_best_at_save": selected},
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)


def verify_checkpoint(path, expected_state, probe, expected_logits, device, config, provenance):
    saved = torch.load(path, map_location=device, weights_only=True)
    require(saved["config"] == config and saved["provenance"] == provenance, "重载配置或来源信息不一致")
    require(file_hash(project_path(saved["provenance"]["normalizer_path"])) == provenance["normalizer_sha256"], "标准化参数校验失败")
    reloaded = LightweightCNN().to(device)
    reloaded.load_state_dict(saved["model_state_dict"], strict=True)
    require(all(torch.equal(expected_state[name], value) for name, value in cpu_state(reloaded).items()), "重载权重不一致")
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(probe.to(device)).cpu()
    finite(actual, "reloaded logits")
    require(torch.allclose(actual, expected_logits, rtol=1e-6, atol=1e-7), "重载前后输出不一致")
    return {"weights_exact_match": True, "forward_matches": True, "config_matches": True, "normalizer_sha256_matches": True, "max_logit_difference": (actual - expected_logits).abs().max().item(), "epoch": saved["epoch"]}


def append_metrics(results_dir, row):
    """每完成一轮就持久化日志，出错时保留此前全部 Epoch。"""
    mode = "x" if row["epoch"] == 1 else "a"
    with (results_dir / "metrics.csv").open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
        if row["epoch"] == 1:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def write_status(results_dir, status):
    # 仅更新本次新建目录中属于本次运行的状态文件。
    with (results_dir / "status.json").open("w", encoding="utf-8") as stream:
        json.dump(status, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def export_summary(results_dir, summary):
    with (results_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)


def run(config_path):
    config = load_config(config_path)
    require(torch.cuda.is_available(), "GPU 不可用；停止，不回退到 CPU")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if config["run_type"] == "formal":
        require("RTX 3060 Laptop GPU" in gpu_name, f"正式实验 GPU 与预期不符: {gpu_name}")
    seed_everything(config["seed"])
    results_dir, checkpoints_dir = (project_path(config[key]) for key in ("results_dir", "checkpoints_dir"))
    require(not results_dir.exists() and not checkpoints_dir.exists(), "输出目录已存在，拒绝覆盖已有结果")
    protected = ["results/dataset_manifest.csv", config["split_csv"], config["normalizer_path"], "src/model.py", "src/dataset.py"]
    hashes = {name: file_hash(project_path(name)) for name in protected}
    provenance = {
        "normalizer_path": config["normalizer_path"], "normalizer_sha256": hashes[config["normalizer_path"]],
        "split_sha256": hashes[config["split_csv"]], "protected_file_sha256": hashes,
        "config_sha256": file_hash(config_path), "runner_sha256": file_hash(Path(__file__)),
        "train_participants": [1, 2, 3, 4, 5], "val_participants": [6], "test_used": False,
        "gpu": gpu_name, "pytorch": str(torch.__version__), "cuda_build": torch.version.cuda,
    }
    loaders = make_loaders(config)
    model = LightweightCNN().to(device)
    require(sum(p.numel() for p in model.parameters()) == 35655, "模型参数量不符合基线")
    initial = cpu_state(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    results_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    with (results_dir / "config_snapshot.yaml").open("xb") as stream:
        stream.write(config_path.read_bytes())
    require(file_hash(results_dir / "config_snapshot.yaml") == provenance["config_sha256"], "配置快照校验失败")
    print(f"RUN_TYPE={config['run_type']}; GPU={gpu_name}; config={json.dumps(config)}", flush=True)
    status = {"state": "running", "run_type": config["run_type"], "started_at_utc": datetime.now(timezone.utc).isoformat(), "last_completed_epoch": 0}
    write_status(results_dir, status)
    try:
        execute_run(config, config_path, model, loaders, optimizer, criterion, device, results_dir, checkpoints_dir, initial, hashes, provenance, gpu_name, status)
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_status(results_dir, status)
        raise


def execute_run(config, config_path, model, loaders, optimizer, criterion, device, results_dir, checkpoints_dir, initial, hashes, provenance, gpu_name, status):
    rows, best, best_path, best_state, best_logits, probe = [], None, None, None, None, None
    best_epoch, no_improvement = 0, 0
    limit = config["max_epochs"] if config["run_type"] == "formal" else config["epochs"]
    patience = config.get("early_stopping_patience")
    stop_reason = "max_epochs_reached"
    for epoch in range(1, limit + 1):
        status.update(current_epoch=epoch, phase="train")
        write_status(results_dir, status)
        epoch_initial = cpu_state(model)
        train, epoch_probe = train_epoch(model, loaders["train"], optimizer, criterion, device)
        changed = any(not torch.equal(epoch_initial[name], value) for name, value in cpu_state(model).items())
        require(changed, f"Epoch {epoch} 权重未更新")
        if probe is None:
            probe = epoch_probe
        status.update(phase="validation")
        write_status(results_dir, status)
        val = validate_epoch(model, loaders["val"], criterion, device)
        selected = is_better(val, best)
        no_improvement = 0 if selected else no_improvement + 1
        checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
        if selected or config["run_type"] == "smoke_test":
            save_checkpoint(checkpoint_path, model, optimizer, epoch, config, train, val, provenance, selected)
        if selected:
            best, best_path, best_state, best_epoch = val.copy(), checkpoint_path, cpu_state(model), epoch
            with torch.no_grad():
                best_logits = model(probe.to(device)).cpu()
        row = {
            "run_type": config["run_type"], "epoch": epoch,
            "train_loss": train["loss"], "train_accuracy": train["accuracy"],
            "val_loss": val["loss"], "val_accuracy": val["accuracy"],
            "train_samples": train["samples"], "val_samples": val["samples"],
            "optimizer_steps": train["steps"], "max_gradient_l2": train["max_gradient_l2"],
            "selected_as_best": selected, "epochs_without_improvement": no_improvement,
            "train_weights_updated": changed, "val_parameters_unchanged": val["parameters_unchanged"],
        }
        append_metrics(results_dir, row)
        rows.append(row)
        status.update(last_completed_epoch=epoch, best_epoch=best_epoch, best_val_metrics=best, epochs_without_improvement=no_improvement, phase="epoch_complete")
        write_status(results_dir, status)
        print(json.dumps(row), flush=True)
        if patience is not None and no_improvement >= patience:
            stop_reason = "early_stopping_patience_exhausted"
            break
    final = cpu_state(model)
    changes = {name: (value.double() - initial[name].double()).norm().item() for name, value in final.items()}
    require(any(norm > 0 for norm in changes.values()), "训练后权重未更新")
    require(any(not torch.equal(initial[name], value) for name, value in best_state.items()), "最佳 checkpoint 权重未更新")
    reload_result = verify_checkpoint(best_path, best_state, probe, best_logits, device, config, provenance)
    require(all(file_hash(project_path(name)) == digest for name, digest in hashes.items()), "受保护文件发生变化")
    require(file_hash(config_path) == provenance["config_sha256"], "运行期间配置发生变化")
    summary = {
        "run_type": config["run_type"], "formal_experiment": config["run_type"] == "formal", "config": config,
        "gpu": gpu_name, "provenance": provenance, "epochs_completed": len(rows),
        "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "best_checkpoint": best_path.relative_to(PROJECT_ROOT).as_posix(), "best_val_metrics": best,
        "best_epoch": best_epoch, "max_epochs": limit, "early_stopping_patience": patience,
        "early_stopping_triggered": stop_reason == "early_stopping_patience_exhausted",
        "stop_reason": stop_reason, "epochs_without_improvement": no_improvement, "run_status": "completed",
        "selection_rule": "higher val_accuracy; exact accuracy tie uses lower val_loss; exact tie is not improvement",
        "checkpoint_reload": reload_result, "final_weight_delta_l2": changes,
        "all_input_logits_loss_gradients_weights_finite": True,
        "protected_files_unchanged": True, "test_used": False,
    }
    export_summary(results_dir, summary)
    status.update(state="completed", phase="complete", stop_reason=stop_reason, finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_status(results_dir, status)
    print("SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"PASS: {config['run_type']} 运行结束；未使用测试集。", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/baseline.yaml")
    args = parser.parse_args()
    try:
        run(args.config.resolve())
    except Exception as error:
        print(f"NEEDS ATTENTION: {error}", file=sys.stderr)
        raise
