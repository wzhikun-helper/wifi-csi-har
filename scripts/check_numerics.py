"""诊断三个真实训练 Batch；可加载固定标准化参数，不训练或更新模型。

运行：python -B scripts/check_numerics.py
标准化诊断：python -B scripts/check_numerics.py --normalizer models/csi_normalizer.npz
固定模型/采样种子 42、batch_size=32，在 CPU 上报告数值和梯度。
统计仅代表抽取的 96 个训练样本，不是全训练集拟合结果。
"""

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CSIDataset
from src.model import LightweightCNN


def require_finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} 包含 NaN/Inf")


def statistics(tensor):
    # 使用 float64 累积统计；std 为总体标准差（correction=0）。
    flat = tensor.detach().double().reshape(-1)
    require_finite(flat, "待统计数值")
    levels = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    quantiles = torch.quantile(flat, torch.tensor(levels, dtype=torch.float64))
    return {
        "min": flat.min().item(), "max": flat.max().item(),
        "mean": flat.mean().item(), "std_population": flat.std(correction=0).item(),
        "quantiles": {f"p{100 * q:g}": value for q, value in zip(levels, quantiles.tolist())},
    }


def report(name, value):
    print(f"{name}: {json.dumps(value, ensure_ascii=False, allow_nan=False)}", flush=True)


def check(normalizer_path=None):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = CSIDataset("train", normalizer_path=normalizer_path)
    if dataset.participant_ids != frozenset(range(1, 6)):
        raise ValueError("训练集人员与固定划分不符")
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    # 模型只初始化一次；三个 Batch 均在相同参数下独立计算梯度。
    model = LightweightCNN()
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    for name, parameter in model.named_parameters():
        require_finite(parameter, name)
    criterion = nn.CrossEntropyLoss()
    print(f"seed={seed}; device=cpu; train_only=True; batches=3; batch_size=32; std=population", flush=True)
    print(f"fixed_normalizer={dataset.normalizer_path}", flush=True)
    uniform_loss = math.log(7)
    print(f"七分类均匀预测理论 Loss=ln(7)={uniform_loss:.8f}；随机模型不必等于该值。", flush=True)
    inputs, logits_list, target_list, loss_list = [], [], [], []
    gradient_norms = []
    for number, (x, y) in enumerate(itertools.islice(loader, 3), start=1):
        require_finite(x, f"Batch {number} input")
        if x.shape != (32, 52, 950) or y.shape != (32,) or y.dtype != torch.long:
            raise ValueError("Batch 形状或标签 dtype 错误")
        if not ((y >= 0) & (y <= 6)).all().item():
            raise ValueError("标签不在 0–6")
        model.zero_grad(set_to_none=True)  # 清除前一个 Batch 的梯度，不更新参数。
        logits = model(x)
        require_finite(logits, f"Batch {number} logits")
        loss = criterion(logits, y)
        require_finite(loss, f"Batch {number} loss")
        if loss.ndim != 0:
            raise ValueError("Loss 必须为标量")
        log_probabilities = logits.detach().log_softmax(dim=1)
        probabilities = log_probabilities.exp()
        true_log_probabilities = log_probabilities.gather(1, y[:, None]).squeeze(1)
        entropy = -(probabilities * log_probabilities).sum(dim=1)
        report(f"batch_{number}_input", statistics(x))
        report(f"batch_{number}_logits", statistics(logits))
        report(f"batch_{number}_prediction", {
            "target_counts_0_to_6": torch.bincount(y, minlength=7).tolist(),
            "prediction_counts_0_to_6": torch.bincount(logits.argmax(dim=1), minlength=7).tolist(),
            "mean_max_probability": probabilities.max(dim=1).values.mean().item(),
            "mean_entropy": entropy.mean().item(),
            "mean_within_sample_logit_span": (logits.max(dim=1).values - logits.min(dim=1).values).mean().item(),
            "true_class_nll_min": (-true_log_probabilities).min().item(),
            "true_class_nll_max": (-true_log_probabilities).max().item(),
            "cross_entropy": loss.item(),
        })
        loss.backward()
        norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise ValueError(f"缺少梯度: {name}")
            require_finite(parameter.grad, f"Batch {number} gradient {name}")
            norms[name] = parameter.grad.detach().double().norm().item()
        overall = math.sqrt(sum(value * value for value in norms.values()))
        if not math.isfinite(overall):
            raise ValueError("整体梯度范数非有限")
        report(f"batch_{number}_gradient_l2", {**norms, "overall": overall})
        gradient_norms.append(overall)
        inputs.append(x.detach())
        logits_list.append(logits.detach())
        target_list.append(y.detach())
        loss_list.append(loss.item())
    if len(inputs) != 3:
        raise ValueError("未读取到三个完整 Batch")
    for name, parameter in model.named_parameters():
        if not torch.equal(initial[name], parameter.detach()):
            raise ValueError(f"参数被意外修改: {name}")
    all_inputs = torch.cat(inputs)
    all_logits = torch.cat(logits_list)
    report("aggregate_96_samples_input", statistics(all_inputs))
    report("aggregate_96_samples_logits", statistics(all_logits))
    report("aggregate_prediction_counts_0_to_6", torch.bincount(all_logits.argmax(dim=1), minlength=7).tolist())
    report("aggregate_target_counts_0_to_6", torch.bincount(torch.cat(target_list), minlength=7).tolist())
    report("summary", {
        "mean_cross_entropy": sum(loss_list) / len(loss_list), "uniform_reference": uniform_loss,
        "overall_gradient_l2_by_batch": gradient_norms,
        "input_logits_loss_gradients_all_finite": True, "all_parameters_unchanged": True,
    })
    print("数值检查完成：未拟合统计量、未执行优化器步骤；仅按选项应用固定标准化，Loss 降低不是通过条件。", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalizer", type=Path, default=None)
    args = parser.parse_args()
    try:
        check(args.normalizer)
    except Exception as error:
        print(f"NEEDS ATTENTION: {error}", file=sys.stderr)
        raise
