"""使用一个真实训练 Batch 检查前向与反向传播；不执行参数更新。

运行：python -B scripts/check_model.py
只读取 train，不读取验证集或测试集，不保存权重或实验结果。
"""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CSIDataset
from src.model import LightweightCNN


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check():
    # 固定初始化和读取顺序，使用 CPU 完成这一次功能检查。
    torch.manual_seed(42)
    dataset = CSIDataset("train")
    require(dataset.participant_ids == frozenset(range(1, 6)), "训练集参与者不符合固定划分")
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(42),
    )
    x, y = next(iter(loader))
    require(tuple(x.shape) == (32, 52, 950) and tuple(y.shape) == (32,), "Batch 形状错误")
    require(x.dtype == torch.float32 and y.dtype == torch.long, "Batch 类型错误")
    require(torch.isfinite(x).all().item(), "输入包含 NaN/Inf")
    require(((y >= 0) & (y <= 6)).all().item(), "标签超出 0–6")
    model = LightweightCNN()
    print(f"Device: {x.device}; split=train; participants={sorted(dataset.participant_ids)}")
    print(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    require(parameter_count == 35655, f"参数总量不符合指定结构: {parameter_count}")
    print(f"Total parameters: {parameter_count}")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    shapes = {}

    # Hook 记录实际 forward 的各层输出，避免为展示形状重复前向传播。
    def record_shape(name):
        def hook(module, inputs, output):
            shapes[name] = tuple(output.shape)
        return hook

    handles = [layer.register_forward_hook(record_shape(name)) for name, layer in model.named_children()]
    try:
        logits = model(x)  # 唯一一次模型前向传播
    finally:
        for handle in handles:
            handle.remove()
    expected_shapes = {
        "conv1": (32, 64, 950), "relu1": (32, 64, 950),
        "pool1": (32, 64, 475), "conv2": (32, 128, 475),
        "relu2": (32, 128, 475), "avg_pool": (32, 128, 1),
        "flatten": (32, 128), "classifier": (32, 7),
    }
    print(f"input: {tuple(x.shape)}")
    for name, expected in expected_shapes.items():
        print(f"{name}: {shapes.get(name)}")
        require(shapes.get(name) == expected, f"{name} 输出形状错误")
    require(tuple(logits.shape) == (32, 7), "最终输出形状错误")
    require(torch.isfinite(logits).all().item(), "logits 包含 NaN/Inf")
    print(f"Logits: dtype={logits.dtype}; all_finite=True")
    loss = nn.CrossEntropyLoss()(logits, y)
    require(loss.ndim == 0 and torch.isfinite(loss).item(), "Loss 不是有限标量")
    print(f"CrossEntropyLoss: value={loss.item():.8f}, shape={tuple(loss.shape)}, finite=True")

    loss.backward()  # 仅检查梯度连通性，不创建优化器，不执行 optimizer.step()
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        require(gradient is not None, f"缺少梯度: {name}")
        require(torch.isfinite(gradient).all().item(), f"非有限梯度: {name}")
        require(torch.count_nonzero(gradient).item() > 0, f"梯度全为零: {name}")
        require(torch.equal(before[name], parameter.detach()), f"参数发生变化: {name}")
        print(f"gradient {name}: shape={tuple(gradient.shape)}, finite=True, nonzero=True")
    print("PASS: shapes, logits, finite scalar loss, connected finite gradients; all parameters unchanged")
    print("本次 Loss 仅用于初始化模型的功能检查，不代表训练结果或识别性能。")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        check()
    except Exception as error:
        print(f"NEEDS ATTENTION: {error}", file=sys.stderr)
        raise
