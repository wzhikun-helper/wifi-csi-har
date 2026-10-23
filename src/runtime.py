"""复现运行的设备与输出保护；不修改实验协议或创建目录。"""
from pathlib import Path

import torch


def require_new_output(path, protected=()):
    path = Path(path).resolve()
    for reserved in protected:
        reserved = Path(reserved).resolve()
        if path == reserved or path.is_relative_to(reserved) or reserved.is_relative_to(path):
            raise ValueError(f"Output overlaps frozen results: {path}")
    if path.exists():
        raise FileExistsError(f"Output already exists; refusing overwrite: {path}")
    return path


def reproduction_output(value, project_root):
    root = Path(project_root).resolve()
    path = (root / value).resolve()
    if path == root / "results" or not path.is_relative_to(root / "results"):
        raise ValueError("Reproduction output must be a new directory under project/results")
    return require_new_output(path, [root / "results/formal_seed42", root / "results/smoke_test"])


def select_device(requested="cuda", allow_cpu_fallback=False):
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu, cuda or cuda:N devices are supported")
    fallback = False
    if device.type == "cuda":
        if not torch.cuda.is_available():
            if not allow_cpu_fallback:
                raise RuntimeError("CUDA unavailable; use --device cpu or --allow-cpu-fallback explicitly")
            device, fallback = torch.device("cpu"), True
        else:
            index = device.index if device.index is not None else 0
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError(f"Invalid CUDA device index: {index}")
            device = torch.device(f"cuda:{index}")
    info = {
        "requested_device": requested, "actual_device": str(device),
        "cpu_fallback": fallback, "cpu_explicit": requested == "cpu",
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pytorch": str(torch.__version__), "cuda_build": torch.version.cuda,
    }
    return device, info
