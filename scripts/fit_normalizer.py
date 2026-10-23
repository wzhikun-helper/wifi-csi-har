"""仅用 P1–P5 的全部训练 MAT 拟合固定的 52 通道标准化参数。

运行：python -B scripts/fit_normalizer.py
float64 分块 Welford 统计，总体标准差 ddof=0；绝不覆盖已有输出。
保存后再次仅读取 train，验证 Dataset 标准化后的全部 1400 个样本。
"""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CSIDataset


class ChannelMoments:
    """合并 (时间位置,52) 数据块，避免平方和相减带来的消减误差。"""

    def __init__(self):
        self.count = 0
        self.mean = np.zeros(52, dtype=np.float64)
        self.m2 = np.zeros(52, dtype=np.float64)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 52 or not np.isfinite(values).all():
            raise ValueError("统计输入必须是有限的 (N,52) 数组")
        count = values.shape[0]
        if count == 0:
            raise ValueError("统计输入为空")
        mean = values.mean(axis=0, dtype=np.float64)
        m2 = np.square(values - mean).sum(axis=0, dtype=np.float64)
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total
        if not np.isfinite(self.mean).all() or not np.isfinite(self.m2).all():
            raise ValueError("float64 累积出现非有限值")

    def std(self):
        if self.count == 0 or (self.m2 < 0).any():
            raise ValueError("方差统计无效")
        return np.sqrt(self.m2 / self.count)


def print_json(label, value):
    print(f"{label}: {json.dumps(value, ensure_ascii=False, allow_nan=False)}", flush=True)


def fit(output):
    if output.exists():
        raise FileExistsError(f"已有结果需要保留，拒绝覆盖: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"输出目录不存在: {output.parent}")
    split_csv = PROJECT_ROOT / "results/dataset_split.csv"
    dataset = CSIDataset("train", split_csv=split_csv)
    if len(dataset) != 1400 or dataset.participant_ids != frozenset(range(1, 6)):
        raise ValueError("训练集样本数或参与者错误")
    if output.is_relative_to(dataset.data_root):
        raise ValueError("不得将输出写入原始数据目录")
    split_digest = hashlib.sha256(split_csv.read_bytes()).hexdigest()
    moments = ChannelMoments()
    content_hash, sample_ids_hash = hashlib.sha256(), hashlib.sha256()
    for index, sample in enumerate(dataset.samples, start=1):
        try:
            # 同一份字节既用于读取，也用于来源哈希；不读取 P6–P8 的 MAT。
            data = sample.mat_path.read_bytes()
            contents = loadmat(io.BytesIO(data), variable_names=["CSIamp"])
            amplitude = contents.get("CSIamp")
            if not isinstance(amplitude, np.ndarray) or amplitude.shape != (950, 52):
                raise ValueError("CSIamp 缺失或形状不是 (950,52)")
            if amplitude.dtype.kind not in "iuf" or not np.isfinite(amplitude).all():
                raise ValueError("CSIamp 不是有限实数数组")
            moments.update(amplitude)
            relative = sample.mat_path.relative_to(dataset.data_root).as_posix()
            content_hash.update(relative.encode("utf-8") + b"\0" + hashlib.sha256(data).digest())
            sample_ids_hash.update((sample.sample_id + "\n").encode("utf-8"))
        except Exception as error:
            raise RuntimeError(f"拟合失败，样本 {sample.sample_id}，路径 {sample.mat_path}: {error}") from error
        if index % 350 == 0:
            print(f"fit progress: {index}/1400", flush=True)
    if moments.count != 1400 * 950:
        raise ValueError("每通道统计位置数错误")
    mean, raw_std = moments.mean, moments.std()
    epsilon = 1e-8
    near_zero = raw_std < epsilon
    safe_std = np.where(near_zero, 1.0, raw_std)
    if hashlib.sha256(split_csv.read_bytes()).hexdigest() != split_digest:
        raise ValueError("拟合期间 split 文件发生变化")
    # 不含绝对路径或 pickle 对象；保存数据协议、统计轴和可核验哈希。
    with output.open("xb") as stream:
        np.savez_compressed(
            stream, schema_version=np.int64(1), mean=mean, std=safe_std, raw_std=raw_std,
            near_zero_mask=near_zero, epsilon=np.float64(epsilon),
            source_split=np.array("train"), participant_ids=np.arange(1, 6, dtype=np.int64),
            n_samples=np.int64(1400), values_per_channel=np.int64(moments.count),
            input_shape=np.array([950, 52], dtype=np.int64), ddof=np.int64(0),
            variable_name=np.array("CSIamp"), statistics_axis=np.array("all train samples and time positions, per channel"),
            split_sha256=np.array(split_digest), train_sample_ids_sha256=np.array(sample_ids_hash.hexdigest()),
            train_mat_tree_sha256=np.array(content_hash.hexdigest()),
            train_mat_hash_recipe=np.array("ordered relative POSIX path UTF-8 + NUL + binary file SHA256"),
            fitter_sha256=np.array(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        )
    print_json("parameters", {
        "path": str(output), "shape": list(mean.shape), "dtype": str(mean.dtype),
        "mean_range": [mean.min(), mean.max()], "raw_std_range": [raw_std.min(), raw_std.max()],
        "safe_std_range": [safe_std.min(), safe_std.max()], "samples": len(dataset),
        "values_per_channel": moments.count, "epsilon": epsilon,
        "near_zero_channel_indices_0_based": np.flatnonzero(near_zero).tolist(),
    })
    # 验证所有训练样本的真实 Dataset 输出；不创建 val/test Dataset。
    normalized = CSIDataset("train", split_csv=split_csv, normalizer_path=output)
    checked = ChannelMoments()
    count = 0
    for x, y in DataLoader(normalized, batch_size=32, shuffle=False, num_workers=0):
        if x.shape[1:] != (52, 950) or y.shape != (x.shape[0],):
            raise ValueError("标准化 Batch 形状错误")
        checked.update(x.numpy().transpose(0, 2, 1).reshape(-1, 52))
        count += x.shape[0]
    normalized_std = checked.std()
    if count != 1400 or checked.count != moments.count:
        raise ValueError("标准化核验样本覆盖不完整")
    if not np.allclose(checked.mean, 0, atol=1e-6):
        raise ValueError("标准化训练通道均值偏离 0")
    if not np.allclose(normalized_std[~near_zero], 1, atol=1e-6):
        raise ValueError("标准化训练通道标准差偏离 1")
    overall_mean = checked.mean.mean()
    overall_std = np.sqrt(np.mean(normalized_std**2 + (checked.mean - overall_mean)**2))
    print_json("normalized_all_train", {
        "samples": count, "overall_mean": overall_mean, "overall_std_population": overall_std,
        "channel_means": checked.mean.tolist(), "channel_stds_population": normalized_std.tolist(),
        "all_finite": True, "full_batch_shape": [32, 52, 950], "last_batch_shape": list(x.shape),
    })
    print("PASS: train-only float64 fitting and full training Dataset normalization verified", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/csi_normalizer.npz")
    args = parser.parse_args()
    try:
        fit(args.output.resolve())
    except Exception as error:
        print(f"NEEDS ATTENTION: {error}", file=sys.stderr)
        raise
