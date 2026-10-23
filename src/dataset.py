"""Fixed-split CSI samples; optional fixed, training-only channel normalization."""

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTICIPANTS = {"train": frozenset(range(1, 6)), "val": frozenset({6}), "test": frozenset({7, 8})}
SPLIT_COUNTS = {"train": 1400, "val": 280, "test": 560}


def load_normalizer(path, split_csv):
    """所有 split 使用同一训练集参数；只加载固定值，绝不重新拟合。"""
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "schema_version", "mean", "std", "raw_std", "near_zero_mask", "epsilon",
            "source_split", "participant_ids", "n_samples", "values_per_channel",
            "input_shape", "ddof", "variable_name", "split_sha256",
        }
        if not required.issubset(saved.files):
            raise ValueError(f"Missing normalizer metadata in {path}")
        if (
            saved["schema_version"].item() != 1
            or saved["source_split"].item() != "train"
            or saved["participant_ids"].tolist() != [1, 2, 3, 4, 5]
            or saved["n_samples"].item() != 1400
            or saved["values_per_channel"].item() != 1400 * 950
            or saved["input_shape"].tolist() != [950, 52]
            or saved["ddof"].item() != 0
            or saved["variable_name"].item() != "CSIamp"
        ):
            raise ValueError(f"Normalizer is not from the required training protocol: {path}")
        if saved["split_sha256"].item() != hashlib.sha256(Path(split_csv).read_bytes()).hexdigest():
            raise ValueError("Normalizer split SHA256 does not match the current split CSV")
        mean, std, raw_std = (saved[key].copy() for key in ("mean", "std", "raw_std"))
        mask = saved["near_zero_mask"].copy()
        epsilon = float(saved["epsilon"].item())
    if any(a.shape != (52,) or a.dtype != np.float64 or not np.isfinite(a).all() for a in (mean, std, raw_std)):
        raise ValueError("Normalizer mean/std must be finite float64 arrays of shape (52,)")
    if not np.isfinite(epsilon) or epsilon <= 0 or (raw_std < 0).any() or (std <= 0).any():
        raise ValueError("Invalid normalizer standard deviations or epsilon")
    if mask.shape != (52,) or mask.dtype != np.bool_ or not np.array_equal(mask, raw_std < epsilon):
        raise ValueError("Normalizer near-zero channel mask is inconsistent")
    if not np.array_equal(std, np.where(mask, 1.0, raw_std)):
        raise ValueError("Normalizer safe standard deviations are inconsistent")
    mean.setflags(write=False)
    std.setflags(write=False)
    return mean, std


@dataclass(frozen=True)
class Sample:
    sample_id: str
    mat_path: Path
    participant_id: int
    activity_id: int
    split: str


class CSIDataset(Dataset):
    """Read one existing split; return (float32 [52, 950], long scalar).

    ``samples`` exposes row metadata for auditing. Labels are activity_id - 1.
    Invalid metadata or MAT arrays raise explicit errors; no samples are dropped.
    """

    def __init__(self, split, split_csv=None, data_root=None, normalizer_path=None):
        if split not in PARTICIPANTS:
            raise ValueError(f"Unknown split {split!r}; expected train, val, or test")
        self.split = split
        self.data_root = Path(data_root if data_root is not None else PROJECT_ROOT / "data/raw/ESP-Fi_dataset").resolve(strict=True)
        if not self.data_root.is_dir():
            raise ValueError(f"Data root is not a directory: {self.data_root}")
        source = Path(split_csv if split_csv is not None else PROJECT_ROOT / "results/dataset_split.csv")
        required = {"sample_id", "mat_path", "participant_id", "activity_id", "environment_id", "repetition_id", "split"}
        all_samples = []
        seen = set()
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not required.issubset(reader.fieldnames):
                raise ValueError(f"Missing or duplicate columns in {source}")
            for line, row in enumerate(reader, start=2):
                try:
                    if None in row or any(not row.get(key) for key in required):
                        raise ValueError("Missing values or malformed row")
                    environment, participant, activity, repetition = (
                        int(row[key]) for key in ("environment_id", "participant_id", "activity_id", "repetition_id")
                    )
                    group = row["split"]
                    if group not in PARTICIPANTS or participant not in PARTICIPANTS[group]:
                        raise ValueError("Participant does not match the fixed split")
                    if environment not in range(1, 5) or activity not in range(1, 8) or repetition not in range(1, 11):
                        raise ValueError("Out-of-range sample metadata")
                    sample_id = row["sample_id"]
                    if sample_id != f"{environment}-{participant}-{activity}-{repetition}":
                        raise ValueError("sample_id does not match row metadata")
                    if sample_id in seen:
                        raise ValueError(f"Duplicate sample ID: {sample_id}")
                    relative = PurePosixPath(row["mat_path"])
                    if relative.is_absolute() or ".." in relative.parts or ":" in row["mat_path"] or "\\" in row["mat_path"]:
                        raise ValueError("Expected a safe data-root-relative POSIX mat_path")
                    path = (self.data_root / relative).resolve(strict=True)
                    if not path.is_relative_to(self.data_root) or not path.is_file():
                        raise ValueError("MAT path is not a file inside data_root")
                    if path.suffix != ".mat" or path.stem != sample_id:
                        raise ValueError("MAT filename does not match sample_id")
                    seen.add(sample_id)
                    all_samples.append(Sample(sample_id, path, participant, activity, group))
                except (OSError, ValueError, TypeError) as error:
                    raise ValueError(f"Invalid split CSV {source}, line {line}: {error}") from error
        counts = Counter(sample.split for sample in all_samples)
        if dict(counts) != SPLIT_COUNTS:
            raise ValueError(f"Unexpected split sizes: {dict(counts)}; expected {SPLIT_COUNTS}")
        self.samples = tuple(sample for sample in all_samples if sample.split == split)
        self.participant_ids = frozenset(sample.participant_id for sample in self.samples)
        if self.participant_ids != PARTICIPANTS[split]:
            raise ValueError(f"Unexpected participant set for {split}: {self.participant_ids}")
        # 默认不标准化，与已有检查脚本兼容。三个 split 都加载同一来源的固定参数。
        self.normalizer_path = Path(normalizer_path).resolve(strict=True) if normalizer_path is not None else None
        self.channel_mean, self.channel_std = (
            load_normalizer(self.normalizer_path, source) if self.normalizer_path is not None else (None, None)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            contents = loadmat(sample.mat_path, variable_names=["CSIamp"])
            if "CSIamp" not in contents:
                raise ValueError("Missing CSIamp variable")
            amplitude = contents["CSIamp"]
            if not isinstance(amplitude, np.ndarray) or amplitude.shape != (950, 52):
                raise ValueError(f"Expected CSIamp shape (950, 52), got {getattr(amplitude, 'shape', None)}")
            if amplitude.dtype.kind not in "iuf" or np.iscomplexobj(amplitude):
                raise ValueError(f"Expected real numeric amplitude, got dtype {amplitude.dtype}")
            if not np.isfinite(amplitude).all():
                raise ValueError("CSIamp contains NaN or Inf")
            with np.errstate(over="ignore", invalid="ignore"):
                if self.channel_mean is None:
                    converted = np.array(amplitude.T, dtype=np.float32, order="C", copy=True)
                else:
                    # 每个通道使用训练集全部样本拟合的统计量；计算后返回 float32。
                    normalized = (amplitude.T.astype(np.float64) - self.channel_mean[:, None]) / self.channel_std[:, None]
                    converted = np.array(normalized, dtype=np.float32, order="C", copy=True)
            if not np.isfinite(converted).all():
                raise ValueError("CSIamp contains non-finite values after float32 conversion")
            x = torch.from_numpy(converted)
            y = torch.tensor(sample.activity_id - 1, dtype=torch.long)
            return x, y
        except Exception as error:
            raise RuntimeError(f"Failed to read sample {sample.sample_id} at {sample.mat_path}: {error}") from error


def create_dataloaders(split_csv=None, data_root=None, batch_size=32, num_workers=0, seed=42, normalizer_path=None):
    """Shuffle training order only; Windows-friendly default uses no subprocesses."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            CSIDataset(split, split_csv=split_csv, data_root=data_root, normalizer_path=normalizer_path),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
            generator=generator if split == "train" else None,
        )
        for split in PARTICIPANTS
    }
