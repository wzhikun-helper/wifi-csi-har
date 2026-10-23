"""Read real samples and validate Dataset/DataLoader behavior without training.

Run: python -B scripts/check_dataloader.py
Only a single sample and the first batch per split are checked numerically.
The validation batch is independently compared against all 32 source MAT arrays.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import RandomSampler, SequentialSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import PARTICIPANTS, SPLIT_COUNTS, create_dataloaders


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check():
    loaders = create_dataloaders(batch_size=32, num_workers=0)
    all_ids = set()
    groups = []
    for split, loader in loaders.items():
        dataset = loader.dataset
        require(len(dataset) == SPLIT_COUNTS[split], f"{split}: wrong dataset size")
        require(dataset.participant_ids == PARTICIPANTS[split], f"{split}: wrong participants")
        ids = {sample.sample_id for sample in dataset.samples}
        require(len(ids) == len(dataset) and not (all_ids & ids), "Duplicate/overlapping samples")
        all_ids.update(ids)
        groups.append(dataset.participant_ids)
        labels = {sample.activity_id - 1 for sample in dataset.samples}
        require(labels == set(range(7)), f"{split}: label metadata must cover 0-6")
        sampler_type = RandomSampler if split == "train" else SequentialSampler
        require(isinstance(loader.sampler, sampler_type), f"{split}: incorrect shuffle setting")
        single_x, single_y = dataset[0]
        require(tuple(single_x.shape) == (52, 950), "Wrong single-sample shape")
        require(single_x.dtype == torch.float32 and single_y.dtype == torch.long, "Wrong sample dtype")
        require(single_y.ndim == 0 and 0 <= single_y.item() <= 6, "Invalid scalar label")
        require(torch.isfinite(single_x).all().item(), "Non-finite sample values")
        batch_x, batch_y = next(iter(loader))
        require(tuple(batch_x.shape) == (32, 52, 950), f"{split}: wrong X batch shape")
        require(tuple(batch_y.shape) == (32,), f"{split}: wrong y batch shape")
        require(batch_x.dtype == torch.float32 and batch_y.dtype == torch.int64, "Wrong batch dtype")
        require(torch.isfinite(batch_x).all().item(), f"{split}: non-finite batch")
        require(((batch_y >= 0) & (batch_y <= 6)).all().item(), "Invalid batch labels")
        print(f"{split}: length={len(dataset)}, participants={sorted(dataset.participant_ids)}, shuffle={split == 'train'}")
        print(f"  single: X={tuple(single_x.shape)}, y={tuple(single_y.shape)}, dtypes={single_x.dtype}/{single_y.dtype}")
        print(f"  batch: X={tuple(batch_x.shape)}, y={tuple(batch_y.shape)}, dtypes={batch_x.dtype}/{batch_y.dtype}, label_range={batch_y.min().item()}..{batch_y.max().item()}, all_finite=True")
        print(f"  all dataset metadata labels={sorted(labels)}")
        if split == "val":
            for index, sample in enumerate(dataset.samples[:32]):
                reference = loadmat(sample.mat_path, variable_names=["CSIamp"])["CSIamp"]
                expected = torch.from_numpy(np.array(reference.T, dtype=np.float32, order="C", copy=True))
                require(torch.equal(batch_x[index], expected), f"Source/batch mismatch: {sample.mat_path}")
                require(batch_y[index].item() == sample.activity_id - 1, f"Label mismatch: {sample.mat_path}")
                if index < 3:
                    print(f"  verified batch[{index}]: path={sample.mat_path}, activity_id={sample.activity_id}, y={batch_y[index].item()}, X={tuple(batch_x[index].shape)}")
            print("  source correspondence: all 32 paths, labels and tensor values PASS")
    require(len(all_ids) == 2240, "Incomplete sample coverage")
    for index, left in enumerate(groups):
        for right in groups[index + 1:]:
            require(not left & right, "Participants overlap between splits")
    print("PASS: fixed splits, sample/batch formats, labels, finite values and source correspondence")
    print("Scope: numerical checks cover only the samples/batches read above, not all 2240 MAT arrays.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        check()
    except Exception as error:
        print(f"NEEDS ATTENTION: {error}", file=sys.stderr)
        raise
