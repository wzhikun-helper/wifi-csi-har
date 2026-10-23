"""Offline regression tests: synthetic MAT/metadata only, no real dataset or fitting."""
import csv
import hashlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from scipy.io import savemat

from scripts import create_split, evaluate, run_experiment
from src.dataset import CSIDataset, Sample, load_normalizer
from src.model import LightweightCNN
from src.runtime import reproduction_output, require_new_output, select_device


class SyntheticFiles(unittest.TestCase):
    def setUp(self):
        # Keep test-owned files in the writable checkout, not a Windows sandbox
        # package Temp directory whose child-file ACLs may deny access.
        temp = tempfile.TemporaryDirectory(prefix=".pipeline_test_", dir=Path(__file__).resolve().parent)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def sample(self, array):
        path = self.root / "synthetic.mat"
        savemat(path, {"CSIamp": array})
        data = CSIDataset.__new__(CSIDataset)
        data.samples = (Sample("1-1-7-1", path, 1, 7, "train"),)
        data.channel_mean = data.channel_std = None
        return data

    def normalizer(self, **changes):
        split = self.root / "synthetic_split.csv"
        split.write_bytes(b"synthetic metadata for SHA256 only\n")
        values = dict(schema_version=np.int64(1), mean=np.zeros(52, dtype=np.float64),
                      std=np.ones(52, dtype=np.float64), raw_std=np.ones(52, dtype=np.float64),
                      near_zero_mask=np.zeros(52, dtype=np.bool_), epsilon=np.float64(1e-8),
                      source_split=np.array("train"), participant_ids=np.arange(1, 6, dtype=np.int64),
                      n_samples=np.int64(1400), values_per_channel=np.int64(1400 * 950),
                      input_shape=np.array([950, 52]), ddof=np.int64(0), variable_name=np.array("CSIamp"),
                      split_sha256=np.array(hashlib.sha256(split.read_bytes()).hexdigest()))
        values.update(changes)
        path = self.root / "synthetic_normalizer.npz"
        np.savez(path, **values)
        return path, split

    def test_valid_sample_shape_dtype_and_label(self):
        x, y = self.sample(np.ones((950, 52)))[0]
        self.assertEqual(tuple(x.shape), (52, 950))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.int64)
        self.assertEqual(y.item(), 6)

    def test_wrong_shape_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "shape"):
            self.sample(np.ones((52, 950)))[0]

    def test_nan_rejected(self):
        x = np.ones((950, 52)); x[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
            self.sample(x)[0]

    def test_inf_rejected(self):
        x = np.ones((950, 52)); x[0, 0] = np.inf
        with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
            self.sample(x)[0]

    def test_complex_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "real numeric"):
            self.sample(np.ones((950, 52), dtype=np.complex128))[0]

    def test_training_normalizer_accepted(self):
        mean, std = load_normalizer(*self.normalizer())
        self.assertEqual(mean.shape, (52,))
        np.testing.assert_array_equal(std, 1)
        self.assertFalse(mean.flags.writeable)

    def test_validation_normalizer_rejected(self):
        with self.assertRaisesRegex(ValueError, "training protocol"):
            load_normalizer(*self.normalizer(source_split=np.array("val")))

    def test_wrong_people_normalizer_rejected(self):
        with self.assertRaisesRegex(ValueError, "training protocol"):
            load_normalizer(*self.normalizer(participant_ids=np.array([6])))

    def test_split_hash_mismatch_rejected(self):
        path, split = self.normalizer()
        split.write_bytes(b"changed synthetic metadata")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_normalizer(path, split)

    def test_zero_std_rejected(self):
        with self.assertRaisesRegex(ValueError, "standard deviations"):
            load_normalizer(*self.normalizer(std=np.zeros(52, dtype=np.float64)))

    def test_nonempty_output_rejected_without_changes(self):
        output = self.root / "existing"; output.mkdir()
        marker = output / "sentinel"; marker.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            require_new_output(output)
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_empty_existing_output_also_rejected(self):
        self.root.joinpath("empty").mkdir()
        with self.assertRaises(FileExistsError):
            require_new_output(self.root / "empty")

    def test_frozen_output_forbidden_even_if_missing(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            reproduction_output("results/formal_seed42/new", self.root)

    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError):
            reproduction_output("../outside", self.root)

    def test_training_plan_creates_no_directory(self):
        with mock.patch.object(run_experiment, "PROJECT_ROOT", self.root):
            results, checkpoints = run_experiment.plan_outputs({}, "results/new_run")
        self.assertEqual(checkpoints, results / "checkpoints")
        self.assertFalse(results.exists())

    def test_training_existing_run_refused(self):
        output = self.root / "results/existing"; output.mkdir(parents=True)
        with mock.patch.object(run_experiment, "PROJECT_ROOT", self.root):
            with self.assertRaises(FileExistsError):
                run_experiment.plan_outputs({}, "results/existing")

    def test_evaluation_existing_run_refused_before_sources(self):
        output = self.root / "results/existing"; output.mkdir(parents=True)
        with mock.patch.object(evaluate, "PROJECT_ROOT", self.root), \
             mock.patch.object(evaluate, "verify_sources") as sources:
            with self.assertRaises(FileExistsError):
                evaluate.run("results/existing", requested_device="cpu")
        sources.assert_not_called()

    def test_training_dry_run_never_loads_dataset(self):
        config = dict(run_type="formal", device="cuda", normalizer_path="models/synthetic.npz",
                      split_csv="results/synthetic.csv", seed=42)
        with mock.patch.object(run_experiment, "PROJECT_ROOT", self.root), \
             mock.patch.object(run_experiment, "load_config", return_value=config), \
             mock.patch.object(run_experiment, "file_hash", return_value="synthetic_hash"), \
             mock.patch.object(run_experiment, "make_loaders") as loaders, \
             mock.patch("src.dataset.load_normalizer"):
            result = run_experiment.run(self.root / "synthetic.yaml", "results/dry_run", "cpu", dry_run=True)
        self.assertEqual(result["actual_device"], "cpu")
        loaders.assert_not_called()
        self.assertFalse((self.root / "results/dry_run").exists())

    def test_evaluation_check_only_never_infers_or_calculates_metrics(self):
        context = dict(checkpoint="models/synthetic.pt", checkpoint_epoch=16)
        with mock.patch.object(evaluate, "PROJECT_ROOT", self.root), \
             mock.patch.object(evaluate, "verify_sources", return_value=({}, {}, {}, context)), \
             mock.patch.object(evaluate, "CSIDataset") as dataset, \
             mock.patch.object(evaluate, "infer") as infer, \
             mock.patch.object(evaluate, "calculate_metrics") as metrics:
            evaluate.run("results/check_only", requested_device="cpu", check_only=True)
        dataset.assert_not_called(); infer.assert_not_called(); metrics.assert_not_called()
        self.assertFalse((self.root / "results/check_only").exists())

    def test_evaluation_default_output_refused_before_loading(self):
        with mock.patch.object(evaluate, "verify_sources") as sources:
            with self.assertRaisesRegex(ValueError, "output-dir"):
                evaluate.run()
        sources.assert_not_called()

    def test_evaluation_rejects_unselected_checkpoint_before_loading(self):
        config = dict(run_type="formal", seed=42, batch_size=32,
                      checkpoint_selection="val_accuracy_then_lower_val_loss",
                      normalizer_path="models/csi_normalizer.npz")
        config_bytes = yaml.safe_dump(config).encode("utf-8")
        for rel in ["configs/formal_seed42.yaml", "results/formal_seed42/config_snapshot.yaml"]:
            path = self.root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(config_bytes)
        summary = dict(run_status="completed", formal_experiment=True, best_epoch=16,
                       best_checkpoint=evaluate.CHECKPOINT, config=config,
                       provenance={"config_sha256": hashlib.sha256(config_bytes).hexdigest()})
        (self.root / "results/formal_seed42/summary.json").write_text(json.dumps(summary), encoding="utf-8")
        selected = self.root / evaluate.CHECKPOINT; selected.parent.mkdir(parents=True)
        selected.write_bytes(b"synthetic selected checkpoint; never loaded")
        alternate = self.root / "models/alternate.pt"; alternate.write_bytes(b"different checkpoint")
        with mock.patch.object(evaluate, "PROJECT_ROOT", self.root), \
             mock.patch.object(torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "Checkpoint"):
                evaluate.verify_sources("models/alternate.pt")
        load.assert_not_called()


class ProtocolAndModel(unittest.TestCase):
    @staticmethod
    def metadata():
        # The production splitter intentionally requires the full grid. Only tiny
        # strings are synthesized here; no real files or CSI arrays are needed.
        fields = ["sample_id", "mat_path", "csv_path", "environment_id", "participant_id",
                  "activity_id", "activity_name", "repetition_id"]
        rows = []
        for e, p, a, r in itertools.product(range(1, 5), range(1, 9), range(1, 8), range(1, 11)):
            sid = f"{e}-{p}-{a}-{r}"
            rows.append(dict(zip(fields, [sid, sid + ".mat", sid + ".csv", str(e), str(p), str(a), "synthetic", str(r)])))
        return fields, rows

    def test_fixed_split_and_deterministic_order(self):
        fields, rows = self.metadata()
        result = create_split.split_rows(fields, rows)
        self.assertEqual(result, create_split.split_rows(fields, list(reversed(rows))))
        for split, people in {"train": {1, 2, 3, 4, 5}, "val": {6}, "test": {7, 8}}.items():
            self.assertEqual({int(r["participant_id"]) for r in result if r["split"] == split}, people)
        self.assertFalse(any("split" in row for row in rows))

    def test_invalid_person_rejected(self):
        fields, rows = self.metadata(); rows[0]["participant_id"] = "9"
        with self.assertRaises(ValueError):
            create_split.split_rows(fields, rows)

    def test_duplicate_sample_rejected(self):
        fields, rows = self.metadata(); rows[1] = rows[0].copy()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            create_split.split_rows(fields, rows)

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in LightweightCNN().parameters()), 35655)

    def test_synthetic_forward_shape(self):
        torch.manual_seed(42)
        with torch.no_grad():
            logits = LightweightCNN().eval()(torch.zeros(2, 52, 950))
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all().item())


class Devices(unittest.TestCase):
    def test_explicit_cpu_recorded(self):
        device, info = select_device("cpu")
        self.assertEqual(device.type, "cpu"); self.assertTrue(info["cpu_explicit"])
        self.assertIsNone(info["gpu"]); self.assertEqual(info["pytorch"], str(torch.__version__))

    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_no_silent_cpu_fallback(self, available):
        with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
            select_device("cuda")

    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_explicit_fallback_recorded(self, available):
        device, info = select_device("cuda", allow_cpu_fallback=True)
        self.assertEqual(device.type, "cpu"); self.assertTrue(info["cpu_fallback"])

    @mock.patch("torch.cuda.get_device_name", return_value="Synthetic Different NVIDIA GPU")
    @mock.patch("torch.cuda.device_count", return_value=2)
    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_other_gpu_and_index_supported(self, *mocks):
        device, info = select_device("cuda:1")
        self.assertEqual(str(device), "cuda:1")
        self.assertEqual(info["gpu"], "Synthetic Different NVIDIA GPU")

    @mock.patch("torch.cuda.device_count", return_value=1)
    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_invalid_gpu_index_rejected(self, *mocks):
        with self.assertRaisesRegex(ValueError, "index"):
            select_device("cuda:3", allow_cpu_fallback=True)


if __name__ == "__main__":
    unittest.main()
