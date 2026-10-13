"""Regression checks for reproducible data handling and experiment checkpoints.

Run with ``python -m unittest discover -s tests -v``. All fixtures are synthetic
and created in a temporary directory; the training check always uses the CPU.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment import (
    evaluate, load_model_state, make_model, model_state, read_records,
    stratified_indices,
)


def write_records(path, texts, labels):
    pq.write_table(pa.table({"text": texts, "label": labels}), path)


class DataProtocolTests(unittest.TestCase):
    def test_read_records_preserves_labels_and_rejects_hidden_labels(self):
        with tempfile.TemporaryDirectory(prefix="claqs-records-") as directory:
            path = Path(directory) / "records.parquet"
            write_records(path, ["positive review", "negative review"], [1, 0])
            texts, labels = read_records(path)
            self.assertEqual(list(texts), ["positive review", "negative review"])
            self.assertEqual(list(labels), [1, 0])
            write_records(path, ["unknown test label"], [-1])
            with self.assertRaises(ValueError):
                read_records(path)

    def test_stratified_split_is_reproducible_disjoint_and_complete(self):
        labels = [0] * 16 + [1] * 8
        train, validation = stratified_indices(labels, 0.25, 123)
        again_train, again_validation = stratified_indices(labels, 0.25, 123)
        self.assertEqual(list(train), list(again_train))
        self.assertEqual(list(validation), list(again_validation))
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(len(labels))))
        self.assertEqual(len(train) + len(validation), len(labels))
        self.assertEqual(sum(labels[i] == 0 for i in validation), 4)
        self.assertEqual(sum(labels[i] == 1 for i in validation), 2)
        other_train, other_validation = stratified_indices(labels, 0.25, 456)
        self.assertNotEqual((list(train), list(validation)),
                            (list(other_train), list(other_validation)))


class EvaluationMetricsTests(unittest.TestCase):
    def test_metrics_count_documents_and_weight_partial_batches_correctly(self):
        class FixedLogits(torch.nn.Module):
            def forward(self, tokens):
                table = torch.tensor([[0., 0.], [4., 0.], [0., 0.],
                                      [0., 2.], [2., 0.], [0., 2.]])
                return table[tokens[:, 0]]

        # The first document has two windows, with two and one real tokens.
        batches = [
            (torch.tensor([[1, 2], [3, 0], [4, 0]]), torch.tensor([0, 1]),
             torch.tensor([0, 0, 1])),
            (torch.tensor([[5, 0]]), torch.tensor([1]), torch.tensor([0])),
        ]
        metrics = evaluate(FixedLogits(), batches, torch.device("cpu"), 0)
        expected_logits = torch.tensor([[8 / 3, 2 / 3], [2., 0.], [0., 2.]])
        expected_loss = torch.nn.functional.cross_entropy(
            expected_logits, torch.tensor([0, 1, 1])).item()
        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["confusion"], [[1, 0], [1, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3, places=6)
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)


class SparsePaddingTests(unittest.TestCase):
    def test_sparse_padding_preserves_logits_and_all_parameter_gradients(self):
        torch.manual_seed(17)
        vocab = {"<pad>": 0, "<unk>": 1, "good": 2, "bad": 3,
                 "movie": 4, "story": 5}
        config = dict(qubits=2, window=4, degree=2, ansatz_layers=1,
                      ff_layers=1, dimension=16, batch_size=4, dropout=0.,
                      head_dropout=0., mask_padding=True, sparse_padding=False)
        dense = make_model(config, vocab, torch.device("cpu"))
        with torch.no_grad():
            dense.base.lcu_coefficients.copy_(
                torch.tensor([0.7 + 0.3j, 0.2 + 0.4j, -0.3 + 0.1j, 0j]))
            dense.base.qsvt_polynomial_coefficients.copy_(
                torch.tensor([0.7, 0.4, -0.2]))
        sparse = make_model(dict(config, sparse_padding=True), vocab,
                            torch.device("cpu"))
        load_model_state(sparse, model_state(dense))
        tokens = torch.tensor([[2, 3, 4, 5], [3, 2, 0, 0],
                               [4, 0, 0, 0], [0, 0, 0, 0]])
        target = torch.tensor([0, 1, 0, 1])
        dense_logits, sparse_logits = dense(tokens), sparse(tokens)
        self.assertTrue(torch.isfinite(dense_logits).all())
        self.assertTrue(torch.isfinite(sparse_logits).all())
        torch.testing.assert_close(dense_logits, sparse_logits, rtol=3e-5, atol=1e-6)
        torch.nn.functional.cross_entropy(dense_logits, target).backward()
        torch.nn.functional.cross_entropy(sparse_logits, target).backward()
        dense_parameters = dict(dense.named_parameters())
        sparse_parameters = dict(sparse.named_parameters())
        self.assertEqual(set(dense_parameters), set(sparse_parameters))
        self.assertFalse(any("output_feedforward" in name for name in dense_parameters))
        self.assertEqual(dense.base.quantum_feedforward_parameters.numel(), 8)
        for name, parameter in dense_parameters.items():
            with self.subTest(parameter=name):
                other = sparse_parameters[name]
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(other.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(torch.isfinite(other.grad).all())
                torch.testing.assert_close(parameter.grad, other.grad,
                                           rtol=3e-5, atol=1e-6)
        for model in (dense, sparse):
            self.assertEqual(torch.count_nonzero(model.base.embedding.weight.grad[0]), 0)
            # Position 3 is valid in row 0, despite its zero LCU coefficient.
            # Skipping that position would incorrectly remove this derivative.
            self.assertGreater(model.base.lcu_coefficients.grad[3].abs().item(), 1e-8)
            self.assertTrue((model.base.quantum_feedforward_parameters.grad.abs()
                             > 1e-10).all())


class ExperimentRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="claqs-experiment-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.train_path = cls.directory / "train.parquet"
        cls.validation_path = cls.directory / "validation.parquet"
        cls.output = cls.directory / "run"
        write_records(cls.train_path,
                      ["great movie excellent acting" if i % 2 else
                       "bad movie boring story" for i in range(16)],
                      [i % 2 for i in range(16)])
        write_records(cls.validation_path,
                      ["great movie validationonlytoken" if i % 2 else
                       "bad movie validationonlytoken" for i in range(8)],
                      [i % 2 for i in range(8)])
        cls.environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        cpu_config = ("--device", "cpu", "--epochs", "1", "--qubits", "2",
                      "--degree", "1", "--ansatz-layers", "1", "--ff-layers", "1",
                      "--dimension", "16", "--window", "4", "--stride", "4",
                      "--batch-size", "4", "--max-windows", "1", "--workers", "0")
        cls.run_cli("train", "--train", str(cls.train_path),
                    "--validation", str(cls.validation_path),
                    "--output", str(cls.output), *cpu_config)
        cls.checkpoint = torch.load(cls.output / "best.pt", map_location="cpu",
                                    weights_only=False)
        cls.holdout_output = cls.directory / "holdout-run"
        cls.run_cli("train", "--train", str(cls.train_path),
                    "--output", str(cls.holdout_output), "--val-ratio", "0.25",
                    "--val-limit", "2", *cpu_config)
        cls.holdout_checkpoint = torch.load(cls.holdout_output / "best.pt",
                                            map_location="cpu", weights_only=False)

    @classmethod
    def run_cli(cls, *arguments, check=True):
        result = subprocess.run([sys.executable, str(ROOT / "experiment.py"),
                                 *arguments], cwd=ROOT, env=cls.environment,
                                text=True, capture_output=True, timeout=180)
        if check and result.returncode:
            raise AssertionError(f"CLI failed ({result.returncode}):\n"
                                 f"{result.stdout}\n{result.stderr}")
        return result

    def test_checkpoint_is_self_contained_and_reload_validates_parameters(self):
        for key in ("config", "vocab", "model", "epoch", "best_validation",
                    "data_protocol"):
            self.assertIn(key, self.checkpoint)
        self.assertNotIn("validationonlytoken", self.checkpoint["vocab"])
        self.assertTrue(self.checkpoint["data_protocol"])
        self.assertTrue(self.checkpoint["model"])
        self.assertFalse(any("torchquantum_device" in key
                             for key in self.checkpoint["model"]))
        model = make_model(self.checkpoint["config"], self.checkpoint["vocab"],
                           torch.device("cpu"))
        load_model_state(model, self.checkpoint["model"])
        current_state = model_state(model)
        self.assertEqual(set(current_state), set(self.checkpoint["model"]))
        for key, value in current_state.items():
            torch.testing.assert_close(value, self.checkpoint["model"][key])
        damaged_state = dict(self.checkpoint["model"])
        del damaged_state[next(iter(dict(model.named_parameters())))]
        with self.assertRaises(ValueError):
            load_model_state(model, damaged_state)

    def test_training_writes_metrics_and_epoch_history(self):
        metrics = json.loads((self.output / "metrics.json").read_text())
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics["evaluation_split"], "validation")
        self.assertFalse(metrics["test_evaluated"])
        self.assertEqual(metrics["train_samples"], 16)
        self.assertEqual(metrics["validation_samples"], 8)
        protocol = self.checkpoint["data_protocol"]
        self.assertEqual(protocol["validation_source"], "external")
        self.assertEqual(len(protocol["train_sha256"]), 64)
        self.assertNotEqual(protocol["train_sha256"], protocol["validation_sha256"])
        history = [json.loads(line) for line in
                   (self.output / "history.jsonl").read_text().splitlines()
                   if line.strip()]
        self.assertEqual(len(history), 1)
        self.assertIsInstance(history[0], dict)

    def test_cli_evaluation_repeats_without_checkpoint_configuration_flags(self):
        results = []
        for index in range(2):
            output = self.directory / f"evaluation-{index}.json"
            self.run_cli("evaluate", "--checkpoint", str(self.output / "best.pt"),
                         "--data", str(self.validation_path), "--split-name",
                         "validation", "--device", "cpu", "--output", str(output))
            payload = json.loads(output.read_text())
            metrics = payload.get("metrics", payload)
            for key in ("loss", "accuracy", "macro_f1"):
                self.assertIn(key, metrics)
                self.assertIsInstance(metrics[key], (int, float))
            self.assertGreaterEqual(metrics["loss"], 0)
            for key in ("accuracy", "macro_f1"):
                self.assertGreaterEqual(metrics[key], 0)
                self.assertLessEqual(metrics[key], 1)
            results.append(metrics)
        for key in ("loss", "accuracy", "macro_f1"):
            self.assertAlmostEqual(results[0][key], results[1][key], places=7)

    def test_cli_evaluation_rejects_hidden_labels(self):
        hidden = self.directory / "hidden.parquet"
        output = self.directory / "invalid-evaluation.json"
        write_records(hidden, ["unlabeled public test example"], [-1])
        result = self.run_cli("evaluate", "--checkpoint", str(self.output / "best.pt"),
                              "--data", str(hidden), "--split-name", "test",
                              "--device", "cpu", "--output", str(output), check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())

    def test_internal_holdout_evaluation_replays_split_and_validation_subset(self):
        protocol = self.holdout_checkpoint["data_protocol"]
        self.assertEqual(protocol["validation_source"], "stratified_train_holdout")
        self.assertEqual(len(protocol["validation_indices"]), 4)
        self.assertEqual(len(protocol["validation_subset_indices"]), 2)
        output = self.directory / "holdout-evaluation.json"
        self.run_cli("evaluate", "--checkpoint", str(self.holdout_output / "best.pt"),
                     "--data", str(self.train_path), "--split-name", "validation",
                     "--device", "cpu", "--output", str(output))
        payload = json.loads(output.read_text())
        self.assertEqual(payload["split"], "validation")
        metrics = payload["metrics"]
        expected = self.holdout_checkpoint["best_validation"]
        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["confusion"], expected["confusion"])
        for key in ("loss", "accuracy", "macro_f1", "macro_precision", "macro_recall"):
            self.assertAlmostEqual(metrics[key], expected[key], places=6)

    def test_internal_holdout_training_file_cannot_be_reported_as_test(self):
        output = self.directory / "invalid-holdout-test.json"
        result = self.run_cli("evaluate", "--checkpoint",
                              str(self.holdout_output / "best.pt"),
                              "--data", str(self.train_path), "--split-name", "test",
                              "--device", "cpu", "--output", str(output), check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())

    def test_external_validation_rejects_a_different_file_hash(self):
        mismatched = self.directory / "different-validation.parquet"
        write_records(mismatched, ["another negative movie", "another positive movie"],
                      [0, 1])
        output = self.directory / "invalid-mismatched-validation.json"
        result = self.run_cli("evaluate", "--checkpoint", str(self.output / "best.pt"),
                              "--data", str(mismatched), "--split-name", "validation",
                              "--device", "cpu", "--output", str(output), check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
