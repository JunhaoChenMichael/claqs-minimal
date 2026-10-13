"""Run a tiny CPU train/save/reload/evaluate cycle using temporary synthetic data."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="claqs-smoke-") as directory:
        tmp = Path(directory)
        for split, size in (("train", 24), ("test", 8)):
            pq.write_table(pa.table({
                "text": ["great movie excellent acting" if i % 2 else "bad movie boring story" for i in range(size)],
                "label": [i % 2 for i in range(size)],
            }), tmp / f"{split}.parquet")
        common = ["--device", "cpu", "--train", str(tmp / "train.parquet"),
                  "--test", str(tmp / "test.parquet"), "--window", "4",
                  "--qubits", "2", "--layers", "1", "--ansatz_layers", "1",
                  "--batch_size", "4", "--window_stride", "4", "--train_max_windows", "1",
                  "--eval_max_windows", "1", "--min_freq", "1", "--val_ratio", "0.25"]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        subprocess.run([sys.executable, str(root / "final.py"), *common,
                        "--epochs", "1", "--checkpoint_dir", str(tmp / "checkpoints")],
                       cwd=root, env=env, check=True)
        checkpoint = next((tmp / "checkpoints").glob("*.pt"))
        subprocess.run([sys.executable, str(root / "eval_checkpoint.py"), *common,
                        "--checkpoint", str(checkpoint)], cwd=root, env=env, check=True)
        from run_classification_quantum import load_text_classification_parquet
        pq.write_table(pa.table({"text": ["hidden label"], "label": [-1]}), tmp / "hidden.parquet")
        try:
            load_text_classification_parquet(str(tmp / "train.parquet"), str(tmp / "hidden.parquet"))
        except ValueError:
            pass
        else:
            raise AssertionError("Hidden labels must be rejected")
    print("PASS: CPU training, checkpoint reload, evaluation, and hidden-label rejection.")


if __name__ == "__main__":
    main()
