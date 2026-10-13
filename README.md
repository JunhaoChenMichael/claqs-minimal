# CLAQS / Quixer — minimal text classification package

A compact source release of the existing Quixer-based classification implementation related to [CLAQS](https://arxiv.org/abs/2510.06532). Includes the quantum model, sliding-window training, checkpoint evaluation, and a synthetic smoke test. No datasets, pretrained weights, logs, or original Git history are included.

## Installation

Use Python 3.11 (Linux recommended):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python smoke_test.py
```

The smoke test creates tiny synthetic parquet files in a temporary directory, trains for one epoch on CPU, reloads the checkpoint, evaluates it, and removes temporary files automatically. It checks execution, not model accuracy. For CUDA, install matching PyTorch 2.3.0 / torchvision 0.18.0 wheels for your GPU before installing the requirements. TorchQuantum imports the legacy Qiskit stack; do not upgrade Qiskit independently to 1.x or newer.

## Data

Supply your own local parquet files. Supported text columns: `text`, `sentence`, `review`, `content`. Supported label columns: `label`, `labels`, `sentiment`, `target`. Labels must be binary 0/1; negative/positive string labels are also accepted. Hidden labels (`-1`) are rejected.

`--train` and `--test` are required paths. The trainer reserves 10% of the supplied training file for internal validation and selects checkpoints using that split. The file passed as `--test` is evaluated after training. This is a random training holdout, not an official validation split.

For SST-2, the public test file has hidden labels: supply the labeled official validation file as `--test`, and describe its resulting score as **validation accuracy**, even though the generic CLI prints “Test”. Do not report an internal random-holdout score as official test accuracy.

## Training

Example for IMDB (adjust paths to your own data):

```bash
python final.py \
  --train data/imdb/train.parquet --test data/imdb/test.parquet \
  --device auto --qubits 8 --layers 5 --ansatz_layers 6 \
  --window 256 --window_stride 64 \
  --train_max_windows 4 --eval_max_windows 4 \
  --readout_head mlp --aggregate mean_logits \
  --batch_size 16 --epochs 15 --lr 0.001 --wd 0 \
  --label_smoothing 0 --max_vocab 100000 --min_freq 1 \
  --checkpoint_dir checkpoints/imdb
```

`--device cpu` works for small configurations; simulation cost grows exponentially with qubits. `python final.py --help` lists all options.

## Evaluation

Use the same data files, vocabulary settings, architecture, window settings, aggregation, mask settings, split seed and validation ratio as training. Substitute the actual checkpoint filename:

```bash
python eval_checkpoint.py \
  --checkpoint checkpoints/imdb/YOUR_CHECKPOINT.pt \
  --train data/imdb/train.parquet --test data/imdb/test.parquet \
  --device auto --qubits 8 --layers 5 --ansatz_layers 6 \
  --window 256 --window_stride 64 \
  --train_max_windows 4 --eval_max_windows 4 \
  --readout_head mlp --aggregate mean_logits \
  --batch_size 16 --max_vocab 100000 --min_freq 1
```

Checkpoints do not store the vocabulary or full CLI configuration; retain your training command. Architecture mismatches are rejected. Use `--train_subset_ratio 1` (the default) when using this evaluator: it rebuilds the vocabulary and internal split from the complete supplied training file.

## Scope and provenance

Core simulation is adapted from **Quixer: A Quantum Transformer Model**, Nikhil Khatri et al., [arXiv:2406.04305](https://arxiv.org/abs/2406.04305). The original Apache-2.0 license is retained in `LICENSE`.

This package preserves the existing model and is not a claim of exact reproduction of the CLAQS paper's reported scores. Known differences include Adam with ReduceLROnPlateau, an MLP head without head dropout, and no target-probability post-selection loss. The inherited quantum feed-forward uses one ansatz layer but allocates parameters using `ansatz_layers`; at 8 qubits / 6 layers only 32 of its 192 allocated parameters are consumed. Complex LCU parameters have two real degrees of freedom per element. These issues are documented rather than changing the architecture while packaging.

Packaging changes: removed absolute machine paths and unrelated runners; made input files explicit; rejected invalid labels; seeded training; ensured the first validation result can save a checkpoint; set the aggregator to evaluation mode; rejected checkpoint architecture mismatches; removed dataset-name-based evaluation skipping. Evaluation is based on the labeled file supplied by the user.

## Files

- `final.py`: sliding-window training and document aggregation.
- `run_classification_quantum.py`: readout heads, vocabulary, and parquet loader.
- `quixer/quixer_model.py`: shared ansatz, complex LCU, polynomial transform, quantum feed-forward.
- `eval_checkpoint.py`: saved-model evaluation.
- `smoke_test.py`: temporary synthetic train/evaluate check.
- `requirements.txt`, `.gitignore`, `LICENSE`: installation, exclusions, and license.

## Upload to GitHub

Create an empty repository under your own GitHub account, then run from this extracted directory:

```bash
git init -b main
git add .
git commit -m "Add minimal quantum text classifier"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

The `.gitignore` excludes common datasets, model weights, checkpoints, logs, caches, and local environments. Review `git status` before committing additional files.
