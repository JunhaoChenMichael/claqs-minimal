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

## Reproducible experiments (recommended)

Use `experiment.py` for new runs. It builds the vocabulary only from training examples, accepts an explicit validation file, saves the vocabulary/configuration/data hashes inside each checkpoint, and never evaluates a test set while selecting a model. Each run requires a fresh output directory.

```bash
# SST-2: use official validation labels, never the hidden-label public test file.
python experiment.py train \
  --train data/sst2/train.parquet --validation data/sst2/validation.parquet \
  --output experiments/sst2/run01 --device cuda \
  --qubits 8 --degree 3 --ansatz-layers 2 --ff-layers 1 \
  --dimension 128 --window 64 --stride 64 --max-windows 1 \
  --batch-size 256 --epochs 4 --seed 42 --sparse-padding \
  --optimizer adam --schedule constant --weight-decay 0 --head-dropout 0

# Restore architecture and vocabulary directly from the checkpoint.
python experiment.py evaluate \
  --checkpoint experiments/sst2/run01/best.pt \
  --data data/sst2/validation.parquet --split-name validation --device cuda

# Regression tests use temporary synthetic data on CPU.
python -m unittest discover -s tests -v
```

For IMDB, omit `--validation` to create a seeded, stratified holdout from the training file, or provide a separate labeled validation file. Use suitable long-document settings, e.g. `--window 256 --stride 128 --max-windows 4`. Evaluate the test file in a separate `evaluate --split-name test` invocation only after selecting the configuration. `--train-limit` provides a repeatable small pilot; its vocabulary is also limited to the chosen training rows. The holdout splitter is stratified, not grouped by duplicate text or original document.

New runs use a compact quantum feed-forward with exactly `4 * qubits * ff_layers` angles and omit the unused vocabulary prediction head. Padding can be excluded from LCU using `--mask-padding` (default); `--no-mask-padding` provides a controlled comparison. `--sparse-padding` additionally skips the masked token circuits while preserving the masked computation; it requires padding masking. Head dropout, Adam/AdamW, constant/cosine learning rates and validation-based early stopping are configurable. These are experimental variants, not claims of reproducing the paper's exact settings. Seeded execution does not guarantee bitwise determinism across GPU/library versions.

Run artifacts: `config.json`, `data_protocol.json`, `history.jsonl`, `metrics.json`, and `best.pt`. Accuracy/F1 are fractions. Metrics explicitly identify validation results; original training configuration is recovered from the checkpoint during evaluation. Validation replays the saved file hash, holdout indices, and optional validation subset; test evaluation rejects known training/validation file hashes. For an internal holdout, pass the original training file with `--split-name validation` to replay only the held-out rows. No optimizer state is stored: `best.pt` supports evaluation, not exact interrupted-training resumption. The legacy scripts below remain available for old checkpoint compatibility.

Detailed measured results and limitations: [SST-2 experiment report](reports/sst2_20261013.md).

## Data (legacy scripts)

Supply your own local parquet files. Supported text columns: `text`, `sentence`, `review`, `content`. Supported label columns: `label`, `labels`, `sentiment`, `target`. Labels must be binary 0/1; negative/positive string labels are also accepted. Hidden labels (`-1`) are rejected.

`--train` and `--test` are required paths. The trainer reserves 10% of the supplied training file for internal validation and selects checkpoints using that split. The file passed as `--test` is evaluated after training. This is a random training holdout, not an official validation split.

For SST-2, the public test file has hidden labels: supply the labeled official validation file as `--test`, and describe its resulting score as **validation accuracy**, even though the generic CLI prints “Test”. Do not report an internal random-holdout score as official test accuracy.

## Training (legacy scripts)

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

## Evaluation (legacy scripts)

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

The legacy entrypoints preserve the existing model and are not a claim of exact reproduction of the CLAQS paper's reported scores. Known differences include Adam with ReduceLROnPlateau, an MLP head without head dropout, and no target-probability post-selection loss. The inherited quantum feed-forward uses one ansatz layer but allocates parameters using `ansatz_layers`; at 8 qubits / 6 layers only 32 of its 192 allocated parameters are consumed. Complex LCU parameters have two real degrees of freedom per element. The new experiment entrypoint corrects the unused-head and feed-forward allocation issues using explicit options; the legacy defaults remain compatible.

Packaging changes: removed absolute machine paths and unrelated runners; made input files explicit; rejected invalid labels; seeded training; ensured the first validation result can save a checkpoint; set the aggregator to evaluation mode; rejected checkpoint architecture mismatches; removed dataset-name-based evaluation skipping. Evaluation is based on the labeled file supplied by the user.

## Files

- `experiment.py`: reproducible new experiments with independent validation and saved vocabulary.
- `tests/test_experiment.py`: CPU regression checks.
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
