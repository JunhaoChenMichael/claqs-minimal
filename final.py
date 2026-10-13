import argparse
import copy
import sys
from typing import Callable, List, Sequence, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import torchtext  # type: ignore

    torchtext.disable_torchtext_deprecation_warning()  # type: ignore
except Exception:
    pass

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import run_classification_quantum as base_runner
from quixer.quixer_model import Quixer


quixer_cls_hparams = base_runner.quixer_cls_hparams.copy()


class MeanLogitsAggregator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        window_to_sample: torch.Tensor,
        num_samples: int,
        window_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str]:
        # Returns logits aggregated by mean (optionally weighted)
        num_classes = logits.size(-1)
        aggregated = logits.new_zeros(num_samples, num_classes)
        counts = logits.new_zeros(num_samples)

        if window_weights is None:
            weights = torch.ones_like(window_to_sample, dtype=logits.dtype)
        else:
            weights = window_weights.to(logits.device, dtype=logits.dtype)

        weighted_logits = logits * weights.unsqueeze(-1)
        aggregated.index_add_(0, window_to_sample, weighted_logits)
        counts.index_add_(0, window_to_sample, weights)

        aggregated = aggregated / counts.clamp_min(1e-6).unsqueeze(-1)
        return aggregated, "logits"


class MeanProbsAggregator(torch.nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        window_to_sample: torch.Tensor,
        num_samples: int,
        window_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str]:
        # Average probabilities then return log-probabilities for CE/NLL compatibility
        probs = torch.softmax(logits, dim=-1)
        num_classes = probs.size(-1)
        aggregated = probs.new_zeros(num_samples, num_classes)
        counts = probs.new_zeros(num_samples)

        if window_weights is None:
            weights = torch.ones_like(window_to_sample, dtype=probs.dtype)
        else:
            weights = window_weights.to(probs.device, dtype=probs.dtype)

        weighted_probs = probs * weights.unsqueeze(-1)
        aggregated.index_add_(0, window_to_sample, weighted_probs)
        counts.index_add_(0, window_to_sample, weights)

        aggregated = aggregated / counts.clamp_min(1e-6).unsqueeze(-1)
        log_probs = torch.log(aggregated.clamp_min(self.eps))
        return log_probs, "log_probs"


class AttnAggregator(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Simple learnable scorer from logits -> scalar score
        self.scorer = torch.nn.Linear(num_classes, 1)

    def forward(
        self,
        logits: torch.Tensor,
        window_to_sample: torch.Tensor,
        num_samples: int,
        window_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str]:
        # Compute attention weights per sample over windows, then weighted sum of logits
        scores = self.scorer(logits).squeeze(-1)  # [W]
        aggregated = logits.new_zeros(num_samples, logits.size(-1))

        if window_weights is not None:
            weights_all = window_weights.to(logits.device, dtype=logits.dtype)
        else:
            weights_all = None

        # Softmax per sample index (segmented softmax)
        for s in range(num_samples):
            mask = window_to_sample == s
            if not torch.any(mask):
                continue
            s_logits = logits[mask]  # [Wi, C]
            s_scores = scores[mask]  # [Wi]
            s_weights = torch.softmax(s_scores, dim=0).unsqueeze(-1)  # [Wi, 1]
            if weights_all is not None:
                w = weights_all[mask].unsqueeze(-1)
                weight_sum = w.sum()
                if weight_sum.item() > 0:
                    w = w / weight_sum
                    s_weights = s_weights * w
                    s_weights = s_weights / s_weights.sum().clamp_min(1e-6)
            aggregated[s] = (s_weights * s_logits).sum(dim=0)

        return aggregated, "logits"


class WindowedIMDBDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: dict,
        window: int,
        stride: int,
        max_windows: int,
    ) -> None:
        self.samples: List[Tuple[List[torch.Tensor], int]] = []
        self.window_counts: List[int] = []

        pad_idx = vocab["<pad>"]
        eos_idx = vocab["<eos>"]

        window_cap = max(1, max_windows)

        for text, label in zip(texts, labels):
            token_ids = _text_to_token_ids(text, vocab, eos_idx)
            windows = _tokens_to_windows(token_ids, window, stride, window_cap, pad_idx)
            self.samples.append((windows, int(label)))
            self.window_counts.append(len(windows))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int]:  # type: ignore[override]
        return self.samples[idx]


def _text_to_token_ids(text: str, vocab: dict, eos_idx: int) -> List[int]:
    tokens = [vocab.get(tok, vocab["<unk>"]) for tok in base_runner.basic_tokenizer(text)]
    tokens.append(eos_idx)
    return tokens


def _tokens_to_windows(
    token_ids: List[int],
    window: int,
    stride: int,
    max_windows: int,
    pad_idx: int,
) -> List[torch.Tensor]:
    if len(token_ids) <= window:
        padded = [pad_idx] * (window - len(token_ids)) + token_ids
        return [torch.tensor(padded, dtype=torch.long)]

    end = len(token_ids)
    step = max(1, stride)

    if end - window <= 0:
        start_positions = [0]
    else:
        start_positions = list(range(0, end - window + 1, step))
        last_start = max(end - window, 0)
        if start_positions[-1] != last_start:
            start_positions.append(last_start)

    windows: List[torch.Tensor] = []
    for start in start_positions:
        slice_tokens = token_ids[start : start + window]
        if len(slice_tokens) < window:
            slice_tokens = [pad_idx] * (window - len(slice_tokens)) + slice_tokens
        windows.append(torch.tensor(slice_tokens[-window:], dtype=torch.long))
        if len(windows) >= max_windows:
            break

    if not windows:
        padded = [pad_idx] * (window - len(token_ids)) + token_ids
        windows = [torch.tensor(padded[-window:], dtype=torch.long)]

    return windows


def _make_collate_fn(max_windows: int, sample_random: bool) -> Callable:
    max_windows = max(1, max_windows)

    def _collate(batch: Sequence[Tuple[List[torch.Tensor], int]]):
        window_tensors: List[torch.Tensor] = []
        labels: List[int] = []
        window_to_sample: List[int] = []

        for sample_idx, (windows, label) in enumerate(batch):
            labels.append(label)
            if len(windows) > max_windows:
                if sample_random:
                    perm = torch.randperm(len(windows))[:max_windows]
                    selected = [windows[i] for i in perm.tolist()]
                else:
                    selected = windows[:max_windows]
            else:
                selected = windows

            for tensor in selected:
                window_tensors.append(tensor)
                window_to_sample.append(sample_idx)

        if not window_tensors:
            raise RuntimeError("Collate produced zero windows; check preprocessing limits.")

        windows_tensor = torch.stack(window_tensors, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        mapping_tensor = torch.tensor(window_to_sample, dtype=torch.long)
        return windows_tensor, labels_tensor, mapping_tensor

    return _collate


def _aggregate_logits(
    logits: torch.Tensor,
    window_to_sample: torch.Tensor,
    num_samples: int,
    window_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Kept for backward compatibility: mean of logits (optionally weighted)
    num_classes = logits.size(-1)
    aggregated = logits.new_zeros(num_samples, num_classes)
    counts = logits.new_zeros(num_samples)

    if window_weights is None:
        weights = torch.ones_like(window_to_sample, dtype=logits.dtype)
    else:
        weights = window_weights.to(logits.device, dtype=logits.dtype)

    weighted_logits = logits * weights.unsqueeze(-1)
    aggregated.index_add_(0, window_to_sample, weighted_logits)
    counts.index_add_(0, window_to_sample, weights)

    aggregated = aggregated / counts.clamp_min(1e-6).unsqueeze(-1)
    return aggregated


def build_windowed_dataloaders(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    device: torch.device,
    window: int,
    batch_size: int,
    val_ratio: float,
    split_seed: int,
    stride: int,
    train_max_windows: int,
    eval_max_windows: int,
    dataset_window_cap: int,
    max_vocab: int,
    min_freq: int,
):
    word_to_idx, _ = base_runner.build_vocab_from_texts(
        train_texts, max_vocab=max_vocab, min_freq=min_freq
    )
    pad_idx = word_to_idx["<pad>"]

    window_cap = max(dataset_window_cap, train_max_windows, eval_max_windows)

    train_dataset_full = WindowedIMDBDataset(
        train_texts,
        train_labels,
        word_to_idx,
        window,
        stride,
        max_windows=window_cap,
    )
    test_dataset = WindowedIMDBDataset(
        test_texts,
        test_labels,
        word_to_idx,
        window,
        stride,
        max_windows=window_cap,
    )

    val_size = max(int(len(train_dataset_full) * val_ratio), 1)
    train_size = len(train_dataset_full) - val_size
    generator = torch.Generator().manual_seed(split_seed)
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset_full, [train_size, val_size], generator=generator
    )

    pin = device.type == "cuda"
    workers = 2

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin,
        num_workers=workers,
        collate_fn=_make_collate_fn(train_max_windows, sample_random=True),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin,
        num_workers=workers,
        collate_fn=_make_collate_fn(eval_max_windows, sample_random=False),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin,
        num_workers=workers,
        collate_fn=_make_collate_fn(eval_max_windows, sample_random=False),
    )

    train_avg_windows = sum(train_dataset_full.window_counts) / max(len(train_dataset_full), 1)
    test_avg_windows = sum(test_dataset.window_counts) / max(len(test_dataset), 1)

    return (
        train_loader,
        val_loader,
        test_loader,
        len(word_to_idx),
        train_avg_windows,
        test_avg_windows,
        pad_idx,
    )


def train_eval(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    wd: float,
    eps: float,
    max_grad_norm: float,
    aggregator: Optional[torch.nn.Module],
    label_smoothing: float,
    lcu_l1: float,
    qsvt_smooth: float,
    qsvt_l2: float,
    use_measure_mask: bool,
    measure_mask_l1: float,
    checkpoint_dir: Optional[str],
    pad_idx: Optional[int] = None,
):
    # Combine model and aggregator params if aggregator is learnable
    params = list(model.parameters()) + (list(aggregator.parameters()) if aggregator else [])
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd, eps=eps)
    criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_nll = torch.nn.NLLLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    # Disable AMP when complex parameters are present (unsupported by GradScaler unscale_)
    has_complex_params = any(p.dtype.is_complex for p in params)
    use_amp = device.type == "cuda" and not has_complex_params
    if has_complex_params and device.type == "cuda":
        print("AMP disabled due to complex parameters; using full precision.", flush=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def _apply_aggregate(
        window_logits: torch.Tensor,
        mapping: torch.Tensor,
        n: int,
        weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str]:
        if aggregator is None:
            # default: mean logits
            return _aggregate_logits(window_logits, mapping, n, weights), "logits"
        return aggregator(window_logits, mapping, n, window_weights=weights)

    def evaluate(loader) -> Tuple[float, float]:
        model.eval()
        if aggregator is not None:
            aggregator.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for window_tokens, labels, mapping in tqdm(loader, desc="Evaluating", leave=False):
                window_tokens = window_tokens.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                mapping = mapping.to(device, non_blocking=True)
                window_weights: Optional[torch.Tensor] = None
                if pad_idx is not None:
                    window_weights = (window_tokens != pad_idx).float().sum(dim=1)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    window_logits = model(window_tokens)
                    sample_out, out_kind = _apply_aggregate(
                        window_logits, mapping, labels.size(0), window_weights
                    )
                    if out_kind == "log_probs":
                        loss = criterion_nll(sample_out, labels)
                    else:
                        loss = criterion_ce(sample_out, labels)

                total_loss += loss.item() * labels.size(0)
                preds = torch.argmax(sample_out, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return avg_loss, accuracy

    best_val_acc = float("-inf")
    best_state = None
    best_agg_state = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        if aggregator is not None:
            aggregator.train()
        running_loss = 0.0
        total_seen = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)

        for window_tokens, labels, mapping in train_iter:
            window_tokens = window_tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mapping = mapping.to(device, non_blocking=True)
            window_weights: Optional[torch.Tensor] = None
            if pad_idx is not None:
                window_weights = (window_tokens != pad_idx).float().sum(dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                window_logits = model(window_tokens)
                sample_out, out_kind = _apply_aggregate(
                    window_logits, mapping, labels.size(0), window_weights
                )
                if out_kind == "log_probs":
                    loss = criterion_nll(sample_out, labels)
                else:
                    loss = criterion_ce(sample_out, labels)

                # Regularization terms
                reg = 0.0
                if hasattr(model, "base") and hasattr(model.base, "lcu_coefficients") and lcu_l1 > 0:
                    reg = reg + lcu_l1 * torch.norm(model.base.lcu_coefficients, p=1)
                if hasattr(model, "base") and hasattr(model.base, "qsvt_polynomial_coefficients"):
                    if qsvt_l2 > 0:
                        reg = reg + qsvt_l2 * torch.norm(model.base.qsvt_polynomial_coefficients, p=2)
                    if qsvt_smooth > 0 and model.base.qsvt_polynomial_coefficients.numel() > 1:
                        diff = model.base.qsvt_polynomial_coefficients[1:] - model.base.qsvt_polynomial_coefficients[:-1]
                        reg = reg + qsvt_smooth * torch.norm(diff, p=2)
                if use_measure_mask and measure_mask_l1 > 0 and hasattr(model, "measure_mask"):
                    reg = reg + measure_mask_l1 * torch.norm(model.measure_mask, p=1)

                loss = loss + reg

            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)
            train_iter.set_postfix({"avg_loss": f"{running_loss / max(total_seen, 1):.4f}"})

        # Evaluate on validation set each epoch
        val_loss, val_acc = evaluate(val_loader)
        print(
            f"Epoch {epoch+1:02d} | TrainLoss {running_loss/max(total_seen,1):.4f} | ValLoss {val_loss:.4f} | ValAcc {val_acc*100:.2f}%",
            flush=True,
        )
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            if aggregator is not None:
                best_agg_state = copy.deepcopy(aggregator.state_dict())
            best_epoch = epoch + 1
            if checkpoint_dir:
                import os
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"best_epoch{best_epoch:02d}_val{best_val_acc*100:.2f}.pt",
                )
                torch.save(
                    {
                        "model": best_state,
                        "aggregator": (best_agg_state if aggregator is not None else None),
                        "epoch": best_epoch,
                        "val_acc": best_val_acc,
                    },
                    ckpt_path,
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        if aggregator is not None and best_agg_state is not None:
            aggregator.load_state_dict(best_agg_state)

    test_loss, test_acc = evaluate(test_loader)
    print(f"Test results | Loss {test_loss:.4f} | Acc {test_acc*100:.2f}%", flush=True)

    return test_acc


def main():
    parser = argparse.ArgumentParser(
        description="Quixer IMDB classification with multi-window aggregation"
    )
    parser.add_argument("--device", "-d", default="auto", help="cpu | cuda | auto")
    parser.add_argument(
        "--train", required=True
    )
    parser.add_argument(
        "--test", required=True
    )
    parser.add_argument("--readout_qubit", type=int, default=0)
    parser.add_argument("--readout_q0", type=int, default=0)
    parser.add_argument("--readout_q1", type=int, default=1)
    parser.add_argument(
        "--readout_head",
        choices=["mlp", "twoz", "singlez"],
        default="mlp",
        help="Readout head to use (mlp combines all measurements)",
    )
    parser.add_argument("--qubits", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--ansatz_layers", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_vocab", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--window_stride", type=int, default=16)
    parser.add_argument("--train_max_windows", type=int, default=3)
    parser.add_argument("--eval_max_windows", type=int, default=6)
    parser.add_argument("--dataset_window_cap", type=int, default=12)
    parser.add_argument(
        "--train_subset_ratio",
        type=float,
        default=1.0,
        help="Fraction of the training split to use (0, 1].",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean_logits", "mean_probs", "attention"],
        default="mean_logits",
        help="How to aggregate window predictions into sample prediction",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    # Measurement selection & regularization & checkpointing
    parser.add_argument("--use_measure_mask", action="store_true")
    parser.add_argument("--measure_mask_l1", type=float, default=0.0)
    parser.add_argument("--lcu_l1", type=float, default=0.0)
    parser.add_argument("--qsvt_smooth", type=float, default=0.0, help="L2 on differences of consecutive QSVT coeffs")
    parser.add_argument("--qsvt_l2", type=float, default=0.0)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.split_seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"
        print(f"Using device: CUDA ({gpu_name})", flush=True)
    else:
        print("Using device: CPU", flush=True)

    h = quixer_cls_hparams.copy()
    if args.qubits is not None:
        h["qubits"] = args.qubits
    if args.layers is not None:
        h["layers"] = args.layers
    if args.ansatz_layers is not None:
        h["ansatz_layers"] = args.ansatz_layers
    if args.window is not None:
        h["window"] = args.window
    if args.epochs is not None:
        h["epochs"] = args.epochs
    if args.batch_size is not None:
        h["batch_size"] = args.batch_size
    if args.dropout is not None:
        h["dropout"] = args.dropout
    if args.lr is not None:
        h["lr"] = args.lr
    if args.wd is not None:
        h["wd"] = args.wd

    print(
        f"Hyperparams: window={h['window']}, stride={args.window_stride}, train_max_windows={args.train_max_windows}, eval_max_windows={args.eval_max_windows}, qubits={h['qubits']}, layers={h['layers']}, ansatz_layers={h['ansatz_layers']}, batch_size={h['batch_size']}, aggregate={args.aggregate}, label_smoothing={args.label_smoothing}",
        flush=True,
    )

    print("Loading parquet datasets...", flush=True)
    train_texts, train_labels, test_texts, test_labels = base_runner.load_text_classification_parquet(
        args.train, args.test
    )

    subset_ratio = max(min(args.train_subset_ratio, 1.0), 0.0)
    if subset_ratio <= 0.0:
        raise ValueError("train_subset_ratio must be > 0.")
    if subset_ratio < 0.9999:
        total_train = len(train_texts)
        target = max(int(total_train * subset_ratio), 1)
        generator = torch.Generator().manual_seed(args.split_seed)
        indices = torch.randperm(total_train, generator=generator)[:target].tolist()
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        print(
            f"Subsampled training data: using {target}/{total_train} samples ({subset_ratio*100:.1f}%).",
            flush=True,
        )
    print(
        f"Loaded train={len(train_texts)} samples, test={len(test_texts)} samples",
        flush=True,
    )

    print("Building vocab and dataloaders...", flush=True)
    (
        train_loader,
        val_loader,
        test_loader,
        vocab_size,
        train_avg_windows,
        test_avg_windows,
        pad_idx,
    ) = build_windowed_dataloaders(
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        device,
        h["window"],
        h["batch_size"],
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        stride=args.window_stride,
        train_max_windows=args.train_max_windows,
        eval_max_windows=args.eval_max_windows,
        dataset_window_cap=args.dataset_window_cap,
        max_vocab=args.max_vocab,
        min_freq=args.min_freq,
    )

    print(
        f"Vocab size={vocab_size} | avg train windows={train_avg_windows:.2f} | avg test windows={test_avg_windows:.2f}",
        flush=True,
    )
    print(
        f"train steps/epoch={len(train_loader)} | val steps={len(val_loader)} | test steps={len(test_loader)}",
        flush=True,
    )

    print("Creating model...", flush=True)
    max_windows_per_sample = max(args.train_max_windows, args.eval_max_windows)
    effective_batch = h["batch_size"] * max_windows_per_sample
    base = Quixer(
        n_qubits=h["qubits"],
        n_tokens=h["window"],
        qsvt_polynomial_degree=h["layers"],
        n_ansatz_layers=h["ansatz_layers"],
        vocabulary_size=vocab_size,
        embedding_dimension=h["dimension"],
        dropout=h["dropout"],
        batch_size=effective_batch,
        device=device,
    )

    base_runner.initialise_weights_local(base)
    base = base.to(device)

    if args.readout_head == "mlp":
        model = base_runner.QuixerMLPReadoutClassifier(base, use_measure_mask=args.use_measure_mask).to(device)
    elif args.readout_head == "twoz":
        model = base_runner.QuixerTwoZReadoutClassifier(
            base, readout_q0=args.readout_q0, readout_q1=args.readout_q1
        ).to(device)
    else:
        model = base_runner.QuixerZReadoutClassifier(base, readout_qubit=args.readout_qubit).to(device)

    # Build aggregator module
    aggregator: Optional[torch.nn.Module]
    if args.aggregate == "mean_logits":
        aggregator = MeanLogitsAggregator()
    elif args.aggregate == "mean_probs":
        aggregator = MeanProbsAggregator()
    else:
        # attention aggregator needs to know num classes: infer by running a tiny dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, h["window"], dtype=torch.long, device=device)
            base_logits = model(dummy) if 'model' in locals() else base(dummy)
            num_classes = base_logits.size(-1)
        aggregator = AttnAggregator(num_classes).to(device)

    print("Starting training...", flush=True)
    best_acc = train_eval(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        epochs=h["epochs"],
        lr=h["lr"],
        wd=h["wd"],
        eps=h["eps"],
        max_grad_norm=h["max_grad_norm"],
        aggregator=aggregator.to(device) if aggregator is not None else None,
        label_smoothing=args.label_smoothing,
        lcu_l1=args.lcu_l1,
        qsvt_smooth=args.qsvt_smooth,
        qsvt_l2=args.qsvt_l2,
        use_measure_mask=args.use_measure_mask,
        measure_mask_l1=args.measure_mask_l1,
        checkpoint_dir=args.checkpoint_dir,
        pad_idx=pad_idx,
    )

    print(f"Best Test Accuracy: {best_acc*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
