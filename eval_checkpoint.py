import argparse
import os
from typing import Optional, Tuple

import torch
from tqdm import tqdm

import final as final_module
import run_classification_quantum as base_runner
from quixer.quixer_model import Quixer


def _build_model_and_aggregator(
    args: argparse.Namespace, h: dict, device: torch.device, vocab_size: int
) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
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
        model = base_runner.QuixerMLPReadoutClassifier(
            base, use_measure_mask=args.use_measure_mask
        ).to(device)
    elif args.readout_head == "twoz":
        model = base_runner.QuixerTwoZReadoutClassifier(
            base, readout_q0=args.readout_q0, readout_q1=args.readout_q1
        ).to(device)
    else:
        model = base_runner.QuixerZReadoutClassifier(
            base, readout_qubit=args.readout_qubit
        ).to(device)

    if args.aggregate == "mean_logits":
        aggregator: Optional[torch.nn.Module] = final_module.MeanLogitsAggregator()
    elif args.aggregate == "mean_probs":
        aggregator = final_module.MeanProbsAggregator()
    else:
        # Need to know number of classes; run dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, h["window"], dtype=torch.long, device=device)
            base_logits = model(dummy)
            num_classes = base_logits.size(-1)
        aggregator = final_module.AttnAggregator(num_classes).to(device)

    if aggregator is not None:
        aggregator = aggregator.to(device)

    return model, aggregator


def _evaluate(
    model: torch.nn.Module,
    aggregator: Optional[torch.nn.Module],
    loader,
    device: torch.device,
    label_smoothing: float,
    desc: str,
    pad_idx: Optional[int] = None,
) -> Tuple[float, float, float, float, float]:
    criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_nll = torch.nn.NLLLoss()

    model.eval()
    if aggregator is not None:
        aggregator.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    num_classes: Optional[int] = None

    with torch.no_grad():
        for window_tokens, labels, mapping in tqdm(loader, desc=desc, leave=False):
            window_tokens = window_tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mapping = mapping.to(device, non_blocking=True)
            window_weights: Optional[torch.Tensor] = None
            if pad_idx is not None:
                window_weights = (window_tokens != pad_idx).float().sum(dim=1)

            window_logits = model(window_tokens)
            if aggregator is None:
                sample_out, out_kind = final_module._aggregate_logits(
                    window_logits, mapping, labels.size(0), window_weights
                ), "logits"
            else:
                sample_out, out_kind = aggregator(
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
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
            if num_classes is None:
                num_classes = sample_out.size(-1)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    precision_macro = recall_macro = f1_macro = 0.0
    if all_preds and num_classes is not None:
        preds_cat = torch.cat(all_preds).long()
        labels_cat = torch.cat(all_labels).long()
        conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for tgt, pred in zip(labels_cat, preds_cat):
            conf[tgt.item(), pred.item()] += 1
        tp = conf.diag().float()
        fp = conf.sum(dim=0).float() - tp
        fn = conf.sum(dim=1).float() - tp
        precision_per = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
        recall_per = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
        denom = precision_per + recall_per
        f1_per = torch.where(denom > 0, 2 * precision_per * recall_per / denom, torch.zeros_like(tp))
        precision_macro = precision_per.mean().item()
        recall_macro = recall_per.mean().item()
        f1_macro = f1_per.mean().item()

    return avg_loss, accuracy, precision_macro, recall_macro, f1_macro


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a Quixer checkpoint and evaluate on the validation/test splits."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="Evaluation device"
    )
    parser.add_argument(
        "--train",
        required=True,
    )
    parser.add_argument(
        "--test",
        required=True,
    )
    parser.add_argument("--readout_head", choices=["mlp", "twoz", "singlez"], default="mlp")
    parser.add_argument("--readout_qubit", type=int, default=0)
    parser.add_argument("--readout_q0", type=int, default=0)
    parser.add_argument("--readout_q1", type=int, default=1)
    parser.add_argument(
        "--aggregate",
        choices=["mean_logits", "mean_probs", "attention"],
        default="mean_logits",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--qubits", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--ansatz_layers", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--window_stride", type=int, default=16)
    parser.add_argument("--train_max_windows", type=int, default=3)
    parser.add_argument("--eval_max_windows", type=int, default=6)
    parser.add_argument("--dataset_window_cap", type=int, default=12)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_vocab", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--use_measure_mask", action="store_true")

    args = parser.parse_args()

    checkpoint_path = os.path.expanduser(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    h = base_runner.quixer_cls_hparams.copy()
    if args.qubits is not None:
        h["qubits"] = args.qubits
    if args.layers is not None:
        h["layers"] = args.layers
    if args.ansatz_layers is not None:
        h["ansatz_layers"] = args.ansatz_layers
    if args.window is not None:
        h["window"] = args.window
    if args.batch_size is not None:
        h["batch_size"] = args.batch_size
    if args.dropout is not None:
        h["dropout"] = args.dropout

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    print("Loading datasets...")
    train_texts, train_labels, test_texts, test_labels = base_runner.load_text_classification_parquet(
        args.train, args.test
    )

    (
        train_loader,
        val_loader,
        test_loader,
        vocab_size,
        train_avg_windows,
        test_avg_windows,
        pad_idx,
    ) = final_module.build_windowed_dataloaders(
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        device,
        h["window"],
        h["batch_size"],
        args.val_ratio,
        args.split_seed,
        args.window_stride,
        args.train_max_windows,
        args.eval_max_windows,
        args.dataset_window_cap,
        args.max_vocab,
        args.min_freq,
    )

    print(
        f"Vocab size={vocab_size} | avg train windows={train_avg_windows:.2f} | avg test windows={test_avg_windows:.2f}"
    )

    model, aggregator = _build_model_and_aggregator(args, h, device, vocab_size)

    print(f"Loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model']
    for drop_key in ("base.torchquantum_device.state", "base.torchquantum_device.states"):
        state_dict.pop(drop_key, None)
    load_res = model.load_state_dict(state_dict, strict=False)
    if load_res.missing_keys:
        print(f"Warning: missing keys during load: {load_res.missing_keys}")
    required_missing = [k for k in load_res.missing_keys if k not in ("base.torchquantum_device.state", "base.torchquantum_device.states")]
    if required_missing or load_res.unexpected_keys:
        raise ValueError("Checkpoint architecture mismatch; match the training CLI options.")
    if load_res.unexpected_keys:
        print(f"Warning: unexpected keys during load: {load_res.unexpected_keys}")

    if aggregator is not None and checkpoint.get("aggregator") is not None:
        agg_state = checkpoint["aggregator"]
        if agg_state is not None:
            load_res = aggregator.load_state_dict(agg_state, strict=True)
            if load_res.missing_keys:
                print(f"Warning: missing aggregator keys: {load_res.missing_keys}")
            required_missing = [k for k in load_res.missing_keys if k not in ("base.torchquantum_device.state", "base.torchquantum_device.states")]
    if required_missing or load_res.unexpected_keys:
        raise ValueError("Checkpoint architecture mismatch; match the training CLI options.")
    if load_res.unexpected_keys:
                print(f"Warning: unexpected aggregator keys: {load_res.unexpected_keys}")

    print("Evaluating on validation set...")
    val_loss, val_acc, val_prec, val_recall, val_f1 = _evaluate(
        model, aggregator, val_loader, device, args.label_smoothing, "Validation", pad_idx
    )
    print("Val metrics | Loss {loss:.4f} | Acc {acc:.2f}% | Precision {prec:.2f}% | Recall {rec:.2f}% | F1 {f1:.2f}%".format(
        loss=val_loss, acc=val_acc * 100, prec=val_prec * 100, rec=val_recall * 100, f1=val_f1 * 100
    ))

    print("Evaluating on test set...")
    test_loss, test_acc, test_prec, test_recall, test_f1 = _evaluate(
        model, aggregator, test_loader, device, args.label_smoothing, "Test", pad_idx
    )
    print("Test metrics | Loss {loss:.4f} | Acc {acc:.2f}% | Precision {prec:.2f}% | Recall {rec:.2f}% | F1 {f1:.2f}%".format(
        loss=test_loss, acc=test_acc * 100, prec=test_prec * 100, rec=test_recall * 100, f1=test_f1 * 100
    ))


if __name__ == "__main__":
    main()
