import argparse
import copy
import re
import sys
from collections import Counter
from typing import Tuple, List

import torch
from tqdm import tqdm

# Optionally silence torchtext deprecation warnings if torchtext gets imported indirectly
try:
    import torchtext  # type: ignore
    torchtext.disable_torchtext_deprecation_warning()  # type: ignore
except Exception:
    pass

# Ensure stdout is line-buffered so progress prints appear immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from quixer.quixer_model import Quixer, apply_qsvt_and_lcu


##################################################
# Default hyperparameters for Quixer classification
##################################################

quixer_cls_hparams = {
    "qubits": 8,
    "layers": 5,
    "ansatz_layers": 6,
    "window": 32,
    "epochs": 30,
    "dropout": 0.20,
    "lr": 0.0005,
    "wd": 0.0005,
    "eps": 1e-10,
    "batch_size": 16,
    "max_grad_norm": 5.0,
    "dimension": 512,
}


def initialise_weights_local(model: torch.nn.Module) -> None:
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    model.apply(_init_weights)


def basic_tokenizer(text: str) -> List[str]:
    # Keep alphanumeric and apostrophes to avoid fragmenting common contractions
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab_from_texts(
    texts: List[str], max_vocab: int = 50000, min_freq: int = 2
) -> Tuple[dict, dict]:
    specials = ["<pad>", "<unk>", "<eos>"]
    counter: Counter[str] = Counter()
    for sent in texts:
        counter.update(basic_tokenizer(sent))

    most_common = [tok for tok, freq in counter.most_common() if freq >= min_freq]
    most_common = most_common[: max(0, max_vocab - len(specials))]

    word_to_idx = {token: idx for idx, token in enumerate(specials)}
    idx_to_word = {idx: token for token, idx in word_to_idx.items()}

    for tok in most_common:
        idx = len(word_to_idx)
        word_to_idx[tok] = idx
        idx_to_word[idx] = tok

    return word_to_idx, idx_to_word


def encode_text(text: str, vocab: dict, window: int) -> torch.Tensor:
    tokens = [vocab.get(tok, vocab["<unk>"]) for tok in basic_tokenizer(text)] + [vocab["<eos>"]]
    if len(tokens) < window:
        tokens = [vocab["<pad>"]] * (window - len(tokens)) + tokens
    else:
        tokens = tokens[-window:]
    return torch.tensor(tokens, dtype=torch.long)  # CPU tensor


class QuixerZReadoutClassifier(torch.nn.Module):
    """All-quantum readout classifier: uses Z expectation on a chosen qubit to produce logits.

    For binary classification, logits = [-z, +z], where z is the Z-expectation of the readout qubit.
    """

    def __init__(self, base: Quixer, readout_qubit: int = 0):
        super().__init__()
        self.base = base
        self.readout_qubit = readout_qubit

    def forward(self, x):
        batch_size = x.shape[0]
        x_emb = self.base.embedding(x)
        pqc_angles = self.base.embedding_to_angles(self.base.dropout(x_emb))

        initial_states = torch.zeros(
            batch_size, 2 ** self.base.n_qubits, dtype=torch.complex64, device=self.base.device
        )
        initial_states[:, 0] = 1.0

        lcu_coefficients = self.base.lcu_coefficients.repeat(batch_size, 1)
        lcu_coefficients = torch.nn.functional.normalize(lcu_coefficients, p=1)

        qsvt_lcu_state = apply_qsvt_and_lcu(
            initial_states,
            pqc_angles,
            self.base.token_parameterized_quantum_circuit,
            self.base.torchquantum_device,
            self.base.n_qubits,
            lcu_coefficients,
            self.base.qsvt_polynomial_coefficients,
        )

        # Normalize and load into TorchQuantum device for feedforward + measurement
        self.base.torchquantum_device.set_states(
            torch.nn.functional.normalize(qsvt_lcu_state, dim=-1)
        )

        self.base.quantum_feedforward(
            self.base.torchquantum_device,
            self.base.quantum_feedforward_parameters.unsqueeze(0).expand(batch_size, -1),
        )

        # Measure X/Y/Z expectations on all qubits (reuses model's measurement stack)
        expectation_values = self.base.measure_all_x_y_z(self.base.torchquantum_device)
        expectation_values = (
            expectation_values.reshape(3, batch_size, self.base.n_qubits)
            .moveaxis(0, 1)
            .reshape(batch_size, -1)
        )

        # Extract Z expectation for the chosen readout qubit
        z_block_offset = 2 * self.base.n_qubits  # X block (n), Y block (n), then Z block (n)
        z = expectation_values[:, z_block_offset + self.readout_qubit]  # [B], in [-1, 1]

        # Binary logits [-z, +z]
        logits = torch.stack([-z, z], dim=-1)
        return logits


class QuixerTwoZReadoutClassifier(torch.nn.Module):
    """All-quantum readout: use Z expectations of two readout qubits as 2D logits.

    This increases expressivity over a single-bit readout while keeping a pure quantum readout.
    """

    def __init__(self, base: Quixer, readout_q0: int = 0, readout_q1: int = 1):
        super().__init__()
        self.base = base
        self.readout_q0 = readout_q0
        self.readout_q1 = readout_q1

    def forward(self, x):
        batch_size = x.shape[0]
        x_emb = self.base.embedding(x)
        pqc_angles = self.base.embedding_to_angles(self.base.dropout(x_emb))

        initial_states = torch.zeros(
            batch_size, 2 ** self.base.n_qubits, dtype=torch.complex64, device=self.base.device
        )
        initial_states[:, 0] = 1.0

        lcu_coefficients = self.base.lcu_coefficients.repeat(batch_size, 1)
        lcu_coefficients = torch.nn.functional.normalize(lcu_coefficients, p=1)

        qsvt_lcu_state = apply_qsvt_and_lcu(
            initial_states,
            pqc_angles,
            self.base.token_parameterized_quantum_circuit,
            self.base.torchquantum_device,
            self.base.n_qubits,
            lcu_coefficients,
            self.base.qsvt_polynomial_coefficients,
        )

        self.base.torchquantum_device.set_states(
            torch.nn.functional.normalize(qsvt_lcu_state, dim=-1)
        )

        self.base.quantum_feedforward(
            self.base.torchquantum_device,
            self.base.quantum_feedforward_parameters.unsqueeze(0).expand(batch_size, -1),
        )

        expectation_values = self.base.measure_all_x_y_z(self.base.torchquantum_device)
        expectation_values = (
            expectation_values.reshape(3, batch_size, self.base.n_qubits)
            .moveaxis(0, 1)
            .reshape(batch_size, -1)
        )

        z_block_offset = 2 * self.base.n_qubits
        z0 = expectation_values[:, z_block_offset + self.readout_q0]
        z1 = expectation_values[:, z_block_offset + self.readout_q1]

        logits = torch.stack([z0, z1], dim=-1)
        return logits


class QuixerMLPReadoutClassifier(torch.nn.Module):
    """Hybrid readout with optional padding masking and hidden-layer dropout.

    Leave pad_idx=None to retain the legacy treatment of padding as real tokens.
    A pad_idx excludes those positions from the LCU and normalizes its coefficients
    independently for each example. Completely padded rows produce a finite result
    through the polynomial's constant term. sparse_padding additionally skips
    quantum simulation for those padding positions, while retaining every valid
    token even when its trainable LCU coefficient is zero.
    """

    def __init__(
        self,
        base: Quixer,
        num_labels: int = 2,
        use_measure_mask: bool = False,
        pad_idx: int | None = None,
        head_dropout: float = 0.0,
        sparse_padding: bool = False,
    ):
        super().__init__()
        self.base = base
        self.use_measure_mask = use_measure_mask
        self.pad_idx = pad_idx
        self.sparse_padding = sparse_padding
        if pad_idx is not None and not 0 <= pad_idx < self.base.embedding.num_embeddings:
            raise ValueError("pad_idx must be a valid vocabulary index")
        if self.use_measure_mask:
            # Learnable mask over measurement features
            self.measure_mask = torch.nn.Parameter(torch.zeros(self.base.nr_of_measurements))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.base.nr_of_measurements, self.base.embedding_dimension),
            # The parameter-free activation block keeps Linear state_dict keys
            # unchanged, including when dropout is enabled for new experiments.
            torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(head_dropout)),
            torch.nn.Linear(self.base.embedding_dimension, num_labels),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_emb = self.base.embedding(x)
        pqc_angles = self.base.embedding_to_angles(self.base.dropout(x_emb))

        initial_states = torch.zeros(
            batch_size, 2 ** self.base.n_qubits, dtype=torch.complex64, device=self.base.device
        )
        initial_states[:, 0] = 1.0

        lcu_coefficients = self.base.lcu_coefficients.unsqueeze(0).expand(batch_size, -1)
        token_mask = None
        if self.pad_idx is not None:
            token_mask = x.ne(self.pad_idx)
            lcu_coefficients = lcu_coefficients * token_mask
        lcu_coefficients = torch.nn.functional.normalize(lcu_coefficients, p=1, dim=-1)

        qsvt_lcu_state = apply_qsvt_and_lcu(
            initial_states,
            pqc_angles,
            self.base.token_parameterized_quantum_circuit,
            self.base.torchquantum_device,
            self.base.n_qubits,
            lcu_coefficients,
            self.base.qsvt_polynomial_coefficients,
            token_mask=token_mask if self.sparse_padding else None,
        )

        self.base.torchquantum_device.set_states(
            torch.nn.functional.normalize(qsvt_lcu_state, dim=-1)
        )

        self.base.quantum_feedforward(
            self.base.torchquantum_device,
            self.base.quantum_feedforward_parameters.unsqueeze(0).expand(batch_size, -1),
        )

        expectation_values = self.base.measure_all_x_y_z(self.base.torchquantum_device)
        expectation_values = (
            expectation_values.reshape(3, batch_size, self.base.n_qubits)
            .moveaxis(0, 1)
            .reshape(batch_size, -1)
        )

        if getattr(self, "use_measure_mask", False):
            mask = torch.sigmoid(self.measure_mask).unsqueeze(0)
            expectation_values = expectation_values * mask
        logits = self.classifier(expectation_values)
        return logits


def load_text_classification_parquet(train_path: str, test_path: str) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Generic parquet loader compatible with IMDB and SST-2 style schemas.

    Accepts common text columns: ["text", "sentence", "review", "content"].
    Accepts common label columns: ["label", "labels", "sentiment", "target"].
    Ensures labels are integers in {0,1} when possible.
    """
    import pyarrow.parquet as pq

    def _extract_xy(table):
        text_cols = ["text", "sentence", "review", "content"]
        label_cols = ["label", "labels", "sentiment", "target"]

        text_col = next((c for c in text_cols if c in table.column_names), None)
        if text_col is None:
            raise KeyError(f"No text column found. Expected one of {text_cols}, got {table.column_names}")

        label_col = next((c for c in label_cols if c in table.column_names), None)
        if label_col is None:
            raise KeyError(f"No label column found. Expected one of {label_cols}, got {table.column_names}")

        xs = table.column(text_col).to_pylist()
        ys_raw = table.column(label_col).to_pylist()

        def _to_int(y):
            if isinstance(y, str):
                named = {"pos": 1, "positive": 1, "true": 1,
                         "neg": 0, "negative": 0, "false": 0}
                y = named.get(y.strip().lower(), y.strip())
            try:
                value = float(y)
            except (TypeError, ValueError):
                raise ValueError(f"Expected binary label 0 or 1, got {y!r}")
            if value not in (0.0, 1.0):
                raise ValueError(
                    f"Expected binary label 0 or 1, got {y!r}. "
                    "Hidden test labels (-1) cannot be evaluated; use a labeled split."
                )
            return int(value)

        ys = [_to_int(y) for y in ys_raw]
        return xs, ys

    train_table = pq.read_table(train_path)
    test_table = pq.read_table(test_path)

    train_texts, train_labels = _extract_xy(train_table)
    test_texts, test_labels = _extract_xy(test_table)

    return train_texts, train_labels, test_texts, test_labels
