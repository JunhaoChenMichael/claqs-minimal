"""Reproducible training and separate evaluation for quantum text classification."""
import argparse
import hashlib
import importlib.metadata
import platform
import json
from pathlib import Path
import random
import time

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from final import WindowedIMDBDataset, _make_collate_fn, MeanLogitsAggregator
from quixer.quixer_model import Quixer
from run_classification_quantum import (
    QuixerMLPReadoutClassifier, build_vocab_from_texts, initialise_weights_local,
)


def read_records(path):
    table = pq.read_table(path)
    text_key = next((k for k in ('text', 'sentence', 'review', 'content') if k in table.column_names), None)
    label_key = next((k for k in ('label', 'labels', 'sentiment', 'target') if k in table.column_names), None)
    if text_key is None or label_key is None:
        raise ValueError('Parquet needs a text/sentence column and a label column')
    texts = table[text_key].to_pylist()
    labels = table[label_key].to_pylist()
    if not texts or any(not isinstance(t, str) for t in texts):
        raise ValueError('Expected nonempty text dataset with no null text')
    if any(y not in (0, 1) for y in labels):
        raise ValueError('Only labeled binary data (0/1) can be evaluated; hidden labels -1 are invalid')
    return texts, [int(y) for y in labels]


def stratified_indices(labels, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError('Validation ratio must be between 0 and 1')
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(set(labels)):
        indices = [i for i, y in enumerate(labels) if y == label]
        if len(indices) < 2:
            raise ValueError('Each class needs at least two examples for splitting')
        rng.shuffle(indices)
        n = max(1, min(len(indices)-1, round(len(indices)*fraction)))
        val.extend(indices[:n])
        train.extend(indices[n:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def take(texts, labels, indices):
    return [texts[i] for i in indices], [labels[i] for i in indices]


def make_model(config, vocab, device):
    base = Quixer(
        n_qubits=config['qubits'], n_tokens=config['window'],
        qsvt_polynomial_degree=config['degree'], n_ansatz_layers=config['ansatz_layers'],
        vocabulary_size=len(vocab), embedding_dimension=config['dimension'],
        dropout=config['dropout'], batch_size=config['batch_size'], device=device,
        n_ff_layers=config['ff_layers'], include_output_head=False,
    )
    model = QuixerMLPReadoutClassifier(
        base, pad_idx=vocab['<pad>'] if config['mask_padding'] else None,
        head_dropout=config['head_dropout'], sparse_padding=config.get('sparse_padding', False),
    )
    initialise_weights_local(model)
    return model.to(device)


def model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if 'torchquantum_device.' not in k}


def load_model_state(model, state):
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if 'torchquantum_device.' not in k]
    if missing or result.unexpected_keys:
        raise ValueError(f'Checkpoint mismatch: missing={missing}, unexpected={result.unexpected_keys}')


def make_loader(texts, labels, vocab, config, train=False):
    dataset = WindowedIMDBDataset(texts, labels, vocab, config['window'], config['stride'],
                                 config['dataset_window_cap'])
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=train, drop_last=False,
                      num_workers=config['workers'], pin_memory=config['device'].startswith('cuda'),
                      generator=torch.Generator().manual_seed(config['seed']),
                      collate_fn=_make_collate_fn(config['max_windows'], sample_random=train))


def batch_logits(model, batch, device, pad_idx):
    tokens, labels, mapping = [x.to(device, non_blocking=True) for x in batch]
    weights = (tokens != pad_idx).sum(dim=1)
    logits, _ = MeanLogitsAggregator()(model(tokens), mapping, len(labels), weights)
    return logits, labels


def evaluate(model, loader, device, pad_idx):
    model.eval()
    loss_sum, count = 0., 0
    conf = torch.zeros(2, 2, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            logits, labels = batch_logits(model, batch, device, pad_idx)
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            count += len(labels)
            conf += torch.bincount((labels*2+logits.argmax(-1)).cpu(), minlength=4).reshape(2, 2)
    tp = conf.diag().float()
    precision = tp / conf.sum(0).clamp_min(1)
    recall = tp / conf.sum(1).clamp_min(1)
    f1 = 2*precision*recall / (precision+recall).clamp_min(1e-12)
    return dict(loss=loss_sum/count, accuracy=tp.sum().item()/count,
                macro_precision=precision.mean().item(), macro_recall=recall.mean().item(),
                macro_f1=f1.mean().item(), samples=count, confusion=conf.tolist())


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def train(args):
    config = {k: v for k, v in vars(args).items() if k != 'func'}
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; previous runs are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    texts, labels = read_records(args.train)
    protocol = dict(train_sha256=digest(args.train), split_seed=args.seed,
                    validation_source='external' if args.validation else 'stratified_train_holdout')
    if args.validation:
        val_texts, val_labels = read_records(args.validation)
        protocol['validation_sha256'] = digest(args.validation)
        if protocol['validation_sha256'] == protocol['train_sha256']:
            raise ValueError('Train and validation files must differ')
    else:
        train_idx, val_idx = stratified_indices(labels, args.val_ratio, args.seed)
        val_texts, val_labels = take(texts, labels, val_idx)
        texts, labels = take(texts, labels, train_idx)
        protocol['train_indices'] = train_idx
        protocol['validation_indices'] = val_idx
    # Subsample only after splitting, and build vocabulary only from actual training rows.
    if args.train_limit and args.train_limit < len(texts):
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(texts)), args.train_limit)
        texts, labels = take(texts, labels, indices)
        protocol['subset_indices'] = indices
    if args.val_limit and args.val_limit < len(val_texts):
        indices = random.Random(args.seed).sample(range(len(val_texts)), args.val_limit)
        val_texts, val_labels = take(val_texts, val_labels, indices)
        protocol['validation_subset_indices'] = indices
    vocab, _ = build_vocab_from_texts(texts, max_vocab=args.max_vocab, min_freq=args.min_freq)
    model = make_model(config, vocab, device)
    train_loader = make_loader(texts, labels, vocab, config, train=True)
    val_loader = make_loader(val_texts, val_labels, vocab, config)
    optimizer_cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.schedule == 'cosine' else None
    params = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
    info = dict(train_samples=len(texts), validation_samples=len(val_texts), vocab_size=len(vocab),
                real_parameters=params, device=str(device), torch_version=str(torch.__version__))
    info['versions'] = {name: importlib.metadata.version(name) for name in ('torchquantum', 'pyarrow')}
    info['versions']['python'] = platform.python_version()
    info['source_sha256'] = {name: digest(Path(__file__).parent/name) for name in
                             ('experiment.py', 'final.py', 'run_classification_quantum.py', 'quixer/quixer_model.py')}
    if device.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(device)
    write_json(output/'config.json', config)
    write_json(output/'data_protocol.json', protocol)
    print(json.dumps({'run': info, 'config': config}), flush=True)
    best_accuracy, best_epoch, stale = -1., 0, 0
    history = []
    start = time.monotonic()
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_start = time.monotonic()
        loss_sum, seen, correct = 0., 0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, target = batch_logits(model, batch, device, vocab['<pad>'])
            loss = torch.nn.functional.cross_entropy(logits, target, label_smoothing=args.label_smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()*len(target)
            correct += (logits.argmax(-1)==target).sum().item()
            seen += len(target)
        metrics = evaluate(model, val_loader, device, vocab['<pad>'])
        record = dict(epoch=epoch, train_loss=loss_sum/seen, train_accuracy=correct/seen,
                      validation=metrics, lr=optimizer.param_groups[0]['lr'], seconds=time.monotonic()-epoch_start)
        history.append(record)
        with (output/'history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        if metrics['accuracy'] > best_accuracy:
            best_accuracy, best_epoch, stale = metrics['accuracy'], epoch, 0
            best_metrics = metrics
            payload = dict(format_version=1, config=config, vocab=vocab, model=model_state(model),
                           epoch=epoch, best_validation=metrics, data_protocol=protocol)
            temp = output/'best.tmp'
            torch.save(payload, temp)
            temp.replace(output/'best.pt')
        else:
            stale += 1
        if scheduler:
            scheduler.step()
        if args.patience and stale >= args.patience:
            break
    result = dict(**info, best_epoch=best_epoch, best_validation=best_metrics,
                  elapsed_seconds=time.monotonic()-start, completed_epochs=len(history),
                  peak_gpu_memory_mb=(torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0),
                  evaluation_split='validation', test_evaluated=False)
    write_json(output/'metrics.json', result)
    print(json.dumps({'result': result}), flush=True)


def eval_checkpoint(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config, vocab = checkpoint['config'].copy(), checkpoint['vocab']
    config.update(device=args.device, workers=0)
    model = make_model(config, vocab, device)
    load_model_state(model, checkpoint['model'])
    texts, labels = read_records(args.data)
    data_hash = digest(args.data)
    protocol = checkpoint['data_protocol']
    if args.split_name == 'validation':
        external = protocol['validation_source'] == 'external'
        expected_hash = protocol['validation_sha256'] if external else protocol['train_sha256']
        if data_hash != expected_hash:
            raise ValueError('Validation file does not match the saved data protocol')
        if not external:
            texts, labels = take(texts, labels, protocol['validation_indices'])
        if 'validation_subset_indices' in protocol:
            texts, labels = take(texts, labels, protocol['validation_subset_indices'])
    elif data_hash in (protocol['train_sha256'], protocol.get('validation_sha256')):
        raise ValueError('Test evaluation cannot use the saved training or validation file')
    metrics = evaluate(model, make_loader(texts, labels, vocab, config), device, vocab['<pad>'])
    result = dict(split=args.split_name, metrics=metrics, data_sha256=data_hash,
                  checkpoint_sha256=digest(args.checkpoint),
                  checkpoint_epoch=checkpoint['epoch'])
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--train', required=True)
    p.add_argument('--validation', help='Official labeled validation file, e.g. SST-2 validation')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    for name, default in [('epochs',5),('qubits',8),('degree',3),('ansatz-layers',2),('ff-layers',1),
                          ('dimension',128),('window',64),('stride',64),('batch-size',16),('max-windows',1),
                          ('dataset-window-cap',12),('workers',0),('seed',42),('cpu-threads',2),
                          ('max-vocab',30000),('min-freq',1),('train-limit',0),('val-limit',0),('patience',3)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('dropout',0.1),('head-dropout',0.1),('lr',0.001),('weight-decay',0.0001),
                          ('label-smoothing',0.),('val-ratio',0.1),('max-grad-norm',5.)]:
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--mask-padding', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--sparse-padding', action='store_true', help='Skip masked PAD token circuits')
    p.add_argument('--optimizer', choices=['adam','adamw'], default='adamw')
    p.add_argument('--schedule', choices=['constant','cosine'], default='cosine')
    p.set_defaults(func=train)
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--split-name', choices=['validation','test'], required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--output')
    p.set_defaults(func=eval_checkpoint)
    args=parser.parse_args()
    if args.command=='train':
        for name in ('epochs','qubits','degree','ansatz_layers','ff_layers','dimension','window','stride',
                     'batch_size','max_windows','dataset_window_cap','cpu_threads','max_vocab'):
            if getattr(args,name)<1:
                parser.error(f'{name} must be positive')
        if args.sparse_padding and not args.mask_padding:
            parser.error('--sparse-padding requires --mask-padding')
        if args.qubits<2:
            parser.error('ansatz-14 requires at least two qubits')
        if args.train_limit<0 or args.val_limit<0 or args.workers<0:
            parser.error('limits and workers must be nonnegative')
    args.func(args)


if __name__ == '__main__':
    main()
