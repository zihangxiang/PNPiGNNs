"""Command-line arguments, classification metrics and plain-text result recording."""
import argparse
import time
from pathlib import Path

import torch


def get_args():
    parser = argparse.ArgumentParser(description='Node-level differentially private GNN training.')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name, see datasets/utils.py:DATASETS')
    parser.add_argument('--expected_batchsize', type=int, default=100, help='expected number of roots per Poisson-sampled batch')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs (each has ceil(N / expected_batchsize) steps)')
    parser.add_argument('--lr', type=float, default=0.001, help='Adam learning rate')
    parser.add_argument('--C', type=float, default=1, help='gradient clipping threshold')
    parser.add_argument('--priv_epsilon', type=float, default=8.8, help='target epsilon (delta = 1 / N^1.1)')
    parser.add_argument('--K', type=int, default=1, help='number of GNN layers / neighbour-sampling rounds')
    parser.add_argument('--num_neighbors', type=int, default=1, help='neighbour-sampling budget M during training (enters the accountant)')
    parser.add_argument('--num_neighbors_test', type=int, default=1, help='max neighbours per test subgraph')
    parser.add_argument('--graph_setting', type=str, default='transductive', help="'transductive' or 'inductive'")
    parser.add_argument('--worker_num', type=int, default=16, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed (also determines the node split)')
    parser.add_argument('--log_dir', type=str, default='logs', help='log directory, relative to this project directory')
    return parser.parse_args()


def show_param(model):
    """Prints and returns ``(summary_str, num_trainable_params)``."""
    lines = [f'\n{"=" * 40}', 'parameter summary:']
    num_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params += param.numel()
            lines.append(f'{name}, {param.data.shape}')
    lines += [f'total number of parameters: {num_params}', f'{"=" * 40}\n']
    info_str = '\n'.join(lines)
    print(info_str)
    return info_str, num_params


class ClassificationMetrics:
    """Running per-class confusion counts, loss and accuracy of one epoch.

    Per-class metrics are properties returning ``(num_classes,)`` tensors. Any
    of them can be prefixed with ``mean_`` (macro average) or ``weighted_``
    (class-frequency weighted average); e.g. ``weighted_recall`` equals accuracy.
    """
    metrics = ('accur', 'recall', 'specif', 'precis', 'npv', 'f1_s', 'iou')

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.tp = self.fn = self.fp = self.tn = 0
        self.hit_count = 0
        self.num_of_prediction = 0
        self.hit_accuracy = 0
        self.num_images = 0
        self.loss = 0

    @torch.no_grad()
    def update(self, pred, true):
        """Adds predicted / true class indices to the confusion counts."""
        pred, true = pred.flatten(), true.flatten()
        classes = torch.arange(0, self.num_classes, device=true.device)
        valid = (0 <= true) & (true < self.num_classes)
        # (num_classes, n) one-hot comparisons
        pred_pos = classes.view(-1, 1) == pred[valid].view(1, -1)
        positive = classes.view(-1, 1) == true[valid].view(1, -1)
        pred_neg, negative = ~pred_pos, ~positive
        self.tp += (pred_pos & positive).sum(dim=1)
        self.fp += (pred_pos & negative).sum(dim=1)
        self.fn += (pred_neg & positive).sum(dim=1)
        self.tn += (pred_neg & negative).sum(dim=1)

        self.hit_count += (pred == true).sum().item()
        self.num_of_prediction += int(pred.numel())
        self.hit_accuracy = self.hit_count / self.num_of_prediction

    def batch_update(self, loss, logits, targets):
        self.num_images += logits.shape[0]
        self.loss += loss.item() * logits.shape[0]
        self.update(logits.data.argmax(dim=1), targets.flatten())

    @property
    def frequency(self):
        count = self.tp + self.fn
        return count / count.sum().clamp(min=1)

    @property
    def total(self):
        return (self.tp + self.fn).sum()

    # denominators are clamped to >= 1 to avoid division by zero
    @property
    def accur(self):
        return (self.tp + self.tn) / self.total.clamp(min=1)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn).clamp(min=1)

    @property
    def specif(self):
        return self.tn / (self.tn + self.fp).clamp(min=1)

    @property
    def npv(self):
        return self.tn / (self.tn + self.fn).clamp(min=1)

    @property
    def precis(self):
        return self.tp / (self.tp + self.fp).clamp(min=1)

    @property
    def f1_s(self):
        tp2 = 2 * self.tp
        return tp2 / (tp2 + self.fp + self.fn).clamp(min=1)

    @property
    def iou(self):
        return self.tp / (self.tp + self.fp + self.fn).clamp(min=1)

    def weighted(self, scores):
        return (self.frequency * scores).sum()

    def __getattr__(self, name):
        """Resolves ``mean_<metric>`` and ``weighted_<metric>``."""
        prefix, _, metric = name.partition('_')
        if prefix in ('mean', 'weighted') and metric:
            values = getattr(self, metric)
            return values.mean() if prefix == 'mean' else self.weighted(values)
        raise AttributeError(name)

    def __repr__(self):
        metrics = torch.stack([getattr(self, m) for m in self.metrics])
        perc = lambda x: f'{float(x) * 100:.2f}%'.ljust(8)
        out = 'Class'.ljust(6) + ''.join(m.ljust(8) for m in self.metrics)
        if self.num_classes <= 20:
            out += '\n' + '-' * 60
            for i, values in enumerate(metrics.t()):
                out += '\n' + str(i).ljust(6) + ''.join(perc(v) for v in values)
        out += '\n' + '-' * 60
        out += '\n' + 'Mean'.ljust(6) + ''.join(perc(m.mean()) for m in metrics)
        out += '\n' + 'Wted'.ljust(6) + ''.join(perc(self.weighted(m)) for m in metrics)
        return out + f'\nhit accuracy: {float(self.hit_accuracy) * 100:.2f}%'


class DataRecorder:
    """Appends lines to text files in ``root``. The first write to a file by
    this recorder starts with a timestamped separator."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.started_files = set()

    def record_data(self, filename, content):
        with open(self.root / filename, 'a') as file:
            if filename not in self.started_files:
                self.started_files.add(filename)
                time_stamp = time.strftime('[%d_%H_%M_%S]', time.localtime(time.time()))
                file.write(f'\n{time_stamp} {"=" * 40}NEW{"=" * 40}\n')
            for item in (content if isinstance(content, list) else [content]):
                file.write(f'{item}\n')
