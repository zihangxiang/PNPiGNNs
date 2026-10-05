"""DP-SGD training loops.

* :class:`NodeDPTrainer`: the paper's node-level DP GNN (``main.py``).
* :class:`NaiveDPSGDTrainer`: graph-free DP-SGD MLP baseline (``main_NaiveDPSGD.py``).

Per-sample gradients are computed with ``vmap(grad(...))`` on a functional copy
of the model, privatised (clipped, averaged, Gaussian noise added) and applied
to the real model by the optimizer; the functional copy is then re-synced.

Outputs (relative to this directory):
    ``<log_dir>/log.txt``              human-readable log of every run
    ``<log_dir>/weighted_recall.csv``  per-epoch train/val/test accuracy
    ``data_records/jd_*.json``         one JSON entry per run (args + accuracy curves)
"""
import enum
import json
import logging
import time
from copy import deepcopy
from pathlib import Path

import torch
from functorch import grad, make_functional_with_buffers, vmap

import datasets.SETUP as SETUP
import utils
from privacy import accounting_analysis, mix

PROJECT_DIR = Path(__file__).resolve().parent


class Phase(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()


class DPTrainer:
    """Shared DP-SGD loop. Subclasses define:

    * :meth:`compute_noise_multiplier`: sigma for the target ``(epsilon, delta)``;
    * :meth:`sample_logits`: logits of shape ``(1, num_classes)`` for one sample;
    * :meth:`clip_per_grad`: bound the norm of each per-sample gradient;
    * :meth:`add_noise`: privatise one averaged parameter gradient;
    * :meth:`json_record_name`: file name of the JSON record.

    ``loaders`` is ``[train_loader, val_loader, test_loader]`` (val/test may be
    ``None``); the train loader's dataset must have ``graph_data_name`` and
    ``graph_data`` attributes.
    """
    metric_name = 'weighted_recall'  # class-frequency weighted recall == accuracy

    def __init__(self, *, model, optimizer, loaders, device, criterion, args):
        self.model = model
        self.optimizer = optimizer
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.device = device
        self.criterion = criterion
        self.args = args

        self.func_model, self.func_params, self.func_buffers = make_functional_with_buffers(
            deepcopy(model), disable_autograd_tracking=True)

        train_set = self.train_loader.dataset
        args.q = args.expected_batchsize / len(train_set)
        args.delta = 1 / len(train_set) ** 1.1
        # the number of noisy steps that training will actually take
        args.num_steps = args.epoch * len(self.train_loader)
        self.std = args.std = self.compute_noise_multiplier()

        self._init_logging(train_set)
        self.json_recorder = JsonRecorder(PROJECT_DIR / 'data_records' / self.json_record_name())
        self.json_recorder.add_record('args', vars(args))
        self.json_recorder.add_record('num_params', args.num_params)
        self.json_recorder.add_record('dataset', train_set.graph_data_name)
        for key in ('train_acc', 'val_acc', 'test_acc', 'test_pre'):
            self.json_recorder.add_record(key, None)

    # --------------------------------------------------------- to override
    def compute_noise_multiplier(self):
        raise NotImplementedError

    def sample_logits(self, params, buffers, x):
        raise NotImplementedError

    def clip_per_grad(self, per_grad):
        raise NotImplementedError

    def add_noise(self, grad, batch_size):
        raise NotImplementedError

    def json_record_name(self):
        raise NotImplementedError

    # ------------------------------------------------------------- logging
    def _init_logging(self, train_set):
        log_dir = PROJECT_DIR / self.args.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_dir / 'log.txt',
            filemode='a',
            datefmt="%H:%M:%S",
            level=logging.INFO,
            format='%(asctime)s[%(levelname)s] ~ %(message)s',
        )
        self.write_log('\n\n' + "VV" * 40 + '\n' + " " * 37 + 'NEW LOG\n' + "^^" * 40)

        param_info, self.args.num_params = utils.show_param(self.model)
        self.write_log(param_info, verbose=False)
        arg_info = '\n'.join(['args:'] + [f'{key} -- {value}' for key, value in vars(self.args).items()])
        arg_info = f'\n{"=" * 40}\n{arg_info}\n{"=" * 40}\n'
        self.write_log(arg_info)
        self.write_log(f'dataset: {train_set.graph_data_name}')
        self.write_log(train_set.graph_data)

        self.data_logger = utils.DataRecorder(log_dir)
        for item in (arg_info, param_info, train_set.graph_data_name, train_set.graph_data):
            self.data_logger.record_data(f'{self.metric_name}.csv', item)

    @staticmethod
    def write_log(info, verbose=True):
        if verbose:
            print(str(info))
        logging.info(str(info))

    # ------------------------------------------------------------ training
    def run(self):
        start_time = time.time()
        phases = ((Phase.TRAIN, self.train_loader, 'train_acc'),
                  (Phase.VAL, self.val_loader, 'val_acc'),
                  (Phase.TEST, self.test_loader, 'test_acc'))

        for epoch in range(self.args.epoch):
            self.write_log(f'\nEpoch: [{epoch}] '.ljust(11) + '#' * 35)

            results = {}
            for phase, loader, json_key in phases:
                if loader is None:
                    continue
                results[phase] = self.one_epoch(phase, loader)
                self.json_recorder.add_record(json_key, float(getattr(results[phase], self.metric_name)))
            if Phase.TEST in results:
                self.json_recorder.add_record('test_pre', float(results[Phase.TEST].weighted_precis))

            cells = [f'{epoch}'] + [
                f'{float(getattr(results[phase], self.metric_name)) * 100:.2f}%'.rjust(7) if phase in results else 'NAN'
                for phase in Phase
            ]
            self.data_logger.record_data(f'{self.metric_name}.csv', (' ' * 3).join(cells))

        self.write_log(f'\n\n=> TIME for ALL: {time.time() - start_time:.2f}  secs')
        self._shutdown_loader_workers()
        self.json_recorder.save()

    def _shutdown_loader_workers(self):
        """Terminates persistent DataLoader workers."""
        for loader in (self.train_loader, self.val_loader, self.test_loader):
            iterator = getattr(loader, '_iterator', None)
            if iterator is not None:
                iterator._shutdown_workers()

    def one_epoch(self, phase, loader):
        metrics = utils.ClassificationMetrics(num_classes=self.args.num_classes)
        is_training = phase is Phase.TRAIN
        self.model.train(is_training)

        start = time.time()
        if is_training:
            print(f'==> have {len(loader)} iterations in this epoch')
        for x, targets in loader:
            x, targets = x.to(self.device), targets.to(self.device)
            if is_training:
                self.optimizer.zero_grad()
                per_grad = vmap(grad(self._sample_loss), in_dims=(None, None, 0, 0))(
                    self.func_params, self.func_buffers, x, targets)
            loss, logits, targets = self._batch_forward(x, targets)
            metrics.batch_update(loss, logits, targets)
            if is_training:
                self._private_step(per_grad)

        metrics.loss /= metrics.num_images
        self.write_log(f'    {phase}: {time.time() - start:.3f} S, '
                       f'{self.metric_name} = {float(getattr(metrics, self.metric_name)) * 100:.2f}%')
        return metrics

    def _sample_loss(self, params, buffers, x, y):
        return self.criterion(self.sample_logits(params, buffers, x), y.reshape(-1)[:1])

    def _batch_forward(self, x, targets):
        logits = vmap(self.sample_logits, in_dims=(None, None, 0))(self.func_params, self.func_buffers, x)[:, 0]
        targets = targets.reshape(-1)
        return self.criterion(logits, targets), logits, targets

    def _private_step(self, per_grad):
        """Clip, average, add noise, update the model, re-sync the functional copy."""
        per_grad = self.clip_per_grad(list(per_grad))
        for p_model, p_per in zip(self.model.parameters(), per_grad):
            p_model.grad = self.add_noise(torch.mean(p_per, dim=0), batch_size=p_per.shape[0])

        self.optimizer.step()
        for p_model, p_func in zip(self.model.parameters(), self.func_params):
            p_func.copy_(p_model.data)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def per_sample_norms(per_grad):
        """L2 norm of each sample's full gradient, shape ``(batch_size,)``."""
        return torch.norm(torch.cat([p.reshape(p.shape[0], -1) for p in per_grad], dim=1), dim=1, p=2)

    @staticmethod
    def _per_sample(values, like):
        """Reshapes ``(batch_size,)`` values to broadcast against ``like``."""
        return values.reshape(-1, *[1] * (like.dim() - 1))


class NodeDPTrainer(DPTrainer):
    """Node-level DP: one sample is a subgraph; only the root is classified.

    Per-subgraph gradients are clipped to norm ``C / 2``, and the noise
    multiplier comes from the node-level accountant (:mod:`privacy.mix`).
    """

    def compute_noise_multiplier(self):
        return mix.get_std_node_dp(
            q=self.args.q,
            num_steps=self.args.num_steps,
            D_out=len(self.train_loader.dataset),
            M_train=self.args.num_neighbors,
            epsilon=self.args.priv_epsilon,
            delta=self.args.delta,
            cache_dir=SETUP.get_dataset_data_path() / 'privacy_node_dp',
        )

    def sample_logits(self, params, buffers, x):
        return self.func_model(params, buffers, x)[:1]

    def clip_per_grad(self, per_grad):
        norms = 2 * self.per_sample_norms(per_grad) + 1e-6
        multiplier = torch.clamp(self.args.C / norms, max=1)
        return [p * self._per_sample(multiplier, p) for p in per_grad]

    def add_noise(self, grad, batch_size):
        grad = grad + torch.randn_like(grad) * self.std * self.args.C / batch_size
        # clamp extreme coordinates (post-processing)
        threshold = self.std * self.args.C / batch_size / 1e2
        return torch.clamp(grad, -threshold, threshold)

    def json_record_name(self):
        name = self.train_loader.dataset.graph_data_name
        return f'jd_{name}_{self.args.graph_setting}_eps{self.args.priv_epsilon}.json'


class NaiveDPSGDTrainer(DPTrainer):
    """Record-level DP-SGD on node features alone (no graph).

    Per-sample gradients are normalised to norm ``C``, and the noise multiplier
    comes from the standard subsampled-Gaussian accountant.
    """

    def compute_noise_multiplier(self):
        return accounting_analysis.get_std(
            q=self.args.q,
            num_steps=self.args.num_steps,
            epsilon=self.args.priv_epsilon,
            delta=self.args.delta,
            verbose=True,
        )

    def sample_logits(self, params, buffers, x):
        return self.func_model(params, buffers, x).unsqueeze(0)

    def clip_per_grad(self, per_grad):
        norms = self.per_sample_norms(per_grad) + 1e-6
        return [p / self._per_sample(norms / self.args.C, p) for p in per_grad]

    def add_noise(self, grad, batch_size):
        return grad + torch.randn_like(grad) * self.std * self.args.C / self.args.expected_batchsize

    def json_record_name(self):
        return f'jd_{self.train_loader.dataset.graph_data_name}_{self.args.graph_setting}.json'


class JsonRecorder:
    """Collects ``{name: [values]}`` for one run and appends it to a JSON list file."""

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.data_dict = {}

    def add_record(self, name, data):
        """Appends ``data`` to ``name`` (``None`` only creates an empty entry)."""
        self.data_dict.setdefault(name, [])
        if data is not None:
            self.data_dict[name].append(data)

    def save(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        if self.file_path.exists():
            with open(self.file_path, 'r') as f:
                data = json.load(f)
        with open(self.file_path, 'w') as f:
            json.dump(data + [self.data_dict], f, indent=4)
