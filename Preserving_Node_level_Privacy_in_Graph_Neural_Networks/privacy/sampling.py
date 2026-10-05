"""Subgraph sampling for node-level DP training (Figure 3 of the paper).

Every example is a small subgraph around a *root* node ``r``:

* **train**: roots are the training nodes, Poisson-sampled with rate
  ``q = expected_batchsize / num_train_nodes`` (:class:`PoissonSampler`).
  Each neighbour ``u`` of ``r`` (``r -> u`` in ``edge_index``) is kept
  independently with probability ``num_neighbors / d_out(u)``, where
  ``d_out(u)`` is the number of distinct nodes with an edge into ``u``, i.e.
  the number of roots that can pick ``u``. Neighbours may be any node
  (transductive) or only training nodes (inductive).
* **test**: roots are the test nodes. At most ``num_neighbors`` neighbours
  are drawn uniformly without replacement, from test nodes only.

A root without (admissible) neighbours instead gets ``num_neighbors`` nodes
drawn uniformly with replacement from the admissible nodes. Each of the ``K``
sampling rounds samples neighbours of the root again.
"""
import os
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SubgraphSampler(Dataset):
    """Dataset of sampled subgraphs; item ``i`` is rooted at the ``i``-th node of ``mask``.

    Args:
        K: Number of neighbour-sampling rounds.
        num_neighbors: Neighbour budget (see module docstring).
        graph_data: The whole graph (``torch_geometric.data.Data``-like with ``x``, ``y``, ``edge_index``).
        graph_data_name: Name used for cache files and logging.
        mask: Boolean node mask selecting the roots (train or test split).
        setting: ``'transductive'`` or ``'inductive'``.
        dataset_mode: ``'train'`` or ``'test'``.
        seed: Seed of the run; part of the cache-file name because the split depends on it.
        cache_file_path: Directory for the cached neighbour lists.
        out_degree_inverse: ``1 / d_out(u)`` per node (training only).
        max_neighbors_train: Only the first this-many admissible neighbours of a
            node (in ``edge_index`` order) are kept in training, to bound memory.
    """

    def __init__(self, *, K, num_neighbors, graph_data, graph_data_name, mask, setting,
                 dataset_mode, seed, cache_file_path, out_degree_inverse=None, max_neighbors_train=None):
        super().__init__()
        assert setting in ('transductive', 'inductive')
        assert dataset_mode in ('train', 'test')
        start = time.time()
        print(f'\n\n{"=" * 40}\ndataset init...')

        self.K = K
        self.num_neighbors = num_neighbors
        self.graph_data = graph_data
        self.graph_data_name = graph_data_name
        self.mask = mask
        self.setting = setting
        self.dataset_mode = dataset_mode
        self.out_degree_inverse = out_degree_inverse
        self.max_neighbors_train = max_neighbors_train

        # Cached: {node: shuffled admissible neighbours} and the mask they were built with.
        file_name = f'{graph_data_name}_{setting}_{dataset_mode}_{seed}_{max_neighbors_train}.pt'
        file_path = cache_file_path / file_name
        os.makedirs(cache_file_path, exist_ok=True)
        if os.path.exists(file_path):
            print('==> loading the neighbors of each node in the graph...')
            self.neighbors, self.mask = torch.load(file_path)
            self._init_ids()
            assert len(self.neighbors) == len(self.admissible_nodes)
        else:
            self._init_ids()
            print('==> concluding the neighbors of each node in the graph, it may take a while...')
            self.neighbors = self._build_neighbor_lists()
            torch.save([self.neighbors, self.mask], file_path)
            print(f'==> file saved to: {file_path}')

        print(f'-> setting: {self.setting}, mode: {self.dataset_mode}')
        print(f'-> graph dataset contains {self.mask.numel()}')
        print(f'-> number of usable nodes: {self.root_ids.numel()}')
        print(f'-> number of rest legit nodes: {self.num_admissible}')
        print(f'==> done, time elapsed = {time.time() - start:.4f} seconds')
        print('=' * 40)

    @property
    def _neighbors_restricted_to_mask(self):
        return not (self.setting == 'transductive' and self.dataset_mode == 'train')

    def _init_ids(self):
        self.root_ids = self.mask.nonzero(as_tuple=True)[0]
        # nodes allowed to appear as non-root members of a subgraph
        if self._neighbors_restricted_to_mask:
            self.admissible_nodes = self.mask.nonzero(as_tuple=True)[0]
        else:
            self.admissible_nodes = torch.arange(self.mask.numel())
        self.num_admissible = self.admissible_nodes.numel()

    def _build_neighbor_lists(self):
        """``{node: admissible out-neighbours in random order}`` for every admissible node."""
        source_nodes, neighbor_nodes = self.graph_data.edge_index
        # group targets by source, keeping edge_index order within each group
        source_sorted, order = torch.sort(source_nodes, stable=True)
        neighbors_of = torch.split(
            neighbor_nodes[order],
            torch.bincount(source_sorted, minlength=self.mask.numel()).tolist(),
        )

        neighbors = {}
        for node in tqdm(self.admissible_nodes.tolist()):
            node_neighbors = neighbors_of[node]
            if self._neighbors_restricted_to_mask:
                node_neighbors = node_neighbors[self.mask[node_neighbors]]
            if self.dataset_mode == 'train':
                node_neighbors = node_neighbors[:self.max_neighbors_train]
            neighbors[node] = node_neighbors[torch.randperm(node_neighbors.numel())]
        return neighbors

    def __len__(self):
        return self.root_ids.numel()

    def __getitem__(self, index):
        """Returns ``(x, y_root, nodes, in_mask)`` of one subgraph.

        ``nodes[0]`` is the root, ``x = features[nodes]``, ``y_root`` has shape
        ``(1,)`` and ``in_mask = mask[nodes]``.
        """
        root = int(self.root_ids[index])
        nodes = [root]
        for _ in range(self.K):
            root_neighbors = self.neighbors[root]
            if root_neighbors.numel() == 0:
                picked = np.random.choice(self.num_admissible, self.num_neighbors, replace=True)
                sampled = self.admissible_nodes[picked.tolist()]
            elif self.dataset_mode == 'test':
                num = min(root_neighbors.numel(), self.num_neighbors)
                picked = np.random.choice(root_neighbors.numel(), num, replace=False)
                sampled = root_neighbors[picked.tolist()]
            else:
                keep_prob = self.out_degree_inverse[root_neighbors] * self.num_neighbors
                sampled = root_neighbors[torch.rand(root_neighbors.numel()) <= keep_prob]
            nodes += sampled.tolist()

        nodes = torch.tensor(nodes, dtype=self.graph_data.edge_index.dtype)
        return self.graph_data.x[nodes], self.graph_data.y[nodes][:1], nodes, self.mask[nodes]


def collate_subgraphs(batch, drop_seen_roots):
    """Zero-pads subgraphs to a common size: returns ``x (B, max_nodes, F)`` and ``y (B, 1)``.

    With ``drop_seen_roots`` (training), a non-root node that is in the mask and
    is the root of an earlier subgraph of the batch is removed.
    """
    xs = [x for x, _, _, _ in batch]
    if drop_seen_roots:
        seen_roots = set()
        for i, (x, _, nodes, in_mask) in enumerate(batch):
            nodes = nodes.tolist()
            keep = [0] + [j for j in range(1, len(nodes)) if not (nodes[j] in seen_roots and in_mask[j])]
            xs[i] = x[keep]
            seen_roots.add(nodes[0])

    max_nodes = max(x.shape[0] for x in xs)
    padded = []
    for x in xs:
        x_pad = torch.zeros(max_nodes, x.shape[1])
        x_pad[:x.shape[0], :] = x
        padded.append(x_pad)
    return torch.stack(padded, dim=0), torch.stack([y for _, y, _, _ in batch], dim=0)


class PoissonSampler(torch.utils.data.Sampler):
    """Yields ``ceil(n / batch_size)`` batches per epoch; each index is in a
    batch independently with probability ``batch_size / n``."""

    def __init__(self, num_examples, batch_size):
        self.inds = np.arange(num_examples)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_examples / batch_size))
        self.sample_rate = self.batch_size / (1.0 * num_examples)
        super().__init__(None)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            batch = self.inds[batch_idxs.astype(bool)]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def get_subgraphs_loader(dataset, expected_batchsize, worker_num=4):
    """Poisson-sampled loader for a train sampler, shuffled fixed-size batches otherwise."""
    assert expected_batchsize <= len(dataset), f'expected_batchsize = {expected_batchsize} > {len(dataset)}'
    is_train = dataset.dataset_mode == 'train'
    print(f'==> initializing {dataset.dataset_mode} dataloader')
    common = dict(
        dataset=dataset,
        num_workers=worker_num,
        pin_memory=True,
        collate_fn=partial(collate_subgraphs, drop_seen_roots=is_train),
        persistent_workers=True,
    )
    if is_train:
        return DataLoader(batch_sampler=PoissonSampler(len(dataset), expected_batchsize), **common)
    return DataLoader(batch_size=expected_batchsize, shuffle=True, drop_last=False, **common)
