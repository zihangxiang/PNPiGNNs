"""Dataset loading, train/val/test splitting and construction of the subgraph loaders."""
import time
from types import SimpleNamespace

import torch

import privacy.sampling as sampling
from . import SETUP

# Cache directories below the data root. Their content is a deterministic
# function of (dataset, split seed, ...), so they can be reused across runs.
DEGREE_CACHE_DIR = '_cache_edge_index_right_dp'
NEIGHBOR_CACHE_DIR = '_cache_neighbors_right_dp'
# Training subgraphs only consider the first this-many neighbours of a node (memory bound).
MAX_NEIGHBORS_TRAIN = 500
TEST_BATCHSIZE = 1000


def _pyg(cls_name, **kwargs):
    """Loader for ``torch_geometric.datasets.<cls_name>(root=..., **kwargs)``."""
    def load(root):
        import torch_geometric.datasets as pyg_datasets
        return getattr(pyg_datasets, cls_name)(root=root, **kwargs)
    return load


def _ogbn_arxiv(root):
    from ogb.nodeproppred import PygNodePropPredDataset
    return PygNodePropPredDataset(name='ogbn-arxiv', root=root)


def _fake_dataset(root):
    from torch_geometric.datasets import FakeDataset
    return FakeDataset(num_graphs=1, num_nodes=20000, num_classes=10, avg_degree=50,
                       num_channels=128, task='node', is_undirected=False)


def _dgl_fraud_amazon(root):
    from dgl.data import FraudAmazonDataset
    return FraudAmazonDataset(raw_dir=root, train_size=0.8, val_size=0.01)


# --dataset name -> function(root directory) -> dataset
DATASETS = {
    'Amazon_Computers': _pyg('Amazon', name='Computers'),
    'Amazon_Photo': _pyg('Amazon', name='Photo'),
    'Amazon_Products': _pyg('AmazonProducts'),
    'PubMed': _pyg('Planetoid', name='PubMed'),
    'Cora': _pyg('Planetoid', name='Cora'),
    'CiteSeer': _pyg('Planetoid', name='CiteSeer'),
    'Reddit': _pyg('Reddit'),
    'Reddit2': _pyg('Reddit2'),
    'NELL': _pyg('NELL'),
    'Coauthor_CS': _pyg('Coauthor', name='CS'),
    'Coauthor_Physics': _pyg('Coauthor', name='physics'),
    'CitationFull_Cora': _pyg('CitationFull', name='Cora'),
    'CitationFull_DBLP': _pyg('CitationFull', name='DBLP'),
    'CitationFull_Cora_ML': _pyg('CitationFull', name='Cora_ML'),
    'Ogbn_Arvix': _ogbn_arxiv,
    'Flickr': _pyg('Flickr'),
    'FakeDataset': _fake_dataset,
    'WikiCS': _pyg('WikiCS', is_undirected=False),
    'facebook': _pyg('FacebookPagePage'),
    'twitch_DE': _pyg('Twitch', name='DE'),
    'twitch_PT': _pyg('Twitch', name='PT'),
    'twitch_EN': _pyg('Twitch', name='EN'),
    'dgl_famazon': _dgl_fraud_amazon,
}


def graph_dataset_summary(dataset, split):
    data = dataset[0]
    num_train, num_val, num_test = (int(m.sum()) for m in (split.train_mask, split.val_mask, split.test_mask))
    print(f'\n==> Summary of the dataset:...\n{"=" * 50}')
    print(f'Datset: {dataset}:')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print('\nsummary on the first graph...')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}\n')
    print(f'Number of training nodes: {num_train}')
    print(f'Number of validation nodes: {num_val}')
    print(f'Number of test nodes: {num_test}')
    print(f'Train Val Test node label rate: {num_train / data.num_nodes:.3f}, '
          f'{num_val / data.num_nodes:.3f}, {num_test / data.num_nodes:.3f}\n')
    if hasattr(data, 'is_undirected'): print(f'Is undirected: {data.is_undirected()}')
    print(f'{"=" * 50}\n\n')


def _adapt_dgl_graph(graph):
    """Exposes a DGL graph through the PyG attributes used in this code base."""
    graph.train_mask, graph.val_mask, graph.test_mask = graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask']
    graph.num_nodes = graph.num_nodes()
    graph.num_edges = graph.num_edges()
    graph.x = graph.ndata['feature']
    graph.y = graph.ndata['label']
    s, d = graph.adj_tensors('coo', etype=graph.etypes[0])
    graph.edge_index = torch.stack([s, d], dim=0)


def get_split_train_val_test(dataset, split_ratio=(0.8, 0.01, 0.19)):
    """Random node split (ignores any predefined masks); the test split takes all remaining nodes."""
    one_graph = dataset[0]
    if hasattr(one_graph, 'ndata'):
        print('==> dgl graph dataset...')
        _adapt_dgl_graph(one_graph)

    print(f'==> spliting dataset into train, val and test sets by ratio: {split_ratio}...')
    num_nodes = one_graph.num_nodes
    train_num, val_num = int(split_ratio[0] * num_nodes), int(split_ratio[1] * num_nodes)

    permuted_indices = torch.randperm(num_nodes)
    masks = [torch.zeros(num_nodes, dtype=torch.bool) for _ in range(3)]
    masks[0][permuted_indices[:train_num]] = True
    masks[1][permuted_indices[train_num:train_num + val_num]] = True
    masks[2][permuted_indices[train_num + val_num:]] = True
    return SimpleNamespace(train_mask=masks[0], val_mask=masks[1], test_mask=masks[2])


def get_raw_dataset(dataset_name):
    """Loads ``dataset_name`` (downloading it below the data root if needed) and splits its nodes."""
    if dataset_name not in DATASETS:
        raise ValueError(f'Invalid dataset name, got {dataset_name}; choose from {sorted(DATASETS)}')

    print(f'==> Using {dataset_name} data')
    dataset = DATASETS[dataset_name](SETUP.get_dataset_data_path() / dataset_name)
    split = get_split_train_val_test(dataset)
    graph_dataset_summary(dataset, split)
    return dataset, split


def compute_degree_inverse(edge_index, dataset_name, cache_dir):
    """``1 / d_out(u)`` for every node ``u`` (0 if ``d_out(u) = 0``), where
    ``d_out(u)`` is the number of distinct sources of edges into ``u``.

    Cached in ``cache_dir / f'{dataset_name}.pt'``.
    """
    start = time.time()
    file_path = cache_dir / f'{dataset_name}.pt'
    cache_dir.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        print(f'==> load processed edge index from {file_path}')
        return torch.load(file_path)

    print('==> computing degree for each node')
    num_slots = int(edge_index.max()) + 1
    distinct_edges = torch.unique(edge_index[0] * num_slots + edge_index[1])
    in_degree = torch.bincount(distinct_edges % num_slots, minlength=num_slots)

    degree_inverse = torch.zeros(num_slots)
    has_in_edges = in_degree > 0
    # 1/d in float64 then rounded to float32, matching a python-float assignment
    degree_inverse[has_in_edges] = (1.0 / in_degree[has_in_edges].double()).float()

    print(f'time used: {time.time() - start}')
    print(f'saving edge_index to {file_path}')
    torch.save(degree_inverse, file_path)
    return degree_inverse


def form_loaders(args):
    """Builds the subgraph loaders of ``main.py``.

    Returns:
        ``(train_loader, val_loader=None, test_loader, dataset, x)`` where ``x``
        are the standardised node features.
    """
    data_root = SETUP.get_dataset_data_path()
    dataset, split = get_raw_dataset(args.dataset)
    graph_data = dataset[0]

    graph_data.y = graph_data.y[:graph_data.x.shape[0]]  # some datasets have more labels than nodes
    graph_data.x = (graph_data.x - graph_data.x.mean()) / (graph_data.x.std() + 1e-6)

    out_degree_inverse = compute_degree_inverse(graph_data.edge_index, str(dataset), data_root / DEGREE_CACHE_DIR)

    assert args.graph_setting in ['inductive', 'transductive']
    common = dict(
        K=args.K,
        graph_data=graph_data,
        graph_data_name=str(dataset),
        setting=args.graph_setting,
        seed=args.seed,
        cache_file_path=data_root / NEIGHBOR_CACHE_DIR,
    )
    train_set = sampling.SubgraphSampler(
        num_neighbors=args.num_neighbors,
        mask=split.train_mask,
        dataset_mode='train',
        out_degree_inverse=out_degree_inverse,
        max_neighbors_train=MAX_NEIGHBORS_TRAIN,
        **common,
    )
    test_set = sampling.SubgraphSampler(
        num_neighbors=args.num_neighbors_test,
        mask=split.test_mask,
        dataset_mode='test',
        **common,
    )

    train_loader = sampling.get_subgraphs_loader(train_set, expected_batchsize=args.expected_batchsize, worker_num=args.worker_num)
    test_loader = sampling.get_subgraphs_loader(test_set, expected_batchsize=TEST_BATCHSIZE, worker_num=args.worker_num)
    return train_loader, None, test_loader, dataset, graph_data.x
