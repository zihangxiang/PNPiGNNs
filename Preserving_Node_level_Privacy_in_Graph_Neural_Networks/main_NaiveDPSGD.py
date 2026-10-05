"""Baseline: DP-SGD on an MLP over node features only (no graph structure).

Each training node is one record, so standard record-level DP-SGD accounting applies.
"""
import time

import torch
from torch import nn

import datasets.model as dms_model
import datasets.SETUP as SETUP
import datasets.utils as dms_utils
import train_scheduler
import utils


def make_mlp(in_dim, num_classes, h_dim):
    return nn.Sequential(
        nn.Linear(in_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, num_classes),
    )


def make_loader(graph, mask, batch_size, shuffle):
    dataset = torch.utils.data.TensorDataset(graph.x[mask], graph.y[mask])
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=4, drop_last=False)


def main():
    args = utils.get_args()
    args.graph_setting = 'naive'
    SETUP.setup_seed(args.seed)
    device = SETUP.get_device()

    dataset, split = dms_utils.get_raw_dataset(args.dataset)
    graph = dataset[0]
    args.num_classes = dataset.num_classes

    h_dim = 16 if args.dataset == 'Reddit' else 32
    model = make_mlp(graph.x.shape[1], args.num_classes, h_dim).to(device)

    train_loader = make_loader(graph, split.train_mask, args.expected_batchsize, shuffle=True)
    train_loader.dataset.graph_data_name = str(dataset)
    train_loader.dataset.graph_data = graph
    test_loader = make_loader(graph, split.test_mask, args.expected_batchsize, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = train_scheduler.NaiveDPSGDTrainer(
        model=model,
        optimizer=optimizer,
        loaders=[train_loader, None, test_loader],
        device=device,
        criterion=dms_model.criterion,
        args=args,
    )
    trainer.run()


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'\n==> ToTaL TiMe FoR OnE RuN: {time.time() - start:.4f}\n\n\n')
