"""Node-level differentially private GNN training (the paper's method).

Example (see run_*.sh for the paper's sweeps):
    python main.py --dataset facebook --expected_batchsize 4096 --epoch 9 --lr 0.01 \
        --priv_epsilon 8 --num_neighbors 3 --num_neighbors_test 7 --seed 1
"""
import time

import torch

import datasets.model as dms_model
import datasets.SETUP as SETUP
import datasets.utils as dms_utils
import train_scheduler
import utils

HIDDEN_CHANNELS = 128


def main():
    args = utils.get_args()
    SETUP.setup_seed(args.seed)
    device = SETUP.get_device()

    train_loader, val_loader, test_loader, dataset, x = dms_utils.form_loaders(args)
    args.num_classes = dataset.num_classes

    model = dms_model.G_net(K=args.K, feat_dim=x.shape[1], num_classes=args.num_classes,
                            hidden_channels=HIDDEN_CHANNELS)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = train_scheduler.NodeDPTrainer(
        model=model,
        optimizer=optimizer,
        loaders=[train_loader, val_loader, test_loader],
        device=device,
        criterion=dms_model.criterion,
        args=args,
    )
    trainer.run()


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'\n==> ToTaL TiMe FoR OnE RuN: {time.time() - start:.4f}\n\n\n')
