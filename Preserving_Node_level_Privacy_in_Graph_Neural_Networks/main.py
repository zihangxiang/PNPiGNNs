import torch
import time
import datasets.SETUP as SETUP

import datasets.utils as dms_utils
import datasets.model as dms_model

import utils
import train_scheduler as tsch

if __name__ == '__main__':
    s_time  = time.time()
    args = utils.get_args()

    SETUP.setup_seed(args.seed)
    device = SETUP.get_device()

    train_loader, val_loader, test_loader, dataset, x = dms_utils.form_loaders(args)
    args.num_classes = dataset.num_classes

    model = dms_model.G_net(K = args.K, feat_dim = x.shape[1], num_classes = args.num_classes, hidden_channels = 128)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, )
    # optimizer = torch.optim.SGD(model.parameters(), lr = 1, momentum = 0.0)

    # args.optimizer = str(optimizer)
    train_master = tsch.trainer(
                        model = model,
                        optimizer = optimizer,
                        loaders = [train_loader, None, test_loader],
                        device = device,
                        criterion = dms_model.criterion,
                        args = args,
                    )
    
    train_master.run()
    print(f'\n==> ToTaL TiMe FoR OnE RuN: {time.time() - s_time:.4f}\n\n\n')


