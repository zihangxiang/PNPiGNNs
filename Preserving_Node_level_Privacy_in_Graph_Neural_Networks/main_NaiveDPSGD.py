import torch
import time
import datasets.SETUP as SETUP
from torch import nn
import datasets.utils as dms_utils
import datasets.model as dms_model

import utils
import train_scheduler_NaiveDPSGD 



if __name__ == '__main__':
    s_time  = time.time()
    args = utils.get_args()
    # torch.multiprocessing.set_start_method('spawn')
    SETUP.setup_seed(args.seed)
    device = SETUP.get_device()

    dataset, split = dms_utils.get_raw_dataset(args.dataset)
    graph = dataset[0]
    args.num_classes = dataset.num_classes

    ''' dummy var assignment '''
    args.max_in_degree = 20
    args.min_out_degree = 10
    args.num_neighbors = 1
    args.num_not_neighbors = 1
    args.graph_setting = 'naive'
    
    h_dim = 32
    if args.dataset == 'Reddit':
        h_dim = 16
    model = nn.Sequential(
                nn.Linear(graph.x.shape[1], h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, args.num_classes)
            )

    model = model.to(device)

    train_dataset = torch.utils.data.TensorDataset(graph.x[split.train_mask], graph.y[split.train_mask])
    train_dataset.graph_data_name = str(dataset)
    train_dataset.graph_data = graph
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=args.expected_batchsize, 
                        shuffle=True, 
                        num_workers=4, 
                        drop_last=False,
                    )
    test_dataset = torch.utils.data.TensorDataset(graph.x[split.test_mask], graph.y[split.test_mask])
    test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=args.expected_batchsize,
                        shuffle=False,
                        num_workers=4,
                        drop_last=False,
                    )

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 1, momentum=0.9)

    # worker_model_func, worker_param_func, worker_buffers_func = make_functional_with_buffers(deepcopy(model), disable_autograd_tracking=True)

    train_scheduler = train_scheduler_NaiveDPSGD.trainer(
        model = model,
        optimizer = optimizer,
        loaders = [train_loader, None, test_loader],
        device = device,
        criterion = dms_model.criterion,
        args = args,
    )
    
    train_scheduler.run()

    print(f'\n==> ToTaL TiMe FoR OnE RuN: {time.time() - s_time:.4f}\n\n\n')


