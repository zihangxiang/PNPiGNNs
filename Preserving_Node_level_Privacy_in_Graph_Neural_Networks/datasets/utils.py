import torch
from pathlib import Path
import numpy as np
import privacy.sampling as sampling
import torch.nn as nn
import torch.nn.functional as F
import time
from functorch import vmap, grad, make_functional_with_buffers
from copy import deepcopy
from tqdm import tqdm
import random
import os
import dgl
#
from . import SETUP
from privacy import accounting_analysis as aa

def graph_dataset_summary(dataset, split):
    print(f'\n==> Summary of the dataset:...')
    print('='*50)
    print(f'Datset: {dataset}:')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    print()
    print(f'summary on the first graph...')
    data = dataset[0]  # Get the first graph object.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')

    print()

    print(f'Number of training nodes: {split.train_mask.sum()}')
    print(f'Number of validation nodes: {split.val_mask.sum()}')
    print(f'Number of test nodes: {split.test_mask.sum()}')
    print(f'Train Val Test node label rate: {int(split.train_mask.sum()) / data.num_nodes:.3f}, {int(split.val_mask.sum()) / data.num_nodes:.3f}, {int(split.test_mask.sum()) / data.num_nodes:.3f}')
    # print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')

    print()
    if hasattr(data, 'is_undirected'): print(f'Is undirected: {data.is_undirected()}')

    print('='*50)
    print('\n')


def get_split_train_val_test(dataset, split_ratio=(0.8, 0.01, 0.19), enfore_re_split=True):
    # Split the dataset into train, validation and test sets.
    # We use a 60-20-20 split.
    one_graph = dataset[0]
    '''dgl dataset'''
    if hasattr(one_graph, 'ndata'):
        print(f'==> dgl graph dataset...')
        one_graph.train_mask, one_graph.val_mask, one_graph.test_mask = one_graph.ndata['train_mask'], one_graph.ndata['val_mask'], one_graph.ndata['test_mask']
        one_graph.num_nodes = one_graph.num_nodes()
        one_graph.num_edges = one_graph.num_edges()
        one_graph.x = one_graph.ndata['feature']
        one_graph.y = one_graph.ndata['label']

        s,d = one_graph.adj_tensors('coo', etype = one_graph.etypes[0])
        one_graph.edge_index = torch.stack([s,d], dim = 0)

    print(f'masks already defined?: {hasattr(one_graph, "train_mask")}, {hasattr(one_graph, "val_mask")}, {hasattr(one_graph, "test_mask")}')
    if hasattr(one_graph, 'train_mask') and hasattr(one_graph, 'val_mask') and hasattr(one_graph, 'test_mask') and not enfore_re_split:
        print(f'==> dataset already has train, val and test sets...')
        train_mask, val_mask, test_mask = one_graph.train_mask, one_graph.val_mask, one_graph.test_mask
        

    else:    
        print(f'==> spliting dataset into train, val and test sets by ratio: {split_ratio}...')
        num_nodes = one_graph.num_nodes
        
        train, val, test = split_ratio
        train_num, val_num, test_num = int(train * num_nodes), int(val * num_nodes), int(test * num_nodes)

        permuted_indices = torch.randperm(num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[permuted_indices[:train_num]] = True
        val_mask[permuted_indices[train_num:train_num+val_num]] = True
        test_mask[permuted_indices[train_num+val_num:]] = True

    class dummy():
        pass
    split = dummy()
    split.train_mask, split.val_mask, split.test_mask = train_mask, val_mask, test_mask
    return split

def get_raw_dataset(dataset_name):
    if dataset_name not in [
        'Amazon_Computers', 
        'Amazon_Photo',
        'AmazonProducts', 
        'Reddit', 
        'Reddit2', 
        'Cora', 
        'PubMed',
        'CiteSeer',
        'NELL',
        'Coauthor_CS',
        'Coauthor_Physics',
        'CitationFull_Cora',
        'CitationFull_DBLP',
        'CitationFull_Cora_ML',
        'Ogbn_Arvix',
        'Flickr',
        'Amazon_Products',
        'FakeDataset',
        'WikiCS',
        'facebook',
        'twitch_DE',
        'dgl_famazon',
        ]:
        raise ValueError(f'Invalid dataset name, got {dataset_name}')

    print(f'==> Using {dataset_name} data')
    data_file_root = Path( SETUP.get_dataset_data_path() ) / f'{dataset_name}'
    # print('==> data_file_root = ', data_file_root)

    # print(f'==> initializing dataset: {dataset_name}')
    if dataset_name == 'Amazon_Products':
        from torch_geometric.datasets import AmazonProducts
        dataset = AmazonProducts(root = data_file_root)
        
    elif dataset_name == 'Amazon_Computers':
        from torch_geometric.datasets import Amazon 
        dataset = Amazon(root = data_file_root, name='Computers')

    elif dataset_name == 'Amazon_Photo':
        from torch_geometric.datasets import Amazon 
        dataset = Amazon(root = data_file_root, name='Photo')

    elif dataset_name == 'PubMed':
        from torch_geometric.datasets import Planetoid 
        dataset = Planetoid(root = data_file_root, name='PubMed')

    elif dataset_name == 'Cora':
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root = data_file_root, name='Cora')

    elif dataset_name == 'CiteSeer':
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root = data_file_root, name='CiteSeer')

    elif dataset_name == 'Reddit':
        from torch_geometric.datasets import Reddit 
        dataset = Reddit(root = data_file_root)

    elif dataset_name == 'Reddit2':
        from torch_geometric.datasets import Reddit2 
        dataset = Reddit2(root = data_file_root)

    elif dataset_name == 'NELL':
        from torch_geometric.datasets import NELL 
        dataset = NELL(root = data_file_root)

    elif dataset_name == 'Coauthor_CS':
        from torch_geometric.datasets import Coauthor 
        dataset = Coauthor(root = data_file_root, name='CS')

    elif dataset_name == 'Coauthor_Physics':
        from torch_geometric.datasets import Coauthor 
        dataset = Coauthor(root = data_file_root, name='physics')

    elif dataset_name == 'CitationFull_Cora':
        from torch_geometric.datasets import CitationFull
        dataset = CitationFull(root = data_file_root, name='Cora')

    elif dataset_name == 'CitationFull_DBLP':
        from torch_geometric.datasets import CitationFull
        dataset = CitationFull(root = data_file_root, name='DBLP')

    elif dataset_name == 'CitationFull_Cora_ML':
        from torch_geometric.datasets import CitationFull
        dataset = CitationFull(root = data_file_root, name='Cora_ML')

    elif dataset_name == 'Ogbn_Arvix':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=data_file_root)

    elif dataset_name == 'Flickr':
       from torch_geometric.datasets import Flickr
       dataset = Flickr(root=data_file_root)
    
    elif dataset_name == 'FakeDataset':
        from torch_geometric.datasets import FakeDataset
        dataset = FakeDataset(num_graphs = 1, num_nodes = 20000, num_classes = 10, avg_degree = 50, num_channels = 128, task = 'node', is_undirected = False)

    elif dataset_name == 'WikiCS':
        from torch_geometric.datasets import WikiCS
        dataset = WikiCS(root=data_file_root, is_undirected = False)

    elif dataset_name == 'facebook':
        from torch_geometric.datasets import FacebookPagePage
        dataset = FacebookPagePage(root=data_file_root)
    elif dataset_name == 'twitch_DE':
        from torch_geometric.datasets import Twitch
        dataset = Twitch(root=data_file_root, name='DE')
    elif dataset_name == 'twitch_PT':
        from torch_geometric.datasets import Twitch
        dataset = Twitch(root=data_file_root, name='PT')
    elif dataset_name == 'twitch_EN':
        from torch_geometric.datasets import Twitch
        dataset = Twitch(root=data_file_root, name='EN')
    elif dataset_name == 'dgl_famazon':
        from dgl.data import FraudAmazonDataset
        dataset = FraudAmazonDataset(raw_dir=data_file_root, train_size = 0.8, val_size = 0.01)
    else:
        raise ValueError('Invalid dataset name')

    split = get_split_train_val_test(dataset)
    graph_dataset_summary(dataset, split)
    return dataset, split


def dim_reduce(data, y, dataset_name, edge_reduced_dim_dir_path, enforce_reduce = True):
    """
    input:
        data: (N, D)
    # using PCA to reduce the dimension of data
    output:
        data: (N, dim)
    """

    if enforce_reduce:
        reducer = dimension_reduction(
                    data, 
                    y, 
                    dataset_name, 
                    edge_reduced_dim_dir_path = edge_reduced_dim_dir_path,
                )
        data = reducer.get_data_with_reduced_dim()
    return data

class dimension_reduction():
    def __init__(self, 
        data, 
        y, 
        dataset_name,
        dim = 256, 
        EPOCH = 3,
        eps = 0.5,
        edge_reduced_dim_dir_path = None
        ):
    
        self.data = data
        self.y = y
        self.dataset_name = dataset_name
        unique_class = y.unique(return_counts=False)
        num_class = unique_class.numel()
        self.dim = dim
        self.EPOCH = EPOCH

        self.eps = eps
        self.dir_path = edge_reduced_dim_dir_path

        self.loader_len = 10
        
        file_name = f'{self.dataset_name}_{self.dim}_eps{eps}_len{self.loader_len}_epoch{self.EPOCH}.pt'
        self.file_path = f'{self.dir_path}/{file_name}'
        os.mkdir(self.dir_path) if not os.path.exists(self.dir_path) else None
        if os.path.exists(self.file_path):
            print(f'==> load data with reduced dimension from {self.file_path}')
            self.data = torch.load(self.file_path)
            
        else:
            # print(y, y[y>9], y.max(), y.min())
            self.std = aa.get_std(
                            q = 1 / self.loader_len,
                            EPOCH = EPOCH, 
                            epsilon = eps, 
                            delta = 1e-5, 
                            verbose = True,
                       )
            
            self.mlp_encoder = nn.Sequential(
                nn.Linear(data.shape[1], self.dim),
            )
            self.mlp_decoder = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.dim, num_class),
            )
            self.model = nn.Sequential(
                self.mlp_encoder,
                self.mlp_decoder,
            )
            self.dataset = torch.utils.data.TensorDataset(self.data, self.y)
            self.dataloader = torch.utils.data.DataLoader(
                                    self.dataset, 
                                    batch_size = len(self.dataset)//self.loader_len, 
                                    shuffle=True, 
                                    num_workers=4, 
                                    drop_last=True
                                )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

            self.model.cuda()

            self.worker_model_func, self.worker_param_func, self.worker_buffers_func = make_functional_with_buffers(deepcopy(self.model), disable_autograd_tracking=True)

            self.criterion = torch.nn.CrossEntropyLoss()
            self.C = 1

    def training_step(self,):
        print(f'==> start training to get data with dimension {self.data.shape[1]} to reduced dimension {self.dim}')
        s_time = time.time()

        def compute_loss(model_para, buffers, x, targets):
            # print(f'==> x shape: {x.shape}, targets shape: {targets.shape}')
            predictions = self.worker_model_func(model_para, buffers, x)
            ''' only use the first node's prediction '''
            # predictions, targets = predictions[:1], targets[:1]
            # predictions.unsqueeze_(0)
            # targets.unsqueeze_(0)

            # print(f'==> predictions shape: {predictions.shape}, targets shape: {targets.shape}, target dtype: {targets.dtype}')
            loss = self.criterion(predictions, targets.flatten()) #* inputs.shape[0]
            return loss

        for _ in tqdm(range(self.EPOCH)):
            for i, (data, targets) in enumerate(self.dataloader):
                data = data.cuda()
                targets = targets.cuda()

                targets = targets.reshape(-1, 1)[:data.shape[0]]
                data = data.unsqueeze(1)
                # print(f'==> data shape: {data.shape}, targets shape: {targets.shape}')
                per_grad = vmap( grad(compute_loss), in_dims=(None, None, 0, 0) )(self.worker_param_func, self.worker_buffers_func, data, targets)

                self.other_routine(per_grad)
        print('==> finish training to get data with reduced dimension, time cost: {:.4f} s'.format(time.time() - s_time))


    def other_routine(self, per_grad):
        per_grad = self.clip_per_grad(per_grad)

        ''' forming gradients'''
        for p_model, p_per in zip(self.model.parameters(), per_grad):
            # print(p_per.shape)
            p_model.grad = torch.mean(p_per, dim=0)
            ''' add noise to gradients'''
            p_model.grad = p_model.grad + torch.randn_like(p_model.grad) * self.std / p_per.shape[0] 

        self.model_update()
    
    def model_update(self):
        '''update parameters'''
        self.optimizer.step()  
        ''' copy parameters to worker '''
        for p_model, p_worker in zip(self.model.parameters(), self.worker_param_func):
            p_worker.copy_(p_model.data)

    def clip_per_grad(self, per_grad):
        per_grad = list(per_grad)
        per_grad_norm = self._compute_per_grad_norm(per_grad) + 1e-6 
        # print(f'==> per_grad_norm shape: {per_grad_norm.shape}')
        # print(f'==> per_grad_norm: {per_grad_norm}')

        ''' clipping/normalizing '''
        multiplier = torch.clamp(self.C / per_grad_norm, max = 1)
        for index, p in enumerate(per_grad):
            ''' normalizing '''
            # per_grad[index] = p / self._make_broadcastable(per_grad_norm / self.arg_setup.C, p) 
            ''' clipping '''
            # print(f'p shape: {p.shape}, mutiplier shape: {multiplier.shape}')
            per_grad[index] = p * self._make_broadcastable( multiplier, p ) 
        return per_grad

    def _compute_per_grad_norm(self, iterator, which_norm = 2):
        all_grad = torch.cat([p.reshape(p.shape[0], -1) for p in iterator], dim = 1)
        per_grad_norm = torch.norm(all_grad, dim = 1, p = which_norm)
        return per_grad_norm
    
    def _make_broadcastable(self, tensor_to_be_reshape, target_tensor):
        broadcasting_shape = (-1, *[1 for _ in target_tensor.shape[1:]])
        return tensor_to_be_reshape.reshape(broadcasting_shape)

    def get_data_with_reduced_dim(self,):

        if os.path.exists(self.file_path):
            return self.data
        else:
            self.training_step()
            with torch.no_grad():
                results = self.mlp_encoder(self.data.cuda()).detach().cpu()
            print(f'==> save data with reduced dimension to {self.file_path}')
            torch.save(results, self.file_path)
            return results


def constrain_edge_index_with_max_in_min_out_degree(
        edge_index, 
        min_in_degree,
        max_out_degree,
        dataset_name,
        edge_reduced_dim_dir_path,
    ):
    """
    input:
        edge_index: shape [2, num_edges]
    output:
        edge_index: shape [2, num_edges] where for each node, we have min(D_out = D, max(D_in) = D, which should be equal to D
    """
    start_time = time.time()
    
    dir_path = edge_reduced_dim_dir_path
    file_name = f'{dataset_name}_edge_index_min_in_{min_in_degree}_max_out_{max_out_degree}.pt'
    file_path = f'{dir_path}/{file_name}'
    os.mkdir(dir_path) if not os.path.exists(dir_path) else None
    if os.path.exists(file_path):
        print(f'==> load processed edge index from {file_path}')
        data = torch.load(file_path)
        return data
    edge_index_dtype = edge_index.dtype
    new_edge_index = edge_index.cuda()

    ''' get unique nodes '''
    unique_nodes = torch.unique(new_edge_index.view(-1) )

    nodes_in_coming_nodes = {}
    nodes_out_going_nodes = {}
    nodes_in_degree = {}
    nodes_out_degree = {}
    ''' rand node order '''
    unique_nodes = unique_nodes[torch.randperm(unique_nodes.numel())]
    print(f'==> constraining max out degree for each node')
    for node in tqdm(unique_nodes):
        node = int(node)

        ''' constraining out degree '''
        source_nodes_index = new_edge_index[1] == node
        out_going_nodes = torch.unique( new_edge_index[0][source_nodes_index] )
        ''' randomly sample max_in_degree incomming nodes for current node '''
        sampled_out_going_nodes = out_going_nodes[torch.randperm(out_going_nodes.numel())][:max_out_degree]
        nodes_out_going_nodes[node] = sampled_out_going_nodes
        nodes_out_degree[node] = nodes_out_going_nodes[node].numel()

    ''' reconstruct edge index from out degree dict'''
    new_edge_index = []
    print(f'==> reconstructing new edge index from out degree dict')
    for node in tqdm(unique_nodes):
        node = int(node)
        new_edge_index.append(
            torch.stack(
                [
                    nodes_out_going_nodes[node],
                    torch.tensor([node for _ in range(nodes_out_going_nodes[node].numel())], device = nodes_out_going_nodes[node].device)
                ],
                dim = 0
            )
        )
    new_edge_index = torch.cat(new_edge_index, dim = 1)

    print(f'==> constraining max in degree for each node')
    for node in tqdm(unique_nodes):
        node = int(node)

        ''' constraining in degree '''
        target_nodes_index = new_edge_index[0] == node
        in_coming_nodes = torch.unique( new_edge_index[1][target_nodes_index] )
        ''' randomly sample max_in_degree incomming nodes for current node '''
        sampled_in_coming_nodes = in_coming_nodes[torch.randperm(in_coming_nodes.numel())][:min_in_degree]
        nodes_in_coming_nodes[node] = sampled_in_coming_nodes
        nodes_in_degree[node] = nodes_in_coming_nodes[node].numel()

    ''' reconstruct edge index from in degree dict'''
    new_edge_index = []
    print(f'==> reconstructing new edge index from in degree dict')
    for node in tqdm(unique_nodes):
        node = int(node)
        new_edge_index.append(
            torch.stack(
                [
                    torch.tensor([node for _ in range(nodes_in_coming_nodes[node].numel())], device = nodes_in_coming_nodes[node].device),
                    nodes_in_coming_nodes[node]
                ],
                dim = 0
            )
        )
    new_edge_index = torch.cat(new_edge_index, dim = 1)

    print(f'==> computing degree for each node')
    for node in tqdm(unique_nodes):
        node = int(node)

        ''' for out degree'''
        target_nodes_index = new_edge_index[1] == node
        nodes_out_going_nodes[node] = torch.unique( new_edge_index[0][target_nodes_index] )
        nodes_out_degree[node] = nodes_out_going_nodes[node].numel()

        ''' for in degree'''
        source_nodes_index = new_edge_index[0] == node
        nodes_in_coming_nodes[node] = torch.unique( new_edge_index[1][source_nodes_index] )
        nodes_in_degree[node] = nodes_in_coming_nodes[node].numel()

    ''' record '''
    nodes_with_in_degree_smaller_than_D = ListDict()
    nodes_with_out_degree_smaller_than_D = ListDict()
    print(f'==> recording nodes with in degree out degree smaller than D')
    for node in tqdm(unique_nodes):
        node = int(node)
        if nodes_in_degree[node] < min_in_degree:
            nodes_with_in_degree_smaller_than_D.add_item(node)
        if nodes_out_degree[node] < max_out_degree:
            nodes_with_out_degree_smaller_than_D.add_item(node)


    ''' rand node order '''
    unique_nodes = unique_nodes[torch.randperm(unique_nodes.numel())]
    ''' then make each node with min out degree version 2 '''
    print(f'==> making each node with min in-degree')
    timer = 0
    while 1:
        timer += 1
        print(f'==> timer: {timer}')
        new_edge_can_be_created = False
        for node in unique_nodes:
            node = int(node)
            if nodes_in_degree[node] < min_in_degree:
                if len(nodes_with_out_degree_smaller_than_D) == 0:
                    new_edge_can_be_created = False
                    break
                repeat_count = 0
                while 1:
                    random_node = nodes_with_out_degree_smaller_than_D.choose_random_item()
                    if torch.any(nodes_in_coming_nodes[node] == random_node):
                        # print('duplicate')
                        repeat_count += 1
                        if repeat_count > 20:
                            random_node = None
                            break
                        else:
                            continue
                    else:
                        break
                if random_node is None:
                    continue
                new_edge_can_be_created = True
                nodes_in_coming_nodes[node] = torch.cat(
                                            [
                                                nodes_in_coming_nodes[node], 
                                                torch.tensor([random_node], device = nodes_in_coming_nodes[node].device)
                                            ], 
                                            dim = 0
                                        )

                nodes_out_degree[random_node] = nodes_out_degree[random_node] + 1
                nodes_in_degree[node] = nodes_in_degree[node] + 1

                if nodes_out_degree[random_node] >= max_out_degree:
                    # print('remove', random_node, len(nodes_with_out_degree_smaller_than_D) ) 
                    nodes_with_out_degree_smaller_than_D.remove_item(random_node)

        if not new_edge_can_be_created:
            break
                    
    ''' reconstruct new edge index '''
    new_edge_index = []
    print(f'==> reconstructing new edge index from in degree dict')
    for node in tqdm(unique_nodes):
        node = int(node)
        new_edge_index.append(
            torch.stack(
                [
                    torch.tensor([node for _ in range(nodes_in_coming_nodes[node].numel())], device = nodes_in_coming_nodes[node].device),
                    nodes_in_coming_nodes[node]
                ],
                dim = 0
            )
        )
    new_edge_index = torch.cat(new_edge_index, dim = 1)
    new_edge_index = new_edge_index.to(dtype=edge_index_dtype)

    ''' find the max D_in and min D_out '''
    max_out_degree_real = 0
    min_out_degree_real = 1000000000
    max_in_degree_real = 0
    min_in_degree_real = 1000000000

    for node in unique_nodes:
        node = int(node)
        if nodes_out_degree[node] > max_out_degree_real:
            max_out_degree_real = nodes_out_degree[node]
        if nodes_out_degree[node] < min_out_degree_real:
            min_out_degree_real = nodes_out_degree[node]
        if nodes_in_degree[node] < min_in_degree_real:
            min_in_degree_real = nodes_in_degree[node]
        if nodes_in_degree[node] > max_in_degree_real:
            max_in_degree_real =  nodes_in_degree[node]

    ''''''
    print(f'min_in_degree_real: {min_in_degree_real}, max_out_degree_real: {max_out_degree_real}, max_in_degree_real: {max_in_degree_real}, min_out_degree_real: {min_out_degree_real}')
    print(f'time used: {time.time() - start_time}')

    ''''''
    print(f'saving edge_index to {file_path}')
    torch.save(
        [ new_edge_index, min_in_degree_real, max_out_degree_real, max_in_degree_real ], 
        file_path,
    )

    return new_edge_index, min_in_degree_real, max_out_degree_real, max_in_degree_real


def store_edge_distribution(edge_index, file_path):
    # return 
    ''' for each edge, find it in-degree and out-degree '''
    edge_index = edge_index.cuda()
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    
    unique_nodes = torch.unique(edge_index.view(-1))
    dict_in_degree_for_each_node = { int(node): 0 for node in unique_nodes }
    dict_out_degree_for_each_node = { int(node): 0 for node in unique_nodes }

    for node in tqdm(unique_nodes):
        node = int(node)
        dict_in_degree_for_each_node[node] = torch.unique(target_nodes[source_nodes == node]).numel()
        dict_out_degree_for_each_node[node] = torch.unique(source_nodes[target_nodes == node]).numel()

    store = {'in_degree': list(dict_in_degree_for_each_node.values()), 'out_degree': list(dict_out_degree_for_each_node.values())}
    print(f'min in-degree: {min(store["in_degree"])}, max out-degree: {max(store["out_degree"])}, max in-degree: {max(store["in_degree"])}, min out-degree: {min(store["out_degree"])}')
    # print(min(store['in_degree']), max(store['out_degree']), max(store['in_degree']), min(store['out_degree'])) 
    torch.save(store, file_path)


def alternative_sr(new_edge_index):
    unique_nodes = torch.unique(new_edge_index.view(-1))
    ''' q '''
    nodes_in_coming_nodes = {}
    nodes_out_going_nodes = {}
    q =  0.022
    k = 1
    sampling_rate = torch.zeros(unique_nodes.max() + 1, dtype = torch.float32, device = new_edge_index.device)
    scale_sr = torch.ones(unique_nodes.max() + 1, dtype = torch.float32, device = new_edge_index.device)
    in_degree_holder = torch.zeros(unique_nodes.max() + 1, device=new_edge_index.device)
    out_degree_holder = torch.zeros(unique_nodes.max() + 1, device=new_edge_index.device)
    for node in tqdm(unique_nodes):
        node = int(node)
        nodes_in_coming_nodes[node] = torch.unique( new_edge_index[1][new_edge_index[0] == node])
        nodes_out_going_nodes[node] = torch.unique( new_edge_index[0][new_edge_index[1] == node])
        in_degree_holder[node] = nodes_in_coming_nodes[node].numel()
        out_degree_holder[node] = nodes_out_going_nodes[node].numel()

    for node in tqdm(unique_nodes):
        node = int(node)
        in_coming_nodes = nodes_in_coming_nodes[node]
        out_degrees_of_in_coming_nodes = out_degree_holder[in_coming_nodes]
        # scale_sr[node] = q / out_degrees_of_in_coming_nodes.max()
        scale_sr[node] = q / out_degrees_of_in_coming_nodes.min()

    for node in tqdm(unique_nodes):
        node = int(node)
        out_going_nodes = nodes_out_going_nodes[node]
        
        in_degrees_of_out_going_nodes = in_degree_holder[out_going_nodes]

        sampling_rate[node] = 1 - ( 1 - scale_sr[node] ) * torch.prod(1 - scale_sr[out_going_nodes] * k / in_degrees_of_out_going_nodes)

    sampling_rate = sampling_rate[sampling_rate>0]
    scale_sr = scale_sr[scale_sr>0]
    tmp = torch.rand_like(scale_sr)
    print(f'sampled nodes: {(tmp < scale_sr).sum()} out of {scale_sr.numel()} nodes')
    print(1111, sampling_rate.max(), sampling_rate.min(), sampling_rate.max() / q, k)
    exit()

def compute_degree_inverse(edge_index, dataset_name, edge_reduced_dim_dir_path):
    ''' compute degree inverse for each node '''
    mult_factor = 3

    start_time = time.time()
    dir_path = edge_reduced_dim_dir_path
    file_name = f'{dataset_name}.pt'
    file_path = f'{dir_path}/{file_name}'
    os.mkdir(dir_path) if not os.path.exists(dir_path) else None
    if os.path.exists(file_path):
        print(f'==> load processed edge index from {file_path}')
        data = torch.load(file_path)
        return data

    edge_index_dtype = edge_index.dtype
    new_edge_index = edge_index.cuda()
    unique_nodes = torch.unique(new_edge_index.view(-1) )
    
    nodes_out_degree_inverse = torch.zeros(unique_nodes.max() + 1, device = new_edge_index.device)
    print(f'==> computing degree for each node')
    for node in tqdm(unique_nodes):
        node = int(node)

        ''' for out degree'''
        target_nodes_index = new_edge_index[1] == node
        nodes_out_going_nodes = torch.unique( new_edge_index[0][target_nodes_index] )
        out_degree = nodes_out_going_nodes.numel()
        out_d_int = int(out_degree)

        if out_degree == 0:
            nodes_out_degree_inverse[node] = 0
        else:
            nodes_out_degree_inverse[node] = 1.0 / out_d_int
        # print(12, nodes_out_degree_inverse[node], nodes_out_going_nodes.numel())

    nodes_out_degree_inverse = nodes_out_degree_inverse.cpu()
    print(f'time used: {time.time() - start_time}')
    print(f'saving edge_index to {file_path}')
    torch.save(nodes_out_degree_inverse, file_path)

    return nodes_out_degree_inverse

def form_loaders(args):

    edge_reduced_dim_dir_path = Path( SETUP.get_dataset_data_path() ) / '_cache_edge_index_right_dp'
    cache_neighboring_data_for_each_graph_path =  Path( SETUP.get_dataset_data_path() ) / '_cache_neighbors_right_dp'

    dataset, split = get_raw_dataset(args.dataset)
    graph_data = dataset[0]

    '''dim reduce and normalize x'''
    graph_data.y = graph_data.y[:graph_data.x.shape[0]] # for some weird reason, the y is longer than x
    graph_data.x = (graph_data.x - graph_data.x.mean()) / (graph_data.x.std() + 1e-6)

    
    out_degree_inverse = compute_degree_inverse(
                            graph_data.edge_index, 
                            dataset_name = str(dataset), 
                            edge_reduced_dim_dir_path = edge_reduced_dim_dir_path
                        )
    
    assert args.graph_setting  in ['inductive', 'transductive']
    subgraph_sampler_train = sampling.subgraph_sampler(
                                K = args.K, 

                                num_neighbors = args.num_neighbors,
                                neighbor_num_constrain_for_training_for_memory = 500,
                                out_degree_inverse = out_degree_inverse,
                                # num_not_neighbors = args.num_not_neighbors,
                                # not_neighbor_num_constrain_for_training_for_memory = 500,

                                graph_data = graph_data,
                                graph_data_name = str(dataset),
                                mask = split.train_mask,

                                setting = args.graph_setting,
                                dataset_mode = 'train',
                                
                                args = args,
                                cache_file_path = cache_neighboring_data_for_each_graph_path
                                # device=device,
                            )  

  
    subgraph_sampler_test = sampling.subgraph_sampler(
                                K = args.K, 

                                num_neighbors = args.num_neighbors_test, # args.num_neighbors,
                                # num_not_neighbors = args.num_not_neighbors,

                                graph_data = graph_data,
                                graph_data_name = str(dataset),

                                mask = split.test_mask,
                                setting = args.graph_setting,
                                dataset_mode = 'test',

                                args = args,
                                cache_file_path = cache_neighboring_data_for_each_graph_path
                                # device=device
                            )   

    
    train_loader = sampling.get_subgraphs_loader(subgraph_sampler_train, expected_batchsize = args.expected_batchsize, worker_num = args.worker_num)
    # val_loader =   sampling.get_subgraphs_loader(subgraph_sampler_val,   expected_batchsize = 1000, worker_num = args.worker_num, drop_last=False)
    test_loader =  sampling.get_subgraphs_loader(subgraph_sampler_test,  expected_batchsize = 1000, worker_num = args.worker_num, drop_last = False, dataset_mode='test')

    val_loader = None
    return train_loader, val_loader, test_loader, dataset, graph_data.x



import random
class ListDict(object):
    def __init__(self):
        self.item_to_position = {}
        self.items = []

    def add_item(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove_item(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        return random.choice(self.items)

    def __str__(self) -> str:
        return str(self.items)

    def __len__(self):
        return len(self.items)