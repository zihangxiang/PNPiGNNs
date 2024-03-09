import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import os
import torch_geometric
from tqdm import tqdm

class subgraph_sampler(Dataset):
    def __init__(self, 
        K = None, 

        num_neighbors = None, 
        neighbor_num_constrain_for_training_for_memory = None,
        out_degree_inverse = None,
        # num_not_neighbors = None, 
        # not_neighbor_num_constrain_for_training_for_memory = 500,

        graph_data = None, 
        graph_data_name = None,

        mask = None,
        setting = None,
        dataset_mode = None,
        device = 'cpu',

        args = None,
        cache_file_path = None,
        ):
        super().__init__()
        s = time.time()

        print(f'\n\n{"="*40}\ndataset init...')
        self.K = K 
        self.num_neighbors = num_neighbors
        self.neighbor_num_constrain_for_training_for_memory = neighbor_num_constrain_for_training_for_memory
        self.out_degree_inverse = out_degree_inverse

        ''' input graph_data is a torch_geometric.data.Data object, it is the whole graph'''
        self.graph_data = graph_data
        self.graph_data_name = graph_data_name

        self.args = args
        self.cache_file_path = cache_file_path

        self.device = device
        self.graph_edge_index = self.graph_data.edge_index.to(device)

        ''' for what type of node, it is determined by the mask '''
        self.mask = mask.to(device)
       
        self.setting = setting
        self.dataset_mode = dataset_mode

        assert self.setting in ['transductive', 'inductive']
        assert self.dataset_mode in ['train', 'val', 'test']

        # print(f'\n==> total mask size = {self.mask.numel()}, type of mask = {self.mask.dtype}')
        ''' storing the neighbors of each node in the graph '''
        ''' check if the cache file exists, if so, load it, or else, create it '''
        file_name = f'{self.graph_data_name}_{self.setting}_{self.dataset_mode}_{self.args.seed}_{self.neighbor_num_constrain_for_training_for_memory}.pt'
        file_path = self.cache_file_path / file_name
        os.mkdir(self.cache_file_path) if not os.path.exists(self.cache_file_path) else None

        if os.path.exists(file_path):
            print(f'==> loading the neighbors of each node in the graph...')
            data = torch.load(file_path)

            self.dict_of_nodes_neighbors = data[0]
            self.mask = data[1]

            self.init_ids()
            assert len(self.dict_of_nodes_neighbors) == len(self.ids_for_legit_rest_nodes_in_graph)
            # assert len(self.dict_of_nodes_not_neighbors) == len(self.ids_for_legit_rest_nodes_in_graph)
            
        else:
            self.init_ids()

            print(f'==> concluding the neighbors of each node in the graph, it may take a while...')
            self.dict_of_nodes_neighbors = {}
            # self.dict_of_nodes_not_neighbors = {}

            source_nodes = self.graph_edge_index[0, :].cuda()
            neighbor_nodes = self.graph_edge_index[1, :].cuda()
            mask = self.mask.cuda()

            self.ids_for_legit_rest_nodes_in_graph_device = self.ids_for_legit_rest_nodes_in_graph.cuda().reshape(-1)

            ''' shuffle the source_nodes and neighbor_nodes'''
            for node in tqdm(self.ids_for_legit_rest_nodes_in_graph):
                node = int(node)
                ''' get the neighbors of node '''
                neighbors = neighbor_nodes[ source_nodes == node ]

                ''' filter the neighbors of node '''
                neighbors = self.filter_to_be_legit_nodes(neighbors, mask)

                if self.dataset_mode == 'train':
                    neighbors = neighbors[ :self.neighbor_num_constrain_for_training_for_memory ]
                ''' store the neighbors of node '''
                self.dict_of_nodes_neighbors[node] = neighbors[torch.randperm(neighbors.numel())].cpu() 
                ''' store the not neighbors of node '''
                # self.dict_of_nodes_not_neighbors[node] = not_neighbors[torch.randperm(not_neighbors.numel())].cpu()

            del source_nodes, neighbor_nodes, mask
            # data = [self.dict_of_nodes_neighbors, self.dict_of_nodes_not_neighbors, self.mask]

            data = [self.dict_of_nodes_neighbors, self.mask]
            torch.save(data, file_path)
            print(f'==> file saved to: {file_path}')

        # print(f'\n==> setting = {self.setting}, dataset_mode = {self.dataset_mode}, K = {self.K}, num_neighbors = {self.num_neighbors}, num_not_neighbors = {self.num_not_neighbors}, neighbor_num_constrain_for_training = {self.neighbor_num_constrain_for_training}, not_neighbor_num_constrain_for_training_for_memory = {self.not_neighbor_num_constrain_for_training_for_memory}')
        print(f'-> setting: {self.setting}, mode: {self.dataset_mode}')
        print(f'-> graph dataset contains {self.mask.numel()}')
        # print(f'-> one sample is on {self.K * (self.num_neighbors + self.num_not_neighbors) + 1} nodes')
        print(f'-> number of usable nodes: {self.ids_for_all_legit_starting_nodes.numel()}')
        print(f'-> number of rest legit nodes: {self.rest_legit_node_num}')
        print(f'==> done, time elapsed = {time.time() - s:.4f} seconds')
        print(f'{"="*40}')

    def init_ids(self):
        self.ids_for_all_legit_starting_nodes = self.mask.nonzero(as_tuple=True)[0]
        self.num_of_nodes =int(self.ids_for_all_legit_starting_nodes.numel())

        if self.setting == 'transductive' and self.dataset_mode == 'train':
            self.ids_for_legit_rest_nodes_in_graph = torch.arange(self.mask.numel(), device = self.device)
        elif self.setting == 'inductive' and self.dataset_mode == 'train':
            self.ids_for_legit_rest_nodes_in_graph = self.mask.nonzero(as_tuple=True)[0].to(self.device)
        elif self.dataset_mode in ['val', 'test']:
            self.ids_for_legit_rest_nodes_in_graph = self.mask.nonzero(as_tuple=True)[0].to(self.device)
        else:
            raise NotImplementedError
        
        self.rest_legit_node_num = self.ids_for_legit_rest_nodes_in_graph.numel()

    def filter_to_be_legit_nodes(self, nodes, mask):
        if self.setting == 'transductive' and self.dataset_mode == 'train':
            pass
        elif self.setting == 'inductive' and self.dataset_mode == 'train':
            nodes = nodes[mask[nodes] == True]
        elif self.dataset_mode in ['val', 'test']:
            nodes = nodes[mask[nodes] == True]
        else:
            raise NotImplementedError
        return nodes

    def __len__(self):
        return self.num_of_nodes

    def __getitem__(self, mapped_root_node):
        '''
        # root_node: int, the root/starting node of the subgraph
        # return: x, edge_index, y
        #         x: torch.tensor, shape = (K * 2 + 1, num_node_features)
        #         edge_index: torch.tensor, shape = (2, K * 2 ), node id starts from 0 to K * 2 + 1
        #         y: torch.tensor, shape = (K * 2 + 1, )
        '''
        # start_time = time.time()
        root_node = int(self.ids_for_all_legit_starting_nodes[mapped_root_node])

        sub_graph_nodes = [root_node]
        sub_graph_edge_index = []

        for i in range(1, self.K+1):
            ''' action for depth '''
            '''1. sample neighbors'''
            all_neighbors = self.dict_of_nodes_neighbors[root_node]
            if all_neighbors.numel() == 0:
                # print(f'==> root_node = {root_node}, dataset_mode = {self.dataset_mode}, no neighbors')
                ''' 1 random spot '''
                sampled_neighbors = self.ids_for_legit_rest_nodes_in_graph[ np.random.choice(self.rest_legit_node_num, self.num_neighbors, replace=True).tolist() ]
                # sampled_neighbors = self.ids_for_legit_rest_nodes_in_graph[ np.random.choice(self.rest_legit_node_num, 1, replace=True).tolist() ]
            else:
                ''' 2 random spot '''
                if self.dataset_mode == 'test':
                    ''''''
                    num_neighbors = min(all_neighbors.numel(), self.num_neighbors)
                    sampled_neighbors = all_neighbors[ np.random.choice(all_neighbors.numel(), num_neighbors, replace=False).tolist() ]

                else:
                    # print(222, self.out_degree_inverse, self.out_degree_inverse[self.out_degree_inverse>0])
                    p = self.out_degree_inverse[all_neighbors] * self.num_neighbors
                    tester = torch.rand(all_neighbors.numel())
                    sampled_neighbors = all_neighbors[ tester<=p ]

            tmp_neighbors_of_root = sampled_neighbors.tolist()
            sub_graph_nodes += tmp_neighbors_of_root

            center_node = [root_node] * ( len(sampled_neighbors) )
            sub_graph_edge_index.append(
                torch.tensor(
                    [
                        center_node,           #+ tmp_neighbors_of_root,
                        tmp_neighbors_of_root #+ center_node,
                    ], 
                    dtype = self.graph_edge_index.dtype,
                )

            )


        sub_graph_nodes = torch.tensor(sub_graph_nodes, dtype=self.graph_edge_index.dtype)

        return [ 
                self.graph_data.x[sub_graph_nodes], 
                0, 
                # self.graph_data.y[sub_graph_nodes].reshape(-1), 
                self.graph_data.y[sub_graph_nodes][:1], 
                sub_graph_nodes, 
                0, 
                0, 
                self.dataset_mode,
                self.mask[sub_graph_nodes],
                ] 

def collate_subgraphs(batch):
    # overlapping = 0

    sub_graph_nodes_set = set()

    ''' processing for train mode '''
    if batch[0][6] == 'train':
        index_to_keep = []
        for i in range(len(batch)):
            current_nodes = batch[i][3].tolist()
            index_to_keep.append([0])
            for j in range(1, len(current_nodes)):
                if current_nodes[j] in sub_graph_nodes_set and batch[i][7][j] == True:
                    # overlapping += 1
                    pass
                    # batch[i][0][j] = torch.zeros_like(batch[i][0][j])
                else:
                    index_to_keep[i].append(j)
            sub_graph_nodes_set.add(current_nodes[0])

        for i in range(len(batch)):
            batch[i][0] = batch[i][0][index_to_keep[i]]

    ''' find the max x.shape[0] '''
    max_x_shape = 0
    for i in range(len(batch)):
        max_x_shape = max(max_x_shape, batch[i][0].shape[0])

    ''''''
    x = []
    # adj = []
    y = []

    for i in range(len(batch)):
        x_tmp = torch.zeros(max_x_shape, batch[i][0].shape[1])
        x_tmp[:batch[i][0].shape[0], :] = batch[i][0]
        x.append(x_tmp)

        y.append(batch[i][2])
    
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    
    # print(f'==> # total: {len(sub_graph_nodes_set)}, # overlapping: {overlapping}, # central: {len(batch)}, {overlapping / len(batch) * 100:.2f}%, {batch[0][6]}')
    return x, y

def get_subgraphs_loader(train_dataset, expected_batchsize, worker_num = 4, drop_last = True, dataset_mode = 'train'):
    assert expected_batchsize <= len(train_dataset), f'expected_batchsize = {expected_batchsize} > {len(train_dataset)}'
    '''poisson sampling'''
    if dataset_mode == 'train':
        print(f'==> initializing {dataset_mode} dataloader')
        return DataLoader(
            dataset = train_dataset,
            batch_sampler = PoissonSampler(len(train_dataset), expected_batchsize),
            num_workers = worker_num,
            pin_memory = True,
            # drop_last = drop_last,
            collate_fn = collate_subgraphs,
            persistent_workers = True,
        )
    else:

        ''' normal loader '''
        print(f'==> initializing {dataset_mode} dataloader')
        return DataLoader(
                    dataset = train_dataset,
                    batch_size = expected_batchsize,
                    shuffle = True,
                    num_workers = worker_num,
                    pin_memory = True,
                    drop_last = drop_last,
                    collate_fn = collate_subgraphs,
                    persistent_workers = True,
                )

class PoissonSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, batch_size):
        self.inds = np.arange(num_examples)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_examples / batch_size))
        self.sample_rate = self.batch_size / (1.0 * num_examples)
        super().__init__(None)

    def __iter__(self):
        # select each data point independently with probability `sample_rate`
        for i in range(self.num_batches):
            batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            batch = self.inds[batch_idxs.astype(np.bool)]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


class sr_calculator:
    def __init__(self, 
        *,
        batch_size,
        N,
        K,
        min_in_degree,
        max_out_degree,
        num_neighbors,
        num_not_neighbors = None,
    ):
        self.batch_size = batch_size
        self.N = N
        self.K = K
        self.min_in_degree = min_in_degree
        self.max_out_degree = max_out_degree
        self.num_neighbors = num_neighbors
        self.num_not_neighbors = num_not_neighbors
        self.q = self.batch_size / self.N
        
    def calculate(self):

        sr_one_node = []
        for i in range(self.N):
            sr_one_node.append(self.sr_of_node_i(i))

        final_prob = self.union_prob(sr_one_node)
        assert final_prob <= 1,  f"final_prob {final_prob} is larger than 1"
        return final_prob, final_prob / self.q

    def sr_of_node_i(self, i):
        if i == 0:
            return self.q
        elif i <= self.max_out_degree:
            return self.q * self.total_prob(
                                self.num_neighbors / self.min_in_degree,
                                1,
                                0
                            )
        else: 
            return 0
            
    @staticmethod
    def total_prob(prob_1, prob_2_when_1_happen, prob_3_when_1_not_happen):
        return prob_1 * prob_2_when_1_happen + (1 - prob_1) * prob_3_when_1_not_happen
    @staticmethod
    def union_prob(probs):
        results = 1
        for prob in probs:
            results = results * (1 - prob)
        return 1 - results



if __name__ == "__main__":
    # print( rate(nodes_at_which_hop = 4) )
    pass
    # print( rate_of_sampled_the_node(q = batch_size / N, K = K) )

    cal = sr_calculator(
            batch_size = 128,
            N = 13752,
            K = 1,
            min_in_degree = 18,
            max_out_degree = 20,
            num_neighbors = 10,
            # num_not_neighbors = 10,
        )

    print(cal.calculate())