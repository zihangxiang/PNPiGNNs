import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class PoissonSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, batch_size):
        super().__init__()
        self.inds = np.arange(num_examples)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_examples / batch_size))
        self.sample_rate = self.batch_size / (1.0 * num_examples)
        
    def __iter__(self):
        # select each data point independently with probability `sample_rate`
        for i in range(self.num_batches):
            batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            batch = self.inds[batch_idxs.astype(np.bool)]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def privatized_loader(train_dataset, expected_batchsize):
    '''poisson sampling'''
    return DataLoader(
        dataset = train_dataset,
        batch_sampler = PoissonSampler(len(train_dataset), expected_batchsize),
        num_workers = 4,
        pin_memory = True,
    )

    ''' normal loader '''
    return DataLoader(
                    dataset = train_dataset,
                    batch_size = expected_batchsize,
                    shuffle = True,
                    num_workers = 4,
                    pin_memory = True,
                    drop_last = False,
            )