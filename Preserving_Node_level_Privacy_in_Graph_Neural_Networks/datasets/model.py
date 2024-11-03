import torch
# from torch_geometric.nn import GCNConv, SAGEConv, GINConv
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import scatter
from torch import nn

NL = F.elu
criterion = torch.nn.CrossEntropyLoss()

''' '''

class G_net(torch.nn.Module):
    def __init__(self, K, feat_dim, num_classes, hidden_channels):
        super().__init__()
        assert isinstance(K, int)
        assert K >=1

        conv = GCN

        self.conv_list =  nn.ModuleList([conv(feat_dim, hidden_channels)])
        for i in range(K-1):
            self.conv_list.append(conv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, out):
        # out = x
        for i in range(len(self.conv_list)):
            out = self.conv_list[i](out)

        out = (out - torch.mean(out)) / (torch.std(out) + 1e-6)

        return self.classifier(out)

class GCN(torch.nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim):
        super().__init__()    
        self.post_process = torch.nn.Linear(input_feat_dim, output_feat_dim)

    def forward(self, x):
        # out = torch.matmul(adj, x) / adj.sum(dim=1, keepdim=True)
        norm = torch.norm(x, dim=1)
        out = torch.sum(x, dim=0, keepdim=True) / (norm > 0).sum()
        out = NL(out)

        out = (out - torch.mean(out)) / (torch.std(out) + 1e-6)
        out = NL(out)
        out = self.post_process(out)

        ''' check to padding '''
        if out.shape[1] > x.shape[1]:
            '''padding x and then add'''
            x = torch.cat([x, torch.zeros(x.shape[0], out.shape[1] - x.shape[1]).to(x.device)], dim=1)
        else:
            x = self.post_process(x)

        out = out + x
        out = NL(out)

        return out
    

class GIN(torch.nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim):
        super().__init__()    
        self.GIN_eps = torch.nn.parameter.Parameter(torch.Tensor([1]))
        self.post_process = torch.nn.Linear(input_feat_dim, output_feat_dim)

    def forward(self, x):
        out = torch.sum(x, dim=0, keepdim=True)
        out = NL(out)

        out = (out - torch.mean(out)) / (torch.std(out) + 1e-6)
        out = NL(out)

        out = out + x * (1 + self.GIN_eps)

        out = self.post_process(out)
        
        ''' check to padding '''
        if out.shape[1] > x.shape[1]:
            '''padding x and then add'''
            x = torch.cat([x, torch.zeros(x.shape[0], out.shape[1] - x.shape[1]).to(x.device)], dim=1)
        else:
            x = self.post_process(x)

        out = out + x
        out = NL(out)
        return out
    

class SAGE(torch.nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim):
        super().__init__()    
        self.sage_conv = torch.nn.Linear(input_feat_dim, output_feat_dim)
        self.post_process = torch.nn.Linear(input_feat_dim, output_feat_dim)

    def forward(self, x):
        self_conv = self.sage_conv(x)
        norm = torch.norm(x, dim=1)
        out = torch.sum(x, dim=0, keepdim=True) / (norm > 0).sum()

        # out = self.post_process(out) + self_conv
        out = NL(out)

        out = (out - torch.mean(out)) / (torch.std(out) + 1e-6)
        out = NL(out)
        out = self.post_process(out) + self_conv

        ''' check to padding '''
        if out.shape[1] > x.shape[1]:
            '''padding x and then add'''
            x = torch.cat([x, torch.zeros(x.shape[0], out.shape[1] - x.shape[1]).to(x.device)], dim=1)
        else:
            x = self.post_process(x)

        out = out + x
        out = NL(out)
        return out


