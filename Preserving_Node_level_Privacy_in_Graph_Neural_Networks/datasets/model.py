"""GNN that classifies the root node of one (zero-padded) sampled subgraph.

All modules take ``x`` of shape ``(num_nodes, feat_dim)`` for a *single*
subgraph (row 0 is the root, all-zero rows are padding); batching is done
with ``functorch.vmap`` in the trainer.
"""
import torch
import torch.nn.functional as F
from torch import nn

NL = F.elu
criterion = torch.nn.CrossEntropyLoss()


def _standardize(x):
    return (x - torch.mean(x)) / (torch.std(x) + 1e-6)


def _add_residual(out, x, post_process):
    """``NL(out + x')`` where ``x'`` is ``x`` zero-padded to ``out``'s width if
    ``out`` is wider, and ``post_process(x)`` otherwise."""
    if out.shape[1] > x.shape[1]:
        x = torch.cat([x, torch.zeros(x.shape[0], out.shape[1] - x.shape[1]).to(x.device)], dim=1)
    else:
        x = post_process(x)
    return NL(out + x)


class G_net(torch.nn.Module):
    """``K`` aggregation layers, global standardisation, then a linear classifier.

    Returns logits for every row of ``x``; the trainer uses row 0 (the root).
    """

    def __init__(self, K, feat_dim, num_classes, hidden_channels, conv=None):
        super().__init__()
        assert isinstance(K, int)
        assert K >= 1

        conv = conv or GCN
        self.conv_list = nn.ModuleList([conv(feat_dim, hidden_channels)])
        for _ in range(K - 1):
            self.conv_list.append(conv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, out):
        for conv in self.conv_list:
            out = conv(out)
        return self.classifier(_standardize(out))


class GCN(torch.nn.Module):
    """Mean over the non-padding rows, broadcast back to every node as a residual update."""

    def __init__(self, input_feat_dim, output_feat_dim):
        super().__init__()
        self.post_process = torch.nn.Linear(input_feat_dim, output_feat_dim)

    def forward(self, x):
        norm = torch.norm(x, dim=1)
        out = torch.sum(x, dim=0, keepdim=True) / (norm > 0).sum()
        out = NL(_standardize(NL(out)))
        out = self.post_process(out)
        return _add_residual(out, x, self.post_process)


class GIN(torch.nn.Module):
    """GIN-style variant (not used by default; pass ``conv=GIN`` to :class:`G_net`)."""

    def __init__(self, input_feat_dim, output_feat_dim):
        super().__init__()
        self.GIN_eps = torch.nn.parameter.Parameter(torch.Tensor([1]))
        self.post_process = torch.nn.Linear(input_feat_dim, output_feat_dim)

    def forward(self, x):
        out = torch.sum(x, dim=0, keepdim=True)
        out = NL(_standardize(NL(out)))
        out = out + x * (1 + self.GIN_eps)
        out = self.post_process(out)
        return _add_residual(out, x, self.post_process)


class SAGE(torch.nn.Module):
    """GraphSAGE-style variant (not used by default; pass ``conv=SAGE`` to :class:`G_net`)."""

    def __init__(self, input_feat_dim, output_feat_dim):
        super().__init__()
        self.sage_conv = torch.nn.Linear(input_feat_dim, output_feat_dim)
        self.post_process = torch.nn.Linear(input_feat_dim, output_feat_dim)

    def forward(self, x):
        self_conv = self.sage_conv(x)
        norm = torch.norm(x, dim=1)
        out = torch.sum(x, dim=0, keepdim=True) / (norm > 0).sum()
        out = NL(_standardize(NL(out)))
        out = self.post_process(out) + self_conv
        return _add_residual(out, x, self.post_process)
