from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter as scatter
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import get_laplacian, remove_self_loops
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GATConv

def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")

class TimeEncoder(torch.nn.Module):
    def __init__(self, dimension):
        super(TimeEncoder, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 1.5, dimension)))
            .float()
            .reshape(dimension, -1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension))

    def reset_parameters(self):
        pass

    def forward(self, t):
        t = torch.log(t + 1)
        t = t.unsqueeze(dim=1)
        output = torch.cos(self.w(t))
        return output

class GATConvPlus(torch.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin_m = Linear(in_channels[0], out_channels, bias=bias)

        self.gat = GATConv(in_channels[1], out_channels, heads=heads, concat=False)


    def reset_parameters(self):
        self.lin_m.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_t: Tensor,
    ) -> Tensor:

        row, col = edge_index

        x_j = torch.cat([x[col], edge_attr, edge_t], dim=1)
        x_j = scatter.scatter(x_j, row, dim=0, dim_size=x.size(0), reduce="sum")
        x_j = self.lin_m(x_j)
        x_i = self.gat(x, edge_index)
        out = 0.5 * x_j + x_i

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

class GATConvPlusModel(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        edge_attr_channels=50,
        time_channels=50,
        num_layers=2,
        dropout=0.0,
        bn=True,
        heads=1,
        activation="elu",
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(
                GATConvPlus(
                    (
                        first_channels + edge_attr_channels + time_channels,
                        first_channels,
                    ),
                    second_channels,
                    heads=heads,
                )
            )
            self.bns.append(bn(second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)
        self.emb_type = nn.Embedding(12, edge_attr_channels)
        self.emb_direction = nn.Embedding(2, edge_attr_channels)
        self.t_enc = TimeEncoder(time_channels)
        self.reset_parameters()

    def reset_parameters(self):

        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

        nn.init.xavier_uniform_(self.emb_type.weight)

        nn.init.xavier_uniform_(self.emb_direction.weight)

    def forward(self, x, edge_index, edge_attr, edge_t, edge_d):
        edge_attr = self.emb_type(edge_attr) + self.emb_direction(edge_d)
        edge_t = self.t_enc(edge_t)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr, edge_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        return x.log_softmax(dim=-1)