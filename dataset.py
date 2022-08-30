import os.path as osp

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import NeighborSampler

def load_data(file_path, data_anme):
    np_data = np.load(file_path + '/' + data_anme)

    x = np_data['x']
    y = np_data['y'].reshape(-1, 1)
    edge_index = np_data['edge_index']
    edge_type = np_data['edge_type']
    train_mask = np_data['train_mask']
    valid_mask = np_data['valid_mask']
    test_mask = np_data['test_mask']

    x = torch.tensor(x, dtype=torch.float).contiguous()
    y = torch.tensor(y, dtype=torch.int64)
    edge_index = torch.tensor(edge_index.transpose(), dtype=torch.int64).contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.float)
    train_mask = torch.tensor(train_mask, dtype=torch.int64)
    valid_mask = torch.tensor(valid_mask, dtype=torch.int64)
    test_mask = torch.tensor(test_mask, dtype=torch.int64)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, y=y)
    data.train_idx = train_mask
    data.valid_idx = valid_mask
    data.test_idx = test_mask

    return data

class GraphDataset(InMemoryDataset):


    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        names = ['dgraphfin.npz']
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data = load_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'

def preprocessing(data, device, batch_size=1024):
    data = data[0]
    data.adj_t = data.adj_t.to_symmetric()
    x = data.x
    x = (x-x.mean(0))/x.std(0)
    data.x = x

    data = data.to(device)
    data.train_idx = data.train_idx.to(device)
    data.valid_idx = data.valid_idx.to(device)
    data.test_idx = data.test_idx.to(device)

    train_loader = NeighborSampler(data.adj_t, node_idx=data.train_idx, sizes=[10, 5], batch_size=batch_size, shuffle=True, num_workers=8)
    layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=batch_size, shuffle=False, num_workers=8)

    return data, train_loader, layer_loader

