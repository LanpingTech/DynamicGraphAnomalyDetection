import argparse
from ast import arg
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json

from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from model import GATConvPlusModel
from dataset import GraphDataset, data_process

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--data_name', default='dgraphfin.npz')

    # model parameters
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=96)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--heads', type=int, default=1)

    parser.add_argument('--ID', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = GraphDataset(root="./dataset", name="DGraphFin")
    args.num_classes = 2

    data = dataset[0]

    split_idx = {
        "train": data.train_mask,
        "valid": data.valid_mask,
        "test": data.test_mask,
    }

    data = data_process(data).to(device)
    train_idx = split_idx["train"].to(device)

    data.train_pos = train_idx[data.y[train_idx] == 1]
    data.train_neg = train_idx[data.y[train_idx] == 0]

    model_params = {
        'in_channels': data.x.size(-1),
        'hidden_channels': args.hidden_channels,
        'out_channels': args.num_classes,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'heads': args.heads,
        'bn': args.batch_norm
    }
    model = GATConvPlusModel(**model_params).to(device)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.eval()
    with torch.no_grad():
        out = model(
            data.x, data.edge_index, data.edge_attr, data.edge_timestamp, data.edge_direct,
        )
        y_pred = out.exp()  # (N,num_classes)

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        y_pred = y_pred.argmax(axis=-1)

        print('predict result:', y_pred[args.ID])

