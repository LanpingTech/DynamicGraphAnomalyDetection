import argparse
from ast import arg
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json

from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from model import GATModel
from dataset import GraphDataset, preprocessing

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--data_name', default='dgraphfin.npz')

    # model parameters
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_heads', type=int, nargs='+', default=[4, 1])

    parser.add_argument('--ID', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data = GraphDataset('data', args.data_name, transform=T.ToSparseTensor())
    args.num_classes = data.num_classes
    data, train_loader, layer_loader = preprocessing(data, device)

    model_params = {
        'in_channels': data.num_features,
        'hidden_channels': args.hidden_channels,
        'out_channels': args.num_classes,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'layer_heads': args.layer_heads,
        'batchnorm': args.batch_norm
    }
    model = GATModel(**model_params).to(device)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.eval()
    with torch.no_grad():
        out = model.inference(data.x, layer_loader, device)
        y_pred = out.exp()  # (N,num_classes)

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        y_pred = y_pred.argmax(axis=-1)

        print('predict result:', y_pred[args.ID])

