import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json

from tqdm import tqdm
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.subgraph import k_hop_subgraph

from model import GATConvPlusModel
from dataset import GraphDataset, data_process
from utils import *

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--data_name', default='dgraphfin.npz')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-7)

    # model parameters
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=96)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--heads', type=int, default=1)

    return parser.parse_args()

def train(model, data, optimizer):
    model.train()

    optimizer.zero_grad()
    neg_idx = data.train_neg[
        torch.randperm(data.train_neg.size(0))[: data.train_pos.size(0)]
    ]
    train_idx = torch.cat([data.train_pos, neg_idx], dim=0)

    nodeandneighbor, edge_index, node_map, mask = k_hop_subgraph(
        train_idx, 3, data.edge_index, relabel_nodes=True, num_nodes=data.x.size(0)
    )

    out = model(
        data.x[nodeandneighbor],
        edge_index,
        data.edge_attr[mask],
        data.edge_timestamp[mask],
        data.edge_direct[mask],
    )
    loss = F.nll_loss(out[node_map], data.y[train_idx])
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    optimizer.step()
    torch.cuda.empty_cache()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(
        data.x, data.edge_index, data.edge_attr, data.edge_timestamp, data.edge_direct,
    )

    y_pred = out.exp()
    return y_pred

def eval_metric(y_true, y_pred):
    if torch is not None and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if torch is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    ## check type
    if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
        raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

    if not y_pred.ndim == 2:
        raise RuntimeError('y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

    eval_result = {
        'accuracy': eval_acc(y_true, y_pred),
        'precision': eval_precision(y_true, y_pred),
        'recall': eval_recall(y_true, y_pred),
        'f1': eval_f1(y_true, y_pred),
        'auc': eval_rocauc(y_true, y_pred),
    }
    return eval_result

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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = 0.0

    checkpoint_dict = {}
    y_train, y_valid = data.y[data.train_mask], data.y[data.valid_mask]
    for epoch in range(1, args.epochs+1):
        loss = train(model, data, optimizer)
        out = test(model, data)
        preds_train, preds_valid = out[data.train_mask], out[data.valid_mask]
        train_result = eval_metric(y_train, preds_train)
        valid_result = eval_metric(y_valid, preds_valid)
        eval_results = {}
        eval_results['train'] = train_result
        eval_results['valid'] = valid_result

        if valid_result['auc'] >= best_auc:
            best_auc = valid_result['auc']
            torch.save(model.state_dict(), "model.pt")
            preds = out[data.test_mask].cpu().numpy()
        
        print(f'Epoch: {epoch:02d}')
        print(eval_results)

        checkpoint_dict[epoch] = eval_results

    
    test_result = eval_metric(data.y[data.test_mask], preds)
    print('Test Result:')
    print(test_result)

    with open('checkpoint.json', 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)










