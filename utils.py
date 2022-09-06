import os
from datetime import datetime
import shutil

import torch
import torch.nn.functional as F
import torch_geometric
import torch_scatter as scatter
from torch import Tensor
import numpy as np

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score


def prepare_folder(name, model_name):
    model_dir = f'./model_files/{name}/{model_name}/'
   
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir

def prepare_tune_folder(name, model_name):
    str_time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    tune_model_dir = f'./tune_results/{name}/{model_name}/{str_time}/'
   
    if os.path.exists(tune_model_dir):
        print(f'rm tune_model_dir {tune_model_dir}')
        shutil.rmtree(tune_model_dir)
    os.makedirs(tune_model_dir)
    print(f'make tune_model_dir {tune_model_dir}')
    return tune_model_dir

def save_preds_and_params(parameters, preds, model, file):
    save_dict = {'parameters':parameters, 'preds': preds, 'params': model.state_dict()
           , 'nparams': sum(p.numel() for p in model.parameters())}
    torch.save(save_dict, file)
    return 
    
def add_degree_feature(x: Tensor, edge_index: Tensor):
    row, col = edge_index
    in_degree = torch_geometric.utils.degree(col, x.size(0), x.dtype)

    out_degree = torch_geometric.utils.degree(row, x.size(0), x.dtype)
    return torch.cat([x, in_degree.view(-1, 1), out_degree.view(-1, 1)], dim=1)


def add_feature_flag(x):
    feature_flag = torch.zeros_like(x[:, :17])
    feature_flag[x[:, :17] == -1] = 1
    x[x == -1] = 0
    return torch.cat((x, feature_flag), dim=1)


def add_label_feature(x, y, valid_mask=None):
    y = y.clone()
    if valid_mask is not None:
        y[valid_mask] = 4
    y[y == 1] = 0
    y[y == 4] = 0
    y_one_hot = F.one_hot(y).squeeze()
    return torch.cat((x, y_one_hot[:, :-1]), dim=1)


def add_label_counts(x, edge_index, y):
    y = y.clone().squeeze()
    background_nodes = torch.logical_or(y == 2, y == 3)
    foreground_nodes = torch.logical_and(y != 2, y != 3)
    y[background_nodes] = 1
    y[foreground_nodes] = 0

    row, col = edge_index
    a = F.one_hot(y[col])
    b = F.one_hot(y[row])
    temp = scatter.scatter(a, row, dim=0, dim_size=y.size(0), reduce="sum")
    temp += scatter.scatter(b, col, dim=0, dim_size=y.size(0), reduce="sum")

    return torch.cat([x, temp.to(x)], dim=1)


def cos_sim_sum(x, edge_index):
    row, col = edge_index
    sim = F.cosine_similarity(x[row], x[col])
    sim_sum = scatter.scatter(sim, row, dim=0, dim_size=x.size(0), reduce="sum")
    return torch.cat([x, torch.unsqueeze(sim_sum, dim=1)], dim=1)


def to_undirected(edge_index, edge_attr, edge_timestamp):

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_timestamp = torch.cat([edge_timestamp, edge_timestamp], dim=0)
    return edge_index, edge_attr, edge_timestamp


def eval_rocauc(y_true, y_pred):

    if y_pred.shape[1] ==2:
        auc = roc_auc_score(y_true, y_pred[:, 1])
    else:
        onehot_code = np.eye(y_pred.shape[1])
        y_true_onehot = onehot_code[y_true]
        auc = roc_auc_score(y_true_onehot, y_pred)

    return auc

def eval_acc(y_true, y_pred):
    y_pred = y_pred.argmax(axis=-1)
    correct = y_true == y_pred
    acc = float(np.sum(correct))/len(correct)

    return acc

def eval_precision(y_true, y_pred):
    y_pred = y_pred.argmax(axis=-1)
    precision = precision_score(y_true, y_pred)
    return precision

def eval_recall(y_true, y_pred):
    y_pred = y_pred.argmax(axis=-1)
    recall = recall_score(y_true, y_pred)
    return recall

def eval_f1(y_true, y_pred):
    y_pred = y_pred.argmax(axis=-1)
    f1 = f1_score(y_true, y_pred)
    return f1