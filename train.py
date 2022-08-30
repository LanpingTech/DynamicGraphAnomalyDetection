import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from model import GATModel
from dataset import GraphDataset, preprocessing

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--data_name', default='dgraphfin.npz')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight_decay', type=float, default=5e-7)

    # model parameters
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_heads', type=int, nargs='+', default=[4, 1])

    return parser.parse_args()

def train(epoch, train_loader, model, data, train_idx, optimizer, device):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(data.x[n_id], adjs)
        loss = F.nll_loss(out, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def test(layer_loader, model, data, device, ):
    model.eval()
    
    out = model.inference(data.x, layer_loader, device)
    y_pred = out.exp()  # (N,num_classes)   
    
    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        node_id = node_id.to(device)
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
            
    return eval_results, losses, y_pred

def eval_metric(y_true, y_pred):
    if torch is not None and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if torch is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    

    


if __name__ == '__main__':
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data = GraphDataset('data', args.data_name, transform=T.ToSparseTensor())
    args.num_classes = 2
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

    print(data.num_features)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)






