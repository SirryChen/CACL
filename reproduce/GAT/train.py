import os
from tqdm import tqdm
import torch
import json
from torch.optim import AdamW
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
from utils import set_random_seed, super_parament_initial
from model import GATModel


def train(step, subgraph_loader):
    model.train()

    total_node_predict = torch.tensor([]).to(device)
    total_node_label = torch.tensor([]).to(device)
    loss_bar = {'loss': [], 'batch_size': []}
    tqdm_bar = tqdm(total=len(subgraph_loader), ncols=125, desc=f'训练模型-{step}')

    for subgraph in subgraph_loader:
        train_user_mask = subgraph['user'].batch_size
        node_predict = model(subgraph.to(device))[:train_user_mask]
        node_label = subgraph['user'].node_label[:train_user_mask].long()

        optimizer.zero_grad()
        loss = compute_loss(node_predict, node_label)
        loss.backward()
        optimizer.step()
        total_node_label = torch.cat((total_node_label, node_label), dim=0)
        total_node_predict = torch.cat((total_node_predict, node_predict.detach()), dim=0)
        loss_bar['loss'].append(loss.item()*train_user_mask)
        loss_bar['batch_size'].append(train_user_mask)
        tqdm_bar.update(1)

    f1 = f1_score(total_node_label.to('cpu'), torch.argmax(total_node_predict, dim=1).to('cpu'))
    acc = accuracy_score(total_node_label.to('cpu'), torch.argmax(total_node_predict, dim=1).to('cpu'))
    loss = sum(loss_bar['loss']) / sum(loss_bar['batch_size'])
    tqdm_bar.set_postfix_str(f'f1: {round(f1, 4)}, acc: {round(acc, 4)}, loss: {round(loss, 4)}')

    return loss


@torch.no_grad()
def valid(subgraph_loader):
    model.eval()

    total_node_predict = torch.tensor([]).to(device)
    total_node_label = torch.tensor([]).to(device)

    tqdm_bar = tqdm(total=len(subgraph_loader), ncols=125, desc=f'验证模型')
    for subgraph in subgraph_loader:
        valid_user_mask = subgraph['user'].batch_size
        node_predict = model(subgraph.to(device).detach())[:valid_user_mask]
        node_label = subgraph['user'].node_label[:valid_user_mask].long()

        total_node_predict = torch.cat((total_node_predict, node_predict.detach()), dim=0)
        total_node_label = torch.cat([total_node_label, node_label], dim=0)
        tqdm_bar.update(1)

    label = total_node_label.to('cpu')
    predict = torch.argmax(total_node_predict, dim=1).to('cpu')
    conf_matrix = confusion_matrix(label, predict)
    f1 = f1_score(label, predict)
    acc = accuracy_score(label, predict)
    mcc = matthews_corrcoef(label, predict)
    tqdm_bar.set_postfix_str(f'f1: {round(f1, 4)}, acc: {round(acc, 4)}, mat: {conf_matrix[0], conf_matrix[1]}')

    return {'f1': f1, 'acc': acc, 'mcc': mcc, 'mat': conf_matrix.tolist()}


if __name__ == "__main__":
    set_random_seed(0)
    s_parament = super_parament_initial()
    args = s_parament.parse_args()

    dataset_name = args.dataset
    device = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')

    predata_file_path = f"../../predata/{dataset_name}/"
    model_save_path = f"./train_result/{dataset_name}/"
    if not os.path.exists("./train_result"):
        os.mkdir("./train_result")
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    graph: HeteroData = torch.load(predata_file_path + 'graph.pt')
    del graph['user', 'post', 'tweet']

    kwargs = {'batch_size': 128, 'num_workers': 6, 'persistent_workers': True}
    num_neighbors = {edge_type: [1000] * 5 if edge_type[0] != 'tweet' else [100] * 5 for edge_type in graph.edge_types}
    train_dataloader = NeighborLoader(graph, num_neighbors=num_neighbors, shuffle=True,
                                      input_nodes=('user', graph['user'].node_split == 0), **kwargs)
    kwargs = {'batch_size': 2000, 'num_workers': 6, 'persistent_workers': True}
    num_neighbors = {edge_type: [50] * 5 if edge_type[0] != 'tweet' else [100] * 5 for edge_type in graph.edge_types}
    valid_dataloader = NeighborLoader(graph, num_neighbors=num_neighbors, shuffle=True,
                                      input_nodes=('user', graph['user'].node_split == 1), **kwargs)
    kwargs = {'batch_size': 2000, 'num_workers': 6, 'persistent_workers': True}
    num_neighbors = {edge_type: [50] * 5 if edge_type[0] != 'tweet' else [100] * 5 for edge_type in graph.edge_types}
    test_dataloader = NeighborLoader(graph, num_neighbors=num_neighbors, shuffle=True,
                                     input_nodes=('user', graph['user'].node_split == 2), **kwargs)
    model = GATModel(graph.metadata(), args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    compute_loss = torch.nn.CrossEntropyLoss()

    best_f1 = 0
    error_times = 0
    epoch_loss = {}
    epoch_score = {}

    for epoch in range(512):
        loss_ = train(epoch, train_dataloader)
        score_ = valid(valid_dataloader)
        epoch_loss[f'epoch-{epoch}'] = loss_
        epoch_score[f'epoch-{epoch}'] = score_

        if score_['f1'] > best_f1:
            torch.save(model.state_dict(), model_save_path + 'model.pth')
            best_f1 = score_['f1']
            error_times = 0
        elif score_['f1'] < best_f1:
            error_times += 1
            if error_times > args.max_error_times:
                print(f'*** stop at epoch-{epoch} ***')
                break

    model.load_state_dict(torch.load(model_save_path + 'model.pth'))
    final_score = valid(test_dataloader)
    epoch_score['final'] = final_score
    print(f'test score: {final_score}')

    fine_tune_info = {'score': epoch_score, 'loss': epoch_loss}
    with open(model_save_path + 'train_info.json', 'w') as fp:
        json.dump(fine_tune_info, fp, indent=4)
