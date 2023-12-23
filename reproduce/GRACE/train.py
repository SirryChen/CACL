import os
from tqdm import tqdm
import torch
import json
from torch.optim import AdamW
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
from utils import set_random_seed, super_parament_initial
from model import GRACEModel, drop_feature
from torch_geometric.utils import dropout_edge
import copy
from eval import label_classification


def train(step, subgraph_loader: NeighborLoader):
    model.train()

    loss_bar = {'loss': [], 'batch_size': []}
    tqdm_bar = tqdm(total=len(subgraph_loader), ncols=125, desc=f'训练模型-{step}')

    for subgraph1 in iter(subgraph_loader):
        train_user_mask = subgraph1['user'].batch_size
        subgraph2 = copy.deepcopy(subgraph1)
        for edge_type_ in subgraph1.edge_types:
            subgraph1[edge_type_].edge_index = dropout_edge(subgraph1[edge_type_].edge_index, p=args.drop_edge_rate_1)[0]
            subgraph2[edge_type_].edge_index = dropout_edge(subgraph2[edge_type_].edge_index, p=args.drop_edge_rate_2)[0]
        subgraph1['user'].x = drop_feature(subgraph1['user'].x, args.drop_feature_rate_1)
        subgraph2['user'].x = drop_feature(subgraph2['user'].x, args.drop_feature_rate_2)

        user_embedding1 = model(subgraph1.to(device))[:train_user_mask]
        user_embedding2 = model(subgraph2.to(device))[:train_user_mask]

        optimizer.zero_grad()
        loss = model.loss(user_embedding1, user_embedding2, batch_size=0)
        loss.backward()
        optimizer.step()
        loss_bar['loss'].append(loss.item()*train_user_mask)
        loss_bar['batch_size'].append(train_user_mask)
        tqdm_bar.update(1)

    loss = sum(loss_bar['loss']) / sum(loss_bar['batch_size'])
    tqdm_bar.set_postfix_str(f'loss: {round(loss, 4)}')

    return loss


@torch.no_grad()
def test(subgraph_loader: NeighborLoader):
    model.eval()

    total_node_embedding = torch.tensor([]).to(device)
    total_node_label = torch.tensor([]).to(device)

    tqdm_bar = tqdm(total=len(subgraph_loader), ncols=125, desc=f'验证模型')
    for subgraph in iter(subgraph_loader):
        test_user_mask = subgraph['user'].batch_size
        node_embedding = model(subgraph.to(device).detach())[:test_user_mask]
        node_label = subgraph['user'].node_label[:test_user_mask].long()
        total_node_label = torch.cat([total_node_label, node_label], dim=0)
        total_node_embedding = torch.cat([total_node_embedding, node_embedding], dim=0)
        tqdm_bar.update(1)

    return label_classification(total_node_embedding, total_node_label, ratio=0.1)


if __name__ == "__main__":
    set_random_seed(0)
    s_parament = super_parament_initial()
    args = s_parament.parse_args()

    dataset_name = args.dataset
    device = torch.device('cuda:4') if torch.cuda.is_available() else torch.device('cpu')

    predata_file_path = f"../../predata/{dataset_name}/"
    model_save_path = f"./train_result/{dataset_name}/"
    if not os.path.exists("./train_result"):
        os.mkdir("./train_result")
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    graph = torch.load(predata_file_path + 'graph.pt')
    for edge_type in graph.edge_types:
        if edge_type[0] != edge_type[2]:
            del graph[edge_type]

    kwargs = {'batch_size': 128, 'num_workers': 6, 'persistent_workers': True}
    num_neighbors = {edge_type: [1000] * 5 if edge_type[0] != 'tweet' else [100] * 5 for edge_type in graph.edge_types}
    train_dataloader = NeighborLoader(graph, num_neighbors=num_neighbors, shuffle=True,
                                      input_nodes=('user', graph['user'].node_split == 0), **kwargs)
    kwargs = {'batch_size': 2000, 'num_workers': 6, 'persistent_workers': True}
    num_neighbors = {edge_type: [50] * 5 if edge_type[0] != 'tweet' else [100] * 5 for edge_type in graph.edge_types}
    test_dataloader = NeighborLoader(graph, num_neighbors=num_neighbors, shuffle=True,
                                     input_nodes=('user', graph['user'].node_split == 2), **kwargs)

    model = GRACEModel(graph.metadata(), args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    compute_loss = torch.nn.CrossEntropyLoss()

    best_f1 = 0
    error_times = 0
    epoch_loss = {}
    epoch_score = {}

    for epoch in range(args.num_epochs):
        loss_ = train(epoch, train_dataloader)
        epoch_loss[f'epoch-{epoch}'] = loss_

    torch.save(model.state_dict(), model_save_path + 'model.pth')
    final_score = test(test_dataloader)
    epoch_score['final'] = final_score
    print(f'test score: {final_score}')

    fine_tune_info = {'score': epoch_score, 'loss': epoch_loss}
    with open(model_save_path + 'train_info.json', 'w') as fp:
        json.dump(fine_tune_info, fp, indent=4)
