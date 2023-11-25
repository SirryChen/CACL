import os
import json
import torch
from CD_model import ModCDModel
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.data import HeteroData
from tqdm import tqdm
from utils import super_parament_initial


def train(step):
    CDModel.train()
    loss_ = 0.0
    train_score_ = {'auc': [], 'ap': []}
    tqdm_bar = tqdm(total=len(dataloader), ncols=150)
    tqdm_bar.set_description(f'train: {step}/{epoch_num}')
    for subgraph in iter(dataloader):
        CDModel(subgraph)

        optimizer_CD.zero_grad()
        cd_loss = CDModel.compute_loss()
        cd_loss.backward()
        optimizer_CD.step()

        loss_ += cd_loss.item()
        score_ = CDModel.compute_score()
        train_score_['auc'].append(score_['auc'])
        train_score_['ap'].append(score_['ap'])
        tqdm_bar.set_postfix_str(f"loss: {round(cd_loss.item(), 3)}, "
                                 f"auc: {round(score_['auc'], 3)}, "
                                 f"ap: {round(score_['ap'], 3)}, "
                                 f"mat: {score_['confusion_matrix'][0].tolist()}"
                                 f"{score_['confusion_matrix'][1].tolist()}")
        tqdm_bar.update(1)

    train_score_['auc'] = sum(train_score_['auc'])/len(train_score_['auc'])
    train_score_['ap'] = sum(train_score_['ap'])/len(train_score_['ap'])
    tqdm_bar.set_postfix_str(f"loss: {round(loss_ / len(dataloader), 3)}, "
                             f"auc: {round(train_score_['auc'], 3)}, "
                             f"ap: {round(train_score_['ap'], 3)}")

    return loss_, train_score_


def val():
    CDModel.eval()
    val_score_ = {'auc': [], 'ap': []}
    with torch.no_grad():
        for subgraph in iter(dataloader):
            CDModel(subgraph)

            score_ = CDModel.compute_score()
            val_score_['auc'].append(score_['auc'])
            val_score_['ap'].append(score_['ap'])
            break

    val_score_['auc'] = sum(val_score_['auc'])/len(val_score_['auc'])
    val_score_['ap'] = sum(val_score_['ap'])/len(val_score_['ap'])
    print(f"val: auc-{val_score_['auc']}, ap-{val_score_['ap']}")
    return val_score_


if __name__ == "__main__":

    s_parament = super_parament_initial()
    args = s_parament.parse_args()

    dataset_name = args.dataset
    device_cd = torch.device("cuda:2") if torch.cuda.is_available() else torch.device('cpu')

    predata_file_path = f"./predata/{dataset_name}/"
    model_save_path = f"./train_result/{dataset_name}/"
    if not os.path.exists('./train_result'):
        os.mkdir('./train_result')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    with open('cd_config.json', 'r') as fp:
        cd_config = json.load(fp)

    graph: HeteroData = torch.load(predata_file_path + 'graph.pt')
    for edge_type in graph.edge_types:
        if edge_type[0] != edge_type[2]:
            graph[edge_type[2], edge_type[1], edge_type[0]].edge_index = graph[edge_type].edge_index[[1, 0]]
    # graph['tweet', 'from', 'user'].edge_index = graph['user', 'post', 'tweet'].edge_index[[1, 0]]

    num_neighbors = {edge_type: [50] * 3 if edge_type[0] != 'tweet' else [0] * 3 for edge_type in graph.edge_types}
    kwargs = {'batch_size': 4000, 'num_workers': 6, 'persistent_workers': True}
    dataloader = NeighborLoader(graph, num_neighbors=num_neighbors, shuffle=True, input_nodes='user', **kwargs)

    CDModel = ModCDModel(graph['user'].x.size(-1), args, cd_config, device_cd, graph.metadata(), pretrain=True)
    CDModel = CDModel.to(device_cd)
    optimizer_CD = torch.optim.Adam(params=CDModel.parameters(), lr=cd_config['lr'])

    epoch_num = cd_config['epoch']
    max_error_time = cd_config['max_error_times']
    record_score = {'train': [], 'val': []}
    record_loss = {}
    best_auc = 0.0
    best_ap = 0.0
    error_time = 0
    for epoch in range(epoch_num):
        train_loss, train_score = train(epoch)
        val_score = val()

        record_score['train'].append(train_score)
        record_loss[epoch] = train_loss
        record_score['val'].append(val_score)

        if val_score['auc'] >= best_auc:
            best_auc = val_score['auc']
            torch.save(CDModel.cd_encoder_decoder.state_dict(),
                       model_save_path + f'pretrain_cd_model_{args.basic_model}.pth')
            error_time = 0
        else:
            error_time += 1
            if error_time >= max_error_time:
                print(f"*** early stop at {epoch - error_time} ***")
                break

    pretrain_info = {'score': record_score, 'loss': record_loss}
    with open(model_save_path + f'pretrain_info_{args.basic_model}.json', 'w') as fp:
        json.dump(pretrain_info, fp, indent=4)

    print('*** finish pretraining ***')
