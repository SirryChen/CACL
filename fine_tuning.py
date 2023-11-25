import os
from tqdm import tqdm
import torch
import json
from torch.optim import AdamW
from torch_geometric.loader import HGTLoader, NeighborLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
from utils import set_random_seed, super_parament_initial
from CL_model import HeteroGraphConvModel, Classifier, adaptive_augment, BinaryLoss, FocalLoss, SubGraphDataset


def train(step, subgraph_loader):
    Encoder.train()
    classifier.train()

    total_node_predict = torch.tensor([]).to(device_ft)
    total_node_label = torch.tensor([]).to(device_ft)
    loss_bar = {'loss': [], 'batch_size': []}
    tqdm_bar = tqdm(total=len(subgraph_loader), ncols=125, desc=f'微调模型-{step}')

    for subgraph_th, subgraph in enumerate(subgraph_loader.subgraph(tqdm_bar)):
        subgraph['tweet'].x = subgraph['tweet'].x1
        train_user_mask = subgraph['user'].batch_size
        node_emb = Encoder(subgraph.to(device_ft))['user'][:train_user_mask]
        node_predict = classifier(node_emb)
        node_label = subgraph['user'].node_label[:train_user_mask].long()

        optimizer.zero_grad()
        loss = compute_loss(node_predict, node_label)
        loss.backward()
        optimizer.step()
        total_node_label = torch.cat((total_node_label, node_label), dim=0)
        total_node_predict = torch.cat((total_node_predict, node_predict.detach()), dim=0)
        loss_bar['loss'].append(loss.item()*train_user_mask)
        loss_bar['batch_size'].append(train_user_mask)

    f1 = f1_score(total_node_label.to('cpu'), torch.argmax(total_node_predict, dim=1).to('cpu'))
    acc = accuracy_score(total_node_label.to('cpu'), torch.argmax(total_node_predict, dim=1).to('cpu'))
    loss = sum(loss_bar['loss']) / sum(loss_bar['batch_size'])
    tqdm_bar.set_postfix_str(f'f1: {round(f1, 4)}, acc: {round(acc, 4)}, loss: {round(loss, 4)}')

    return loss


@torch.no_grad()
def valid(subgraph_loader):
    Encoder.eval()
    classifier.eval()

    total_node_predict = torch.tensor([]).to(device_ft)
    total_node_label = torch.tensor([]).to(device_ft)
    tqdm_bar = tqdm(total=len(subgraph_loader), ncols=125, desc='验证模型')

    for graph_th, subgraph in enumerate(subgraph_loader.subgraph(tqdm_bar)):
        subgraph['tweet'].x = subgraph['tweet'].x1
        valid_user_mask = subgraph['user'].batch_size
        node_emb = Encoder(subgraph.to(device_ft).detach())['user'][:valid_user_mask]
        node_predict = classifier(node_emb)
        node_label = subgraph['user'].node_label[:valid_user_mask].long()

        total_node_predict = torch.cat((total_node_predict, node_predict.detach()), dim=0)
        total_node_label = torch.cat([total_node_label, node_label], dim=0)

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

    device_cd = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    device_ft = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')

    predata_file_path = f"./predata/{dataset_name}/"
    model_save_path = f"./train_result/{dataset_name}/"
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    with open('cd_config.json', 'r') as fp:
        cd_config = json.load(fp)

    graph = torch.load(predata_file_path + 'graph.pt')
    graph['tweet', 'from', 'user'].edge_index = graph['user', 'post', 'tweet'].edge_index[[1, 0]]
    # del graph['user', 'post', 'tweet'], graph['tweet']

    # graph = torch.load("/data1/botdet/CDCL_homogenesis/predata/twibot20/graph.pt")
    # graph['user'].x = graph['user'].x[:, 1:2]

    kwargs = {'batch_size': 128, 'num_workers': 6, 'persistent_workers': True}
    num_neighbors = {edge_type: [50] * 5 if edge_type[0] != 'tweet' else [100] * 5 for edge_type in graph.edge_types}
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

    train_dataset = SubGraphDataset(None, train_dataloader)
    valid_dataset = SubGraphDataset(None, valid_dataloader)
    test_dataset = SubGraphDataset(None, test_dataloader)

    Encoder = HeteroGraphConvModel(model='HGT',
                                   in_channels=train_dataset.get_in_channels(),
                                   hidden_channels=args.encoder_hidden_channel,
                                   out_channels=args.encoder_out_channel,
                                   metadata=train_dataset.get_meta_data(),
                                   num_layers=args.num_layer).to(device_ft)
    # Encoder.load_state_dict(torch.load(model_save_path + 'encoder.pth'))
    classifier = Classifier(input_dim=args.encoder_out_channel + train_dataset.get_in_channels()['user']).to(device_ft)

    params = [{'params': classifier.parameters(), 'lr': args.ft_learning_rate},
              {'params': Encoder.parameters(), 'lr': args.ft_learning_rate}]
    optimizer = AdamW(params, lr=args.ft_learning_rate, weight_decay=args.weight_decay)

    if args.classifier_loss_function == 'binary':
        # weight = train_dataset.get_loss_weight()
        # weight = torch.tensor([0.57, 0.43])
        # compute_loss = BinaryLoss(weight=weight.to(device_ft))
        # compute_loss = BinaryLoss()
        compute_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([4, 3], dtype=torch.float).to(device_ft))
    elif args.classifier_loss_function == 'focal':
        compute_loss = FocalLoss()

    best_f1 = 0
    error_times = 0
    epoch_loss = {}
    epoch_score = {}

    for epoch in range(512):
        loss_ = train(epoch, train_dataset)
        score_ = valid(valid_dataset)
        epoch_loss[f'epoch-{epoch}'] = loss_
        epoch_score[f'epoch-{epoch}'] = score_

        if score_['f1'] > best_f1:
            torch.save(Encoder.state_dict(), model_save_path + 'fine_tuned_encoder.pth')
            torch.save(classifier.state_dict(), model_save_path + 'classifier.pth')
            best_f1 = score_['f1']
            error_times = 0
        elif score_['f1'] < best_f1:
            error_times += 1
            if error_times > args.max_error_times:
                print(f'*** stop at epoch-{epoch} ***')
                break

    Encoder.load_state_dict(torch.load(model_save_path + 'fine_tuned_encoder.pth'))
    classifier.load_state_dict(torch.load(model_save_path + 'classifier.pth'))
    final_score = valid(test_dataset)
    epoch_score['final'] = final_score
    print(f'test score: {final_score}')

    fine_tune_info = {'score': epoch_score, 'loss': epoch_loss}
    with open(model_save_path + 'fine_tune_info.json', 'w') as fp:
        json.dump(fine_tune_info, fp, indent=4)
