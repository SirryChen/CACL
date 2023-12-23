import json
import copy
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from CD_model import ModCDModel
from torch_geometric.data import HeteroData
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import NeighborLoader
from utils import set_random_seed, super_parament_initial
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
from CL_model import HeteroGraphConvModel, MLPProjector, HGcnCLModel, Classifier, adaptive_augment, \
    SubGraphDataset, compute_loss, compute_cross_view_loss


def train(step, subgraph_loader):
    CLModel.train()

    loss_bar = {'loss': [], 'user_num': []}

    tqdm_bar = tqdm(total=len(subgraph_loader), ncols=125)
    tqdm_bar.set_description(f'epoch: {step}/{args.epochs}')
    for graph_th, subgraph in enumerate(subgraph_loader.subgraph(tqdm_bar)):
        batch_user_mask = subgraph['user'].batch_mask

        tqdm_bar.set_postfix_str(f'progress: 自适应图增强')
        augmented_graph1, augmented_graph2 = adaptive_augment(subgraph)
        augmented_graph1 = augmented_graph1.to(device_cl).detach()
        augmented_graph2 = augmented_graph2.to(device_cl).detach()

        tqdm_bar.set_postfix_str(f'progress: 图编码')
        node_projections_1, node_prediction = CLModel(augmented_graph1)
        node_projections_2, _ = CLModel(augmented_graph2)

        tqdm_bar.set_postfix_str(f'progress: 损失计算更新')
        node_label = subgraph['user'].node_label.to(device_cl)
        node_split = subgraph['user'].node_split.to(device_cl)
        total_loss = compute_loss(node_projections_1['user'][batch_user_mask],
                                  node_projections_2['user'][batch_user_mask],
                                  node_prediction[batch_user_mask],
                                  node_label[batch_user_mask], node_split[batch_user_mask],
                                  args.tau, args.alpha, args.beta)
        optimizer_CL.zero_grad()
        total_loss.backward()
        optimizer_CL.step()
        subgraph_loader.cd_model.cd_encoder_decoder.update_embedding_layer(CLModel.update_cd_model())
        loss_bar['user_num'].append(subgraph['user'].x.size(0))
        loss_bar['loss'].append(total_loss.to('cpu').detach().item() * loss_bar['user_num'][-1])

    tqdm_bar.set_postfix(None)
    avg_loss = sum(loss_bar['loss']) / sum(loss_bar['user_num'])
    tqdm_bar.set_postfix_str(f'average loss: {avg_loss}')
    # lr_scheduler.step()

    return avg_loss


def train_hard(step, subgraph_loader):
    CLModel.train()

    loss_bar = {'loss': [], 'user_num': []}

    tqdm_bar = tqdm(total=len(subgraph_loader), ncols=125)
    tqdm_bar.set_description(f'epoch: {step}/{args.epochs}')
    for subgraph_a, subgraph_b in subgraph_loader.subgraph_matched(tqdm_bar):
        batch_user_mask_a = subgraph_a['user'].batch_mask
        batch_user_mask_b = subgraph_b['user'].batch_mask

        tqdm_bar.set_postfix_str(f'progress: 自适应图增强')
        augment_graph_a_1, augment_graph_a_2 = adaptive_augment(copy.deepcopy(subgraph_a))
        augment_graph_b_1, augment_graph_b_2 = adaptive_augment(copy.deepcopy(subgraph_b))

        augment_graph_a_1 = augment_graph_a_1.to(device_cl).detach()
        augment_graph_a_2 = augment_graph_a_2.to(device_cl).detach()
        augment_graph_b_1 = augment_graph_b_1.to(device_cl).detach()
        augment_graph_b_2 = augment_graph_b_2.to(device_cl).detach()

        tqdm_bar.set_postfix_str(f'progress: 图编码')
        node_projections_a_1, node_prediction_a = CLModel(augment_graph_a_1)
        node_projections_a_2, _ = CLModel(augment_graph_a_2)
        node_projections_b_1, node_prediction_b = CLModel(augment_graph_b_1)
        node_projections_b_2, _ = CLModel(augment_graph_b_2)

        tqdm_bar.set_postfix_str(f'progress: 损失计算更新')
        node_label_a = subgraph_a['user'].node_label.to(device_cl)
        node_label_b = subgraph_b['user'].node_label.to(device_cl)
        total_loss = compute_cross_view_loss(node_projections_a_1['user'][batch_user_mask_a],
                                             node_projections_a_2['user'][batch_user_mask_a],
                                             node_projections_b_1['user'][batch_user_mask_b],
                                             node_projections_b_2['user'][batch_user_mask_b],
                                             node_prediction_a[batch_user_mask_a],
                                             node_prediction_b[batch_user_mask_b],
                                             node_label_a[batch_user_mask_a], node_label_b[batch_user_mask_b],
                                             args.tau, args.alpha, args.beta, mean_flag=False)
        optimizer_CL.zero_grad()
        total_loss.backward()
        optimizer_CL.step()
        subgraph_loader.cd_model.cd_encoder_decoder.update_embedding_layer(CLModel.update_cd_model())
        loss_bar['user_num'].append(subgraph_a['user'].x.size(0) + subgraph_b['user'].x.size(0))
        loss_bar['loss'].append(total_loss.to('cpu').detach().item() * loss_bar['user_num'][-1])

        tqdm_bar.set_postfix_str(f'progress: 生成子图')

    tqdm_bar.set_postfix(None)
    avg_loss = sum(loss_bar['loss']) / sum(loss_bar['user_num'])
    tqdm_bar.set_postfix_str(f'average loss: {avg_loss}')
    # lr_scheduler.step()

    return avg_loss


@torch.no_grad()
def valid(subgraph_loader):
    CLModel.eval()

    total_node_predict = torch.tensor([]).to(device_cl)
    total_node_label = torch.tensor([]).to(device_cl)
    tqdm_bar = tqdm(total=len(subgraph_loader), ncols=125, desc='验证模型')

    for graph_th, subgraph in enumerate(subgraph_loader.subgraph(tqdm_bar)):
        subgraph['tweet'].x = subgraph['tweet'].x1
        batch_user_mask = subgraph['user'].batch_mask
        _, node_predict = CLModel(subgraph.to(device_cl).detach())
        node_predict = node_predict[batch_user_mask]
        node_label = subgraph['user'].node_label[batch_user_mask].long()

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
    # 参数初始化
    set_random_seed(0)
    s_parament = super_parament_initial()
    args = s_parament.parse_args()

    dataset_name = args.dataset

    device_cd = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device_cl = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')

    predata_file_path = f"./predata/{dataset_name}/"
    model_save_path = f"./train_result/{dataset_name}/"
    experiment_result_save_path = f"./train_result/{dataset_name}/dynamic_cd_hard_positive_negative_individual_cl_5/"

    if not os.path.exists(experiment_result_save_path):
        os.mkdir(experiment_result_save_path)

    with open('cd_config.json', 'r') as fp:
        cd_config = json.load(fp)

    # 数据加载，反转关系
    graph: HeteroData = torch.load(predata_file_path + 'graph.pt')
    for edge_type in graph.edge_types:
        if edge_type[0] != edge_type[2]:
            graph[edge_type[2], edge_type[1], edge_type[0]].edge_index = graph[edge_type].edge_index[[1, 0]]

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

    # kwargs = {'batch_size': 100, 'num_workers': 6, 'persistent_workers': True}
    # num_neighbors = {edge_type: [5] * 3 if edge_type[0] != 'tweet' else [100] * 3 for edge_type in graph.edge_types}
    # train_dataloader = NeighborLoader(graph, num_neighbors=num_neighbors, shuffle=True,
    #                                   input_nodes=('user', graph['user'].node_split == 0), **kwargs)
    # kwargs = {'batch_size': 2000, 'num_workers': 6, 'persistent_workers': True}
    # num_neighbors = {edge_type: [5] * 3 if edge_type[0] != 'tweet' else [100] * 3 for edge_type in graph.edge_types}
    # valid_dataloader = NeighborLoader(graph, num_neighbors=num_neighbors, shuffle=True,
    #                                   input_nodes=('user', graph['user'].node_split == 1), **kwargs)
    # kwargs = {'batch_size': 2000, 'num_workers': 6, 'persistent_workers': True}
    # num_neighbors = {edge_type: [5] * 3 if edge_type[0] != 'tweet' else [100] * 3 for edge_type in graph.edge_types}
    # test_dataloader = NeighborLoader(graph, num_neighbors=num_neighbors, shuffle=True,
    #                                  input_nodes=('user', graph['user'].node_split == 2), **kwargs)

    CDModel = ModCDModel(args, cd_config, device_cd, pretrain=False, ensure_comm_num=True)
    CDModel = CDModel.to(device_cd)

    train_dataset = SubGraphDataset(CDModel, train_dataloader)
    valid_dataset = SubGraphDataset(None, valid_dataloader)
    test_dataset = SubGraphDataset(None, test_dataloader)

    # 模型初始化
    Encoder = HeteroGraphConvModel(model=args.basic_model,
                                   in_channels=train_dataset.get_in_channels(),
                                   hidden_channels=args.encoder_hidden_channel,
                                   out_channels=args.encoder_out_channel,
                                   metadata=train_dataset.get_meta_data(),
                                   args=args)
    Projector = MLPProjector(node_types=train_dataset.get_in_channels().keys(),
                             input_size={node_type: args.encoder_out_channel + feature_dim
                                         if node_type != 'user' else args.encoder_out_channel + args.embedding_dim
                                         for node_type, feature_dim in train_dataset.get_in_channels().items()},
                             output_size=args.encoder_out_channel,
                             hidden_size=args.projector_hidden_size)
    classifier = Classifier(input_dim=args.encoder_out_channel + args.embedding_dim)
    CLModel = HGcnCLModel(Encoder, Projector, classifier).to(device_cl)

    valid(valid_dataset)    # first forward for initializing model
    train_dataset.cd_model.cd_encoder_decoder.load_state_dict(
        torch.load(model_save_path + f"pretrain_cd_model_{args.basic_model}.pth"))
    CLModel.encoder.init_first_layer(train_dataset.cd_model.cd_encoder_decoder.state_dict())

    optimizer_CL = AdamW(CLModel.parameters(), lr=args.cl_learning_rate, weight_decay=args.weight_decay)
    # lr_scheduler = CosineAnnealingLR(optimizer_CL, args.epochs//10, 0)

    best_target = 0
    target_name = 'f1'
    error_times = 0
    epoch_loss = {}
    epoch_score = {}
    restrain_flag = 0
    for epoch in range(args.epochs):
        loss_ = train_hard(epoch, train_dataset)
        score_ = valid(valid_dataset)
        epoch_loss[f'epoch-{epoch}'] = loss_
        epoch_score[f'epoch-{epoch}'] = score_
        if epoch < args.lr_warmup_epochs:
            args.alpha = 1.0
            args.beta = 0.0
            continue
        else:
            args.alpha = 0.1
            args.beta = 1.0
        if score_[target_name] > best_target:
            torch.save(CDModel.state_dict(), experiment_result_save_path + 'cd_model.pth')
            torch.save(Encoder.state_dict(), experiment_result_save_path + 'encoder.pth')
            torch.save(classifier.state_dict(), experiment_result_save_path + 'classifier.pth')
            best_target = score_[target_name]
            error_times = 0
        elif score_[target_name] < best_target:
            error_times += 1
            if error_times > args.max_error_times:
                print(f'*** stop at epoch-{epoch} ***')
                break
    CDModel.load_state_dict(torch.load(experiment_result_save_path + 'cd_model.pth'))
    Encoder.load_state_dict(torch.load(experiment_result_save_path + 'encoder.pth'))
    Encoder.dropout = nn.Dropout(0)
    classifier.load_state_dict(torch.load(experiment_result_save_path + 'classifier.pth'))
    CLModel = HGcnCLModel(Encoder, Projector, classifier).to(device_cl)
    final_score = valid(test_dataset)
    epoch_score['final'] = final_score
    print(f'test score: {final_score}')

    train_info = {'score': epoch_score, 'loss': epoch_loss}
    with open(experiment_result_save_path + 'train_info.json', 'w') as fp:
        json.dump(train_info, fp, indent=4)
    print('********** Finish **********')
