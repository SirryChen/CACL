import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.nn.conv import HeteroConv, HGTConv, RGCNConv, FastRGCNConv, GATConv, SAGEConv
from torch_geometric.nn.norm import BatchNorm
import numpy as np
from sklearn.metrics import confusion_matrix
from torch_geometric.data import HeteroData
from torch.nn.functional import cross_entropy
from copy import deepcopy
from utils import drop_feature, feature_drop_weights, compute_page_rank, add_self_loop, tweet_augment, information_entropy
from CL_utils import compute_hard_loss, compute_pro_loss, unsupervised_cl_loss, traditional_cl_loss, \
    compute_cross_mean_view_loss, compute_cross_individual_loss, RGTLayer


class MLPProjector(nn.Module):
    """MLP used for predictor. The MLP has one hidden layer.
    @input_size (int): Size of input features.
    @output_size (int): Size of output features.
    @hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """

    def __init__(self, node_types, input_size, output_size, hidden_size, dropout=0.5):
        super(MLPProjector, self).__init__()
        self.projector = nn.ModuleDict()
        for node_type in node_types:
            net = nn.Sequential(
                nn.Linear(input_size[node_type], hidden_size, bias=True),
                nn.PReLU(num_parameters=1),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size, bias=True)
            )
            self.projector[node_type] = net
        self.init_weight()

    def forward(self, x):
        return {node_type: self.projector[node_type](node_feature) for node_type, node_feature in x.items()}

    def init_weight(self):
        for net in self.projector.values():
            for layer in net:
                if isinstance(net, nn.Linear):
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0.0)


# 创建一个 HeteroConv 模型作为编码器
class HeteroGraphConvModel(nn.Module):
    def __init__(self, model, in_channels, hidden_channels, out_channels, metadata, args):
        super(HeteroGraphConvModel, self).__init__()
        self.model = model
        self.num_prop_size = args.num_prop_size
        self.cat_prop_size = args.cat_prop_size
        self.des_size = args.des_size
        self.tweet_size = args.tweet_size
        self.linear_relu_des = nn.Sequential(nn.Linear(args.des_size, int(args.embedding_dim / 4)),
                                             nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.linear_relu_tweet = nn.Sequential(nn.Linear(args.tweet_size, int(args.embedding_dim / 4)),
                                               nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(args.num_prop_size, int(args.embedding_dim / 4)),
                                                  nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(args.cat_prop_size, int(args.embedding_dim / 4)),
                                                  nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.linear = nn.ModuleDict({node_type: nn.Linear(in_channels[node_type], hidden_channels)
                                     if node_type != 'user' else nn.Linear(args.embedding_dim, hidden_channels)
                                     for node_type in in_channels})
        self.LReLU = nn.LeakyReLU()
        self.Hetero_Conv_list = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(args.num_layer - 1):
            self.Hetero_Conv_list.append(self.choice_basic_model(hidden_channels, hidden_channels, metadata))
            self.batch_norms.append(BatchNorm(hidden_channels, momentum=args.momentum))
        self.Hetero_Conv_list.append(self.choice_basic_model(hidden_channels, out_channels, metadata))
        self.batch_norms.append(BatchNorm(out_channels, momentum=args.momentum))
        self.dropout = nn.Dropout(p=args.dropout)

        self.init_weight()

    def forward(self, graph):
        num_prop = graph['user'].x[:, :self.num_prop_size]
        cat_prop = graph['user'].x[:, self.num_prop_size:(self.num_prop_size+self.cat_prop_size)]
        des = graph['user'].x[:, (self.num_prop_size+self.cat_prop_size):
                                 (self.num_prop_size+self.cat_prop_size+self.des_size)]
        tweet = graph['user'].x[:, (self.num_prop_size+self.cat_prop_size+self.des_size):]
        user_features_des = self.linear_relu_des(des)
        user_features_tweet = self.linear_relu_tweet(tweet)
        user_features_numeric = self.linear_relu_num_prop(num_prop)
        user_features_bool = self.linear_relu_cat_prop(cat_prop)
        graph['user'].x = torch.cat((user_features_numeric, user_features_bool,
                                     user_features_des, user_features_tweet), dim=1)
        node_feature = graph.x_dict
        edge_index = graph.edge_index_dict

        node_embedding = {node_type: self.dropout(self.LReLU(self.linear[node_type](feature)))
                          for node_type, feature in node_feature.items()}
        # 卷积、批量归一化
        for Conv, batch_norm in zip(self.Hetero_Conv_list, self.batch_norms):
            node_embedding = Conv(node_embedding, edge_index)
            # node_embedding = {node_type: batch_norm(embedding) for node_type, embedding in node_embedding.items()}

        node_embedding = {node_type: torch.cat((embedding, node_feature[node_type]), dim=1)
                          for node_type, embedding in node_embedding.items()}
        return node_embedding

    def choice_basic_model(self, in_channels, out_channels, metadata):
        if self.model == "HGT":
            # return HGTConv(in_channels=in_channels, out_channels=out_channels, metadata=metadata, heads=1)
            return HGTConv(in_channels=in_channels, out_channels=out_channels, metadata=metadata, heads=2, group='sum')
        elif self.model == "GAT":
            return HeteroConv({edge_type: GATConv((-1, -1), out_channels, add_self_loops=False)
                               for edge_type in metadata[1]}, aggr='sum')
        elif self.model == "SAGE":
            return HeteroConv({edge_type: SAGEConv((-1, -1), out_channels) for edge_type in metadata[1]}, aggr='sum')
        elif self.model == "RGT":
            return RGTLayer(num_edge_type=len(metadata[1]), in_channel=in_channels, out_channel=out_channels)

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
            elif isinstance(module, dict):
                for layer in module.values():
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0.0)

    def init_first_layer(self, weight):
        self.load_state_dict(weight, strict=False)


# 创建HGcnCLModel模型
class HGcnCLModel(nn.Module):
    def __init__(self, encoder, projector, classifier=None):
        super(HGcnCLModel, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.classifier = classifier

    def forward(self, graph):
        node_embeddings = self.encoder(graph)
        node_projections = self.projector(node_embeddings)
        if self.classifier is not None:
            node_predicts = self.classifier(node_embeddings['user'])
            return node_projections, node_predicts
        else:
            return node_projections, None

    def update_cd_model(self):
        return self.encoder.state_dict()

    def get_node_embeddings(self, node_features):
        return self.encoder(node_features)


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.LReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.init_weight()

    def forward(self, node_embedding):
        hidden_embedding = self.dropout(self.LReLU(self.fc1(node_embedding)))
        predict = self.fc2(hidden_embedding)

        return torch.softmax(predict, dim=1)

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
            elif isinstance(module, dict):
                for layer in module.values():
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0.0)


class BinaryLoss(nn.Module):
    """
    bot:1,human:0, human为第一列，bot为第2列
    """

    def __init__(self, reduction='mean', weight=None):
        super(BinaryLoss, self).__init__()
        self.compute_loss = nn.CrossEntropyLoss(reduction=reduction, weight=weight)

    def forward(self, predicts, labels):
        assert predicts.dim() == 2 and labels.dim() == 1
        labels = torch.stack([1 - labels, labels]).t().float()
        loss = self.compute_loss(predicts, labels)

        return loss


class FocalLoss(nn.Module):
    """
    bot:1,human:0, human为第1列，bot为第2列
    """

    def __init__(self, alpha=1, gamma=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.compute_loss = nn.functional.binary_cross_entropy
        self.reduction = reduction

    def forward(self, predicts, labels):
        assert predicts.dim() == 2 and labels.dim() == 1
        labels = torch.stack([1 - labels, labels]).t()
        loss = self.compute_loss(predicts, labels, reduction='none')
        pt = torch.exp(-loss)
        loss = self.alpha * (1 - pt) ** self.gamma * loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class SubGraphDataset:
    def __init__(self, cd_model, dataloader):
        self.graph = dataloader.data
        self.cd_model = cd_model
        self.dataloader = dataloader
        self.init_cd_model()
        self.entropy_list = []
        self.comm_num = []

    def __len__(self):
        return len(self.dataloader)

    def init_cd_model(self):
        if self.cd_model is not None:
            for graph in self.dataloader:
                self.cd_model(graph)
                break

    def subgraph(self, tqdm_bar):
        # generate one subgraph for one loop
        for graph in self.dataloader:
            graph['user'].batch_mask = torch.tensor([True] * graph['user'].batch_size +
                                                    [False] * (graph['user'].x.size(0) - graph['user'].batch_size))
            if self.cd_model is None:
                yield graph
                tqdm_bar.update(1)
            else:
                new_edge_index, k_partition = self.cd_model(graph)
                for comm_id in set(k_partition.values()):
                    yield generate_subgraph(graph, comm_id, k_partition, new_edge_index)
                tqdm_bar.update(1)

    def subgraph_matched(self, tqdm_bar):
        # generate matched subgraph_1 and subgraph_2 for one loop
        for graph in self.dataloader:
            graph['user'].batch_mask = torch.tensor([True] * graph['user'].batch_size +
                                                    [False] * (graph['user'].x.size(0) - graph['user'].batch_size))
            new_edge_index, k_partition = self.cd_model(graph)
            match_pair = self.cd_model.match_community_pair(k_partition)
            entropy_batch = []
            for comm_1, comm_2 in match_pair.items():
                subgraph_1 = generate_subgraph(graph, comm_1, k_partition, new_edge_index)
                subgraph_2 = generate_subgraph(graph, comm_2, k_partition, new_edge_index)

                # node_label_a = subgraph_1['user'].node_label
                # node_label_b = subgraph_2['user'].node_label
                # batch_user_mask_a = subgraph_1['user'].batch_mask
                # batch_user_mask_b = subgraph_2['user'].batch_mask
                # entropy_batch.append((information_entropy(node_label_a[batch_user_mask_a].to('cpu').numpy())
                #                       + information_entropy(node_label_b[batch_user_mask_b].to('cpu').numpy()))/2)
                yield subgraph_1, subgraph_2
            tqdm_bar.update(1)
            # self.entropy_list.append(entropy_batch)
            # self.comm_num.append(len(match_pair)*2)

    def get_loss_weight(self):
        node_label = torch.tensor(self.graph['user'].node_label)
        weight = torch.sum(node_label) / node_label.size(-1)
        return torch.tensor([weight, 1 - weight])

    def get_meta_data(self):
        meta_data = (self.graph.node_types, self.graph.edge_types)
        return meta_data

    def get_in_channels(self):
        in_channel = {}
        for node_type in self.graph.node_types:
            if node_type == 'tweet':
                in_channel[node_type] = self.graph[node_type].x1.size(-1)
            else:
                in_channel[node_type] = self.graph[node_type].x.size(-1)
        return in_channel
    
    def get_comm_change(self):
        return self.entropy_list, self.comm_num


def compute_loss(emb1, emb2, pred, label, split, tau, alpha=0.01, beta=1):
    # cl_loss = 0                                                                                 # 无对比损失
    cl_loss = unsupervised_cl_loss(emb1, emb2, split, tau)                                    # 无监督的对比损失函数
    # cl_loss = traditional_cl_loss(emb1, emb2, label, split, tau)                              # 传统的有监督对比损失函数
    # cl_loss = compute_pro_loss(emb1, emb2, label, split, tau)                                   # 改进的对比损失
    if pred is not None:
        train_mask = split == 0
        # bot_num = torch.sum(label[train_mask] == 1)
        # human_num = torch.sum(label[train_mask] == 0)
        # weight = torch.tensor([bot_num/(bot_num+human_num), human_num/(bot_num+human_num)]).to(pred.device)
        # weight = torch.tensor([2, 5], dtype=torch.float).to(pred.device)
        # pred_loss = cross_entropy(pred[train_mask], label[train_mask], weight=weight)
        pred_loss = cross_entropy(pred[train_mask], label[train_mask])
        loss = alpha * cl_loss + beta * pred_loss
        # conf_matrix = confusion_matrix(label[train_mask].to('cpu'), torch.argmax(pred[train_mask].to('cpu'), dim=1))
        # print(conf_matrix)
        return loss
    else:
        return cl_loss


def compute_cross_view_loss_experiment(emb_a_1, emb_a_2, emb_b_1, emb_b_2, pred_a, pred_b,
                                       label_a, label_b, tau, alpha, beta, mean_flag=True):
    contrastive_loss_a, p_sim_a, n_sim_a = compute_cross_individual_loss(emb_a_1, emb_a_2, emb_b_1, emb_b_2,
                                                                         pred_a, label_a, label_b, tau)
    contrastive_loss_b, p_sim_b, n_sim_b = compute_cross_individual_loss(emb_b_1, emb_b_2, emb_a_1, emb_a_2,
                                                                         pred_b, label_b, label_a, tau)

    contrastive_loss = (contrastive_loss_a + contrastive_loss_b) / 2
    # weight = torch.tensor([2, 5], dtype=torch.float).to(emb_a_1.device)       # twibot-22
    # weight = torch.tensor([3, 2], dtype=torch.float).to(emb_a_1.device)       # cresci-15
    # prediction_loss = cross_entropy(pred_a, label_a, weight=weight)
    prediction_loss = cross_entropy(pred_a, label_a)

    total_loss = alpha * contrastive_loss + beta * prediction_loss

    norms = torch.norm(emb_a_1, dim=1, keepdim=True)
    unit_vectors_a = emb_a_1 / norms
    similarity_matrix = torch.matmul(unit_vectors_a, unit_vectors_a.T)
    sim_a = (torch.sum(similarity_matrix) - similarity_matrix.shape[0]) / (similarity_matrix.shape[0]
                                                                           * (similarity_matrix.shape[0]-1))

    norms = torch.norm(emb_b_1, dim=1, keepdim=True)
    unit_vectors_b = emb_a_1 / norms
    similarity_matrix = torch.matmul(unit_vectors_b, unit_vectors_b.T)
    sim_b = (torch.sum(similarity_matrix) - similarity_matrix.shape[0]) / (similarity_matrix.shape[0]
                                                                           * (similarity_matrix.shape[0]-1))

    similarity_matrix = torch.matmul(unit_vectors_a, unit_vectors_b.T)
    sim_ab = (torch.sum(similarity_matrix) - similarity_matrix.shape[0]) / (similarity_matrix.shape[0]
                                                                           * (similarity_matrix.shape[0]-1))

    return total_loss, (p_sim_a + p_sim_b)/2, (n_sim_a + n_sim_b)/2, (sim_a + sim_b)/2, sim_ab


def compute_cross_view_loss(emb_a_1, emb_a_2, emb_b_1, emb_b_2, pred_a, pred_b,
                            label_a, label_b, tau, alpha, beta, mean_flag=True):
    if mean_flag:
        contrastive_loss_a = compute_cross_mean_view_loss(emb_a_1, emb_a_2, emb_b_1, emb_b_2,
                                                          pred_a, label_a, label_b, tau)
        contrastive_loss_b = compute_cross_mean_view_loss(emb_b_1, emb_b_2, emb_a_1, emb_a_2,
                                                          pred_b, label_b, label_a, tau)
    else:
        contrastive_loss_a, _, _ = compute_cross_individual_loss(emb_a_1, emb_a_2, emb_b_1, emb_b_2,
                                                                             pred_a, label_a, label_b, tau)
        contrastive_loss_b, _, _ = compute_cross_individual_loss(emb_b_1, emb_b_2, emb_a_1, emb_a_2,
                                                                             pred_b, label_b, label_a, tau)

    contrastive_loss = (contrastive_loss_a + contrastive_loss_b) / 2
    # weight = torch.tensor([2, 5], dtype=torch.float).to(emb_a_1.device)       # twibot-22
    # weight = torch.tensor([3, 2], dtype=torch.float).to(emb_a_1.device)       # cresci-15
    # prediction_loss = cross_entropy(pred_a, label_a, weight=weight)
    prediction_loss = cross_entropy(pred_a, label_a)

    total_loss = alpha * contrastive_loss + beta * prediction_loss

    return total_loss


def adaptive_augment(graph: HeteroData, drop_feature_rate=0.2):
    augment_graph1 = graph
    augment_graph2 = deepcopy(graph)

    # Link Prediction for edge augmentation
    for edge_type in graph.edge_types:
        if hasattr(graph[edge_type], 'augmented_edge_index'):
            augment_graph2[edge_type].edge_index = augment_graph2[edge_type].augmented_edge_index
            del augment_graph1[edge_type].augmented_edge_index, augment_graph2[edge_type].augmented_edge_index

    # Synonymy Substitution for text augmentation
    for node_type in graph.node_types:
        if node_type == 'tweet':
            augment_graph1['tweet'].x = augment_graph1['tweet'].x1
            augment_graph2['tweet'].x = augment_graph2['tweet'].x2
            # augment_graph2['tweet'].x = augment_graph2['tweet'].x1
            del augment_graph1['tweet'].x1, augment_graph1['tweet'].x2
            del augment_graph2['tweet'].x1, augment_graph2['tweet'].x2

    # Node Feature Shifting based on PageRank
    augmented_edge_type = []
    for edge_type in graph.edge_types:
        if edge_type[0] == edge_type[2] and (edge_type[0] not in augmented_edge_type) \
                and augment_graph2[edge_type].edge_index.numel() > 0:
            node_page_rank = compute_page_rank(augment_graph2[edge_type].edge_index,
                                               augment_graph2[edge_type[0]].x.shape[0])
            feature_weight = feature_drop_weights(augment_graph2[edge_type[0]].x, node_c=node_page_rank)
            augment_graph2[edge_type[0]].x = drop_feature(augment_graph2[edge_type[0]].x,
                                                          feature_weight, drop_feature_rate)
            augmented_edge_type.append(edge_type[0])

    # NOTE experiment no text
    # for edge_type in graph.edge_types:
    #     if edge_type[0] == 'tweet' or edge_type[2] == 'tweet':
    #         del augment_graph1[edge_type], augment_graph2[edge_type]
    
    # NOTE experiment no metadata
    # augment_graph1['user'].x[:, :6] = 0.0
    # augment_graph2['user'].x[:, :6] = 0.0
    
    # NOTE experiment no heterogeneous
    # for edge_type in graph.edge_types:
    #     if edge_type[0] == 'tweet' or edge_type[2] == 'tweet':
    #         del augment_graph1[edge_type], augment_graph2[edge_type]
    # hetero_edge = []
    # for edge_type in graph.edge_types:
    #     if edge_type[0] == edge_type[2]:
    #         hetero_edge.append(edge_type)
    # for edge_type in hetero_edge[1:]:
    #     augment_graph1[hetero_edge[0]].edge_index = torch.cat((augment_graph1[hetero_edge[0]].edge_index, augment_graph1[edge_type].edge_index), dim=1)
    #     augment_graph2[hetero_edge[0]].edge_index = torch.cat((augment_graph2[hetero_edge[0]].edge_index, augment_graph2[edge_type].edge_index), dim=1)
    #     del augment_graph1[edge_type], augment_graph2[edge_type]

    return augment_graph1, augment_graph2


def total_tweet_augment(graph_path, augment_method=None, tweet_dim=32):
    """
    对文本进行增强，保存图，graph['tweet']第一维是原文向量，第二维是增强向量
    """
    origin_graph_path = graph_path + 'graph.pt'
    save_path = graph_path + f'tweet_augment_{augment_method}_{tweet_dim}_graph.pt'
    if not os.path.exists(save_path):
        graph = torch.load(origin_graph_path)
        torch.save(tweet_augment(graph, augment_method, tweet_dim=tweet_dim), save_path)
    new_graph = torch.load(save_path)
    return new_graph


def generate_subgraph(graph: HeteroData, comm_id, partition: dict, new_user_edge_index=None):
    """
    根据社区分类进行分割子图
    @param graph: 原图
    @param comm_id: 社区编号
    @param partition: 社区检测结果{node_id: comm_id}
    @param new_user_edge_index: 增强的用户之间的边
    @return: 子图
    """

    def generate_edge_index(edge_index):
        for source, target in edge_index:
            if partition.get(source) == comm_id or partition.get(target) == comm_id:
                if node_mask[edge_type[0]][source] is None:
                    node_mask[edge_type[0]][source] = node_num[edge_type[0]]
                    node_num[edge_type[0]] += 1
                sub_edge_index[0].append(node_mask[edge_type[0]][source])
                if node_mask[edge_type[2]][target] is None:
                    node_mask[edge_type[2]][target] = node_num[edge_type[2]]
                    node_num[edge_type[2]] += 1
                sub_edge_index[1].append(node_mask[edge_type[2]][target])

    node_types = graph.node_types
    edge_types = graph.edge_types

    node_mask = {node_type: [None] * graph[node_type].x.size(-2) if node_type != 'tweet' else
                 [None] * graph[node_type].x1.size(-2) for node_type in node_types}
    node_num = {node_type: 0 for node_type in node_types}
    subgraph = HeteroData()
    for edge_type in edge_types:
        sub_edge_index = [[], []]
        generate_edge_index(graph[edge_type].edge_index.t().numpy())
        subgraph[edge_type].edge_index = torch.tensor(sub_edge_index, dtype=torch.long)
        if edge_type[0] == edge_type[2] and new_user_edge_index is not None:
            generate_edge_index(new_user_edge_index.t().numpy())
            subgraph[edge_type].augmented_edge_index = torch.tensor(sub_edge_index, dtype=torch.long)
            subgraph[edge_type].augmented_edge_index = add_self_loop(subgraph[edge_type].augmented_edge_index,
                                                                     node_num[edge_type[0]])
    for node_type in node_types:
        trans = [[ind, mask] for ind, mask in enumerate(node_mask[node_type]) if mask is not None]
        sample_ind = [0] * len(trans)
        for tran in trans:
            sample_ind[tran[1]] = tran[0]
        if node_type == 'tweet':
            subgraph[node_type].x1 = graph[node_type].x1[sample_ind]
            subgraph[node_type].x2 = graph[node_type].x2[sample_ind]
        else:
            subgraph[node_type].x = graph[node_type].x[sample_ind]
        if node_type == 'user':
            subgraph[node_type].batch_mask = graph[node_type].batch_mask[sample_ind]
        if hasattr(graph[node_type], 'node_label'):
            subgraph[node_type].node_label = graph[node_type].node_label[sample_ind]
        if hasattr(graph[node_type], 'node_split'):
            subgraph[node_type].node_split = graph[node_type].node_split[sample_ind]
    return subgraph
