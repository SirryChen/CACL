import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import normalize
from torch_geometric.nn.conv import HeteroConv, HGTConv, RGCNConv, FastRGCNConv, GATConv, SAGEConv
from torch_geometric.nn.norm import BatchNorm
import numpy as np
from sklearn.metrics import confusion_matrix
from torch_geometric.data import HeteroData
from torch.nn.functional import cosine_similarity, cross_entropy
from copy import deepcopy
from utils import drop_feature, feature_drop_weights, compute_page_rank, add_self_loop, tweet_augment


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
        for HGT, batch_norm in zip(self.Hetero_Conv_list, self.batch_norms):
            node_embedding = HGT(node_embedding, edge_index)
            node_embedding = {node_type: batch_norm(embedding) for node_type, embedding in node_embedding.items()}

        node_embedding = {node_type: torch.cat((embedding, node_feature[node_type]), dim=1)
                          for node_type, embedding in node_embedding.items()}
        return node_embedding

    def choice_basic_model(self, in_channels, out_channels, metadata):
        if self.model == "HGT":
            # return HGTConv(in_channels=in_channels, out_channels=out_channels, metadata=metadata, heads=2)
            return HGTConv(in_channels=in_channels, out_channels=out_channels, metadata=metadata, heads=1)
        elif self.model == "GAT":
            return HeteroConv({edge_type: GATConv((-1, -1), out_channels) for edge_type in metadata[1]}, aggr='sum')
        elif self.model == "SAGE":
            return HeteroConv({edge_type: SAGEConv((-1, -1), out_channels) for edge_type in metadata[1]}, aggr='sum')

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
        self.Hetero_Conv_list[0].load_state_dict(weight, strict=False)


# 创建HGcnCLModel模型
class HGcnCLModel(nn.Module):
    def __init__(self, encoder, projector, classifier=None):
        super(HGcnCLModel, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.classifier = classifier

    def forward(self, node_features):
        node_embeddings = self.encoder(node_features)
        node_projections = self.projector(node_embeddings)
        if self.classifier is not None:
            node_predicts = self.classifier(node_embeddings['user'])
            return node_projections, node_predicts
        else:
            return node_projections, None

    def update_cd_model(self):
        return self.encoder.Hetero_Conv_list[0].state_dict()


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

    def __len__(self):
        return len(self.dataloader)

    def subgraph(self, tqdm_bar):
        for graph in self.dataloader:
            graph['user'].batch_mask = torch.tensor([True] * graph['user'].batch_size +
                                                    [False] * (graph['user'].x.size(0) - graph['user'].batch_size))
            if self.cd_model is None:
                yield graph
                tqdm_bar.update(1)
            else:
                new_edge_index, k_partition = self.cd_model(graph)
                for subgraph in generate_subgraph(graph, k_partition, new_edge_index):
                    yield subgraph
                tqdm_bar.update(1)

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


def compute_cl_loss(emb1, emb2, label: torch.tensor, split: torch.tensor, tau):
    def trans(x, t=tau):
        return torch.exp(x / t)

    def cos_loss(node_emb, node_emb_opposite, node_emb_diff_label):
        self_loss = trans(cosine_similarity(node_emb.unsqueeze(0), node_emb_opposite.unsqueeze(0)).squeeze(0))
        if len(node_emb_diff_label) != 0:
            between_loss = torch.sum(trans(torch.matmul(node_emb, node_emb_diff_label.t())
                                           / (torch.norm(node_emb) * torch.norm(node_emb_diff_label, dim=1))))
        else:
            between_loss = torch.tensor(1e-5)
        loss_ = self_loss / (self_loss + between_loss)
        return loss_

    label[split != 0] = 2           # 分离训练集
    bot_emb1 = emb1[label == 1]
    human_emb1 = emb1[label == 0]
    bot_emb2 = emb2[label == 1]
    human_emb2 = emb2[label == 0]
    total_loss = []
    for node in range(label.size(0)):
        node1_emb = emb1[node]
        node2_emb = emb2[node]
        if label[node] == 1:
            loss1 = cos_loss(node1_emb, emb2[node], human_emb2)
            loss2 = cos_loss(node2_emb, emb1[node], human_emb1)
            loss = (loss1 + loss2) / 2
        elif label[node] == 0:
            loss1 = cos_loss(node1_emb, emb2[node], bot_emb2)
            loss2 = cos_loss(node2_emb, emb1[node], bot_emb1)
            loss = (loss1 + loss2) / 2
        else:
            loss1 = cos_loss(node1_emb, emb2[node], torch.cat((emb2[:node], emb2[node+1:]), dim=0))
            loss2 = cos_loss(node2_emb, emb1[node], torch.cat((emb1[:node], emb1[node+1:]), dim=0))
            loss = (loss1 + loss2) / 2
        total_loss.append(- torch.log(loss))
    average_loss = sum(total_loss) / len(total_loss)

    return average_loss


def traditional_cl_loss(emb1, emb2, label, split, tau):
    emb1 = normalize(emb1)
    emb2 = normalize(emb2)
    all_sim = torch.exp(torch.matmul(emb1, emb2.t()) / tau)
    label[split != 0] = 2  # 分离训练集
    bot_emb1 = emb1[label == 1]
    human_emb1 = emb1[label == 0]
    bot_emb2 = emb2[label == 1]
    human_emb2 = emb2[label == 0]
    total_loss = []
    for node in range(label.size(0)):
        if label[node] == 0:
            loss1 = torch.sum(- torch.log(
                torch.exp(torch.matmul(emb1[node], bot_emb2.t()) / tau) / torch.sum(all_sim[node]))
                              ) / bot_emb2.size(0)
            loss2 = torch.sum(- torch.log(
                torch.exp(torch.matmul(emb2[node], bot_emb1.t()) / tau) / torch.sum(all_sim[node]))
                              ) / bot_emb1.size(0)
        elif label[node] == 1:
            loss1 = torch.sum(- torch.log(
                torch.exp(torch.matmul(emb1[node], human_emb2.t()) / tau) / torch.sum(all_sim[node]))
                              ) / human_emb2.size(0)
            loss2 = torch.sum(- torch.log(
                torch.exp(torch.matmul(emb2[node], human_emb1.t()) / tau) / torch.sum(all_sim[node]))
                              ) / human_emb1.size(0)
        else:
            continue
        loss = (loss1 + loss2) / 2
        if torch.isinf(loss):
            pass
        total_loss.append(loss)
    average_loss = sum(total_loss) / len(total_loss)

    return average_loss


def unsupervised_cl_loss(emb1, emb2, split, tau):
    def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = normalize(z1)
        z2 = normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
        refl_sim = torch.exp(sim(z1, z1) / tau)      # intra-view
        between_sim = torch.exp(sim(z1, z2) / tau)   # inter-view

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    emb1 = emb1[split == 0]
    emb2 = emb2[split == 0]
    loss1 = semi_loss(emb1, emb2)
    loss2 = semi_loss(emb2, emb1)
    total_loss = ((loss1 + loss2) * 0.5).mean()
    return total_loss


def compute_loss(emb1, emb2, pred, label: torch.tensor, split: torch.tensor, tau, alpha=0.01, beta=1):
    cl_loss = compute_cl_loss(emb1, emb2, label, split, tau)      # 改进的对比损失函数
    # cl_loss = unsupervised_cl_loss(emb1, emb2, split, tau)          # 无监督的对比损失函数
    # cl_loss = traditional_cl_loss(emb1, emb2, label, split, tau)  # 传统的有监督对比损失函数
    if pred is not None:
        train_mask = split == 0
        # bot_num = torch.sum(label[train_mask] == 1)
        # human_num = torch.sum(label[train_mask] == 0)
        # weight = torch.tensor([bot_num/(bot_num+human_num), human_num/(bot_num+human_num)]).to(pred.device)
        # weight = torch.tensor([5, 3], dtype=torch.float).to(pred.device)
        pred_loss = cross_entropy(pred[train_mask], label[train_mask], weight=None)
        loss = alpha * cl_loss + beta * pred_loss
        conf_matrix = confusion_matrix(label[train_mask].to('cpu'), torch.argmax(pred[train_mask].to('cpu'), dim=1))
        print(conf_matrix)
        return loss
    else:
        return cl_loss


# 计算节点的PageRank centrality并执行自适应增强
def adaptive_augment(graph: HeteroData, drop_feature_rate=0.2, random_co=0.02):
    augment_graph1 = deepcopy(graph)
    augment_graph2 = deepcopy(graph)
    for edge_type in graph.edge_types:
        if hasattr(graph[edge_type], 'augmented_edge_index'):
            augment_graph1[edge_type].edge_index = augment_graph1[edge_type].edge_index
            augment_graph2[edge_type].edge_index = augment_graph2[edge_type].augmented_edge_index
            del augment_graph1[edge_type].augmented_edge_index, augment_graph2[edge_type].augmented_edge_index

    # feature增强，对图1进行随机增强
    for node_type in graph.node_types:
        if node_type == 'tweet':
            augment_graph1['tweet'].x = augment_graph1['tweet'].x1
            augment_graph2['tweet'].x = augment_graph2['tweet'].x2
            del augment_graph1['tweet'].x1, augment_graph1['tweet'].x2
            del augment_graph2['tweet'].x1, augment_graph2['tweet'].x2
        elif augment_graph1[node_type].x.size(0) > 0:
            col_max, _ = torch.max(augment_graph1[node_type].x, dim=0)
            col_min, _ = torch.min(augment_graph1[node_type].x, dim=0)
            random_matrix = random_co * (torch.rand(augment_graph1[node_type].x.size()) * (col_max - col_min) + col_min)
            augment_graph1[node_type].x = augment_graph1[node_type].x + random_matrix

    # feature增强，对图2基于pagerank自适应增强
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


def generate_subgraph(graph: HeteroData, partition: dict, new_user_edge_index=None):
    """
    根据社区分类进行分割子图
    @param graph: 原图
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

    community_ids = set(partition.values())
    node_types = graph.node_types
    edge_types = graph.edge_types

    for comm_id in community_ids:
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

        yield subgraph
