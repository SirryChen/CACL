import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
from torch_geometric.nn.conv import HGTConv
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import HeteroConv, HGTConv, RGCNConv, FastRGCNConv, GATConv, SAGEConv
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from CD_utils import load_user_graph, louvain_cluster, preprocess_adj, preprocess_degree, pairwise_distance, \
    clusters, init_weight, node_sample
from utils import pr_drop_weights, drop_edge_weighted, compress_graph


def preprocess_graph(graph: HeteroData, cd_config: dict, graph_type='train'):
    torch_user_adj, user_feature, num_users = load_user_graph(graph)
    s_adj_louvain, num_communities, partition = louvain_cluster(torch_user_adj, cd_config['s_rec'])
    o_normalized_adj = preprocess_adj(torch_user_adj, num_users)
    s_normalized_adj = preprocess_adj(torch_user_adj + cd_config['lamb'] * s_adj_louvain, num_users)
    if num_users > cd_config['fastGAE_threshold'] and graph_type == 'train':
        # 节点个数过多时抽样，便于计算重构损失
        fast_gae = True
        sample_nodes, sample_adj = node_sample(torch_user_adj, cd_config['fastGAE_threshold'])
        sample_num_edge = sample_adj.values().size(0)
        sample_num_user = len(sample_nodes)
        # 计算损失权重
        pos_weights = torch.tensor(float(sample_num_user * sample_num_user - sample_num_edge) / sample_num_edge)
        # 稀疏的度矩阵
        deg_matrix_sparse = preprocess_degree(sample_adj)
    else:
        fast_gae = False
        sample_nodes = None
        sample_adj = None
        num_edges = torch_user_adj.values().size(0)
        pos_weights = torch.tensor(float(num_users * num_users - num_edges) / num_edges)
        deg_matrix_sparse = preprocess_degree(torch_user_adj)
    adj_ = (torch_user_adj, sample_adj, o_normalized_adj, s_normalized_adj)
    param_ = (num_communities, num_users, pos_weights, fast_gae)
    return adj_, param_, user_feature, sample_nodes, deg_matrix_sparse


class ModCDModel(nn.Module):
    """
    社区检测模型接口
    """
    def __init__(self, input_dim, args, cd_config, cd_device, metadata, pretrain=False):
        super(ModCDModel, self).__init__()
        self.cd_encoder_decoder = CDEncoderDecoder(input_dim, args.encoder_hidden_channel, args.output_dim, metadata,
                                                   args.basic_model)
        self.cd_config = cd_config
        self.pretrain = pretrain
        self.cluster_choice = args.cluster
        self.device = cd_device
        self.torch_user_adj = None
        self.normalized_adj = None
        self.deg_matrix_sparse = None
        self.clusters = None
        self.num_users = None
        self.pos_weights = None
        self.reconstructions = None
        self.sample_adj = None
        self.fast_gae = False
        self.sample_nodes = None

    def forward(self, graph):
        """
        @param graph: 完整图
        @return: 增强的边，划分的子图
        """
        # 提取用户子图
        adj_, param_, user_feature, self.sample_nodes, self.deg_matrix_sparse = preprocess_graph(graph, self.cd_config)
        self.torch_user_adj, self.sample_adj, o_normalized_adj, s_normalized_adj = adj_
        num_communities, self.num_users, self.pos_weights, self.fast_gae = param_

        # 传入社区检测模型
        user_feature = user_feature.to(self.device)
        self.torch_user_adj = self.torch_user_adj.to(self.device)
        s_normalized_adj = s_normalized_adj.to(self.device)
        o_normalized_adj = o_normalized_adj.to(self.device)
        if self.fast_gae:
            self.sample_nodes = self.sample_nodes.to(self.device)
            self.sample_adj = self.sample_adj.to(self.device)

        self.reconstructions, self.clusters, emb = self.cd_encoder_decoder(
            user_feature, s_normalized_adj, o_normalized_adj, self.cd_config['gamma'], self.fast_gae, self.sample_nodes)

        if not self.pretrain:
            new_edge_index = []
            edge_index = self.torch_user_adj.indices()
            edge_number = edge_index.size(1)
            edge_set = set(tuple(edge) for edge in self.torch_user_adj.indices().t().tolist())

            if self.fast_gae:
                for i, j in (torch.sigmoid(self.reconstructions) > 0.5).nonzero().tolist():
                    if (self.sample_nodes[i], self.sample_nodes[j]) not in edge_set:
                        new_edge_index.append([self.sample_nodes[i], self.sample_nodes[j]])
            else:
                for i, j in (torch.sigmoid(self.reconstructions) > 0.5).nonzero().tolist():
                    if (i, j) not in edge_set:
                        new_edge_index.append([i, j])
            new_edge_num = min(edge_number//10, len(new_edge_index))
            new_edge_index = torch.tensor(random.sample(new_edge_index, new_edge_num)).t()

            # NOTE:返回实际使用的社区分类情况，以及用于图增强的连接预测
            k_partition = clusters(emb.detach().to('cpu'), edge_index.to('cpu'), num_communities, self.cluster_choice)

            return new_edge_index, k_partition

    def compute_loss(self):
        if self.fast_gae:
            return cd_loss(self.reconstructions, self.sample_adj, self.deg_matrix_sparse.to(self.device),
                           self.clusters, self.pos_weights, self.cd_config['beta'])
        else:
            return cd_loss(self.reconstructions, self.torch_user_adj, self.deg_matrix_sparse.to(self.device),
                           self.clusters, self.pos_weights, self.cd_config['beta'])

    def compute_score(self):
        scores_dict = {}
        if self.fast_gae:
            label_ = self.sample_adj.to_dense().flatten().to('cpu')
        else:
            label_ = self.torch_user_adj.to_dense().flatten().to('cpu')
        predict_ = torch.sigmoid(self.reconstructions.detach().flatten().to('cpu'))
        scores_dict['auc'] = roc_auc_score(label_, predict_)
        scores_dict['ap'] = 0
        scores_dict['confusion_matrix'] = np.array([[0, 0], [0, 0]])
        # scores_dict['ap'] = average_precision_score(label_, predict_)
        # scores_dict['confusion_matrix'] = confusion_matrix(label_, predict_ > 0.5)
        return scores_dict

    def parameters(self, recurse: bool = True):
        return self.cd_encoder_decoder.parameters()


class CDEncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, metadata, basic_model="HGT"):
        super(CDEncoderDecoder, self).__init__()
        self.model = basic_model
        self.metadata = (['user'], [edge_type for edge_type in metadata[1] if edge_type[0] == edge_type[2]])
        self.linear = nn.Linear(input_dim, hidden_dim)
        # self.HGT = HGTConv(in_channels={'user': hidden_dim}, out_channels=hidden_dim, metadata=metadata)
        self.Hetero_Conv = self.choice_basic_model(in_channels={'user': hidden_dim}, out_channels=hidden_dim,
                                                   metadata=metadata)
        self.weight_layer1 = torch.nn.Parameter(init_weight(hidden_dim, hidden_dim))
        self.weight_layer2 = torch.nn.Parameter(init_weight(hidden_dim, output_dim))
        nn.init.xavier_uniform_(self.weight_layer1)
        nn.init.xavier_uniform_(self.weight_layer2)

        self.init_weight()

    def choice_basic_model(self, in_channels, out_channels, metadata):
        if self.model == "HGT":
            # return HGTConv(in_channels=in_channels, out_channels=out_channels, metadata=metadata, heads=2)
            return HGTConv(in_channels=in_channels, out_channels=out_channels, metadata=metadata, heads=1)
        elif self.model == "GAT":
            return HeteroConv({edge_type: GATConv((-1, -1), out_channels) for edge_type in metadata[1]
                               if edge_type[0] == edge_type[2]}, aggr='sum')
        elif self.model == "SAGE":
            return HeteroConv({edge_type: SAGEConv((-1, -1), out_channels) for edge_type in metadata[1]
                               if edge_type[0] == edge_type[2]}, aggr='sum')

    def update_conv_layer(self, param):
        self.Hetero_Conv.load_state_dict(param, strict=False)

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data)

    def forward(self, user_feature, s_normalized_adj, o_normalized_adj, gamma, fast_gae=False, sample_user=None):
        user_feature = self.linear(user_feature)
        x_dict = {'user': user_feature}
        edge_dict = {edge_type: o_normalized_adj.indices() for edge_type in self.metadata[1]}
        user_feature = self.Hetero_Conv(x_dict, edge_dict)['user']
        # graph encoder
        z_hidden = torch.sparse.mm(s_normalized_adj, torch.mm(user_feature, self.weight_layer1))
        z_mean = torch.sparse.mm(o_normalized_adj, torch.mm(z_hidden, self.weight_layer2))

        if fast_gae:
            sample_z_mean = z_mean[sample_user, :]
            # graph decoder
            reconstructions = torch.matmul(sample_z_mean, sample_z_mean.t())
            # modularity inspired decoder
            clusters_distance = pairwise_distance(sample_z_mean, gamma)

        else:
            # graph decoder
            reconstructions = torch.matmul(z_mean, z_mean.t())
            # modularity inspired decoder
            clusters_distance = pairwise_distance(z_mean, gamma)

        return reconstructions, clusters_distance, z_mean


def cd_loss(predicts, labels, degree_matrix, clusters_distance, pos_weights, beta):
    """
    计算损失
    @param predicts: 重构的邻接矩阵，n*n tensor密集矩阵
    @param labels: 原始的邻接矩阵，稀疏矩阵
    @param degree_matrix: 稀疏的度矩阵
    @param clusters_distance:
    @param pos_weights: 应对不平衡的边的情况，使得存在边的损失在训练中被加权，以处理类别不平衡
    @param beta: 模块度损失权重系数
    @return:
    """
    node_num = labels.values().size(0)
    labels = torch.flatten(labels.to_dense())
    degree_matrix = torch.flatten(degree_matrix.to_dense())
    clusters_distance = torch.flatten(clusters_distance)
    # 重构损失
    reconstruct_loss = func.binary_cross_entropy_with_logits(predicts.flatten(),
                                                             labels,
                                                             pos_weight=pos_weights)
    # conf_matrix = confusion_matrix(labels.to('cpu'), torch.sigmoid(predicts.flatten().detach().to('cpu')) > 0.5)
    # print(conf_matrix)
    # 模块度损失
    modularity_loss = (1.0 / node_num) * torch.sum((labels - degree_matrix) * clusters_distance)

    all_loss = reconstruct_loss - beta * modularity_loss
    return all_loss


class TraditionalModel:
    def __init__(self, method='louvain'):
        self.method = method

    def __call__(self, graph: HeteroData):
        torch_user_adj, user_feature, num_users = load_user_graph(graph)
        if self.method == 'louvain':
            _, community_num, k_partition = louvain_cluster(torch_user_adj, None)
        else:
            _, num_communities, _ = louvain_cluster(torch_user_adj, None)
            k_partition = clusters(graph['user'].x, torch_user_adj.indices(), num_communities, self.method)
        _, k_partition = compress_graph(set(k_partition.values()), k_partition, node_num_threshold=1000)
        drop_weights = pr_drop_weights(torch_user_adj.indices(), num_users, aggr='sink', k=200)

        masked_edge_index = drop_edge_weighted(torch_user_adj.indices(), drop_weights, p=0.3)

        return masked_edge_index, k_partition
