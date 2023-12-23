import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import networkx as nx
import scipy.sparse as sp
from community import best_partition
from torch_geometric.data import HeteroData
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score
from CD_utils import load_user_graph, louvain_cluster, preprocess_adj, preprocess_degree, pairwise_distance, \
    clusters, init_weight
from utils import pr_drop_weights, drop_edge_weighted, compress_graph


def preprocess_graph(graph: HeteroData):
    # extract user graph
    torch_user_adj, user_feature, num_users = load_user_graph(graph)
    normalized_adj = preprocess_adj(torch_user_adj, num_users)
    deg_matrix_sparse = preprocess_degree(torch_user_adj)
    num_edges = torch_user_adj.values().size(0)
    pos_weights = torch.tensor(float(num_users * num_users - num_edges) / num_edges)

    # louvain community detection
    sp_user_adj = sp.coo_matrix((torch_user_adj.values(), (torch_user_adj.indices()[0], torch_user_adj.indices()[1])),
                                shape=torch_user_adj.shape)
    partition = best_partition(nx.from_scipy_sparse_array(sp_user_adj))
    communities_louvain = list(partition.values())
    num_communities = max(communities_louvain) + 1

    adj_ = (normalized_adj, torch_user_adj)
    param_ = (num_communities, pos_weights)

    return user_feature, adj_, deg_matrix_sparse, param_


class ModCDModel(nn.Module):
    """
    社区检测模型接口
    """
    def __init__(self, args, cd_config, cd_device, pretrain=False, ensure_comm_num=False):
        super(ModCDModel, self).__init__()
        self.cd_encoder_decoder = CDEncoderDecoder(args.encoder_hidden_channel, args.output_dim, args)
        self.cd_config = cd_config
        self.pretrain = pretrain
        self.ensure_comm_num = ensure_comm_num
        self.cluster_choice = args.cluster
        self.device = cd_device
        self.origin_adj = None
        self.deg_matrix_sparse = None
        self.reconstruct = None
        self.clusters_distance = None
        self.pos_weights = None
        self.node_emb = None

    def forward(self, graph: HeteroData):
        """
        @param graph: 完整图
        @return: 增强的边，划分的子图
        """
        user_feature, adj, self.deg_matrix_sparse, param = preprocess_graph(graph)
        normalized_adj, self.origin_adj = adj
        num_communities, self.pos_weights = param
        self.reconstruct, self.clusters_distance, self.node_emb = self.cd_encoder_decoder(user_feature.to(self.device),
                                                                                          normalized_adj.to(self.device),
                                                                                          self.cd_config['gamma'])
        if not self.pretrain:
            new_edge_index = []
            edge_index = self.origin_adj.indices()
            edge_number = edge_index.size(1)
            edge_set = set(tuple(edge) for edge in self.origin_adj.indices().t().tolist())

            for i, j in (torch.sigmoid(self.reconstruct) > 0.5).nonzero().tolist():
                if (i, j) not in edge_set:
                    new_edge_index.append([i, j])
            new_edge_num = min(edge_number//10, len(new_edge_index))
            new_edge_index = torch.tensor(random.sample(new_edge_index, new_edge_num)).t()

            # NOTE:返回实际使用的社区分类情况，以及用于图增强的连接预测
            k_partition = clusters(self.node_emb.detach().to('cpu'), edge_index.to('cpu'),
                                   num_communities, self.cluster_choice, self.ensure_comm_num)

            return new_edge_index, k_partition

    def compute_loss(self):
        return cd_loss(self.reconstruct, self.origin_adj.to(self.device), self.deg_matrix_sparse.to(self.device),
                       self.clusters_distance, self.pos_weights.to(self.device), self.cd_config['beta'])

    def compute_score(self):
        scores_dict = {}
        label_ = self.origin_adj.to_dense().flatten().to('cpu')
        predict_ = torch.sigmoid(self.reconstruct.detach().flatten().to('cpu'))
        scores_dict['f1-score'] = f1_score(label_, predict_ >= 0.5)
        scores_dict['auc'] = 0
        scores_dict['ap'] = 0
        scores_dict['confusion_matrix'] = np.array([[0, 0], [0, 0]])
        # scores_dict['auc'] = roc_auc_score(label_, predict_)
        # scores_dict['ap'] = average_precision_score(label_, predict_)
        # scores_dict['confusion_matrix'] = confusion_matrix(label_, predict_ > 0.5)
        return scores_dict

    def parameters(self, recurse: bool = True):
        return self.cd_encoder_decoder.parameters()

    def match_community_pair(self, k_partition: dict):
        """
        @param k_partition: the result of community detection
        @return: the matched communities {comm_id1: comm_id2,}
        """
        index_id_map = {index: comm_id for index, comm_id in enumerate(set(k_partition.values()))}
        id_index_map = {comm_id: index for index, comm_id in enumerate(set(k_partition.values()))}
        comm_num = len(index_id_map)
        comm_rep = torch.zeros(comm_num, self.node_emb.size(1)).to(self.device)
        node_num = torch.zeros(comm_num).to(self.device)
        for node, comm_id in k_partition.items():
            comm_rep[id_index_map[comm_id]] += self.node_emb[node]
            node_num[id_index_map[comm_id]] += 1
        comm_rep = comm_rep/node_num.reshape(-1, 1)
        comm_rep = func.normalize(comm_rep, p=2, dim=1)
        similarity_matrix = torch.mm(comm_rep, comm_rep.t())
        similarity_matrix.fill_diagonal_(-torch.inf)
        match_pair = {}
        paired_comm = []
        for i in range(comm_num):   # 删除重复的社区对
            if i not in paired_comm:
                matched_comm = torch.argmax(similarity_matrix[i, :]).item()
                match_pair[index_id_map[i]] = index_id_map[matched_comm]
                paired_comm.append(i)
                paired_comm.append(matched_comm)
        return match_pair


class CDEncoderDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, args):
        super(CDEncoderDecoder, self).__init__()
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

        self.weight_layer1 = torch.nn.Parameter(init_weight(hidden_dim, hidden_dim))
        self.weight_layer2 = torch.nn.Parameter(init_weight(hidden_dim, output_dim))

        self.init_weight()

    def update_embedding_layer(self, param):
        self.load_state_dict(param, strict=False)

    def init_weight(self):
        nn.init.xavier_uniform_(self.weight_layer1)
        nn.init.xavier_uniform_(self.weight_layer2)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data)

    def forward(self, user_feature, adj, gamma):
        num_prop = user_feature[:, :self.num_prop_size]
        cat_prop = user_feature[:, self.num_prop_size:(self.num_prop_size + self.cat_prop_size)]
        des = user_feature[:, (self.num_prop_size + self.cat_prop_size):
                              (self.num_prop_size + self.cat_prop_size + self.des_size)]
        tweet = user_feature[:, (self.num_prop_size + self.cat_prop_size + self.des_size):]
        user_features_des = self.linear_relu_des(des)
        user_features_tweet = self.linear_relu_tweet(tweet)
        user_features_numeric = self.linear_relu_num_prop(num_prop)
        user_features_bool = self.linear_relu_cat_prop(cat_prop)
        user_feature = torch.cat((user_features_numeric, user_features_bool,
                                  user_features_des, user_features_tweet), dim=1)

        # graph encoder
        z_hidden = torch.sparse.mm(adj, torch.mm(user_feature, self.weight_layer1))
        z_mean = torch.sparse.mm(adj, torch.mm(z_hidden, self.weight_layer2))

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
    # reconstruction loss
    reconstruct_loss = func.binary_cross_entropy_with_logits(predicts.flatten(), labels, pos_weight=pos_weights)
    # conf_matrix = confusion_matrix(labels.to('cpu'), torch.sigmoid(predicts.flatten().detach().to('cpu')) > 0.5)
    # print(conf_matrix)
    # modularity loss
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
