import random
import networkx as nx
import numpy as np
from torch_geometric.data import HeteroData
import torch
from community import best_partition
from sklearn.cluster import KMeans
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from utils import compress_graph


def load_user_graph(graph: HeteroData):
    """
    考虑用户之间的关系，提取用户子图，并将图转为无向图
    @param graph: torch异质图格式的图
    @return: 用户邻接稀疏矩阵(COO格式)，用户的特征向量(n*m torch.tensor)
    """
    total_edge_index = torch.tensor([[], []])
    for edge_type in graph.edge_types:
        if edge_type[0] != edge_type[2]:
            continue
        user_edge_index = torch.cat((graph[edge_type].edge_index, graph[edge_type].edge_index[[1, 0]]), dim=1)
        total_edge_index = torch.cat((total_edge_index, user_edge_index), dim=1)
    user_feature = graph['user'].x
    user_num = user_feature.shape[0]
    # torch格式的稀疏矩阵，去除重复边，并将value置为1
    torch_user_adj = torch.sparse_coo_tensor(total_edge_index, torch.ones(total_edge_index.size(1)),
                                             (user_num, user_num)).coalesce()
    torch_user_adj = torch.sparse_coo_tensor(torch_user_adj.indices(), torch.ones(torch_user_adj.values().size()),
                                             (user_num, user_num)).coalesce()
    return torch_user_adj, user_feature, user_num


def louvain_cluster(torch_user_adj, s_rec):
    """
    @param torch_user_adj: torch格式的稀疏矩阵
    @param s_rec: 稀疏化系数（度数阈值，大于则需要稀疏）
    @return: 稀疏化的社区邻接矩阵（在同一个社区为1），社区数量，节点的社区分类
    """
    sp_user_adj = sp.coo_matrix((torch_user_adj.values(), (torch_user_adj.indices()[0], torch_user_adj.indices()[1])),
                                shape=torch_user_adj.shape)
    partition = best_partition(nx.from_scipy_sparse_array(sp_user_adj))  # 进行社区检测
    communities_louvain = list(partition.values())

    num_communities_louvain = max(communities_louvain) + 1  # 社区数量

    communities_louvain_onehot = torch.eye(num_communities_louvain)[communities_louvain].to_sparse()
    # n*n社区邻接矩阵，两点在同一社区时为1
    adj_louvain = torch.sparse.mm(communities_louvain_onehot, communities_louvain_onehot.t())
    # 移除对角自环
    adj_louvain = adj_louvain - torch.sparse.spdiags(torch.ones(adj_louvain.shape[0]), torch.tensor(0),
                                                     adj_louvain.shape)

    # s_adj_louvain = sparsification(adj_louvain.coalesce(), s_rec)  # 抽样后的社区邻接矩阵
    s_adj_louvain = adj_louvain.coalesce()
    return s_adj_louvain, num_communities_louvain, partition


def sparsification(adj_louvain, s=5):
    """
    对louvain_cluster产生的模块计量邻接矩阵进行稀疏化（抽样）
    @param adj_louvain: n*n社区邻接矩阵
    @param s: 抽样阈值，邻居数大于s时进行抽样
    @return: 抽样后的社区邻接矩阵torch格式
    """
    # 节点个数
    n = adj_louvain.shape[0]
    # 计算每个节点的度数，竖加
    degrees = torch.sparse.sum(adj_louvain, dim=0).to_dense()
    s_adj_louvain = [[], []]
    indices = adj_louvain.indices()
    front = 0
    back = 0
    for i in range(n):
        for j in range(front, n):
            if indices[0, j] != indices[0, back]:
                back = j
                break
        edges = indices[1, front:back]
        nonzero_degree = degrees[edges]
        if len(edges) > 0:
            choice_edges = edges[torch.multinomial(nonzero_degree, min(len(edges), s), replacement=False)]
            s_adj_louvain[0].append([i]*len(choice_edges) + choice_edges.tolist())
            s_adj_louvain[1].append(choice_edges.tolist() + [i]*len(choice_edges))
        front = back
    s_adj_louvain = torch.sparse_coo_tensor(indices=torch.tensor(s_adj_louvain),
                                            values=torch.ones(len(s_adj_louvain[0])), size=(n, n)).coalesce()
    s_adj_louvain = torch.sparse_coo_tensor(s_adj_louvain.indices(), torch.ones(s_adj_louvain.values().size()),
                                            (n, n)).coalesce()

    return s_adj_louvain


def preprocess_adj(torch_user_adj, num_user):
    """
    归一化处理邻接矩阵，避免邻接矩阵和特征矩阵内积相乘改变特征原本的分布
    @param torch_user_adj: torch格式的稀疏化邻接矩阵
    @param num_user: 用户节点个数
    @return: 归一化的稀疏邻接矩阵
    """
    # 对角矩阵
    adj_diag = torch.sparse_coo_tensor(torch.tensor([range(torch_user_adj.size(0))]).repeat(2, 1),
                                       torch.ones(torch_user_adj.size(0)), (num_user, num_user))
    torch_user_adj = torch_user_adj + adj_diag
    # 计算度矩阵
    degree_matrix = torch.sparse.sum(torch_user_adj, dim=0).to_dense()
    # 计算度矩阵的逆平方根
    degree_matrix_sqrt_inv = torch.pow(degree_matrix, -0.5)
    # 将度矩阵逆平方根转换为对角矩阵
    degree_matrix_sqrt_inv = torch.diag(degree_matrix_sqrt_inv).to_sparse_coo()
    # 归一化邻接矩阵
    normalized_adj = torch.sparse.mm(torch.sparse.mm(degree_matrix_sqrt_inv, torch_user_adj), degree_matrix_sqrt_inv)

    return normalized_adj


def preprocess_degree(torch_user_adj):
    # 计算每个节点的度
    degrees = torch.sparse.sum(torch_user_adj, dim=0).to_dense().unsqueeze(dim=1)
    edge_num = torch_user_adj.values().size(0)
    deg_matrix = (1.0 / edge_num) * torch.matmul(degrees, degrees.t())
    # 将度矩阵转换为稀疏矩阵
    deg_matrix_sparse = deg_matrix.to_sparse()

    return deg_matrix_sparse


def node_sample(torch_user_adj, threshold):
    """
    返回抽取的节点索引、抽取节点构成的稀疏邻接矩阵
    @param torch_user_adj: torch.sparse_coo_tensor格式用户邻接矩阵
    @param threshold: 抽取节点个数阈值
    @return:
    """
    node_probability = torch.sparse.sum(torch_user_adj, dim=0).to_dense().squeeze()
    sample_node_num = min(int(threshold + (torch_user_adj.shape[0] - threshold) / 10), 10000)
    sample_node = torch.multinomial(node_probability, sample_node_num, replacement=False)
    sample_user_adj = torch_user_adj.to_dense()[sample_node, :][:, sample_node]

    nonzero_indices = torch.nonzero(sample_user_adj).t()
    sample_user_adj = torch.sparse_coo_tensor(nonzero_indices, torch.ones(nonzero_indices.size(1)),
                                              (sample_node_num, sample_node_num)).coalesce()
    nonzero_indices = sample_user_adj.indices()
    sample_user_adj = torch.sparse_coo_tensor(nonzero_indices, torch.ones(nonzero_indices.size(1)),
                                              (sample_node_num, sample_node_num)).coalesce()
    return sample_node, sample_user_adj


def preprocess_val_graph(graph: HeteroData, val_ratio):
    """
    只保留图中关于user部分，并返回用于验证link prediction的两幅图
    @param graph: 原始完整图
    @param val_ratio: 用于预测的边的数量
    """
    val_graph = HeteroData()
    val_graph['user'].x = graph['user'].x
    true_edge = torch.tensor([[], []])
    for edge_type in graph.edge_types:
        if edge_type[0] == edge_type[2]:  # user->user
            edge_index = graph[edge_type].edge_index
            val_edge_index_ind = set(random.sample(range(edge_index.size(1)),
                                                   int(val_ratio * edge_index.size(1))))
            train_edge_index = []
            val_edge_index = []
            for ind, edge in enumerate(edge_index.t().tolist()):
                if ind in val_edge_index_ind:
                    val_edge_index.append(edge)
                else:
                    train_edge_index.append(edge)
            val_graph[edge_type].edge_index = torch.tensor(train_edge_index).t()
            true_edge = torch.cat((true_edge, torch.tensor(val_edge_index).t()), dim=1)
    true_adj = torch.zeros(val_graph['user'].x.size(0), val_graph['user'].x.size(0))
    for i, j in true_edge.t().int():
        true_adj[i, j] = 1

    return val_graph, true_adj


def pairwise_distance(x, gamma, epsilon=0.1):
    """
    节点对的欧式距离
    @param x: n*d 向量矩阵
    @param gamma: 调控系数
    @param epsilon: 维稳小量
    @return: n*n欧式距离矩阵
    """
    x1 = torch.sum(x * x, dim=1, keepdim=True)
    # x2 = torch.matmul(x, x.t())

    return torch.exp(- gamma * (x1 - 2 * torch.matmul(x, x.t()) + x1.t() + epsilon))


def split_large_partition(partition, node_num_threshold=5000):
    """
    对节点数量较大的图进行划分
    @param partition: 节点社区聚类结果字典{node_ind: community_id}
    @param node_num_threshold: 社区最大节点个数
    @return: 社区聚类字典{node_ind: community_id}
    """

    def split_label(comm, label, m_label):
        node_number = len(comm[label])
        comm[m_label] = comm[label][node_number // 2:]
        comm[label] = comm[label][:node_number // 2]
        m_label += 1
        mm_label = m_label
        if len(comm[label]) > node_num_threshold:
            comm, mm_label = split_label(comm, label, m_label)
        if len(comm[mm_label - 1]) > node_num_threshold:
            comm, mm_label = split_label(comm, mm_label - 1, mm_label)
        return comm, mm_label

    max_label = max(list(partition.values())) + 1
    communities = {label: [] for label in set(partition.values())}
    for node, label in partition.items():
        communities[label].append(node)
    label_list = list(communities.keys())
    for label in label_list:
        node_num = len(communities[label])
        if node_num > node_num_threshold:
            communities, max_label = split_label(communities, label, max_label)
    new_partition = {}
    for label in communities.keys():
        for node_ind in communities[label]:
            new_partition[node_ind] = label
    return new_partition


def clusters(embedding, edge_index, num_communities, cluster, ensure_comm_num):
    assert cluster in ['randomwalk', 'k_guide', 'kmeans', 'hierachical'], f'wrong cluster {cluster}'
    partition = {}
    if cluster == 'randomwalk':
        partition = random_walk_cluster(embedding, edge_index, num_communities)
    elif cluster == 'k_guide':
        partition = k_guide_cluster(embedding, edge_index, num_communities)
    elif cluster == 'kmeans':
        partition = k_means_cluster(embedding, num_communities)
    elif cluster == 'hierachical':
        partition = hierarchical_cluster(embedding, num_communities)

    if ensure_comm_num:
        if len(set(partition.values())) <= 1:
            partition = split_large_partition(partition, node_num_threshold=len(partition)//2)
    return partition


def k_means_cluster(embedding: np.array, num_communities):
    """
    利用K近邻算法进行社区检测（欧氏距离等价于余弦距离）
    @param embedding: 节点特征向量
    @param num_communities: 分割的社区数，由louvain算法指定
    @return: 社区聚类字典{node_ind: community_id}
    """
    cluster_labels = KMeans(n_clusters=num_communities, init='k-means++',
                            n_init=200, max_iter=500).fit(embedding).labels_

    partition = {node: label for node, label in enumerate(cluster_labels)}

    return split_large_partition(partition)


def hierarchical_cluster(embedding: np.array, num_communities):
    """
    利用层次聚类进行社区检测（使用离差平方链接构建聚类树，根据单调准则切割聚类树）
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
    @param embedding: 节点特征向量
    @param num_communities: 最大社区个数/控制单调准则切割标准的一个参数，越小要求簇类离散度越低
    @return: 划分的社区
    """
    # linkage_matrix = linkage(embedding, method='ward')
    # cosine_similarity_matrix = 1 - pdist(embedding, metric='cosine')
    # mr = np.max(cosine_similarity_matrix) - cosine_similarity_matrix
    # cluster_labels = fcluster(linkage_matrix, t=c, criterion='monocrit', monocrit=mr)

    distance_matrix = pdist(embedding, metric='cosine')
    linkage_matrix = linkage(distance_matrix, method='single')
    cluster_labels = fcluster(linkage_matrix, num_communities, criterion='maxclust')
    partition = {node: label for node, label in enumerate(cluster_labels)}

    return split_large_partition(partition)


def k_guide_cluster(embedding: torch.tensor, edge_index, num_communities):
    def cosine_similarity(emb1, emb2):
        if torch.norm(emb1) == 0 or torch.norm(emb2) == 0:
            return 0
        return torch.dot(emb1, emb2) / (torch.norm(emb1) * torch.norm(emb2))

    similarities = []
    for edge_ in edge_index.t():
        similarities.append(cosine_similarity(embedding[edge_[0]], embedding[edge_[1]]))
    similarities = torch.tensor(similarities)
    indices = torch.argsort(similarities, descending=True)
    cluster_labels = {node: node for node in range(embedding.size(0))}
    edge_index_list = edge_index.t().numpy()
    label_same = {node: {node} for node in range(embedding.size(0))}
    for ind in indices.tolist():
        edge = edge_index_list[ind]
        if cluster_labels[edge[0]] != cluster_labels[edge[1]]:
            label_same[cluster_labels[edge[0]]] = label_same[cluster_labels[edge[0]]].union(
                label_same[cluster_labels[edge[1]]])
            label1 = cluster_labels[edge[1]]
            for node in label_same[label1]:
                cluster_labels[node] = cluster_labels[edge[0]]
            del label_same[label1]
        if len(label_same) <= num_communities:
            break
    # 处理仍未进入社区的节点
    community_ids = list(label_same.keys())
    for node in cluster_labels.keys():
        if len(label_same[cluster_labels[node]]) == 1:
            indices = (edge_index == node).nonzero()
            if indices.size(0) == 0:
                cluster_labels[node] = random.choice(community_ids)
                continue
            max_edge = torch.argmax(similarities[indices[:, 1]])
            belong_to_label = cluster_labels[int(indices[max_edge, indices[max_edge, 0]])]
            cluster_labels[node] = belong_to_label if len(label_same[belong_to_label]) != 1 \
                else random.choice(community_ids)
    _, new_partition = compress_graph(community_ids=community_ids, partition=cluster_labels, node_num_threshold=200)
    return split_large_partition(new_partition, 1000)


def random_walk_cluster(embedding: torch.tensor, edge_index, num_communities):
    """
    基于随机游走进行社区划分
    @param embedding: 节点特征向量
    @param edge_index: 边矩阵，2*n
    @param num_communities: louvain算法给出的参考社区数量
    """

    def cosine_similarity(emb1, emb2):
        if torch.norm(emb1) == 0 or torch.norm(emb2) == 0:
            return 0
        return torch.dot(emb1, emb2) / (torch.norm(emb1) * torch.norm(emb2))

    def random_walk_by_similarity(max_label_, threshold_):
        for similarity, edge in zip(similarities, edge_index.t().numpy()):
            if similarity >= threshold_ and edge[0] != edge[1]:
                if cluster_labels[edge[0]] is None and cluster_labels[edge[1]] is None:
                    cluster_labels[edge[0]] = max_label_
                    cluster_labels[edge[1]] = max_label_
                    label_same[max_label_] = set()
                    label_same[max_label_].add(edge[0])
                    label_same[max_label_].add(edge[1])
                    max_label_ += 1
                elif cluster_labels[edge[0]] is None and cluster_labels[edge[1]] is not None:
                    cluster_labels[edge[0]] = cluster_labels[edge[1]]
                    label_same[cluster_labels[edge[1]]].add(edge[0])
                elif cluster_labels[edge[0]] is not None and cluster_labels[edge[1]] is None:
                    cluster_labels[edge[1]] = cluster_labels[edge[0]]
                    label_same[cluster_labels[edge[0]]].add(edge[1])
                elif cluster_labels[edge[0]] != cluster_labels[edge[1]]:
                    label2 = cluster_labels[edge[1]]
                    for node in label_same[label2]:
                        cluster_labels[node] = cluster_labels[edge[0]]
                    label_same[cluster_labels[edge[0]]] = (label_same[cluster_labels[edge[0]]]).union(
                        label_same[label2])
                    del label_same[label2]
        return max_label_

    similarities = []
    for edge_ in edge_index.t():
        similarities.append(cosine_similarity(embedding[edge_[0]], embedding[edge_[1]]))
    similarities = torch.tensor(similarities)
    threshold = (lambda up, down: up - (up - down) / num_communities)(torch.max(similarities), torch.min(similarities))
    cluster_labels = {node: None for node in range(embedding.size(0))}
    max_label = 0
    label_same = {}
    for i in range(1):
        max_label = random_walk_by_similarity(max_label, threshold)
        if len(label_same) <= num_communities:
            break
        else:
            threshold = (lambda up, down: up - (i + 2) * (up - down) / num_communities)(
                torch.max(similarities), torch.min(similarities))
    # 处理仍未进入社区的节点
    community_ids = list(label_same.keys())
    for node in cluster_labels.keys():
        if cluster_labels[node] is None:
            indices = (edge_index == node).nonzero()
            if indices.size(0) == 0:
                cluster_labels[node] = random.choice(community_ids)
                continue
            max_edge = torch.argmax(similarities[indices[:, 1]])
            belong_to_label = cluster_labels[int(indices[max_edge, indices[max_edge, 0]])]
            cluster_labels[node] = belong_to_label if belong_to_label is not None else random.choice(community_ids)

    _, new_partition = compress_graph(community_ids=community_ids, partition=cluster_labels, node_num_threshold=200)
    return split_large_partition(new_partition, 1000)


def init_weight(in_dim, out_dim):
    """
    Xavier初始化权重
    """
    init_range = torch.sqrt(torch.tensor(6.0 / (in_dim + out_dim)))
    weight = torch.rand(in_dim, out_dim, dtype=torch.float32) * 2 * init_range - init_range
    return weight
