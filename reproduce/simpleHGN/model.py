from abc import ABC

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
import torch.nn.functional as F


class SimpleHGN(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels, num_edge_type, rel_dim, beta=None, final_layer=False):
        super(SimpleHGN, self).__init__(aggr="add", node_dim=0)
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.W_r = torch.nn.Linear(rel_dim, out_channels, bias=False)
        self.a = torch.nn.Linear(3 * out_channels, 1, bias=False)
        self.W_res = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.rel_emb = torch.nn.Embedding(num_edge_type, rel_dim)
        self.beta = beta
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.ELU = torch.nn.ELU()
        self.final = final_layer
        self.alpha = None

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x, edge_index, edge_type, pre_alpha=None):

        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, pre_alpha=pre_alpha)
        output = node_emb + self.W_res(x)
        output = self.ELU(output)
        if self.final:
            output = F.normalize(output, dim=1)

        return output, self.alpha.detach()

    def message(self, x_i, x_j, edge_type, pre_alpha, index, ptr, size_i):
        out = self.W(x_j)
        rel_emb = self.rel_emb(edge_type)
        alpha = self.leaky_relu(self.a(torch.cat((self.W(x_i), self.W(x_j), self.W_r(rel_emb)), dim=1)))
        alpha = softmax(alpha, index, ptr, size_i)
        if pre_alpha is not None and self.beta is not None:
            self.alpha = alpha * (1 - self.beta) + pre_alpha * self.beta
        else:
            self.alpha = alpha
        out = out * alpha.view(-1, 1)
        return out

    def update(self, aggr_out):
        return aggr_out


class SimpleHGNModel(nn.Module):
    def __init__(self, num_edge_type, args):
        super(SimpleHGNModel, self).__init__()
        embedding_dim = args.embedding_dim
        num_layer = args.num_layer
        self.num_prop_size = args.num_prop_size
        self.cat_prop_size = args.cat_prop_size
        self.des_size = args.des_size
        self.tweet_size = args.tweet_size
        self.dropout = args.dropout
        self.linear_relu_des = nn.Sequential(nn.Linear(args.des_size, int(embedding_dim / 4)),
                                             nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.linear_relu_tweet = nn.Sequential(nn.Linear(args.tweet_size, int(embedding_dim / 4)),
                                               nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(args.num_prop_size, int(embedding_dim / 4)),
                                                  nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(args.cat_prop_size, int(embedding_dim / 4)),
                                                  nn.LeakyReLU(), nn.Dropout(args.dropout))

        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                               nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.SHGNs = nn.ModuleList()
        for _ in range(num_layer - 1):
            self.SHGNs.append(SimpleHGN(embedding_dim, embedding_dim, num_edge_type, args.rel_dim, args.beta))
        self.SHGNs.append(SimpleHGN(embedding_dim, embedding_dim, num_edge_type,
                                    args.rel_dim, args.beta, final_layer=True))

        self.linear_relu_output1 = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                                 nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.linear_output2 = nn.Linear(embedding_dim, 2)
        self.ReLU = nn.LeakyReLU()

    def forward(self, graph: HeteroData):
        num_prop = graph['user'].x[:, :self.num_prop_size]
        cat_prop = graph['user'].x[:, self.num_prop_size:(self.num_prop_size+self.cat_prop_size)]
        des = graph['user'].x[:, (self.num_prop_size+self.cat_prop_size):
                                 (self.num_prop_size+self.cat_prop_size+self.des_size)]
        tweet = graph['user'].x[:, (self.num_prop_size+self.cat_prop_size+self.des_size):]
        edge_index = torch.cat([graph[edge_type].edge_index for edge_type in graph.edge_types], dim=1).to(tweet.device)
        edge_type = torch.cat(
            [torch.zeros(graph[graph.edge_types[0]].edge_index.size(-1), dtype=torch.int64),
             torch.ones(graph[graph.edge_types[1]].edge_index.size(-1), dtype=torch.int64)]
        ).to(tweet.device)
        user_features_des = self.linear_relu_des(des)
        user_features_tweet = self.linear_relu_tweet(tweet)
        user_features_numeric = self.linear_relu_num_prop(num_prop)
        user_features_bool = self.linear_relu_cat_prop(cat_prop)
        user_feature = torch.cat((user_features_numeric, user_features_bool,
                                  user_features_des, user_features_tweet), dim=1)
        user_embedding = self.linear_relu_input(user_feature)

        alpha = None
        for i, simpleHGN in enumerate(self.SHGNs):
            user_embedding, alpha = simpleHGN(user_embedding, edge_index, edge_type, alpha)

        user_embedding = self.linear_relu_output1(user_embedding)
        user_embedding = self.linear_output2(user_embedding)

        return user_embedding
