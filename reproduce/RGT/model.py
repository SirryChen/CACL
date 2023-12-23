import torch
from torch import nn
from torch_geometric.nn.conv import TransformerConv
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class SemanticAttention(torch.nn.Module):
    def __init__(self, in_channel, num_head, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.num_head = num_head
        self.att_layers = torch.nn.ModuleList()
        # multi-head attention
        for i in range(num_head):
            self.att_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(in_channel, hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_size, 1, bias=False))
            )

    def forward(self, z):
        w = self.att_layers[0](z).mean(0)
        beta = torch.softmax(w, dim=0)

        beta = beta.expand((z.shape[0],) + beta.shape)
        output = (beta * z).sum(1)

        for i in range(1, self.num_head):
            w = self.att_layers[i](z).mean(0)
            beta = torch.softmax(w, dim=0)

            beta = beta.expand((z.shape[0],) + beta.shape)
            temp = (beta * z).sum(1)
            output += temp

        return output / self.num_head


class RGTLayer(torch.nn.Module):
    def __init__(self, num_edge_type, in_channel, out_channel, trans_heads=2, semantic_head=2, dropout=0.5):
        super(RGTLayer, self).__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_channel + out_channel, in_channel),
            torch.nn.Sigmoid()
        )

        self.activation = torch.nn.ELU()
        self.transformer_list = torch.nn.ModuleList()
        for i in range(int(num_edge_type)):
            self.transformer_list.append(
                TransformerConv(in_channels=in_channel, out_channels=out_channel, heads=trans_heads, dropout=dropout,
                                concat=False))

        self.num_edge_type = num_edge_type
        self.semantic_attention = SemanticAttention(in_channel=out_channel, num_head=semantic_head)

    def forward(self, features, edge_index, edge_type):
        r"""
        feature: input node features
        edge_index: all edge index, shape (2, num_edges)
        edge_type: same as RGCNconv in torch_geometric
        num_rel: number of relations
        beta: return cross relation attention weight
        agg: aggregation type across relation embedding
        """

        edge_index_list = []
        for i in range(self.num_edge_type):
            tmp = edge_index[:, edge_type == i]
            edge_index_list.append(tmp)

        u = self.transformer_list[0](features, edge_index_list[0].squeeze(0)).flatten(1)  # .unsqueeze(1)
        a = self.gate(torch.cat((u, features), dim=1))

        semantic_embeddings = (torch.mul(torch.tanh(u), a) + torch.mul(features, (1 - a))).unsqueeze(1)

        for i in range(1, len(edge_index_list)):
            u = self.transformer_list[i](features, edge_index_list[i].squeeze(0)).flatten(1)
            a = self.gate(torch.cat((u, features), dim=1))
            output = torch.mul(torch.tanh(u), a) + torch.mul(features, (1 - a))
            semantic_embeddings = torch.cat((semantic_embeddings, output.unsqueeze(1)), dim=1)

            return self.semantic_attention(semantic_embeddings)


class RGTModel(nn.Module):
    def __init__(self, num_edge_type, args):
        super(RGTModel, self).__init__()
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

        self.RGTs = nn.ModuleList()
        for _ in range(num_layer - 1):
            self.RGTs.append(RGTLayer(num_edge_type, embedding_dim, embedding_dim))
        self.RGTs.append(RGTLayer(num_edge_type, embedding_dim, embedding_dim))

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
        edge_index = torch.cat([graph[edge_type].edge_index for edge_type in graph.edge_types], dim=1)
        edge_type = torch.cat([torch.zeros(graph[graph.edge_types[0]].edge_index.size(-1), dtype=torch.int64),
                               torch.ones(graph[graph.edge_types[1]].edge_index.size(-1), dtype=torch.int64)])
        user_features_des = self.linear_relu_des(des)
        user_features_tweet = self.linear_relu_tweet(tweet)
        user_features_numeric = self.linear_relu_num_prop(num_prop)
        user_features_bool = self.linear_relu_cat_prop(cat_prop)
        user_feature = torch.cat((user_features_numeric, user_features_bool,
                                  user_features_des, user_features_tweet), dim=1)
        user_embedding = self.linear_relu_input(user_feature)

        for RGT in self.RGTs:
            user_embedding = self.ReLU(RGT(user_embedding, edge_index, edge_type))

        user_embedding = self.linear_relu_output1(user_embedding)
        user_embedding = self.linear_output2(user_embedding)

        return user_embedding
