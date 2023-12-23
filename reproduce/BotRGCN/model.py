import torch
from torch import nn
from torch_geometric.nn.conv import RGCNConv
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class BotRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dim=128,
                 dropout=0.3):
        super(BotRGCN, self).__init__()
        self.num_prop_size = num_prop_size
        self.cat_prop_size = cat_prop_size
        self.des_size = des_size
        self.tweet_size = tweet_size
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(nn.Linear(des_size, int(embedding_dim / 4)), nn.LeakyReLU())
        self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, int(embedding_dim / 4)), nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, int(embedding_dim / 4)), nn.LeakyReLU())
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, int(embedding_dim / 4)), nn.LeakyReLU())

        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())

        self.RGCNConv = RGCNConv(embedding_dim, embedding_dim, num_relations=2)

        self.linear_relu_output1 = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())
        self.linear_output2 = nn.Linear(embedding_dim, 2)

    def forward(self, graph: HeteroData):
        num_prop = graph['user'].x[:, :self.num_prop_size]
        cat_prop = graph['user'].x[:, self.num_prop_size:(self.num_prop_size+self.cat_prop_size)]
        des = graph['user'].x[:, (self.num_prop_size+self.cat_prop_size):
                                 (self.num_prop_size+self.cat_prop_size+self.des_size)]
        tweet = graph['user'].x[:, (self.num_prop_size+self.cat_prop_size+self.des_size):]
        edge_index = torch.cat([graph[edge_type].edge_index for edge_type in graph.edge_types], dim=1)
        edge_type = torch.cat([torch.zeros(graph[graph.edge_types[0]].edge_index.size(-1), dtype=torch.int64),
                               torch.ones(graph[graph.edge_types[1]].edge_index.size(-1), dtype=torch.int64)])
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.RGCNConv(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.RGCNConv(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x
