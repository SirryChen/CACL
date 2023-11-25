import torch
from torch import nn
from torch_geometric.nn.conv import HGTConv, HeteroConv, GATConv, SAGEConv
from torch_geometric.data import HeteroData


class SAGEModel(nn.Module):
    def __init__(self, metadata, args):
        super(SAGEModel, self).__init__()
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

        self.ConvList = nn.ModuleList()
        for _ in range(num_layer - 1):
            self.ConvList.append(HeteroConv({edge_type: SAGEConv((-1, -1), embedding_dim)
                                 for edge_type in metadata[1]}, aggr='sum'))
        self.ConvList.append(HeteroConv({edge_type: SAGEConv((-1, -1), embedding_dim)
                             for edge_type in metadata[1]}, aggr='sum'))

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
        user_features_des = self.linear_relu_des(des)
        user_features_tweet = self.linear_relu_tweet(tweet)
        user_features_numeric = self.linear_relu_num_prop(num_prop)
        user_features_bool = self.linear_relu_cat_prop(cat_prop)
        user_feature = torch.cat((user_features_numeric, user_features_bool,
                                  user_features_des, user_features_tweet), dim=1)
        user_embedding = self.linear_relu_input(user_feature)

        edge_index = graph.edge_index_dict
        x_dict = {'user': user_embedding}
        for Conv in self.ConvList:
            x_dict = Conv(x_dict, edge_index)
        user_embedding = x_dict['user']

        user_embedding = self.linear_relu_output1(user_embedding)
        user_embedding = self.linear_output2(user_embedding)

        return user_embedding