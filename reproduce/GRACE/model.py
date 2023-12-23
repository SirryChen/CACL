import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, HeteroConv


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class GRACEModel(torch.nn.Module):
    def __init__(self, metadata, args):
        super(GRACEModel, self).__init__()
        embedding_dim = args.embedding_dim
        num_layer = args.num_layer
        num_proj_hidden = args.num_proj_hidden

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

        self.ReLU = torch.relu
        self.ConvList = nn.ModuleList()
        for _ in range(num_layer - 1):
            self.ConvList.append(HeteroConv({edge_type: GCNConv(-1, embedding_dim)
                                             for edge_type in metadata[1]}, aggr='sum'))
        self.ConvList.append(HeteroConv({edge_type: GCNConv(-1, embedding_dim)
                                         for edge_type in metadata[1]}, aggr='sum'))

        self.tau = args.tau

        self.fc1 = torch.nn.Linear(embedding_dim, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_proj_hidden)

    def forward(self, graph):
        num_prop = graph['user'].x[:, :self.num_prop_size]
        cat_prop = graph['user'].x[:, self.num_prop_size:(self.num_prop_size + self.cat_prop_size)]
        des = graph['user'].x[:, (self.num_prop_size + self.cat_prop_size):
                                 (self.num_prop_size + self.cat_prop_size + self.des_size)]
        tweet = graph['user'].x[:, (self.num_prop_size + self.cat_prop_size + self.des_size):]
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
            x_dict = {node_type: self.ReLU(x) for node_type, x in Conv(x_dict, edge_index).items()}

        user_embedding = x_dict['user']

        return user_embedding

    def projection(self, user_embedding):
        z = F.elu(self.fc1(user_embedding))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
