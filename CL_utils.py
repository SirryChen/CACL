import torch
from torch.nn.functional import cosine_similarity, normalize
from torch_geometric.nn.conv import TransformerConv


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

    def forward(self, features, edge_index):
        r"""
        feature: input node features
        edge_index: all edge index, shape (2, num_edges)
        edge_type: same as RGCNconv in torch_geometric
        num_rel: number of relations
        beta: return cross relation attention weight
        agg: aggregation type across relation embedding
        """

        edge_index_list = []
        for edge_type in sorted(list(edge_index.keys())):
            tmp = edge_index[edge_type]
            edge_index_list.append(tmp)

        u = self.transformer_list[0](features, edge_index_list[0].squeeze(0)).flatten(1)
        a = self.gate(torch.cat((u, features), dim=1))

        semantic_embeddings = (torch.mul(torch.tanh(u), a) + torch.mul(features, (1 - a))).unsqueeze(1)

        for i in range(1, len(edge_index_list)):
            u = self.transformer_list[i](features, edge_index_list[i].squeeze(0)).flatten(1)
            a = self.gate(torch.cat((u, features), dim=1))
            output = torch.mul(torch.tanh(u), a) + torch.mul(features, (1 - a))
            semantic_embeddings = torch.cat((semantic_embeddings, output.unsqueeze(1)), dim=1)

            return self.semantic_attention(semantic_embeddings)


def positive_generate(node_projection, node_label, node_split):
    node_label[node_split != 0] = 2  # 分离训练集
    bot_emb = node_projection[node_label == 1]
    human_emb = node_projection[node_label == 0]
    positive_sample_bot = torch.mean(bot_emb, dim=0) if bot_emb.size(0) != 0 else torch.zeros(bot_emb.size[1])
    positive_sample_human = torch.mean(human_emb, dim=0) if human_emb.size(0) != 0 else torch.zeros(human_emb.size[1])
    return positive_sample_bot, positive_sample_human


def compute_pro_loss(emb1, emb2, label: torch.tensor, split: torch.tensor, tau):
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

    label[split != 0] = 2  # 分离训练集
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
            loss1 = cos_loss(node1_emb, emb2[node], torch.cat((emb2[:node], emb2[node + 1:]), dim=0))
            loss2 = cos_loss(node2_emb, emb1[node], torch.cat((emb1[:node], emb1[node + 1:]), dim=0))
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
        refl_sim = torch.exp(sim(z1, z1) / tau)  # intra-view
        between_sim = torch.exp(sim(z1, z2) / tau)  # inter-view

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    emb1 = emb1[split == 0]
    emb2 = emb2[split == 0]
    loss1 = semi_loss(emb1, emb2)
    loss2 = semi_loss(emb2, emb1)
    total_loss = ((loss1 + loss2) * 0.5).mean()
    return total_loss


def compute_hard_loss(emb1, emb2, label, split, positive_bot, positive_human, tau):
    # consider both negative and positive hard samples
    def trans(x: torch.tensor, t=tau):
        return torch.exp(torch.div(x, t))

    def cos_loss(node_emb, node_emb_opposite, node_emb_diff_label):
        self_loss = trans(cosine_similarity(node_emb.unsqueeze(0), node_emb_opposite.unsqueeze(0)).squeeze(0))
        if len(node_emb_diff_label) != 0:
            negative_loss = torch.sum(trans(torch.matmul(node_emb, node_emb_diff_label.t())
                                            / (torch.norm(node_emb) * torch.norm(node_emb_diff_label, dim=1))))
        else:
            negative_loss = torch.tensor(1e-5)
        loss_ = (self_loss + positive_loss) / (self_loss + negative_loss + positive_loss)
        return loss_

    label[split != 0] = 2  # 分离训练集
    bot_emb1 = emb1[label == 1]
    human_emb1 = emb1[label == 0]
    bot_emb2 = emb2[label == 1]
    human_emb2 = emb2[label == 0]
    total_loss = []
    for node in range(label.size(0)):
        node1_emb = emb1[node]
        node2_emb = emb2[node]
        if label[node] == 1:
            positive_loss = (trans(cosine_similarity(node1_emb.reshape(1, -1), positive_bot.reshape(1, -1))) +
                             trans(cosine_similarity(node2_emb.reshape(1, -1), positive_bot.reshape(1, -1)))) / 2
            loss1 = cos_loss(node1_emb, emb2[node], human_emb2)
            loss2 = cos_loss(node2_emb, emb1[node], human_emb1)
            loss = (loss1 + loss2) / 2
        elif label[node] == 0:
            positive_loss = (trans(cosine_similarity(node1_emb.reshape(1, -1), positive_human.reshape(1, -1))) +
                             trans(cosine_similarity(node2_emb.reshape(1, -1), positive_human.reshape(1, -1)))) / 2
            loss1 = cos_loss(node1_emb, emb2[node], bot_emb2)
            loss2 = cos_loss(node2_emb, emb1[node], bot_emb1)
            loss = (loss1 + loss2) / 2
        else:
            positive_loss = 0
            loss1 = cos_loss(node1_emb, emb2[node], torch.cat((emb2[:node], emb2[node + 1:]), dim=0))
            loss2 = cos_loss(node2_emb, emb1[node], torch.cat((emb1[:node], emb1[node + 1:]), dim=0))
            loss = (loss1 + loss2) / 2

        total_loss.append(- torch.log(loss))
    average_loss = sum(total_loss) / len(total_loss)

    return average_loss


def compute_cross_mean_view_loss(emb_a_1, emb_a_2, emb_b_1, emb_b_2, pred_a, label_a, label_b, tau):
    def normalization(x):
        return torch.div(x, torch.norm(x, dim=1, keepdim=True))

    def trans(x: torch.tensor, t=tau):
        return torch.exp(torch.div(x, t))

    emb_a_1 = normalization(emb_a_1)
    emb_a_2 = normalization(emb_a_2)
    emb_b_1 = normalization(emb_b_1)
    emb_b_2 = normalization(emb_b_2)

    human_num_a = torch.sum(label_a == 0)
    robot_num_a = torch.sum(label_a == 1)
    human_num_b = torch.sum(label_b == 0)
    robot_num_b = torch.sum(label_b == 1)

    device = emb_a_1.device
    human_mean_emb_a_1 = emb_a_1[label_a == 0].mean(0) if human_num_a != 0 else torch.zeros(emb_a_1.size(1)).to(device)
    robot_mean_emb_a_1 = emb_a_1[label_a == 1].mean(0) if robot_num_a != 0 else torch.zeros(emb_a_1.size(1)).to(device)
    human_mean_emb_b_1 = emb_b_1[label_b == 0].mean(0) if human_num_b != 0 else torch.zeros(emb_b_1.size(1)).to(device)
    robot_mean_emb_b_1 = emb_b_1[label_b == 1].mean(0) if robot_num_b != 0 else torch.zeros(emb_b_1.size(1)).to(device)
    human_mean_emb_a_2 = emb_a_2[label_a == 0].mean(0) if human_num_a != 0 else torch.zeros(emb_a_2.size(1)).to(device)
    robot_mean_emb_a_2 = emb_a_2[label_a == 1].mean(0) if robot_num_a != 0 else torch.zeros(emb_a_2.size(1)).to(device)
    human_mean_emb_b_2 = emb_b_2[label_b == 0].mean(0) if human_num_b != 0 else torch.zeros(emb_b_2.size(1)).to(device)
    robot_mean_emb_b_2 = emb_b_2[label_b == 1].mean(0) if robot_num_b != 0 else torch.zeros(emb_b_2.size(1)).to(device)

    contrastive_loss = torch.tensor(0.0, requires_grad=True).to(pred_a.device)
    for node in range(label_a.size(0)):
        if label_a[node] == 0:
            self_sim = trans(torch.matmul(emb_a_1[node], emb_a_2[node].t()))

            positive_sim_1 = (trans(torch.matmul(emb_a_1[node], human_mean_emb_b_1.t())) +
                              trans(torch.matmul(emb_a_1[node], human_mean_emb_b_2.t()))) / 2 * human_num_b
            negative_sim_1 = (trans(torch.matmul(emb_a_1[node], robot_mean_emb_a_1.t())) +
                              trans(torch.matmul(emb_a_1[node], robot_mean_emb_a_2.t()))) / 2 * robot_num_a

            positive_sim_2 = (trans(torch.matmul(emb_a_2[node], human_mean_emb_b_1.t())) +
                              trans(torch.matmul(emb_a_2[node], human_mean_emb_b_2.t()))) / 2 * human_num_b
            negative_sim_2 = (trans(torch.matmul(emb_a_2[node], robot_mean_emb_a_1.t())) +
                              trans(torch.matmul(emb_a_2[node], robot_mean_emb_a_2.t()))) / 2 * robot_num_a

        elif label_a[node] == 1:
            self_sim = trans(torch.matmul(emb_a_1[node], emb_a_2[node].t()))

            positive_sim_1 = (trans(torch.matmul(emb_a_1[node], robot_mean_emb_b_1.t())) +
                              trans(torch.matmul(emb_a_1[node], robot_mean_emb_b_2.t()))) / 2 * robot_num_b
            negative_sim_1 = (trans(torch.matmul(emb_a_1[node], human_mean_emb_a_1.t())) +
                              trans(torch.matmul(emb_a_1[node], human_mean_emb_a_2.t()))) / 2 * human_num_a

            positive_sim_2 = (trans(torch.matmul(emb_a_2[node], robot_mean_emb_b_1.t())) +
                              trans(torch.matmul(emb_a_2[node], robot_mean_emb_b_2.t()))) / 2 * robot_num_b
            negative_sim_2 = (trans(torch.matmul(emb_a_2[node], human_mean_emb_a_1.t())) +
                              trans(torch.matmul(emb_a_2[node], human_mean_emb_a_2.t()))) / 2 * human_num_a
        else:
            self_sim = trans(torch.matmul(emb_a_1[node], emb_a_2[node].t()))
            positive_sim_1 = 0
            negative_sim_1 = 0
            positive_sim_2 = 0
            negative_sim_2 = 0

        loss_1 = torch.div((self_sim + positive_sim_1), (self_sim + positive_sim_1 + negative_sim_1))
        loss_2 = torch.div((self_sim + positive_sim_2), (self_sim + positive_sim_2 + negative_sim_2))
        contrastive_loss += (- torch.log(loss_1) - torch.log(loss_2))

    contrastive_loss = contrastive_loss / (2 * label_a.size(0))

    return contrastive_loss


def compute_cross_individual_loss(emb_a_1, emb_a_2, emb_b_1, emb_b_2, pred_a, label_a, label_b, tau):
    def normalization(x):
        return torch.div(x, torch.norm(x, dim=1, keepdim=True))

    def trans(x: torch.tensor, t=tau):
        return torch.exp(torch.div(x, t))

    emb_a_1 = normalization(emb_a_1)
    emb_a_2 = normalization(emb_a_2)
    emb_b_1 = normalization(emb_b_1)
    emb_b_2 = normalization(emb_b_2)

    device = emb_a_1.device
    human_emb_a_1 = emb_a_1[label_a == 0] if torch.sum(label_a == 0) != 0 else torch.zeros(emb_a_1.size(1)).to(device)
    robot_emb_a_1 = emb_a_1[label_a == 1] if torch.sum(label_a == 1) != 0 else torch.zeros(emb_a_1.size(1)).to(device)
    human_emb_b_1 = emb_b_1[label_b == 0] if torch.sum(label_b == 0) != 0 else torch.zeros(emb_b_1.size(1)).to(device)
    robot_emb_b_1 = emb_b_1[label_b == 1] if torch.sum(label_b == 1) != 0 else torch.zeros(emb_b_1.size(1)).to(device)
    human_emb_a_2 = emb_a_2[label_a == 0] if torch.sum(label_a == 0) != 0 else torch.zeros(emb_a_2.size(1)).to(device)
    robot_emb_a_2 = emb_a_2[label_a == 1] if torch.sum(label_a == 1) != 0 else torch.zeros(emb_a_2.size(1)).to(device)
    human_emb_b_2 = emb_b_2[label_b == 0] if torch.sum(label_b == 0) != 0 else torch.zeros(emb_b_2.size(1)).to(device)
    robot_emb_b_2 = emb_b_2[label_b == 1] if torch.sum(label_b == 1) != 0 else torch.zeros(emb_b_2.size(1)).to(device)

    contrastive_loss = torch.tensor(0.0, requires_grad=True).to(pred_a.device)

    p_sim = []
    n_sim = []

    for node in range(label_a.size(0)):
        if label_a[node] == 0:
            self_sim = trans(torch.matmul(emb_a_1[node], emb_a_2[node].t()))

            positive_sim_1 = torch.sum(trans(torch.matmul(emb_a_1[node], human_emb_b_1.t())) +
                                       trans(torch.matmul(emb_a_1[node], human_emb_b_2.t()))) / 2
            negative_sim_1 = torch.sum(trans(torch.matmul(emb_a_1[node], robot_emb_a_1.t())) +
                                       trans(torch.matmul(emb_a_1[node], robot_emb_a_2.t()))) / 2

            positive_sim_2 = torch.sum(trans(torch.matmul(emb_a_2[node], human_emb_b_1.t())) +
                                       trans(torch.matmul(emb_a_2[node], human_emb_b_2.t()))) / 2
            negative_sim_2 = torch.sum(trans(torch.matmul(emb_a_2[node], robot_emb_a_1.t())) +
                                       trans(torch.matmul(emb_a_2[node], robot_emb_a_2.t()))) / 2

        elif label_a[node] == 1:
            self_sim = trans(torch.matmul(emb_a_1[node], emb_a_2[node].t()))

            positive_sim_1 = torch.sum(trans(torch.matmul(emb_a_1[node], robot_emb_b_1.t())) +
                                       trans(torch.matmul(emb_a_1[node], robot_emb_b_2.t()))) / 2
            negative_sim_1 = torch.sum(trans(torch.matmul(emb_a_1[node], human_emb_a_1.t())) +
                                       trans(torch.matmul(emb_a_1[node], human_emb_a_2.t()))) / 2

            positive_sim_2 = torch.sum(trans(torch.matmul(emb_a_2[node], robot_emb_b_1.t())) +
                                       trans(torch.matmul(emb_a_2[node], robot_emb_b_2.t()))) / 2
            negative_sim_2 = torch.sum(trans(torch.matmul(emb_a_2[node], human_emb_a_1.t())) +
                                       trans(torch.matmul(emb_a_2[node], human_emb_a_2.t()))) / 2
        else:
            self_sim = trans(torch.matmul(emb_a_1[node], emb_a_2[node].t()))
            positive_sim_1 = 0
            negative_sim_1 = 0
            positive_sim_2 = 0
            negative_sim_2 = 0

        loss_1 = torch.div((self_sim + positive_sim_1), (self_sim + positive_sim_1 + negative_sim_1))
        loss_2 = torch.div((self_sim + positive_sim_2), (self_sim + positive_sim_2 + negative_sim_2))
        contrastive_loss += (- torch.log(loss_1) - torch.log(loss_2)) * 0.5

        p_sim.append((positive_sim_1/label_b.size(0) + positive_sim_2/label_b.size(0))/2)
        n_sim.append((negative_sim_1/label_a.size(0) + negative_sim_2/label_a.size(0))/2)

    contrastive_loss = contrastive_loss / label_a.size(0)

    return contrastive_loss, sum(p_sim)/len(p_sim), sum(n_sim)/len(n_sim)
