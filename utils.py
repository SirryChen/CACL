from datetime import datetime
import numpy as np
import openai
from argparse import ArgumentParser
import torch
from torch_geometric.data import HeteroData
from gensim.models import Doc2Vec
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import nltk
nltk.data.path.append(r'/data1/botdet/LLM/nltk/nltk_data-gh-pages/packages')
nltk.data.path.append(r"/data1/botdet/LLM/nltk/wordnet31")
nltk.data.path.append(r"/data1/botdet/LLM/nltk/twitter_samples")
from nltk.corpus import wordnet


def calc_activate_days(created_at):
    created_at = created_at.strip()
    create_date = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y')
    crawl_date = datetime.strptime('2020 09 28 +0000', '%Y %m %d %z')
    delta_date = crawl_date - create_date
    return delta_date.days


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))


def tweet_augment(graph: HeteroData, augment_method='wordnet', tweet_dim=32):
    """
    基于词义转换的文本增强
    @return: graph['tweet'].x第一维是原文向量，第二维为增强向量
    """
    augment_model = augment_tokenizer = None
    device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    model = Doc2Vec.load(f"./predata/wiki_doc2vec/pretrained_wiki_doc2vec_{tweet_dim}.model")
    if augment_method == 'phi':
        augment_model = AutoModelForCausalLM.from_pretrained("/data1/botdet/LLM/microsoft/phi-1_5",
                                                             trust_remote_code=True, local_files_only=True).to(device)
        augment_tokenizer = AutoTokenizer.from_pretrained("/data1/botdet/LLM/microsoft/phi-1_5",
                                                          trust_remote_code=True, local_files_only=True)
    origin_tweets_vector = []
    augment_tweets_vector = []
    tqdm_bar = tqdm(total=len(graph['tweet'].x), desc=f'文本增强')
    threshold = 25000000
    for i, (label, text) in enumerate(zip(graph['tweet'].node_label, graph['tweet'].x)):
        if i < 20000001:
            continue
        origin_tweets_vector.append(list(model.infer_vector(text.split())))
        if augment_method == 'gpt' and random.random() > 0.9:
            text = gpt3_paraphrase(text)
        elif augment_method == 'phi' and random.random() > 0.9:
            text = phi_paraphrase(text, augment_model, augment_tokenizer, device)
        elif augment_method == 'wordnet' and random.random() > 0.5:
            text = wordnet_paraphrase(text)
        augment_tweets_vector.append(list(model.infer_vector(text.split())))
        tqdm_bar.update(1)
        if i > threshold:
            torch.save(torch.tensor(origin_tweets_vector), f"/data1/botdet/CDCL/predata/twibot20/cache/origin{i}.pt")
            torch.save(torch.tensor(augment_tweets_vector), f"/data1/botdet/CDCL/predata/twibot20/cache/augment{i}.pt")
            origin_tweets_vector = []
            augment_tweets_vector = []
            threshold += threshold
    torch.save(torch.tensor(origin_tweets_vector), f"/data1/botdet/CDCL/predata/twibot20/cache/origin{threshold}.pt")
    torch.save(torch.tensor(augment_tweets_vector), f"/data1/botdet/CDCL/predata/twibot20/cache/augment{threshold}.pt")
    origin_tweets_vector = torch.tensor([])
    augment_tweets_vector = torch.tensor([])
    for i in [5000001, 10000001, 20000001]:
        origin_tweets_vector = torch.cat((origin_tweets_vector, torch.load(f"/data1/botdet/CDCL/predata/twibot20/cache/origin{i}.pt")), dim=0)
        augment_tweets_vector = torch.cat((augment_tweets_vector, torch.load(f"/data1/botdet/CDCL/predata/twibot20/cache/augment{i}.pt")), dim=0)
    graph['tweet'].x1 = origin_tweets_vector
    graph['tweet'].x2 = augment_tweets_vector

    return graph


def gpt3_paraphrase(text):
    openai.api_key = "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"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Paraphrase the following text: '{text}'",
        max_tokens=50,  # 设置生成的最大标记数
        n=1,  # 生成一个转述
        stop=None,  # 不设置停止标记
        temperature=0.7  # 控制生成的多样性，可根据需要调整
    )
    augmented_text = response.choices[0].text
    return augmented_text


def phi_paraphrase(text, phi_model, phi_tokenizer, device):
    pattern = r"Answer:(.*?)Exercise 2"
    inputs = phi_tokenizer(f'''paraphrase the following text: {text}''', return_tensors="pt",
                           return_attention_mask=False).to(device)
    outputs = phi_model.generate(**inputs, max_length=400)
    outputs = phi_tokenizer.batch_decode(outputs)[0]
    result = re.search(pattern, outputs, re.DOTALL)
    augmented_text = result.group(1).strip() if result else text
    return augmented_text


def wordnet_paraphrase(text):
    """
    https://www.nltk.org/nltk_data/
    """
    words = text.split()
    new_words = []
    for word in words:
        word_synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                word_synonyms.append(lemma.name())
        if word_synonyms and random.random() > 0.5:
            new_word = word_synonyms.pop()
            new_words.append(new_word)
        else:
            new_words.append(word)
    augmented_text = ' '.join(new_words)
    return augmented_text


def drop_feature(node_feature, weight, probability: float, threshold: float = 0.7):
    weight = weight / weight.mean() * probability
    weight = weight.where(weight < threshold, torch.ones_like(weight) * threshold)
    drop_prob = weight

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    node_feature = node_feature.clone()
    node_feature[:, drop_mask] = 0.

    return node_feature


def feature_drop_weights(node_feature, node_c):
    node_feature = node_feature.abs()  # 元素取绝对值
    node_feature = torch.where(node_feature == 0, torch.tensor(1e-4), node_feature)  # 将0替换为维稳小量
    weight = node_feature.t() @ node_c
    weight = weight.log()
    weight = (weight.max() - weight) / (weight.max() - weight.mean())

    return weight


def compute_page_rank(edge_index, node_num, damp: float = 0.85, k: int = 10):
    pr = torch.ones(node_num) / node_num

    adj_matrix = torch.zeros(node_num, node_num)
    adj_matrix[edge_index[0], edge_index[1]] = 1

    for _ in range(k):
        new_pr = (1 - damp) / node_num + damp * torch.matmul(adj_matrix, pr)
        pr = new_pr

    return pr


def pr_drop_weights(edge_index, node_num, aggr: str = 'sink', k: int = 10):
    pv = compute_page_rank(edge_index, node_num, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)      # FIXME 应该是大于号
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


def add_self_loop(edge_index, node_num):
    self_loop_edge_index = torch.tensor([range(node_num)], dtype=torch.long).repeat(2, 1)
    edge_index = torch.cat((edge_index, self_loop_edge_index), dim=1)
    return edge_index


def compress_graph(community_ids, partition, node_num_threshold=1500):
    """
    将小规模的图合并
    """
    part_dict = {}
    for community_id in community_ids:
        part_dict[community_id] = []
    # 按社区编号将用户归类
    for user_id in partition.keys():
        part_dict[partition[user_id]].append(user_id)
    # 找出小规模社区分配为0号社区，重新分配其他社区标号
    community_id_map = {}
    new_id = 1
    for community_id in community_ids:
        if len(part_dict[community_id]) < node_num_threshold:
            community_id_map[community_id] = 0
        else:
            community_id_map[community_id] = new_id
            new_id += 1
    new_partition = {}
    for node, comm_id in partition.items():
        new_partition[node] = community_id_map[comm_id]
    return set(new_partition.values()), new_partition


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def super_parament_initial():
    dataset_name = 'twibot20'
    s_parament = ArgumentParser()
    if dataset_name == 'twibot22':
        s_parament.add_argument('--dataset', type=str, default='twibot22', help='使用数据集名称')
        s_parament.add_argument('--tweet_dim', type=int, default=128, help='推文编码长度')
        s_parament.add_argument('--hidden_dim', type=int, default=64, help='社区检测模型的隐藏层维度')
        s_parament.add_argument('--output_dim', type=int, default=32, help='社区检测模型的输出层维度')
        s_parament.add_argument('--encoder_hidden_channel', type=int, default=128, help='对比学习的隐藏层维度')
        s_parament.add_argument('--encoder_out_channel', type=int, default=128, help='对比学习的输出层维度')
        s_parament.add_argument('--num_layer', type=int, default=3, help='图卷积层数')
        s_parament.add_argument('--projector_hidden_size', type=int, default=128, help='投影头的隐藏层维度')
        s_parament.add_argument('--learning_rate', type=float, default=0.001, help='对比学习模型学习率')
        s_parament.add_argument('--weight_decay', type=float, default=1e-5, help='对比学习模型优化器超参')
        s_parament.add_argument('--lr_warmup_epochs', type=int, default=64, help='对比学习模型预热期')
        s_parament.add_argument('--epochs', type=int, default=128, help='训练次数')
        s_parament.add_argument('--momentum', type=float, default=0.99, help='梯度下降惯性动量')
        s_parament.add_argument('--tau', type=float, default=0.4, help='对比损失温度参数')
        s_parament.add_argument('--tweet_augment_method', type=str, default='wordnet', help='文本增强方法')
        s_parament.add_argument('--max_error_times', type=int, default=5, help='最大错误早停次数')
        s_parament.add_argument('--local_rank', default=0, type=int)
        s_parament.add_argument('--classifier_loss_function', type=str, default='binary', help='分类器训练使用的损失函数',
                                choices=['binary', 'focal'])
        s_parament.add_argument('--cluster', type=str, default='randomwalk', help='聚类方法',
                                choices=['kmeans', 'randomwalk', 'hierachical'])

    elif dataset_name == 'twibot20':
        s_parament.add_argument('--dataset', type=str, default='twibot20', help='使用数据集名称')
        s_parament.add_argument('--basic_model', type=str, default="HGT", help='基础模型', choices=['HGT', 'GAT', 'SAGE'])
        s_parament.add_argument('--des_size', default=768, help='用户描述特征维度')
        s_parament.add_argument('--tweet_size', default=768, help='用户推文特征维度')
        s_parament.add_argument('--num_prop_size', default=6, help='用户数字信息维度')
        s_parament.add_argument('--cat_prop_size', default=11, help='用户布尔信息维度')
        s_parament.add_argument('--embedding_dim', default=128, help='用户编码长度')
        s_parament.add_argument('--tweet_dim', type=int, default=128, help='推文编码长度')
        s_parament.add_argument('--hidden_dim', type=int, default=64, help='社区检测模型的隐藏层维度')
        s_parament.add_argument('--output_dim', type=int, default=32, help='社区检测模型的输出层维度')
        s_parament.add_argument('--encoder_hidden_channel', type=int, default=128, help='对比学习的隐藏层维度')
        s_parament.add_argument('--encoder_out_channel', type=int, default=128, help='对比学习的输出层维度')
        s_parament.add_argument('--num_layer', type=int, default=2, help='图卷积层数')
        s_parament.add_argument('--projector_hidden_size', type=int, default=128, help='投影头的隐藏层维度')
        s_parament.add_argument('--cl_learning_rate', type=float, default=0.0005, help='对比学习模型学习率')
        s_parament.add_argument('--weight_decay', type=float, default=3e-5, help='对比学习模型优化器超参')
        s_parament.add_argument('--lr_warmup_epochs', type=int, default=36, help='对比学习模型预热期')
        s_parament.add_argument('--ft_learning_rate', type=float, default=0.001, help='微调学习率')
        s_parament.add_argument('--epochs', type=int, default=200, help='训练次数')
        s_parament.add_argument('--momentum', type=float, default=0.99, help='梯度下降惯性动量')
        s_parament.add_argument('--tau', type=float, default=0.07, help='对比损失温度参数')
        s_parament.add_argument('--tweet_augment_method', type=str, default='wordnet', help='文本增强方法')
        s_parament.add_argument('--max_error_times', type=int, default=5, help='最大错误早停次数')
        s_parament.add_argument('--alpha', type=int, default=0.01, help='对比学习损失权重系数')
        s_parament.add_argument('--beta', type=int, default=1, help='分类损失权重系数')
        s_parament.add_argument('--dropout', type=int, default=0.5, help='dropout')
        s_parament.add_argument('--local_rank', default=0, type=int)
        s_parament.add_argument('--classifier_loss_function', type=str, default='binary', help='分类器训练使用的损失函数',
                                choices=['binary', 'focal'])
        s_parament.add_argument('--cluster', type=str, default='randomwalk', help='聚类方法',
                                choices=['kmeans', 'randomwalk', 'hierachical', 'k_guide'])

    return s_parament
