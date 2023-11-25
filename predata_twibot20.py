import json
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
import numpy as np
import ijson
from tqdm import tqdm
from utils import calc_activate_days
from gensim.models.doc2vec import Doc2Vec
import nltk
nltk.data.path.append(r'/data1/botdet/LLM/nltk/nltk_data-gh-pages/packages')
nltk.data.path.append(r"/data1/botdet/LLM/nltk/wordnet31")
nltk.data.path.append(r"/data1/botdet/LLM/nltk/twitter_samples")
from nltk.corpus import wordnet


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
        if word_synonyms:
            new_word = word_synonyms.pop()
            new_words.append(new_word)
        else:
            new_words.append(word)
    augmented_text = ' '.join(new_words)
    return augmented_text


def get_users_feature():
    """
    提取用户的特征，数字特征+文本特征（描述+名字）
    @return: 用户特征字典{user_id: user_feature}
    """
    # if os.path.exists(predata_file_path + 'users_feature.json'):
    #     with open(predata_file_path + 'users_feature.json', 'r') as fp:
    #         users_feature_dict = json.load(fp)
    #         return users_feature_dict

    model = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_128.model")
    properties_list = ['created_at',
                       {'public_metrics': ['followers_count', 'following_count', 'tweet_count', 'listed_count']},
                       'description', 'entities', 'location',
                       'pinned_tweet_id', 'profile_image_url', 'protected',
                       'url', 'username', 'verified', 'withheld']
    properties_number = 15

    with open(dataset_file_path + 'node.json', 'r') as file:
        users = ijson.items(file, 'item')
        users_feature_dict = {}
        for user in tqdm(users, desc='加载用户特征'):
            if user['id'].find('u') == -1:
                continue
            users_feature_dict[user['id']] = []
            num_feature = []
            text_str = ''
            for user_property in properties_list:
                if isinstance(user_property, dict):
                    for count_property in user_property["public_metrics"]:
                        if user["public_metrics"][count_property] is None:
                            num_feature.append(0)
                        else:
                            num_feature.append(user["public_metrics"][count_property])
                elif user[user_property] is None:  # 属性为空则补0
                    if user_property not in ['username', 'description']:
                        num_feature.append(0)
                elif user_property in ['withheld', 'url', 'profile_image_url',  # bool属性值，非None就输入1
                                       'pinned_tweet_id', 'entities', 'location']:
                    num_feature.append(1)
                elif user_property in ['verified', 'protected']:
                    num_feature.append(int(user[user_property] == 'True'))
                elif user_property in ['username', 'description']:
                    text_str += user[user_property]
                elif user_property in ['created_at']:
                    num_feature.append(calc_activate_days(user[user_property].strip()))
            text_feature = [float(i) for i in model.infer_vector(text_str.split())]
            users_feature_dict[user['id']] = num_feature + text_feature
            # 属性值数目小于规定值时报错
            assert len(users_feature_dict[user['id']]) == properties_number + len(text_feature) - 2, \
                'user:{}, properties_number:{} < {}'.format(user['id'], len(users_feature_dict[user['id']]),
                                                            properties_number + len(text_feature) - 2)
    with open(predata_file_path + 'users_feature.json', 'w') as fp:
        json.dump(users_feature_dict, fp)

    users_id = users_feature_dict.keys()
    properties = [users_feature_dict[user_id] for user_id in users_id]

    properties = np.array(properties)
    for column in tqdm(range(properties_number - 2), desc='用户特征z-score归一化', ncols=50):
        mean = np.mean(properties[:, column])  # 求平均值
        std = np.std(properties[:, column])  # 求标准差
        if std == 0:
            properties[:, column] = mean
        else:
            properties[:, column] = (properties[:, column] - mean) / std  # z-score归一化

    for index, user_id in enumerate(users_id):
        users_feature_dict[user_id] = properties[index].tolist()
    with open(predata_file_path + 'users_feature.json', 'w') as fp:
        json.dump(users_feature_dict, fp)

    return users_feature_dict


def get_tweets_feature():
    """
    寻找并保存其包含的tweet节点的文本
    """
    if os.path.exists(predata_file_path + 'tweets_text.json'):
        with open(predata_file_path + 'tweets_text.json', 'r') as fp:
            return json.load(fp)

    tweets_text_dict = {}
    with open(dataset_file_path + 'node.json', 'r') as file:
        tweets = ijson.items(file, 'item')
        for tweet in tqdm(tweets, desc='加载推文文本'):
            if tweet.get('text') is not None:
                tweets_text_dict[tweet['id']] = tweet['text']

    with open(predata_file_path + 'tweets_text.json', 'w') as fp:
        json.dump(tweets_text_dict, fp)
    return tweets_text_dict


def get_node_label_split(load_from_local=False):
    """
    @param load_from_local: 是否从本地加载
    @return: user_label{user_id: (bot1,human0)}, user_split{user_id: (train0,val1,test2,support3)}
    """
    if load_from_local:
        with open(predata_file_path + 'user_label.json', 'r') as fp:
            user_labels = json.load(fp)
        with open(predata_file_path + 'user_split.json', 'r') as fp:
            user_splits = json.load(fp)
    else:
        labels = pd.read_csv(dataset_file_path + 'label.csv')
        splits = pd.read_csv(dataset_file_path + 'split.csv')
        user_labels = {}
        user_splits = {}
        for _, user in tqdm(splits.iterrows(), desc='加载用户分类'):
            user_splits[user['id']] = graph_types_map[user['split']]
        for _, user in tqdm(labels.iterrows(), desc='加载用户标签'):
            user_labels[user['id']] = int(user['label'] == 'bot')
        with open(predata_file_path + 'user_label.json', 'w') as fp:
            json.dump(user_labels, fp)
        with open(predata_file_path + 'user_split.json', 'w') as fp:
            json.dump(user_splits, fp)

    return user_labels, user_splits


def pytorch_edge_build(user_labels_dict, user_splits_dict, users_feature_dict, tweets_text_dict):
    torch_graph = HeteroData()
    user_id_index_map = {}
    tweet_id_index_map = {}
    max_user_index = 0
    max_tweet_index = 0
    user_features = []
    tweet_texts = []
    user_label = []
    user_split = []
    tweet_label = []
    edge_index_dict = {}
    chunk_size = 10000000
    edges = pd.read_csv(dataset_file_path + 'edge.csv', chunksize=chunk_size)
    for i, chunk_edges in enumerate(edges):
        for _, edge in tqdm(chunk_edges.iterrows(), desc=f'构建用户异质图-{i}'):
            source, relation, target = edge["source_id"], edge["relation"], edge["target_id"]
            if relation in ['follow', 'friend']:
                if edge_index_dict.get(('user', relation, 'user')) is None:
                    edge_index_dict['user', relation, 'user'] = []
                if user_id_index_map.get(source) is None:
                    user_id_index_map[source] = max_user_index
                    max_user_index += 1
                    user_features.append(users_feature_dict[source])
                    user_label.append(user_labels_dict.get(source))
                    user_split.append(graph_types_map[user_splits_dict.get(source)])
                if user_id_index_map.get(target) is None:
                    user_id_index_map[target] = max_user_index
                    max_user_index += 1
                    user_features.append(users_feature_dict[target])
                    user_label.append(user_labels_dict.get(target))
                    user_split.append(graph_types_map[user_splits_dict.get(target)])
                edge_index_dict['user', relation, 'user'].append([user_id_index_map[source],
                                                                  user_id_index_map[target]])
            elif relation in ['post']:
                if edge_index_dict.get(('user', relation, 'tweet')) is None:
                    edge_index_dict['user', relation, 'tweet'] = []
                if user_id_index_map.get(source) is None:
                    user_id_index_map[source] = max_user_index
                    max_user_index += 1
                    user_features.append(users_feature_dict[source])
                    user_label.append(user_labels_dict.get(source))
                    user_split.append(graph_types_map[user_splits_dict.get(source)])
                if tweet_id_index_map.get(target) is None:
                    tweet_id_index_map[target] = max_tweet_index
                    max_tweet_index += 1
                    tweet_texts.append(tweets_text_dict[target])
                    tweet_label.append(user_labels_dict.get(source))
                edge_index_dict['user', relation, 'tweet'].append([user_id_index_map[source],
                                                                   tweet_id_index_map[target]])

    torch_graph['user'].x = torch.tensor(user_features, dtype=torch.float)
    torch_graph['tweet'].x = tweet_texts
    torch_graph['user'].node_label = torch.tensor([label if label is not None else -1 for label in user_label])
    torch_graph['tweet'].node_label = torch.tensor([label if label is not None else -1 for label in tweet_label])
    torch_graph['user'].node_split = torch.tensor(user_split, dtype=torch.int8)
    for edge_type in edge_index_dict.keys():
        torch_graph[edge_type].edge_index = torch.tensor(edge_index_dict[edge_type]).t()

    torch.save(torch_graph, predata_file_path + 'cache/pre_graph.pt')


def aggregate_tweet_feature():
    model_32 = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_32.model")
    model_128 = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_128.model")

    graph = torch.load(predata_file_path + 'graph.pt')

    user_tweet_feature = torch.zeros(graph['user'].x.size(0), 128)
    user_tweet_num = torch.zeros(graph['user'].x.size(0))
    for edge in tqdm(graph['user', 'post', 'tweet'].edge_index.t().tolist(), desc='聚合推文信息'):
        user_tweet_feature[edge[0], :] += torch.tensor(model_128.infer_vector(graph['tweet'].x[edge[1]].split()))
        user_tweet_num[edge[0]] += 1
    torch.save(user_tweet_feature, predata_file_path + 'cache/user_tweet_feature.pt')
    user_tweet_num = torch.where(torch.tensor(user_tweet_num == 0), 1, user_tweet_num)
    graph['user'].x = torch.cat((graph['user'].x, user_tweet_feature / user_tweet_num.view(-1, 1)), dim=1)

    origin_tweet_feature = []
    augment_tweet_feature = []
    for text in tqdm(graph['tweet'].x, desc='编码推文'):
        origin_tweet_feature.append(model_32.infer_vector(text.split()))
        augment_tweet_feature.append(model_32.infer_vector(wordnet_paraphrase(text).split()))
    graph['tweet'].x1 = torch.tensor(origin_tweet_feature)
    graph['tweet'].x2 = torch.tensor(augment_tweet_feature)
    del graph['tweet'].x

    torch.save(graph, predata_file_path + 'graph.pt')


if __name__ == "__main__":
    dataset_file_path = "/data1/botdet/datasets/Twibot-20-Format/"
    predata_file_path = "./predata/twibot20/"
    doc2vec_model_path = "./predata/wiki_doc2vec/"
    if not os.path.exists('./predata'):
        os.mkdir('./predata')
    if not os.path.exists(predata_file_path):
        os.mkdir(predata_file_path)
        os.mkdir(predata_file_path + 'cache')

    graph_types_map = {'train': 0, 'val': 1, 'test': 2, 'support': 3}

    user_feature = get_users_feature()
    tweet_text = get_tweets_feature()
    users_label, users_split = get_node_label_split(load_from_local=True)

    pytorch_edge_build(users_label, users_split, user_feature, tweet_text)
    aggregate_tweet_feature()
