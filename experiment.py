# import torch
# from torch_geometric.data import HeteroData
#
# graph: HeteroData = torch.load("/data1/botdet/CDCL_heterogeneous/predata/twibot20/graph2.pt")
# for node_type in graph.node_types:
#     print(node_type)
#     if hasattr(graph[node_type], 'node_split'):
#         print('node_split', type(graph[node_type].node_split))
#         print(graph[node_type].node_split.size())
#     if hasattr(graph[node_type], 'node_label'):
#         print('node_label', type(graph[node_type].node_label))
#         print(graph[node_type].node_label.size())
#     if hasattr(graph[node_type], 'x'):
#         print('x', type(graph[node_type].x))
#         print(graph[node_type].x.size())
#     else:
#         print('x1', type(graph[node_type].x1))
#         print('x2', type(graph[node_type].x2))
#         print(graph[node_type].x1.size())
#         print(graph[node_type].x2.size())
#
# for edge_type in graph.edge_types:
#     print(type(graph[edge_type].edge_index))
#     print(graph[edge_type].edge_index.size())

# from gensim.models.doc2vec import Doc2Vec
#
# # text = 'this is a dog with a tail.'
# text = 'Day 1 Trump supporter. I rode the escalator! Constitutionalist traditionalist conservative. My 1st vote was Reagan! America, family first. #1A #2A #MAGA #KAG '
# doc2vec_model_path = "./predata/wiki_doc2vec/"
# model_32 = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_32.model")
# result = model_32.infer_vector(text.split())
#
# print(result)

# import torch
# from transformers import AutoTokenizer, AutoModel
#
# text = 'Day 1 Trump supporter. I rode the escalator! Constitutionalist traditionalist conservative. My 1st vote was Reagan! America, family first. #1A #2A #MAGA #KAG '
# bert_model_path = "/data1/botdet/LLM/bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
# encoder = AutoModel.from_pretrained(bert_model_path)
# encode_text = encoder(**tokenizer(text, return_tensors='pt')).last_hidden_state[:, 0, :][0].tolist()
# print(encode_text)



# import json
# import ijson
#
#
# dataset_file_path = "/data1/botdet/datasets/Twibot-20-Format/"
# with open(dataset_file_path + 'node_info.json', 'r') as file:
#     users = ijson.items(file, 'item')
#     for user in users:
#         if f"u{user['ID']}".strip() == 'u972552968212955136':
#             print('yes')


# import json
# import ijson
# from transformers import AutoTokenizer, AutoModel
#
# bert_model_path = "/data1/botdet/LLM/bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
# encoder = AutoModel.from_pretrained(bert_model_path)
#
# dataset_file_path = "/data1/botdet/datasets/Twibot-20-Format/"
# with open(dataset_file_path + 'node_info.json', 'r') as file:
#     users = ijson.items(file, 'item')
#     each_user_tweet = ''
#     for user in users:
#         for tweet in user['tweet']:
#             each_user_tweet += tweet
#         encode_text = encoder(**tokenizer(each_user_tweet, return_tensors='pt', truncation=True, max_length=512, padding=True)).last_hidden_state[:, 0, :][0].tolist()
#         print(encode_text)
#         break

import json
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
import numpy as np
import ijson
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from gensim.models.doc2vec import Doc2Vec
import nltk
nltk.data.path.append(r'/data1/botdet/LLM/nltk/nltk_data-gh-pages/packages')
nltk.data.path.append(r"/data1/botdet/LLM/nltk/wordnet31")
nltk.data.path.append(r"/data1/botdet/LLM/nltk/twitter_samples")
from nltk.corpus import wordnet


def calc_activate_days(created_at):
    created_at = created_at.strip()
    create_date = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S%z')
    crawl_date = datetime.strptime('2020 09 28 +0000', '%Y %m %d %z')
    delta_date = crawl_date - create_date
    return delta_date.days


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
    if os.path.exists(predata_file_path + 'users_feature.json'):
        with open(predata_file_path + 'users_feature.json', 'r') as fp:
            users_feature_dict = json.load(fp)
            return users_feature_dict

    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    encoder = AutoModel.from_pretrained(bert_model_path)
    num_properties_list = ['created_at', 'username', 'description', 'name',
                           {'public_metrics': ['followers_count', 'following_count', 'tweet_count', 'listed_count']}]

    bool_properties_list = ['entities', 'location', 'pinned_tweet_id', 'profile_image_url', 'protected',
                            'url', 'verified', 'withheld']
    text_properties_list = ['username', 'description']
    properties_number = len(num_properties_list) + 3 + len(bool_properties_list)

    with open(dataset_file_path + 'user.json', 'r') as file:
        users = ijson.items(file, 'item')
        users_feature_dict = {}
        for user in tqdm(users, desc='加载用户特征'):
            each_user_feature = []
            for user_property in num_properties_list:
                if isinstance(user_property, dict):
                    for count_property in user_property['public_metrics']:
                        if user["public_metrics"][count_property] is None:
                            each_user_feature.append(0)
                        else:
                            each_user_feature.append(user["public_metrics"][count_property])
                elif user.get(user_property) is None:
                    each_user_feature.append(0)
                else:
                    if user_property == 'created_at':
                        each_user_feature.append(calc_activate_days(user[user_property].strip()))
                    else:
                        each_user_feature.append(len(user[user_property]))

            for user_property in bool_properties_list:
                if user.get(user_property) is None:
                    each_user_feature.append(0)
                elif user_property in ['verified', 'protected']:
                    each_user_feature.append(int(user[user_property]))
                else:
                    each_user_feature.append(1)

            user_text = []
            for user_property in text_properties_list:
                user_text.append(user.get(user_property))
            user_text = f'{user_text[0]}: {user_text[1]}'
            encode_text = encoder(**tokenizer(user_text, return_tensors='pt')).last_hidden_state[:, 0, :][0].tolist()
            each_user_feature.extend(encode_text)
            users_feature_dict[user['id']] = each_user_feature

            # 属性值数目小于规定值时报错
            assert len(users_feature_dict[user['id']]) == properties_number + 768, \
                'user:{}, properties_number:{} < {}'.format(user['id'], len(users_feature_dict[user['id']]),
                                                            properties_number + 768)
    with open(predata_file_path + 'users_feature.json', 'w') as fp:
        json.dump(users_feature_dict, fp)

    users_id = users_feature_dict.keys()
    properties = [users_feature_dict[user_id] for user_id in users_id]

    properties = np.array(properties)
    for column in tqdm(range(properties_number), desc='用户特征z-score归一化', ncols=50):
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


def get_tweets_text():
    """
    寻找并保存tweet节点的文本
    """
    def custom_object_hook(dct):
        keys_to_filter = ["author_id", 'id', "text"]
        return {key: dct[key] for key in keys_to_filter if key in dct}

    if os.path.exists(predata_file_path + "tweets_text.json"):
        with open(predata_file_path + 'tweets_text.json', 'r') as fp:
            return json.load(fp)

    if not os.path.exists(predata_file_path + 'cache'):
        os.mkdir(predata_file_path + 'cache')

    each_user_tweet_num = {}
    for i in range(9):
        tweets_text_dict = {}
        with open(dataset_file_path + f'tweet_{i}.json', 'r') as fp:
            tweets = json.load(fp, object_hook=custom_object_hook)
            for tweet in tqdm(tweets, desc=f'tweet-{i}/8: 加载推文文本'):
                if each_user_tweet_num.get(tweet['id']) is None:
                    each_user_tweet_num[tweet['id']] = 0
                each_user_tweet_num[tweet['id']] += 1
                tweets_text_dict[tweet['id']] = tweet['text']
        with open(predata_file_path + f'cache/tweets_text{i}.json', 'w') as fp:
            json.dump(tweets_text_dict, fp)
    tweets_text_dict = {}
    for i in range(9):
        with open(predata_file_path + f'cache/tweets_text{i}.json', 'r') as fp:
            tweets_text_dict.update(json.load(fp))
    with open(predata_file_path + 'tweets_text.json', 'w') as fp:
        json.dump(tweets_text_dict, fp)
    with open(predata_file_path + 'each_user_tweet_num.json', 'w') as fp:
        json.dump(each_user_tweet_num, fp)
    return tweets_text_dict


def get_lists_feature():
    if os.path.exists(predata_file_path + 'lists_feature.json'):
        with open(predata_file_path + 'lists_feature.json', 'r') as fp:
            lists_feature_dict = json.load(fp)
            return lists_feature_dict

    # 加载预训练的句向量模型
    model = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_32.model")

    properties_list = ['created_at', 'follower_count', 'member_count', 'private', 'description', 'name']
    lists_feature_dict = {}
    with open(dataset_file_path + 'list.json', 'r') as fp:
        lists = ijson.items(fp, 'item')
        for tlist in tqdm(lists, desc='加载list特征'):
            feature_list = [calc_activate_days(tlist['created_at'].strip()),
                            tlist['follower_count'],
                            tlist['member_count'],
                            int(tlist['private'])]
            text_str = ''
            if tlist['name'] is not None:
                text_str += tlist['name']
            if tlist['description'] is not None:
                text_str += tlist['description']
            feature_list += model.infer_vector(text_str.split()).tolist()
            lists_feature_dict[tlist['id']] = feature_list

    users_id = list(lists_feature_dict.keys())
    properties = [lists_feature_dict[user_id] for user_id in users_id]

    properties = np.array(properties)
    for column in tqdm(range(len(properties_list) - 2), desc='list特征z-score归一化'):
        mean = np.mean(properties[:, column])  # 求平均值
        std = np.std(properties[:, column])  # 求标准差
        if std == 0:
            properties[:, column] = mean
        else:
            properties[:, column] = (properties[:, column] - mean) / std  # z-score归一化

    for index, user_id in enumerate(users_id):
        lists_feature_dict[user_id] = properties[index].tolist()

    with open(predata_file_path + 'lists_feature.json', 'w') as fp:
        json.dump(lists_feature_dict, fp)

    return lists_feature_dict


def get_node_label_split():
    """
    @return: user_label{user_id: (bot1,human0)}, user_split{user_id: (train0,val1,test2)}
    """
    if os.path.exists(predata_file_path + 'user_label.json') and os.path.exists(predata_file_path + 'user_split.json'):
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


def pytorch_edge_build(user_labels_dict, user_splits_dict, users_feature_dict, tweets_text_dict, lists_feature_dict):
    # if os.path.exists(predata_file_path + 'graph.pt'):
    #     return
    torch_graph = HeteroData()
    with open(predata_file_path + "/cache/user_id_index_map.json", 'r') as fp:
        user_id_index_map = json.load(fp)
    tweet_id_index_map = {}
    list_id_index_map = {}
    max_user_index = 0
    max_tweet_index = 0
    max_list_index = 0
    user_features = []
    tweet_texts = []
    list_features = []
    user_label = []
    user_split = []
    tweet_label = []
    edge_index_dict = {}
    chunk_size = 10000000
    edges = pd.read_csv(dataset_file_path + 'edge.csv', chunksize=chunk_size)
    for i, chunk_edges in enumerate(edges):
        for _, edge in tqdm(chunk_edges.iterrows(), desc=f'构建用户异质图-{i}'):
            source, relation, target = edge["source_id"], edge["relation"], edge["target_id"]
            if relation in ['followers', 'following']:
                if edge_index_dict.get(('user', relation, 'user')) is None:
                    edge_index_dict['user', relation, 'user'] = []
                if user_id_index_map.get(source) is None:
                    user_id_index_map[source] = max_user_index
                    max_user_index += 1
                    user_features.append(users_feature_dict[source])
                    user_label.append(user_labels_dict.get(source))
                    user_split.append(user_splits_dict.get(source))
                if user_id_index_map.get(target) is None:
                    user_id_index_map[target] = max_user_index
                    max_user_index += 1
                    user_features.append(users_feature_dict[target])
                    user_label.append(user_labels_dict.get(target))
                    user_split.append(user_splits_dict.get(target))
                edge_index_dict['user', relation, 'user'].append([user_id_index_map[source],
                                                                  user_id_index_map[target]])
            elif relation in ['post', 'pinned'] and tweets_text_dict.get(target) is not None:
                if edge_index_dict.get(('user', relation, 'tweet')) is None:
                    edge_index_dict['user', relation, 'tweet'] = []
                if user_id_index_map.get(source) is None:
                    user_id_index_map[source] = max_user_index
                    max_user_index += 1
                    user_features.append(users_feature_dict[source])
                    user_label.append(user_labels_dict.get(source))
                    user_split.append(user_splits_dict.get(source))
                if tweet_id_index_map.get(target) is None:
                    tweet_id_index_map[target] = max_tweet_index
                    max_tweet_index += 1
                    tweet_texts.append(tweets_text_dict[target])
                    tweet_label.append(user_labels_dict.get(source))
                edge_index_dict['user', relation, 'tweet'].append([user_id_index_map[source],
                                                                   tweet_id_index_map[target]])
            elif relation in ['own']:
                if edge_index_dict.get(('user', relation, 'list')) is None:
                    edge_index_dict['user', relation, 'list'] = []
                if user_id_index_map.get(source) is None:
                    user_id_index_map[source] = max_user_index
                    max_user_index += 1
                    user_features.append(users_feature_dict[source])
                    user_label.append(user_labels_dict.get(source))
                    user_split.append(user_splits_dict.get(source))
                if list_id_index_map.get(target) is None:
                    list_id_index_map[target] = max_list_index
                    max_list_index += 1
                    list_features.append(lists_feature_dict[target])
                edge_index_dict['user', relation, 'list'].append([user_id_index_map[source],
                                                                  list_id_index_map[target]])

    # with open(predata_file_path + "/cache/user_id_index_map.json", 'w') as fp:
    #     json.dump(user_id_index_map, fp)

    torch_graph['user'].x = torch.tensor(user_features, dtype=torch.float)
    torch_graph['tweet'].x = tweet_texts
    torch_graph['user'].node_label = torch.tensor([label if label is not None else -1 for label in user_label])
    torch_graph['tweet'].node_label = torch.tensor([label if label is not None else -1 for label in tweet_label])
    torch_graph['user'].node_split = torch.tensor(user_split, dtype=torch.int8)
    for edge_type in edge_index_dict.keys():
        torch_graph[edge_type].edge_index = torch.tensor(edge_index_dict[edge_type]).t()

    torch.save(torch_graph, predata_file_path + 'cache/pre_graph2.pt')


def buding(user_tweet_feature):
    user_tweet_feature = user_tweet_feature.reshape(-1, 128)
    return user_tweet_feature


def aggregate_tweet_feature():
    model_32 = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_32.model")
    model_128 = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_128.model")
    if not os.path.exists(predata_file_path + 'cache/user_tweet_feature.pt'):
        graph = torch.load(predata_file_path + 'cache/pre_graph.pt')
        each_user_tweet = ['' for _ in range(graph['user'].x.size(0))]
        user_tweet_num = torch.zeros(graph['user'].x.size(0))
        for edge in tqdm(graph['user', 'post', 'tweet'].edge_index.t().tolist(), desc='聚合推文信息'):
            each_user_tweet[edge[0]] += graph['tweet'].x[edge[1]]
            user_tweet_num[edge[0]] += 1
        del graph
        torch.save(user_tweet_num, predata_file_path + 'cache/user_tweet_num.pt')

        user_tweet_feature = torch.tensor([])
        for i, user_tweet in tqdm(enumerate(each_user_tweet), total=len(each_user_tweet), desc='编码每个用户的推文集合'):
            each_user_tweet_feature = torch.tensor(model_128.infer_vector(user_tweet.split()))
            user_tweet_feature = torch.stack((user_tweet_feature, each_user_tweet_feature), dim=0)
        torch.save(user_tweet_feature, predata_file_path + 'cache/user_tweet_feature.pt')
    else:
        user_tweet_feature = torch.load(predata_file_path + 'cache/user_tweet_feature.pt')
        user_tweet_num = torch.load(predata_file_path + 'cache/user_tweet_num.pt')

    user_tweet_feature = buding(user_tweet_feature)

    user_tweet_num = torch.where(torch.tensor(user_tweet_num == 0), 1, user_tweet_num)
    graph = torch.load(predata_file_path + 'cache/pre_graph.pt')
    user_tweet_feature = user_tweet_feature / user_tweet_num.view(-1, 1)
    graph['user'].x = torch.cat((graph['user'].x, user_tweet_feature), dim=1)

    origin_tweet_feature = []
    augment_tweet_feature = []
    for text in tqdm(graph['tweet'].x, desc='编码推文'):
        origin_tweet_feature.append(model_32.infer_vector(text.split()))
        augment_tweet_feature.append(model_32.infer_vector(wordnet_paraphrase(text).split()))
    graph['tweet'].x1 = torch.tensor(origin_tweet_feature)
    graph['tweet'].x2 = torch.tensor(augment_tweet_feature)
    del graph['tweet'].x

    torch.save(graph, predata_file_path + 'graph.pt')


def ggg():
    pre_graph = torch.load(predata_file_path + 'cache/pre_graph2.pt')
    # print(pre_graph['user'].x.size())
    # print(pre_graph['tweet'].x.size())
    # print(pre_graph.edge_types)
    print(pre_graph)

    graph = torch.load(predata_file_path + 'graph.pt')
    # print(graph['user'].x.size())
    # print(graph['tweet'].x.size())
    print(graph)


if __name__ == "__main__":
    dataset_file_path = "/data1/botdet/datasets/Twibot-22-datasets/"
    predata_file_path = "./predata/twibot22/"
    doc2vec_model_path = "./predata/wiki_doc2vec/"
    bert_model_path = "/data1/botdet/LLM/bert-base-uncased"
    if not os.path.exists('./predata'):
        os.mkdir('./predata')
    if not os.path.exists(predata_file_path):
        os.mkdir(predata_file_path)

    graph_types_map = {'train': 0, 'val': 1, 'test': 2}

    # user_feature = get_users_feature()
    # tweet_text = get_tweets_text()
    # list_feature = get_lists_feature()
    # users_label, users_split = get_node_label_split()
    # pytorch_edge_build(users_label, users_split, user_feature, tweet_text, list_feature)
    # aggregate_tweet_feature()

    # ggg()

    graph: HeteroData = torch.load(predata_file_path + 'graph.pt')
    print(graph)

