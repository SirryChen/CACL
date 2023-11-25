import networkx as nx
import json
import pandas as pd
import community
import torch
from torch_geometric.data import HeteroData
import os
import numpy as np
import ijson
from tqdm import tqdm
from utils import calc_activate_days
from CD_utils import compress_graph
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


def get_node_label_split(load_from_local=False):
    """
    获取所有节点标签，推文的标签与所有者的标签一致，bot为1，human为0;获取所有节点的分类
    @param load_from_local: 是否从本地加载
    @return: 用户分类{user_id: user_split}，推文分类{tweet_id: tweet_split}，用户标签，推文标签
    """
    if load_from_local:
        print("从本地加载数据...")
        with open(predata_file_path + 'user_label.json', 'r') as fp:
            user_labels = json.load(fp)
        with open(predata_file_path + 'tweet_label.json', 'r') as fp:
            tweet_labels = json.load(fp)
        with open(predata_file_path + 'user_split.json', 'r') as fp:
            user_splits = json.load(fp)
        with open(predata_file_path + 'tweet_split.json', 'r') as fp:
            tweet_splits = json.load(fp)
    else:
        labels = pd.read_csv(dataset_file_path + 'label.csv')
        splits = pd.read_csv(dataset_file_path + 'split.csv')
        user_labels = {}
        user_splits = {}
        for _, user in tqdm(splits.iterrows(), desc='加载用户分类'):
            user_splits[user['id']] = user['split']
        for _, user in tqdm(labels.iterrows(), desc='加载用户标签'):
            user_labels[user['id']] = int(user['label'] == 'bot')
        tweet_labels, tweet_splits = get_tweet_label_split(user_labels, user_splits)

        with open(predata_file_path + 'user_label.json', 'w') as fp:
            json.dump(user_labels, fp)
        with open(predata_file_path + 'tweet_label.json', 'w') as fp:
            json.dump(tweet_labels, fp)
        with open(predata_file_path + 'user_split.json', 'w') as fp:
            json.dump(user_splits, fp)
        with open(predata_file_path + 'tweet_split.json', 'w') as fp:
            json.dump(tweet_splits, fp)

    return user_splits, tweet_splits, user_labels, tweet_labels


def get_tweet_label_split(author_label, author_split):
    """
    通过作者的标签，获取所有推文的标签
    @param author_label: 作者标签字典{user_id: user_label}
    @param author_split: 作者分类字典{user_id: user_split}
    @return: 推文标签{tweet_id: tweet_label}, 推文分类{tweet_id: tweet_split}
    """

    def custom_object_hook(dct):
        keys_to_filter = ['author_id', 'id']
        return {key: dct[key] for key in keys_to_filter if key in dct}

    tweet_labels = {}
    tweet_splits = {}
    for i in range(9):
        with open(dataset_file_path + f'tweet_{i}.json', 'r') as fp:
            tweets = json.load(fp, object_hook=custom_object_hook)
            for tweet in tqdm(tweets, desc=f'{i}/8: 加载推文类别和标签', ncols=50):
                tweet_labels[tweet['id']] = author_label['u' + str(tweet['author_id'])]
                tweet_splits[tweet['id']] = author_split['u' + str(tweet['author_id'])]

    return tweet_labels, tweet_splits


def edge_build(load_from_local=False):
    """
    生成本次对应类的图，并进行社区检测分类为子图
    @param load_from_local: 是否从本地加载
    """
    save_file_path = predata_file_path
    # 构建异质图
    if os.path.exists(save_file_path + 'graph.gml'):
        print('从本地加载初始异质图...')
        graph = nx.read_gml(save_file_path + 'graph.gml')
    else:
        graph = nx.MultiDiGraph()
        chunk_size = 10000000
        edges = pd.read_csv(dataset_file_path + 'edge.csv', chunksize=chunk_size)
        for i, chunk_edges in enumerate(edges):
            for _, edge in tqdm(chunk_edges.iterrows(), desc=f'构建用户初始异质图-{i}'):
                source, relation, target = edge["source_id"], edge["relation"], edge["target_id"]

                if relation in ["followers", "following"]:
                    graph.add_node(source, type=1)  # type=用户
                    graph.add_node(target, type=1)  # type=用户
                    graph.add_edge(source, target, type=relation_int_map[relation])

        # 去除可能存在的孤立点
        isolates = list(nx.isolates(graph))
        graph.remove_nodes_from(isolates)
        nx.write_gml(graph, save_file_path + 'graph.gml')

    networkx_file_path = save_file_path + 'networkx_subgraph_without_feature/'
    if load_from_local:
        with open(networkx_file_path + 'partition.json', 'r') as fp:
            partition = json.load(fp)
            community_ids = set(partition.values())
    else:
        print('保存networkx格式子图')
        random_seed = 17
        partition = community.best_partition(graph.to_undirected(), random_state=random_seed, resolution=1.1)
        community_ids = set(partition.values())
        community_ids, partition = compress_graph(community_ids, partition)
        if not os.path.exists(networkx_file_path):
            os.mkdir(networkx_file_path)
        for community_id in community_ids:
            community_nodes = [node for node, comm_id in partition.items() if comm_id == community_id]
            subgraph = graph.subgraph(community_nodes).copy()  # 无需deepcopy
            nx.write_gml(subgraph, networkx_file_path + f'subgraph_{community_id}.gml')
        with open(networkx_file_path + 'partition.json', 'w') as fp:
            json.dump(partition, fp)

    return community_ids, partition


def complete_edge_build(community_ids, partition):
    networkx_file_path = predata_file_path + 'networkx_subgraph_without_feature/'
    for community_id in community_ids:
        subgraph = nx.read_gml(networkx_file_path + f'subgraph_{community_id}.gml')
        chunk_size = 10000000
        edges = pd.read_csv(dataset_file_path + 'edge.csv', chunksize=chunk_size)
        for i, chunk_edges in enumerate(edges):
            for _, edge in tqdm(chunk_edges.iterrows(),
                                desc=f'构建完整初始异质图-{community_id}/{len(community_ids)}-{i}'):
                source, relation, target = edge["source_id"], edge["relation"], edge["target_id"]
                if partition.get(source) == community_id:
                    if relation == 'own':
                        subgraph.add_node(target, type=2)
                        subgraph.add_edge(source, target, type=relation_int_map[relation])
                    elif relation in ['pinned', 'post']:
                        subgraph.add_node(target, type=3)
                        subgraph.add_edge(source, target, type=relation_int_map[relation])

        nx.write_gml(subgraph, networkx_file_path + f'complete_subgraph_{community_id}.gml')


def pytorch_edge_build(community_ids, user_labels, tweet_labels, user_splits, tweet_splits):
    """
    将图保存为pytorch格式
    @param community_ids: 每个社区的id[id1,id2,]
    @param user_labels: 用户标签{user_id: user_label}
    @param tweet_labels: 推文标签{tweet_id: tweet_label}
    @param user_splits: 用户分类{user_id: user_split}
    @param tweet_splits: 推文分类{tweet_id: tweet_split}
    """
    save_file_path = predata_file_path
    networkx_file_path = save_file_path + 'networkx_subgraph_without_feature/'
    pytorch_file_path = save_file_path + "pytorch_subgraph_without_feature/"
    if not os.path.exists(pytorch_file_path):
        os.mkdir(pytorch_file_path)
    # 社区检测分割子图
    subgraph_scale = {}
    for community_id in tqdm(community_ids, desc='转换为pytorch子图'):
        subgraph = nx.read_gml(networkx_file_path + f'complete_subgraph_{community_id}.gml')
        # 转化为pytorch格式异质图
        torch_subgraph = HeteroData()
        for node_type in set(nx.get_node_attributes(subgraph, 'type').values()):
            node_id = [node for node in subgraph.nodes(data=False) if subgraph.nodes[node]['type'] == node_type]
            if node_type == 1:       # user
                torch_subgraph[node_map[node_type]].node_label = [user_labels[node] for node in node_id]
                torch_subgraph[node_map[node_type]].node_split = [user_splits[node] for node in node_id]
            elif node_type == 3:     # tweet
                torch_subgraph[node_map[node_type]].node_label = [tweet_labels[node] for node in node_id]
                torch_subgraph[node_map[node_type]].node_label = [tweet_splits[node] for node in node_id]
            torch_subgraph[node_map[node_type]].node_id = node_id

        for edge_type in set(nx.get_edge_attributes(subgraph, 'type').values()):
            edges = [(source, target) for source, target, data in subgraph.edges(data=True)
                     if data['type'] == edge_type]
            edge_index = []
            source_type = subgraph.nodes(data=True)[edges[0][0]]['type']
            target_type = subgraph.nodes(data=True)[edges[0][1]]['type']
            source_index_map = {node_id: i for i, node_id in enumerate(torch_subgraph[node_map[source_type]].node_id)}
            target_index_map = {node_id: i for i, node_id in enumerate(torch_subgraph[node_map[target_type]].node_id)}
            for source, target in edges:
                u = source_index_map[source]
                v = target_index_map[target]
                edge_index.append((u, v))
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            torch_subgraph[node_map[source_type], int_relation_map[edge_type], node_map[target_type]].edge_index \
                = edge_index  # [2,num_edges]

        # 保存为pytorch格式
        torch.save(torch_subgraph, pytorch_file_path + f"subgraph_{community_id}.pt")
        for node_type in set(nx.get_node_attributes(subgraph, 'type').values()):
            subgraph_scale[f"subgraph_{community_id}_{node_type}_num"] = \
                len(torch_subgraph.node_id_dict[node_map[node_type]])
    subgraph_scale['subgraph_num'] = len(community_ids)
    with open(save_file_path + 'subgraph_scale.json', 'w') as fp:
        json.dump(subgraph_scale, fp, indent=4)


def get_users_feature():
    """
    提取用户的特征，数字特征+文本特征（描述+名字）
    @return: 用户特征字典{user_id: user_feature}
    """
    if os.path.exists(predata_file_path + 'users_feature.json'):
        with open(predata_file_path + 'users_feature.json', 'r') as fp:
            users_feature_dict = json.load(fp)
            return users_feature_dict

    model = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_32.model")
    properties_list = ['created_at', 'description', 'entities', 'location',
                       'pinned_tweet_id', 'profile_image_url', 'protected',
                       'url', 'username', 'verified', 'withheld',
                       {'public_metrics': ['followers_count', 'following_count', 'tweet_count', 'listed_count']}]
    properties_number = 15

    with open(dataset_file_path + 'user.json', 'r') as file:
        users = ijson.items(file, 'item')
        users_feature_dict = {}
        for user in tqdm(users, desc='加载用户特征'):
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
                elif user.get(user_property) is None:  # 属性为空则补0
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


def get_tweets_text():
    """
    为每一个子图寻找并保存其包含的tweet节点的文本
    """
    def custom_object_hook(dct):
        keys_to_filter = ['text', 'id']
        return {key: dct[key] for key in keys_to_filter if key in dct}

    origin_subgraph_file_path = predata_file_path + '/pytorch_subgraphs_without_feature/'
    for n, origin_subgraph_file in enumerate(os.listdir(origin_subgraph_file_path)):
        if os.path.exists(origin_subgraph_file_path + origin_subgraph_file.replace('.pt', '_') + 'tweets_text.json') \
                or origin_subgraph_file.find('.pt') == -1:
            continue
        subgraph = torch.load(origin_subgraph_file_path + origin_subgraph_file)
        sub_tweet_id = set(subgraph['tweet'].node_id)
        del subgraph

        sub_tweets_text_dict = {}
        for i in range(9):
            with open(dataset_file_path + f'tweet_{i}.json', 'r') as fp:
                tweets = json.load(fp, object_hook=custom_object_hook)
                for tweet in tqdm(tweets, desc=f'graph-{n}-tweet-{i}/8: 加载推文文本'):
                    if tweet['id'] in sub_tweet_id:
                        sub_tweets_text_dict[tweet['id']] = tweet['text']

        with open(origin_subgraph_file_path + origin_subgraph_file.replace('.pt', '_') + 'tweets_text.json', 'w') as fp:
            json.dump(sub_tweets_text_dict, fp)


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


def graph_with_feature(users_feature_dict, lists_feature_dict):
    """
    为图中各节点添加属性特征，保存特征维度
    @param users_feature_dict: {user_id: user_feature}
    @param lists_feature_dict: {list_id: list_feature}
    """
    origin_subgraph_file_path = predata_file_path + '/pytorch_subgraph_without_feature/'
    save_file_path = predata_file_path + '/pytorch_subgraph/'
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)

    for origin_subgraph_file in tqdm(os.listdir(origin_subgraph_file_path),
                                     desc=f'{origin_subgraph_file_path}'):
        if origin_subgraph_file.find('.pt') == -1:
            continue
        # 加载子图以及其包含的tweet
        subgraph = torch.load(origin_subgraph_file_path + origin_subgraph_file)
        with open(origin_subgraph_file_path + origin_subgraph_file.replace('.pt', '_') + 'tweets_text.json', 'r') as fp:
            sub_tweets_text_dict = json.load(fp)

        for node_type in subgraph.node_types:
            if node_type == 'user':
                subgraph[node_type].x = torch.tensor([users_feature_dict[node_id]
                                                     for node_id in subgraph[node_type].node_id])
            elif node_type == 'tweet':
                subgraph[node_type].x = [sub_tweets_text_dict[node_id]
                                         for node_id in subgraph[node_type].node_id]
            elif node_type == 'list':
                subgraph[node_type].x = torch.tensor([lists_feature_dict[node_id]
                                                     for node_id in subgraph[node_type].node_id])
            del subgraph[node_type].node_id     # 后续不需要该属性
        torch.save(subgraph, save_file_path + origin_subgraph_file)

    feature_dim_dict = {'user': len(list(users_feature_dict.values())[0]),
                        'tweet': 32,
                        'list': len(list(lists_feature_dict.values())[0])}
    with open(predata_file_path + 'feature_dim_dict.json', 'w') as fp:
        json.dump(feature_dim_dict, fp)


def full_graph(subgraph_type):
    edge = [('user', 'followers', 'user'), ('user', 'following', 'user'), ('user', 'post', 'tweet'),
            ('user', 'pinned', 'tweet'), ('user', 'own', 'list')]
    graph_path = f"/data1/botdet/ContrastiveLearningOnGraph/my_CL/predata/twibot22/{subgraph_type}"
    subgraph_path = graph_path + '/pytorch_subgraphs/'
    graph = HeteroData()
    index_map = {}
    for node_type in node_map.values():
        graph[node_type].x = []
        if node_type != 'list':
            graph[node_type].node_label = []
        index_map[node_type] = 0
    graph['user'].node_split = torch.tensor([])
    for edge_type in edge:
        graph[edge_type].edge_index = torch.tensor([[], []])
    for sub_path in tqdm(os.listdir(subgraph_path), desc='融合子图'):
        subgraph = torch.load(subgraph_path + sub_path)
        for edge_type in subgraph.edge_types:
            edge_index = subgraph[edge_type].edge_index
            source_type = edge_type[0]
            target_type = edge_type[2]
            for j in range(edge_index.size(1)):
                edge_index[0, j] = edge_index[0, j] + index_map[source_type]
                edge_index[1, j] = edge_index[1, j] + index_map[target_type]
            graph[edge_type].edge_index = torch.cat((graph[edge_type].edge_index, edge_index), dim=1)
        for node_type in subgraph.node_types:
            graph[node_type].x += subgraph[node_type].x
            if hasattr(subgraph[node_type], "node_label"):
                graph[node_type].node_label += subgraph[node_type].node_label
            index_map[node_type] = len(graph[node_type].x)
    for node_type in graph.node_types:
        if node_type == 'user':
            graph[node_type].node_split = torch.ones(len(graph[node_type].x)) * user_split[subgraph_type]
        if node_type != 'tweet':
            graph[node_type].x = torch.stack(graph[node_type].x, dim=0)
    torch.save(graph, graph_path + '/graph.pt')


def aggregate_tweet_feature():
    if not os.path.exists(predata_file_path + 'cache/'):
        os.mkdir(predata_file_path + 'cache/')
    model_32 = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_32.model")
    model_128 = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_128.model")

    graph = torch.load(predata_file_path + 'graph.pt')

    for edge_type in graph.edge_types:
        graph[edge_type].edge_index = graph[edge_type].edge_index.long()

    user_tweet_feature = torch.zeros(graph['user'].x.size(0), 128)
    user_tweet_num = torch.zeros(graph['user'].x.size(0))
    for edge in tqdm(graph['user', 'post', 'tweet'].edge_index.t().tolist(), desc='聚合推文信息至用户节点'):
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


def total_graph():
    edge = [('user', 'followers', 'user'), ('user', 'following', 'user'), ('user', 'post', 'tweet'),
            ('user', 'pinned', 'tweet'), ('user', 'own', 'list')]
    each_full_graph_path = ['/data1/botdet/ContrastiveLearningOnGraph/my_CL/predata/twibot22/train/graph.pt',
                            '/data1/botdet/ContrastiveLearningOnGraph/my_CL/predata/twibot22/val/graph.pt',
                            '/data1/botdet/ContrastiveLearningOnGraph/my_CL/predata/twibot22/test/graph.pt']
    graph = HeteroData()
    index_map = {}
    for node_type in node_map.values():
        graph[node_type].x = []
        if node_type != 'list':
            graph[node_type].node_label = []
        if node_type == 'user':
            graph[node_type].node_split = torch.tensor([])
        index_map[node_type] = 0
    for edge_type in edge:
        graph[edge_type].edge_index = torch.tensor([[], []])
    for full_graph_path in each_full_graph_path:
        subgraph = torch.load(full_graph_path)
        print(subgraph.edge_types)
        for edge_type in subgraph.edge_types:
            edge_index = subgraph[edge_type].edge_index
            source_type = edge_type[0]
            target_type = edge_type[2]
            for j in range(edge_index.size(1)):
                edge_index[0, j] = edge_index[0, j] + index_map[source_type]
                edge_index[1, j] = edge_index[1, j] + index_map[target_type]
            graph[edge_type].edge_index = torch.cat((graph[edge_type].edge_index, edge_index), dim=1)
        for node_type in subgraph.node_types:
            graph[node_type].x += subgraph[node_type].x
            if hasattr(subgraph[node_type], "node_label"):
                graph[node_type].node_label += subgraph[node_type].node_label
            if hasattr(subgraph[node_type], "node_split"):
                graph[node_type].node_split = torch.cat((graph[node_type].node_split,
                                                         subgraph[node_type].node_split), dim=0)
            index_map[node_type] = len(graph[node_type].x)
    for node_type in graph.node_types:
        if node_type != 'tweet':
            graph[node_type].x = torch.stack(graph[node_type].x, dim=0)
    torch.save(graph, predata_file_path + 'graph.pt')


if __name__ == "__main__":
    node_map = {1: 'user', 2: 'list', 3: 'tweet'}
    relation_int_map = {'followers': 1, 'following': 2, 'own': 3, 'pinned': 4, 'post': 5}

    int_relation_map = {v: k for k, v in relation_int_map.items()}

    dataset_file_path = "/data1/botdet/datasets/Twibot-22-datasets/"
    predata_file_path = "./predata/twibot22/"
    doc2vec_model_path = "./predata/wiki_doc2vec/"

    # user_split, tweet_split, user_label, tweet_label = get_node_label_split()
    # communities_id, partitions = edge_build()
    # complete_edge_build(communities_id, partitions)
    # pytorch_edge_build(communities_id, user_label, tweet_label, user_split, tweet_split)
    #
    # user_feature_dict = get_users_feature()
    # list_feature_dict = get_lists_feature()
    # get_tweets_text()
    # graph_with_feature(user_feature_dict, list_feature_dict)
    # full_graph()
    user_split = {'train': 0, 'val': 1, 'test': 2}
    # for graph_type in ['train', 'val', 'test']:
    #     full_graph(graph_type)

    # total_graph()     # NOTE 修补图数据使用
    aggregate_tweet_feature()
