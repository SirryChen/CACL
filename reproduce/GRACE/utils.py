import random
import numpy as np
import torch
from argparse import ArgumentParser


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def super_parament_initial():
    s_parament = ArgumentParser()
    s_parament.add_argument('--dataset', default='twibot20', choices=['twibot20', 'twibot22'])
    args = s_parament.parse_args()
    if args.dataset == "twibot20":
        s_parament.add_argument('--des_size', default=768)
        s_parament.add_argument('--tweet_size', default=768)
        s_parament.add_argument('--num_prop_size', default=6)
        s_parament.add_argument('--cat_prop_size', default=11)
        s_parament.add_argument('--learning_rate', default=1e-3)
        s_parament.add_argument('--embedding_dim', default=256)
        s_parament.add_argument('--num_proj_hidden', default=256)
        s_parament.add_argument('--num_layer', default=2)
        s_parament.add_argument('--dropout', default=0.3)
        s_parament.add_argument('--drop_edge_rate_1', default=0.1)
        s_parament.add_argument('--drop_edge_rate_2', default=0.4)
        s_parament.add_argument('--drop_feature_rate_1', default=0.1)
        s_parament.add_argument('--drop_feature_rate_2', default=0.0)
        s_parament.add_argument('--tau', default=0.7)
        s_parament.add_argument('--num_epochs', default=500)
        s_parament.add_argument('--weight_decay', default=1e-5)

    elif args.dataset == 'twibot22':
        s_parament.add_argument('--des_size', default=768)
        s_parament.add_argument('--tweet_size', default=128)
        s_parament.add_argument('--num_prop_size', default=8)
        s_parament.add_argument('--cat_prop_size', default=8)
        s_parament.add_argument('--learning_rate', default=1e-3)
        s_parament.add_argument('--num_hidden', default=256)
        s_parament.add_argument('--num_proj_hidden', default=256)
        s_parament.add_argument('--num_layer', default=2)
        s_parament.add_argument('--dropout', default=0.3)
        s_parament.add_argument('--drop_edge_rate_1', default=0.1)
        s_parament.add_argument('--drop_edge_rate_2', default=0.4)
        s_parament.add_argument('--drop_feature_rate_1', default=0.1)
        s_parament.add_argument('--drop_feature_rate_2', default=0.0)
        s_parament.add_argument('--tau', default=0.7)
        s_parament.add_argument('--num_epochs', default=250)
        s_parament.add_argument('--weight_decay', default=1e-5)

    return s_parament
