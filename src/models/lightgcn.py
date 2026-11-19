"""LightGCN (ID-based baseline).

Reference:
    Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang.
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.

Ported from the AlphaFree research repo (``models/LightGCN.py``): ID embeddings, normalized bipartite graph,
mean of layers 0..L, BPR with one sampled negative, L2 on the batch's ego embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_graph


class LightGCN(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(LightGCN, self).__init__(config, dataloader)
        self.embedding_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.norm_adj = build_graph(
            dataloader.inter_matrix(form="coo"), self.n_users, self.n_items, self.device, config["deterministic"]
        )
        self._eval_emb = None

    def forward(self):
        ego_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_emb, emb_sum = ego_emb, ego_emb
        for _ in range(self.n_layers):
            layer_emb = self.norm_adj.mm(layer_emb)
            emb_sum = emb_sum + layer_emb
        all_emb = emb_sum / (self.n_layers + 1)
        return torch.split(all_emb, [self.n_users, self.n_items])

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        if neg_items.dim() == 2:
            if neg_items.shape[1] != 1:
                raise ValueError("LightGCN uses one BPR negative; set neg_sample_num: 1")
            neg_items = neg_items[:, 0]
        user_emb, item_emb = self.forward()
        u_emb, pos_emb, neg_emb = user_emb[users], item_emb[pos_items], item_emb[neg_items]
        neg_scores = (u_emb * neg_emb).sum(-1)
        pos_scores = (u_emb * pos_emb).sum(-1)
        bpr_loss = F.softplus(neg_scores - pos_scores).mean()

        ego_l2 = (
            self.user_embedding(users).pow(2).sum()
            + self.item_embedding(pos_items).pow(2).sum()
            + self.item_embedding(neg_items).pow(2).sum()
        )
        reg_loss = 0.5 * ego_l2 / users.shape[0]
        return bpr_loss + self.reg_weight * reg_loss

    def train(self, mode=True):
        self._eval_emb = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        if self._eval_emb is None:
            self._eval_emb = self.forward()
        user_emb, item_emb = self._eval_emb
        return user_emb[interaction[0]] @ item_emb.T
