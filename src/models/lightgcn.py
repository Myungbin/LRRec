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
        self._cache = None

    def forward(self):
        emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        summed = emb
        for _ in range(self.n_layers):
            emb = self.norm_adj.mm(emb)
            summed = summed + emb
        final = summed / (self.n_layers + 1)
        return torch.split(final, [self.n_users, self.n_items])

    def calculate_loss(self, interaction):
        users, pos, neg = interaction[0], interaction[1], interaction[2]
        if neg.dim() == 2:
            if neg.shape[1] != 1:
                raise ValueError("LightGCN uses one BPR negative; set neg_sample_num: 1")
            neg = neg[:, 0]
        user_e, item_e = self.forward()
        u, p, n = user_e[users], item_e[pos], item_e[neg]
        bpr = F.softplus((u * n).sum(-1) - (u * p).sum(-1)).mean()
        l2 = (
            self.user_embedding(users).pow(2).sum()
            + self.item_embedding(pos).pow(2).sum()
            + self.item_embedding(neg).pow(2).sum()
        )
        reg = 0.5 * l2 / users.shape[0]
        return bpr + self.reg_weight * reg

    def train(self, mode=True):
        self._cache = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        if self._cache is None:
            self._cache = self.forward()
        user_e, item_e = self._cache
        return user_e[interaction[0]] @ item_e.T
