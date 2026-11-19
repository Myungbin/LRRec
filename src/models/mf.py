"""MF (MF-BPR, ID-based baseline).

Reference:
    Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme.
    "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI 2009.

Matrix factorization trained with BPR: score(u, i) = e_u . e_i, one sampled negative, L2 on the batch's
embeddings with the same scaling as LightGCN (so ``reg_weight`` means the same thing in both models;
MF is LightGCN with 0 layers).
"""

import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class MF(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(MF, self).__init__(config, dataloader)
        self.embedding_dim = config["embedding_size"]
        self.reg_weight = config["reg_weight"]

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        if neg_items.dim() == 2:
            if neg_items.shape[1] != 1:
                raise ValueError("MF uses one BPR negative; set neg_sample_num: 1")
            neg_items = neg_items[:, 0]
        u_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        neg_scores = (u_emb * neg_emb).sum(-1)
        pos_scores = (u_emb * pos_emb).sum(-1)
        bpr_loss = F.softplus(neg_scores - pos_scores).mean()
        reg_loss = 0.5 * (u_emb.pow(2).sum() + pos_emb.pow(2).sum() + neg_emb.pow(2).sum()) / users.shape[0]
        return bpr_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        return self.user_embedding(interaction[0]) @ self.item_embedding.weight.T
