# coding: utf-8
r"""
SimGCL (ID-based baseline)
################################################
Reference:
    Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, Quoc Viet Hung Nguyen.
    "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation." SIGIR 2022.
    Official code: https://github.com/Coder-Yu/SELFRec

LightGCN encoder without E^(0) in the final representation (mean of layers 1..L). Three propagations per step:

    recommendation  E   = mean_{l=1..L} A^l E^(0)                       noise-free, BPR loss
    two CL views    E'  = mean_{l=1..L} E'^(l),  E'^(l) = A E'^(l-1) + Delta'^(l)    (same for E'')
                    Delta = eps * normalize(U(0,1)^d) * sign(E^(l))       different noise per view, node, layer
    loss = BPR + lambda_cl * (InfoNCE(E'_u, E''_u) + InfoNCE(E'_i, E''_i)) + reg_weight * L2

InfoNCE runs over the unique users / positive items of the batch with temperature ``tau_cl``. The L2 term has
the same scaling as LightGCN / XSimGCL in this library. XSimGCL (``models/xsimgcl.py``) is the variant with a
single perturbed propagation shared by both losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_graph


class SimGCL(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(SimGCL, self).__init__(config, dataloader)
        self.embedding_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.eps = config["eps"]
        self.lambda_cl = config["lambda_cl"]
        self.tau_cl = config["tau_cl"]
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

    def _noise(self, emb):
        return torch.sign(emb) * F.normalize(torch.rand_like(emb), dim=-1) * self.eps

    def forward(self, perturbed=False):
        emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layers = []
        for _ in range(self.n_layers):
            emb = self.norm_adj.mm(emb)
            if perturbed and self.eps > 0:
                emb = emb + self._noise(emb)
            layers.append(emb)
        final = torch.stack(layers, dim=1).mean(dim=1)
        return torch.split(final, [self.n_users, self.n_items])

    def _info_nce(self, a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = a @ b.T / self.tau_cl
        return F.cross_entropy(logits, torch.arange(a.shape[0], device=a.device))

    def calculate_loss(self, interaction):
        users, pos, neg = interaction[0], interaction[1], interaction[2]
        if neg.dim() == 2:
            if neg.shape[1] != 1:
                raise ValueError("SimGCL uses one BPR negative; set neg_sample_num: 1")
            neg = neg[:, 0]
        user_e, item_e = self.forward(perturbed=False)
        u, p, n = user_e[users], item_e[pos], item_e[neg]
        bpr = F.softplus((u * n).sum(-1) - (u * p).sum(-1)).mean()

        user_v1, item_v1 = self.forward(perturbed=True)
        user_v2, item_v2 = self.forward(perturbed=True)
        uu, ui = torch.unique(users), torch.unique(pos)
        cl = self._info_nce(user_v1[uu], user_v2[uu]) + self._info_nce(item_v1[ui], item_v2[ui])

        reg = 0.5 * (
            self.user_embedding(users).pow(2).sum()
            + self.item_embedding(pos).pow(2).sum()
            + self.item_embedding(neg).pow(2).sum()
        ) / users.shape[0]
        return bpr + self.lambda_cl * cl + self.reg_weight * reg

    def train(self, mode=True):
        self._cache = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        if self._cache is None:
            self._cache = self.forward(perturbed=False)
        user_e, item_e = self._cache
        return user_e[interaction[0]] @ item_e.T
