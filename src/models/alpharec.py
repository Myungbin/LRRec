# coding: utf-8
r"""
AlphaRec
################################################
Reference:
    Leheng Sheng, An Zhang, Yi Zhang, Yuxin Chen, Xiang Wang, Tat-Seng Chua.
    "Language Representations Can be What Recommenders Need: Findings and Potentials." ICLR 2025.
    Official code: https://github.com/LehengTHU/AlphaRec

ID-free collaborative filtering on frozen item language representations z_i:

    z_u   = mean_{i in I_u} z_i                     (user LR = mean of the training history)
    e^(0) = MLP(z)                                   2-layer MLP + LeakyReLU, shared by users and items
    e     = mean_{l=0..L} A_hat^l e^(0)             LightGCN propagation on the user-item graph
    loss  = InfoNCE on cosine scores, temperature tau, sampled negatives (neg_sample_num)
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender, lr_mlp_hidden
from utils.utils import build_norm_adj, scipy_to_torch_sparse


class AlphaRec(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(AlphaRec, self).__init__(config, dataloader)

        self.embedding_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.tau = config["tau"]

        inter = dataloader.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = build_norm_adj(inter, self.n_users, self.n_items).to(self.device)

        # user LR = mean of the history's item LRs (users without history get a zero vector)
        deg = np.asarray(inter.sum(axis=1)).ravel()
        d_inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
        R_mean = scipy_to_torch_sparse(sp.diags(d_inv) @ inter).to(self.device)
        self.register_buffer("Z", self.lr_feat, persistent=False)
        self.register_buffer("Z_user", torch.sparse.mm(R_mean, self.lr_feat), persistent=False)
        del self.lr_feat

        d = self.Z.shape[1]
        h = lr_mlp_hidden(d, self.lm_model)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.LeakyReLU(), nn.Linear(h, self.embedding_dim))

        self._cache = None

    def forward(self):
        all_e = torch.cat([self.mlp(self.Z_user), self.mlp(self.Z)], dim=0)
        layers = [all_e]
        for _ in range(self.n_layers):
            all_e = torch.sparse.mm(self.norm_adj, all_e)
            layers.append(all_e)
        all_e = torch.stack(layers, dim=1).mean(dim=1)
        return torch.split(all_e, [self.n_users, self.n_items])

    def calculate_loss(self, interaction):
        users, pos = interaction[0], interaction[1]
        user_e, item_e = self.forward()
        u = F.normalize(user_e[users], dim=-1)
        p = F.normalize(item_e[pos], dim=-1)
        if len(interaction) < 3:  # in-batch negatives
            logits = u @ p.T / self.tau
            return F.cross_entropy(logits, torch.arange(u.shape[0], device=u.device))
        neg = interaction[2]
        if neg.dim() == 1:
            neg = neg.unsqueeze(1)
        n = F.normalize(item_e[neg], dim=-1)
        logits = torch.cat([(u * p).sum(-1, keepdim=True), torch.einsum("bd,bnd->bn", u, n)], dim=1) / self.tau
        return F.cross_entropy(logits, torch.zeros(u.shape[0], dtype=torch.long, device=u.device))

    def train(self, mode=True):
        self._cache = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        if self._cache is None:
            user_e, item_e = self.forward()
            self._cache = (F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1))
        user_e, item_e = self._cache
        return user_e[interaction[0]] @ item_e.T
