"""AlphaRec.

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
from utils.utils import build_graph, scipy_to_torch_sparse


class AlphaRec(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(AlphaRec, self).__init__(config, dataloader)

        self.embedding_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.tau = config["tau"]

        inter_mat = dataloader.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = build_graph(inter_mat, self.n_users, self.n_items, self.device, config["deterministic"])

        # user LR = mean of the history's item LRs (users without history get a zero vector)
        user_deg = np.asarray(inter_mat.sum(axis=1)).ravel()
        inv_deg = np.where(user_deg > 0, 1.0 / np.maximum(user_deg, 1e-12), 0.0)
        hist_mean = scipy_to_torch_sparse(sp.diags(inv_deg) @ inter_mat).to(self.device)
        self.register_buffer("item_lr", self.lr_feat, persistent=False)
        self.register_buffer("user_lr", torch.sparse.mm(hist_mean, self.lr_feat), persistent=False)
        del self.lr_feat

        lr_dim = self.item_lr.shape[1]
        hidden_dim = lr_mlp_hidden(lr_dim, self.lm_model)
        self.mlp = nn.Sequential(
            nn.Linear(lr_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, self.embedding_dim)
        )

        self._eval_emb = None

    def forward(self):
        ego_emb = torch.cat([self.mlp(self.user_lr), self.mlp(self.item_lr)], dim=0)
        layer_emb, layer_embs = ego_emb, [ego_emb]
        for _ in range(self.n_layers):
            layer_emb = self.norm_adj.mm(layer_emb)
            layer_embs.append(layer_emb)
        all_emb = torch.stack(layer_embs, dim=1).mean(dim=1)
        return torch.split(all_emb, [self.n_users, self.n_items])

    def calculate_loss(self, interaction):
        users, pos_items = interaction[0], interaction[1]
        user_emb, item_emb = self.forward()
        u_emb = F.normalize(user_emb[users], dim=-1)
        pos_emb = F.normalize(item_emb[pos_items], dim=-1)
        if len(interaction) < 3:  # in-batch negatives
            logits = u_emb @ pos_emb.T / self.tau
            return F.cross_entropy(logits, torch.arange(u_emb.shape[0], device=u_emb.device))
        neg_items = interaction[2]
        if neg_items.dim() == 1:
            neg_items = neg_items.unsqueeze(1)
        neg_emb = F.normalize(item_emb[neg_items], dim=-1)
        pos_scores = (u_emb * pos_emb).sum(-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", u_emb, neg_emb)
        logits = torch.cat([pos_scores, neg_scores], dim=1) / self.tau
        return F.cross_entropy(logits, torch.zeros(u_emb.shape[0], dtype=torch.long, device=u_emb.device))

    def train(self, mode=True):
        self._eval_emb = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        if self._eval_emb is None:
            user_emb, item_emb = self.forward()
            self._eval_emb = (F.normalize(user_emb, dim=-1), F.normalize(item_emb, dim=-1))
        user_emb, item_emb = self._eval_emb
        return user_emb[interaction[0]] @ item_emb.T
