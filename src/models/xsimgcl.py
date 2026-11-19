"""XSimGCL (ID-based baseline).

Reference:
    Junliang Yu, Xin Xia, Tong Chen, Lizhen Cui, Nguyen Quoc Viet Hung, Hongzhi Yin.
    "XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation." TKDE 2023.

Ported from the AlphaFree research repo (``models/XSimGCL.py``). The recommendation and contrastive objectives
share one perturbed propagation: signed uniform noise of magnitude ``eps`` is added after every layer during
training; layer ``layer_cl`` is contrasted with the final representation (InfoNCE, ``tau_cl``). The final
representation is the mean of layers 1..L (E^(0) excluded, Eq. 10). Inference uses noise-free propagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_graph


class XSimGCL(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(XSimGCL, self).__init__(config, dataloader)
        self.embedding_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.eps = config["eps"]
        self.layer_cl = config["layer_cl"]
        self.lambda_cl = config["lambda_cl"]
        self.tau_cl = config["tau_cl"]
        self.reg_weight = config["reg_weight"]
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if not 1 <= self.layer_cl <= self.n_layers:
            raise ValueError("layer_cl must be in [1, n_layers]")

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
        layers, cl_emb = [], None
        for layer in range(1, self.n_layers + 1):
            emb = self.norm_adj.mm(emb)
            if perturbed and self.eps > 0:
                emb = emb + self._noise(emb)
            layers.append(emb)
            if layer == self.layer_cl:
                cl_emb = emb
        final = torch.stack(layers, dim=1).mean(dim=1)
        user_e, item_e = torch.split(final, [self.n_users, self.n_items])
        cl_user, cl_item = torch.split(cl_emb, [self.n_users, self.n_items])
        return user_e, item_e, cl_user, cl_item

    def _info_nce(self, a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = a @ b.T / self.tau_cl
        return F.cross_entropy(logits, torch.arange(a.shape[0], device=a.device))

    def calculate_loss(self, interaction):
        users, pos, neg = interaction[0], interaction[1], interaction[2]
        if neg.dim() == 2:
            if neg.shape[1] != 1:
                raise ValueError("XSimGCL uses one BPR negative; set neg_sample_num: 1")
            neg = neg[:, 0]
        user_e, item_e, cl_user, cl_item = self.forward(perturbed=True)
        u, p, n = user_e[users], item_e[pos], item_e[neg]
        bpr = F.softplus((u * n).sum(-1) - (u * p).sum(-1)).mean()
        uu, ui = torch.unique(users), torch.unique(pos)
        cl = self._info_nce(user_e[uu], cl_user[uu]) + self._info_nce(item_e[ui], cl_item[ui])
        l2 = (
            self.user_embedding(users).pow(2).sum()
            + self.item_embedding(pos).pow(2).sum()
            + self.item_embedding(neg).pow(2).sum()
        )
        reg = 0.5 * l2 / users.shape[0]
        return bpr + self.lambda_cl * cl + self.reg_weight * reg

    def train(self, mode=True):
        self._cache = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        if self._cache is None:
            user_e, item_e, _, _ = self.forward(perturbed=False)
            self._cache = (user_e, item_e)
        user_e, item_e = self._cache
        return user_e[interaction[0]] @ item_e.T
