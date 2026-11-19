"""SimGCL (ID-based baseline).

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
        self._eval_emb = None

    def _noise(self, emb):
        return torch.sign(emb) * F.normalize(torch.rand_like(emb), dim=-1) * self.eps

    def forward(self, perturbed=False):
        layer_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_embs = []
        for _ in range(self.n_layers):
            layer_emb = self.norm_adj.mm(layer_emb)
            if perturbed and self.eps > 0:
                layer_emb = layer_emb + self._noise(layer_emb)
            layer_embs.append(layer_emb)
        all_emb = torch.stack(layer_embs, dim=1).mean(dim=1)
        return torch.split(all_emb, [self.n_users, self.n_items])

    def _info_nce(self, view1, view2):
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        logits = view1 @ view2.T / self.tau_cl
        return F.cross_entropy(logits, torch.arange(view1.shape[0], device=view1.device))

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        if neg_items.dim() == 2:
            if neg_items.shape[1] != 1:
                raise ValueError("SimGCL uses one BPR negative; set neg_sample_num: 1")
            neg_items = neg_items[:, 0]
        user_emb, item_emb = self.forward(perturbed=False)
        u_emb, pos_emb, neg_emb = user_emb[users], item_emb[pos_items], item_emb[neg_items]
        neg_scores = (u_emb * neg_emb).sum(-1)
        pos_scores = (u_emb * pos_emb).sum(-1)
        bpr_loss = F.softplus(neg_scores - pos_scores).mean()

        user_view1, item_view1 = self.forward(perturbed=True)
        user_view2, item_view2 = self.forward(perturbed=True)
        uniq_users, uniq_items = torch.unique(users), torch.unique(pos_items)
        user_cl_loss = self._info_nce(user_view1[uniq_users], user_view2[uniq_users])
        item_cl_loss = self._info_nce(item_view1[uniq_items], item_view2[uniq_items])
        cl_loss = user_cl_loss + item_cl_loss

        ego_l2 = (
            self.user_embedding(users).pow(2).sum()
            + self.item_embedding(pos_items).pow(2).sum()
            + self.item_embedding(neg_items).pow(2).sum()
        )
        reg_loss = 0.5 * ego_l2 / users.shape[0]
        return bpr_loss + self.lambda_cl * cl_loss + self.reg_weight * reg_loss

    def train(self, mode=True):
        self._eval_emb = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        if self._eval_emb is None:
            self._eval_emb = self.forward(perturbed=False)
        user_emb, item_emb = self._eval_emb
        return user_emb[interaction[0]] @ item_emb.T
