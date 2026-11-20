"""RLMRec (LR alignment on top of an ID-based graph recommender).

Reference:
    Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang.
    "Representation Learning with Large Language Models for Recommendation." WWW 2024.
    Official code: https://github.com/HKUDS/RLMRec

Model-agnostic: an ID-based backbone (``backbone``: LightGCN / SimGCL / XSimGCL, the models of this library with
the hyper-parameters of their own yaml) is trained with its own loss plus a cross-view alignment between its final
embeddings e and frozen semantic representations s:

    con (RLMRec-Con)  L = InfoNCE(e_u -> MLP(s_u), all users) + InfoNCE(e_i -> MLP(s_i), batch items)
                          for the positive and for the negative items;            loss += kd_weight * L
    gen (RLMRec-Gen)  mask_ratio of all nodes get a learnable token as layer-0 embedding (in every propagation
                      of the step), L = InfoNCE(MLP(e_v) -> s_v) over the masked nodes v; loss += recon_weight * L

InfoNCE uses cosine similarity with temperature kd_temperature / re_temperature. Inference uses e only.

Deviations from the official code:
- s_i = the dataset's item LR (lr_feature_file), s_u = mean LR of the user's training items (as AlphaRec). The
  paper encodes LLM-generated user / item profiles (GPT-3.5 + text-embedding-ada-002), not available here.
- The backbone part is this library's implementation, line for line (with kd_weight 0 a run reproduces the
  backbone's numbers exactly), not SSLRec's: no edge dropout, layer mean instead of sum, L2 on the batch's
  layer-0 embeddings instead of all parameters, SimGCL / XSimGCL contrast over the batch's unique users / items.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_graph, csr_to_torch

BACKBONES = ("LightGCN", "SimGCL", "XSimGCL")


class RLMRec(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(RLMRec, self).__init__(config, dataloader)
        self.backbone = config["backbone"]
        self.align = config["align"]
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}")
        if self.align not in ("con", "gen"):
            raise ValueError("align must be 'con' or 'gen'")
        if self.lr_feat is None:
            raise ValueError("RLMRec needs the item LRs; set is_lr_model: True")

        self.embedding_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if self.backbone != "LightGCN":
            self.eps = config["eps"]
            self.lambda_cl = config["lambda_cl"]
            self.tau_cl = config["tau_cl"]
        self.layer_cl = config["layer_cl"] if self.backbone == "XSimGCL" else None
        if self.layer_cl is not None and not 1 <= self.layer_cl <= self.n_layers:
            raise ValueError("layer_cl must be in [1, n_layers]")

        # created before the alignment modules so that the backbone gets the same initial weights as when it is
        # trained alone
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        inter_mat = dataloader.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = build_graph(inter_mat, self.n_users, self.n_items, self.device, config["deterministic"])

        self.register_buffer("item_lr", self.lr_feat, persistent=False)
        self.register_buffer("user_lr", self._history_mean(inter_mat, self.lr_feat), persistent=False)
        del self.lr_feat

        lr_dim = self.item_lr.shape[1]
        hidden_dim = (lr_dim + self.embedding_dim) // 2
        if self.align == "con":
            self.kd_weight = config["kd_weight"]
            self.kd_temperature = config["kd_temperature"]
            self.mlp = nn.Sequential(
                nn.Linear(lr_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, self.embedding_dim)
            )
        else:
            self.mask_ratio = config["mask_ratio"]
            self.recon_weight = config["recon_weight"]
            self.re_temperature = config["re_temperature"]
            self.mask_token = nn.Parameter(torch.zeros(1, self.embedding_dim))
            self.mlp = nn.Sequential(
                nn.Linear(self.embedding_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, lr_dim)
            )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self._eval_emb = None

    def _history_mean(self, inter_mat, item_lr):
        """Row-normalized train matrix times the item LRs; embedding_bag keeps it deterministic (zero rows for
        users without training items)."""
        deg = np.asarray(inter_mat.sum(axis=1)).ravel()
        inv_deg = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
        indptr, indices, weights = csr_to_torch(sp.diags(inv_deg) @ inter_mat.tocsr(), self.device)
        return F.embedding_bag(
            indices, item_lr, indptr, mode="sum", per_sample_weights=weights, include_last_offset=True
        )

    def _noise(self, emb):
        return torch.sign(emb) * F.normalize(torch.rand_like(emb), dim=-1) * self.eps

    def forward(self, ego_emb=None, perturbed=False):
        """(user_emb, item_emb, user_cl_emb, item_cl_emb); the last two are None except for XSimGCL."""
        if ego_emb is None:
            ego_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        cl_emb = None
        if self.backbone == "LightGCN":  # mean of layers 0..L
            layer_emb, emb_sum = ego_emb, ego_emb
            for _ in range(self.n_layers):
                layer_emb = self.norm_adj.mm(layer_emb)
                emb_sum = emb_sum + layer_emb
            all_emb = emb_sum / (self.n_layers + 1)
        else:  # mean of layers 1..L, signed uniform noise after every layer when perturbed
            layer_emb, layer_embs = ego_emb, []
            for layer_idx in range(1, self.n_layers + 1):
                layer_emb = self.norm_adj.mm(layer_emb)
                if perturbed and self.eps > 0:
                    layer_emb = layer_emb + self._noise(layer_emb)
                layer_embs.append(layer_emb)
                if layer_idx == self.layer_cl:
                    cl_emb = layer_emb
            all_emb = torch.stack(layer_embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items])
        if cl_emb is None:
            return user_emb, item_emb, None, None
        user_cl_emb, item_cl_emb = torch.split(cl_emb, [self.n_users, self.n_items])
        return user_emb, item_emb, user_cl_emb, item_cl_emb

    @staticmethod
    def _info_nce(query, keys, tau, targets=None):
        """Cross-entropy of query i over cosine(query i, all keys) / tau; the positive key is keys[targets[i]]
        (keys[i] when targets is None)."""
        query = F.normalize(query, dim=-1)
        keys = F.normalize(keys, dim=-1)
        logits = query @ keys.T / tau
        if targets is None:
            targets = torch.arange(query.shape[0], device=query.device)
        return F.cross_entropy(logits, targets)

    def _mask_nodes(self):
        """Layer-0 table with a random mask_ratio of the nodes replaced by the mask token, and the node mask."""
        ego_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        n_nodes = ego_emb.shape[0]
        n_masked = max(int(n_nodes * self.mask_ratio), 1)
        masked = torch.zeros(n_nodes, dtype=torch.bool, device=ego_emb.device)
        masked[torch.randperm(n_nodes, device=ego_emb.device)[:n_masked]] = True
        return torch.where(masked.unsqueeze(1), self.mask_token, ego_emb), masked

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        if neg_items.dim() == 2:
            if neg_items.shape[1] != 1:
                raise ValueError("RLMRec uses one BPR negative; set neg_sample_num: 1")
            neg_items = neg_items[:, 0]
        ego_emb, masked = self._mask_nodes() if self.align == "gen" else (None, None)

        # backbone loss: the statement order of models/lightgcn.py, simgcl.py and xsimgcl.py (autograd
        # accumulation order) so that kd_weight 0 reproduces them bit for bit
        perturbed = self.backbone == "XSimGCL"  # XSimGCL trains on its perturbed propagation
        user_emb, item_emb, user_cl_emb, item_cl_emb = self.forward(ego_emb, perturbed=perturbed)
        u_emb, pos_emb, neg_emb = user_emb[users], item_emb[pos_items], item_emb[neg_items]
        neg_scores = (u_emb * neg_emb).sum(-1)
        pos_scores = (u_emb * pos_emb).sum(-1)
        bpr_loss = F.softplus(neg_scores - pos_scores).mean()

        cl_loss = None
        if self.backbone == "SimGCL":
            user_view1, item_view1, _, _ = self.forward(ego_emb, perturbed=True)
            user_view2, item_view2, _, _ = self.forward(ego_emb, perturbed=True)
            uniq_users, uniq_items = torch.unique(users), torch.unique(pos_items)
            user_cl_loss = self._info_nce(user_view1[uniq_users], user_view2[uniq_users], self.tau_cl)
            item_cl_loss = self._info_nce(item_view1[uniq_items], item_view2[uniq_items], self.tau_cl)
            cl_loss = user_cl_loss + item_cl_loss
        elif self.backbone == "XSimGCL":
            uniq_users, uniq_items = torch.unique(users), torch.unique(pos_items)
            user_cl_loss = self._info_nce(user_emb[uniq_users], user_cl_emb[uniq_users], self.tau_cl)
            item_cl_loss = self._info_nce(item_emb[uniq_items], item_cl_emb[uniq_items], self.tau_cl)
            cl_loss = user_cl_loss + item_cl_loss

        ego_l2 = (
            self.user_embedding(users).pow(2).sum()
            + self.item_embedding(pos_items).pow(2).sum()
            + self.item_embedding(neg_items).pow(2).sum()
        )
        reg_loss = 0.5 * ego_l2 / users.shape[0]
        if cl_loss is None:
            rec_loss = bpr_loss + self.reg_weight * reg_loss
        else:
            rec_loss = bpr_loss + self.lambda_cl * cl_loss + self.reg_weight * reg_loss

        if self.align == "con":
            user_sem = self.mlp(self.user_lr)  # every user is a negative of the user term (official code)
            pos_sem, neg_sem = self.mlp(self.item_lr[pos_items]), self.mlp(self.item_lr[neg_items])
            kd_loss = (
                self._info_nce(u_emb, user_sem, self.kd_temperature, targets=users)
                + self._info_nce(pos_emb, pos_sem, self.kd_temperature)
                + self._info_nce(neg_emb, neg_sem, self.kd_temperature)
            )
            return rec_loss, self.kd_weight * kd_loss

        # node ids ascending = masked users first, then masked items
        masked_users = masked[: self.n_users].nonzero(as_tuple=True)[0]
        masked_items = masked[self.n_users :].nonzero(as_tuple=True)[0]
        pred_lr = self.mlp(torch.cat([user_emb[masked_users], item_emb[masked_items]], dim=0))
        target_lr = torch.cat([self.user_lr[masked_users], self.item_lr[masked_items]], dim=0)
        recon_loss = self._info_nce(pred_lr, target_lr, self.re_temperature)
        return rec_loss, self.recon_weight * recon_loss

    def train(self, mode=True):
        self._eval_emb = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        if self._eval_emb is None:
            user_emb, item_emb, _, _ = self.forward()
            self._eval_emb = (user_emb, item_emb)
        user_emb, item_emb = self._eval_emb
        return user_emb[interaction[0]] @ item_emb.T
