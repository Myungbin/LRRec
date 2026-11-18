# coding: utf-8
r"""
AlphaFree
################################################
Reference:
    Minseo Jeon, Junwoo Jung, Daewon Gwak, Jinhong Jung.
    "AlphaFree: Recommendation Free from Users, IDs, and GNNs." WWW 2026.
    Official code: https://github.com/Minseojeonn/AlphaFree

No user embeddings, no item IDs, no GNN. Everything is computed from frozen item language representations
(LRs) z_i and the user's interaction history:

    user LR  z_u      = sum_{i in I_u} R_ui z_i                 R = L1-row-normalized train interactions
    origin   e = MLP(z)                                         used for inference
    aug view e+ = MLP+(z+),  z+_i = mean_{j in S_i} z_j         representation augmentation
             z_u^+ = sum_{i in I_u^+} R+_ui z_i,  I_u^+ = I_u U (U_{i in I_u} S_i)   interaction augmentation
    S_i  = top-K_c co-occurring items of i whose LR similarity to i is >= the mean LR similarity of i

    loss = InfoNCE(origin) + InfoNCE(aug) + lambda_align * (Align(users) + Align(positive items))

Only MLP (``mlp_origin``) is used at inference. Parameter names match the official checkpoints
(``data/pretrained/<dataset>.pth.tar``), which therefore load with ``strict=True``.

The augmentation (R, R+, z+) is built once and cached under
``<data_path>/<dataset>/preprocessed/<lm_model>_Kc<K_c>_simb<sim_b>_selfexcl<0|1>/`` (same layout as the
AlphaFree research repo, so caches are interchangeable). With ``split_ratio`` the directory name also gets
``_split<train>-<valid>-<test>_seed<split_seed>``, because the augmentation depends on the training split.
"""

import os
from logging import getLogger

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from common.abstract_recommender import GeneralRecommender, load_lr_feature, lr_mlp_hidden
from utils.dataset import split_tag
from utils.utils import csr_to_torch, gather_csr_rows


class AlphaFree(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(AlphaFree, self).__init__(config, dataloader)
        self.logger = getLogger()

        self.embedding_dim = config["embedding_size"]
        self.tau = config["tau_r"]
        self.align_tau = config["tau_a"]
        self.lambda_align = config["lambda_align"]
        self.K_c = config["K_c"]
        self.sim_b = config["sim_b"] or "raw"
        self.exclude_self = bool(config["exclude_self"])
        self.aug_mode = config["aug_mode"] or "both"  # both | inter | rep | none (ablation)
        if self.aug_mode not in ("both", "inter", "rep", "none"):
            raise ValueError("aug_mode must be one of both / inter / rep / none")

        R, R_aug, z_aug = self._load_or_build_augmentation(config, dataloader)
        self.R = csr_to_torch(R, self.device)
        self.R_aug = csr_to_torch(R_aug, self.device)
        self.register_buffer("Z", self.lr_feat, persistent=False)
        self.register_buffer("Z_aug", torch.from_numpy(z_aug).float().to(self.device), persistent=False)
        del self.lr_feat

        d = self.Z.shape[1]
        h = lr_mlp_hidden(d, self.lm_model)
        self.mlp_origin = nn.Sequential(nn.Linear(d, h), nn.LeakyReLU(), nn.Linear(h, self.embedding_dim))
        self.mlp_aug = nn.Sequential(nn.Linear(d, h), nn.LeakyReLU(), nn.Linear(h, self.embedding_dim))

        self._item_cache = None  # mlp_origin(Z) during evaluation

    # ------------------------------------------------------------------------------ augmentation
    def _load_or_build_augmentation(self, config, dataloader):
        cache_key = "{}_Kc{}_simb{}_selfexcl{}{}".format(
            self.lm_model, self.K_c, self.sim_b, int(self.exclude_self), split_tag(config)
        )
        cache_dir = os.path.join(os.path.abspath(config["data_path"] + config["dataset"]), "preprocessed", cache_key)
        files = ["user_interact_matrix.npz", "aug_user_interact_matrix.npz", "item_cf_embeds_augmented.npy"]
        if all(os.path.isfile(os.path.join(cache_dir, f)) for f in files):
            self.logger.info(f"AlphaFree: loading cached augmentation {cache_dir}")
            R = sp.load_npz(os.path.join(cache_dir, files[0])).tocsr()
            R_aug = sp.load_npz(os.path.join(cache_dir, files[1])).tocsr()
            z_aug = np.load(os.path.join(cache_dir, files[2]))
        else:
            self.logger.info(f"AlphaFree: building augmentation (K_c={self.K_c}, sim_b={self.sim_b}, "
                             f"exclude_self={self.exclude_self}) -> {cache_dir}")
            R, R_aug, z_aug = build_augmentation(
                dataloader.inter_matrix(form="csr"),
                load_lr_feature(config),
                K_c=self.K_c,
                sim_b=self.sim_b,
                exclude_self=self.exclude_self,
                device=self.device,
            )
            os.makedirs(cache_dir, exist_ok=True)
            sp.save_npz(os.path.join(cache_dir, files[0]), R)
            sp.save_npz(os.path.join(cache_dir, files[1]), R_aug)
            np.save(os.path.join(cache_dir, files[2]), z_aug)
        if R.shape != (self.n_users, self.n_items) or z_aug.shape[0] != self.n_items:
            raise ValueError(f"augmentation in {cache_dir} does not match the dataset size")
        return R, R_aug, z_aug

    # ------------------------------------------------------------------------------ encoders
    def _pool(self, csr, users):
        """sum_i R_ui z_i for each user (embedding_bag: deterministic, unlike cuSPARSE SpMM)."""
        cols, vals, offsets = gather_csr_rows(*csr, users)
        return F.embedding_bag(cols, self.Z, offsets, mode="sum", per_sample_weights=vals)

    def compute_origin(self, users):
        return self.mlp_origin(self._pool(self.R, users)), self.mlp_origin(self.Z)

    def compute_aug(self, users):
        user_csr = self.R_aug if self.aug_mode in ("both", "inter") else self.R
        item_lr = self.Z_aug if self.aug_mode in ("both", "rep") else self.Z
        return self.mlp_aug(self._pool(user_csr, users)), self.mlp_aug(item_lr)

    # ------------------------------------------------------------------------------ losses
    def _infonce(self, u, items, pos, neg):
        u = F.normalize(u, dim=-1)
        p = F.normalize(items[pos], dim=-1)
        if neg is None:  # in-batch negatives
            logits = u @ p.T / self.tau
            return F.cross_entropy(logits, torch.arange(u.shape[0], device=u.device))
        n = F.normalize(items[neg], dim=-1)
        pos_s = (u * p).sum(-1, keepdim=True)
        neg_s = torch.einsum("bd,bnd->bn", u, n)
        logits = torch.cat([pos_s, neg_s], dim=1) / self.tau
        return F.cross_entropy(logits, torch.zeros(u.shape[0], dtype=torch.long, device=u.device))

    def _align(self, a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = a @ b.T / self.align_tau
        labels = torch.arange(a.shape[0], device=a.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def calculate_loss(self, interaction):
        users, pos = interaction[0], interaction[1]
        neg = interaction[2] if len(interaction) > 2 else None
        if neg is not None and neg.dim() == 1:
            neg = neg.unsqueeze(1)

        u_o, items_o = self.compute_origin(users)
        u_a, items_a = self.compute_aug(users)
        loss_origin = self._infonce(u_o, items_o, pos, neg)
        loss_aug = self._infonce(u_a, items_a, pos, neg)
        loss_align = self._align(u_o, u_a) + self._align(items_o[pos], items_a[pos])
        return loss_origin + loss_aug + self.lambda_align * loss_align

    # ------------------------------------------------------------------------------ inference
    def train(self, mode=True):
        self._item_cache = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        users = interaction[0]
        if self._item_cache is None:
            self._item_cache = F.normalize(self.mlp_origin(self.Z), dim=-1)
        u = F.normalize(self.mlp_origin(self._pool(self.R, users)), dim=-1)
        return u @ self._item_cache.T


# ------------------------------------------------------------------------------------------ preprocessing
def build_augmentation(train_csr, z, K_c, sim_b="raw", exclude_self=False, device="cpu", batch_size=1024):
    """AlphaFree preprocessing (Algorithm 1 of the paper; ``AlphaFreeData`` in the official code).

    Args:
        train_csr: (n_users, n_items) training interactions (counts)
        z: (n_items, d) item LRs, original dtype (computations follow it, as in the official code)
    Returns:
        R      (n_users, n_items) L1-row-normalized train interactions
        R_aug  (n_users, n_items) L1-row-normalized augmented interactions I+ = I U (U_{i in I} S_i)
        z_aug  (n_items, d) z+_i = sum_{j in S_i} w_ij z_j / sum_j w_ij  (all-zero when S_i is empty)
    """
    n_users, n_items = train_csr.shape
    A = sp.csr_matrix(train_csr, dtype=np.float64)
    # (re-)built from the binary pattern exactly like the official csr_matrix((1.0, (u, i))) incl. duplicates
    R = normalize(sp.csr_matrix(train_csr, dtype=np.float32), norm="l1", axis=1)

    # behavioral similarity: item-item co-occurrence (diagonal = popularity)
    sim = (A.T @ A).tocsr().astype(np.float64)
    if exclude_self:
        sim = sim - sp.diags(sim.diagonal())
        sim.eliminate_zeros()
    if sim_b == "cosine":
        deg = np.asarray(A.sum(axis=0)).ravel()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D = sp.diags(d_inv_sqrt)
        sim = (D @ sim @ D).tocsr()
    elif sim_b != "raw":
        raise ValueError(f"unknown sim_b: {sim_b}")
    # the official code keeps the self-loop and reserves one extra slot for it
    topk = K_c if exclude_self else K_c + 1

    # semantic threshold: mean LR similarity of each item to all items
    zt = torch.from_numpy(z).to(device)
    sim_sum = torch.zeros(n_items, device=device, dtype=zt.dtype)
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        sim_sum[start:end] = (zt[start:end] @ zt.T).sum(dim=1)
    sem_mean = (sim_sum / n_items).cpu().numpy()
    del zt

    # S_i: top-k co-occurring items that pass the semantic filter
    rows, cols, vals = [], [], []
    for i in range(n_items):
        lo, hi = sim.indptr[i], sim.indptr[i + 1]
        if hi == lo:
            continue
        indices = sim.indices[lo:hi]
        values = sim.data[lo:hi]
        for j in np.argsort(values)[-topk:][::-1]:
            nb = indices[j]
            if float(np.dot(z[i], z[nb])) >= sem_mean[i]:
                rows.append(i)
                cols.append(nb)
                vals.append(float(values[j]))
    S = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))

    # representation augmentation: weighted mean of neighbor LRs
    deg = np.asarray(S.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        deg_inv = np.power(deg, -1.0)
    deg_inv[np.isinf(deg_inv)] = 0.0
    z_aug = (sp.diags(deg_inv) @ S).dot(z)

    # interaction augmentation: I+ = I U neighbors of I (paper Eq. 6), uniform weights.
    # The official code builds I+ from the S_i alone and relies on the self-loop to keep I; that drops the
    # rare (u, i) whose i lost its self-loop to ties (1 of 1.34M entries on amazon_movie). Everything else
    # (R, S_i, z+) matches the official preprocessed files exactly.
    B = (A > 0).astype(np.float32)
    S_bin = (S > 0).astype(np.float32)
    I_plus = ((B + B @ S_bin) > 0).astype(np.float32)
    R_aug = normalize(sp.csr_matrix(I_plus, dtype=np.float32), norm="l1", axis=1)
    return R.tocsr(), R_aug.tocsr(), z_aug
