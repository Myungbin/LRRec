"""AlphaFree.

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
        self.aug_mode = config["aug_mode"] or "both"
        if self.aug_mode not in ("both", "inter", "rep", "none"):
            raise ValueError("aug_mode must be one of both / inter / rep / none")

        inter_norm, inter_aug_norm, item_lr_aug = self._load_or_build_augmentation(config, dataloader)
        self.inter_norm = csr_to_torch(inter_norm, self.device)
        self.inter_aug_norm = csr_to_torch(inter_aug_norm, self.device)
        self.register_buffer("item_lr", self.lr_feat, persistent=False)
        self.register_buffer("item_lr_aug", torch.from_numpy(item_lr_aug).float().to(self.device), persistent=False)
        del self.lr_feat

        lr_dim = self.item_lr.shape[1]
        hidden_dim = lr_mlp_hidden(lr_dim, self.lm_model)
        self.mlp_origin = nn.Sequential(
            nn.Linear(lr_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, self.embedding_dim)
        )
        self.mlp_aug = nn.Sequential(
            nn.Linear(lr_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, self.embedding_dim)
        )

        self._eval_item_emb = None

    def _load_or_build_augmentation(self, config, dataloader):
        cache_key = "{}_Kc{}_simb{}_selfexcl{}{}".format(
            self.lm_model, self.K_c, self.sim_b, int(self.exclude_self), split_tag(config)
        )
        cache_dir = os.path.join(os.path.abspath(config["data_path"] + config["dataset"]), "preprocessed", cache_key)
        cache_files = ["user_interact_matrix.npz", "aug_user_interact_matrix.npz", "item_cf_embeds_augmented.npy"]
        if all(os.path.isfile(os.path.join(cache_dir, f)) for f in cache_files):
            self.logger.info(f"AlphaFree: loading cached augmentation {cache_dir}")
            inter_norm = sp.load_npz(os.path.join(cache_dir, cache_files[0])).tocsr()
            inter_aug_norm = sp.load_npz(os.path.join(cache_dir, cache_files[1])).tocsr()
            item_lr_aug = np.load(os.path.join(cache_dir, cache_files[2]))
        else:
            self.logger.info(
                f"AlphaFree: building augmentation (K_c={self.K_c}, sim_b={self.sim_b}, "
                f"exclude_self={self.exclude_self}) -> {cache_dir}"
            )
            inter_norm, inter_aug_norm, item_lr_aug = build_augmentation(
                dataloader.inter_matrix(form="csr"),
                load_lr_feature(config),
                K_c=self.K_c,
                sim_b=self.sim_b,
                exclude_self=self.exclude_self,
                device=self.device,
            )
            os.makedirs(cache_dir, exist_ok=True)
            sp.save_npz(os.path.join(cache_dir, cache_files[0]), inter_norm)
            sp.save_npz(os.path.join(cache_dir, cache_files[1]), inter_aug_norm)
            np.save(os.path.join(cache_dir, cache_files[2]), item_lr_aug)
        if inter_norm.shape != (self.n_users, self.n_items) or item_lr_aug.shape[0] != self.n_items:
            raise ValueError(f"augmentation in {cache_dir} does not match the dataset size")
        return inter_norm, inter_aug_norm, item_lr_aug

    def _pool_user_lr(self, inter_csr, users):
        """sum_i R_ui z_i for each user (embedding_bag: deterministic, unlike cuSPARSE SpMM)."""
        item_idx, weights, offsets = gather_csr_rows(*inter_csr, users)
        return F.embedding_bag(item_idx, self.item_lr, offsets, mode="sum", per_sample_weights=weights)

    def compute_origin(self, users):
        return self.mlp_origin(self._pool_user_lr(self.inter_norm, users)), self.mlp_origin(self.item_lr)

    def compute_aug(self, users):
        user_inter = self.inter_aug_norm if self.aug_mode in ("both", "inter") else self.inter_norm
        item_lr = self.item_lr_aug if self.aug_mode in ("both", "rep") else self.item_lr
        return self.mlp_aug(self._pool_user_lr(user_inter, users)), self.mlp_aug(item_lr)

    def _infonce(self, user_emb, item_emb, pos_items, neg_items):
        u_emb = F.normalize(user_emb, dim=-1)
        pos_emb = F.normalize(item_emb[pos_items], dim=-1)
        if neg_items is None:  # in-batch negatives
            logits = u_emb @ pos_emb.T / self.tau
            return F.cross_entropy(logits, torch.arange(u_emb.shape[0], device=u_emb.device))
        neg_emb = F.normalize(item_emb[neg_items], dim=-1)
        pos_scores = (u_emb * pos_emb).sum(-1, keepdim=True)
        neg_scores = torch.einsum("bd,bnd->bn", u_emb, neg_emb)
        logits = torch.cat([pos_scores, neg_scores], dim=1) / self.tau
        return F.cross_entropy(logits, torch.zeros(u_emb.shape[0], dtype=torch.long, device=u_emb.device))

    def _align(self, origin_emb, aug_emb):
        origin_emb = F.normalize(origin_emb, dim=-1)
        aug_emb = F.normalize(aug_emb, dim=-1)
        logits = origin_emb @ aug_emb.T / self.align_tau
        labels = torch.arange(origin_emb.shape[0], device=origin_emb.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def calculate_loss(self, interaction):
        users, pos_items = interaction[0], interaction[1]
        neg_items = interaction[2] if len(interaction) > 2 else None
        if neg_items is not None and neg_items.dim() == 1:
            neg_items = neg_items.unsqueeze(1)

        user_origin, item_origin = self.compute_origin(users)
        user_aug, item_aug = self.compute_aug(users)
        rec_loss_origin = self._infonce(user_origin, item_origin, pos_items, neg_items)
        rec_loss_aug = self._infonce(user_aug, item_aug, pos_items, neg_items)
        align_loss = self._align(user_origin, user_aug) + self._align(item_origin[pos_items], item_aug[pos_items])
        return rec_loss_origin + rec_loss_aug + self.lambda_align * align_loss

    def train(self, mode=True):
        self._eval_item_emb = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        users = interaction[0]
        if self._eval_item_emb is None:
            self._eval_item_emb = F.normalize(self.mlp_origin(self.item_lr), dim=-1)
        user_emb = F.normalize(self.mlp_origin(self._pool_user_lr(self.inter_norm, users)), dim=-1)
        return user_emb @ self._eval_item_emb.T


def build_augmentation(train_csr, item_lr, K_c, sim_b="raw", exclude_self=False, device="cpu", batch_size=1024):
    """AlphaFree preprocessing (Algorithm 1 of the paper; ``AlphaFreeData`` in the official code).

    Args:
        train_csr: (n_users, n_items) training interactions (counts)
        item_lr: (n_items, d) item LRs z, original dtype (computations follow it, as in the official code)
    Returns:
        inter_norm      R  (n_users, n_items) L1-row-normalized train interactions
        inter_aug_norm  R+ (n_users, n_items) L1-row-normalized augmented interactions I+ = I U (U_{i in I} S_i)
        item_lr_aug     z+ (n_items, d) z+_i = sum_{j in S_i} w_ij z_j / sum_j w_ij  (all-zero when S_i is empty)
    """
    n_users, n_items = train_csr.shape
    inter_count = sp.csr_matrix(train_csr, dtype=np.float64)
    # (re-)built from the binary pattern exactly like the official csr_matrix((1.0, (u, i))) incl. duplicates
    inter_norm = normalize(sp.csr_matrix(train_csr, dtype=np.float32), norm="l1", axis=1)

    # behavioral similarity: item-item co-occurrence (diagonal = popularity)
    cooc = (inter_count.T @ inter_count).tocsr().astype(np.float64)
    if exclude_self:
        cooc = cooc - sp.diags(cooc.diagonal())
        cooc.eliminate_zeros()
    if sim_b == "cosine":
        item_deg = np.asarray(inter_count.sum(axis=0)).ravel()
        with np.errstate(divide="ignore"):
            inv_sqrt_deg = np.where(item_deg > 0, 1.0 / np.sqrt(item_deg), 0.0)
        inv_sqrt_deg_mat = sp.diags(inv_sqrt_deg)
        cooc = (inv_sqrt_deg_mat @ cooc @ inv_sqrt_deg_mat).tocsr()
    elif sim_b != "raw":
        raise ValueError(f"unknown sim_b: {sim_b}")
    # the official code keeps the self-loop and reserves one extra slot for it
    n_candidates = K_c if exclude_self else K_c + 1

    # semantic threshold: mean LR similarity of each item to all items
    item_lr_t = torch.from_numpy(item_lr).to(device)
    lr_sim_sum = torch.zeros(n_items, device=device, dtype=item_lr_t.dtype)
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        lr_sim_sum[start:end] = (item_lr_t[start:end] @ item_lr_t.T).sum(dim=1)
    lr_sim_mean = (lr_sim_sum / n_items).cpu().numpy()
    del item_lr_t

    # S_i (neighbors): top-K_c co-occurring items that pass the semantic filter
    rows, cols, vals = [], [], []
    for i in range(n_items):
        row_start, row_end = cooc.indptr[i], cooc.indptr[i + 1]
        if row_end == row_start:
            continue
        cand_items = cooc.indices[row_start:row_end]
        cand_cooc = cooc.data[row_start:row_end]
        for k in np.argsort(cand_cooc)[-n_candidates:][::-1]:
            j = cand_items[k]
            if float(np.dot(item_lr[i], item_lr[j])) >= lr_sim_mean[i]:
                rows.append(i)
                cols.append(j)
                vals.append(float(cand_cooc[k]))
    neighbors = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))

    # representation augmentation: weighted mean of neighbor LRs
    nb_weight_sum = np.asarray(neighbors.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        nb_weight_inv = np.power(nb_weight_sum, -1.0)
    nb_weight_inv[np.isinf(nb_weight_inv)] = 0.0
    item_lr_aug = (sp.diags(nb_weight_inv) @ neighbors).dot(item_lr)

    # interaction augmentation: I+ = I U neighbors of I (paper Eq. 6), uniform weights.
    # The official code builds I+ from the S_i alone and relies on the self-loop to keep I; that drops the
    # rare (u, i) whose i lost its self-loop to ties (1 of 1.34M entries on amazon_movie). Everything else
    # (R, S_i, z+) matches the official preprocessed files exactly.
    inter_bin = (inter_count > 0).astype(np.float32)
    neighbors_bin = (neighbors > 0).astype(np.float32)
    inter_aug = ((inter_bin + inter_bin @ neighbors_bin) > 0).astype(np.float32)
    inter_aug_norm = normalize(sp.csr_matrix(inter_aug, dtype=np.float32), norm="l1", axis=1)
    return inter_norm.tocsr(), inter_aug_norm.tocsr(), item_lr_aug
