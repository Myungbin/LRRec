# coding: utf-8
"""
Utility functions
##########################
"""

import importlib
import os
import random
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import scipy.sparse as sp
import torch


KST = timezone(timedelta(hours=9))


def get_local_time():
    """Current time (KST) as a sortable string."""
    return datetime.now(KST).strftime("%Y%m%d-%H%M%S")


_cached_time = None


def _sanitize_tag(tag):
    """Keep only filesystem-safe characters so the tag can be a path component."""
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tag).strip())
    return tag.strip("._-")


def log_directory(config):
    """./log/<model>/<dataset>/<time>[_<tag>]/ -- one directory per process (all hyper-parameter combos)."""
    global _cached_time
    if _cached_time is None:
        _cached_time = get_local_time()
    run_name = _cached_time
    tag = _sanitize_tag(config["log_tag"]) if config["log_tag"] else ""
    if tag:
        run_name = f"{_cached_time}_{tag}"
    return f"./log/{config['model']}/{config['dataset']}/{run_name}/"


def get_model(model_name):
    r"""``--model AlphaFree`` -> class ``AlphaFree`` in ``models/alphafree.py`` (naming convention, no registry)."""
    module_path = ".".join(["models", model_name.lower()])
    if importlib.util.find_spec(module_path) is None:
        raise ValueError(f"model file models/{model_name.lower()}.py not found")
    model_module = importlib.import_module(module_path)
    return getattr(model_module, model_name)


def get_trainer():
    return getattr(importlib.import_module("common.trainer"), "Trainer")


def init_seed(seed, deterministic=True):
    """Seed every RNG. ``deterministic`` also makes CUDA index/scatter backward kernels deterministic
    (torch.use_deterministic_algorithms, warn_only for ops without a deterministic version such as sparse.mm),
    so two runs with the same seed and config produce the same numbers on the same GPU type."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r"""Validation-based early stopping.

    Returns:
        (best, cur_step, stop_flag, update_flag)
    """
    stop_flag = False
    update_flag = False
    improved = value > best if bigger else value < best
    if improved:
        cur_step = 0
        best = value
        update_flag = True
    else:
        cur_step += 1
        if cur_step > max_step:
            stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r"""{'recall@20': 0.1, ...} -> 'recall@20: 0.1000    ...'"""
    return "    ".join(f"{metric}: {value:.4f}" for metric, value in result_dict.items())


# ---------------------------------------------------------------------------------------- graphs
def scipy_to_torch_sparse(mat):
    """scipy sparse matrix -> coalesced torch sparse COO (float32, on CPU)."""
    mat = sp.coo_matrix(mat)
    indices = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    values = torch.from_numpy(mat.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(mat.shape)).coalesce()


def build_norm_adj(inter, n_users, n_items):
    r"""Symmetrically normalized user-item bipartite adjacency D^{-1/2} A D^{-1/2} (LightGCN), as torch sparse.

    Nodes are stacked as ``[users (0..n_users-1) ; items (n_users..n_users+n_items-1)]``.
    Isolated nodes get degree 0 and an all-zero row.
    """
    inter = sp.coo_matrix(inter)
    users = inter.row.astype(np.int64)
    items = inter.col.astype(np.int64) + n_users
    rows = np.concatenate([users, items])
    cols = np.concatenate([items, users])
    values = np.ones(rows.shape[0], dtype=np.float32)
    size = n_users + n_items
    adj = sp.coo_matrix((values, (rows, cols)), shape=(size, size), dtype=np.float32)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    nz = deg > 0
    inv_sqrt[nz] = np.power(deg[nz], -0.5)
    norm = sp.diags(inv_sqrt) @ adj @ sp.diags(inv_sqrt)
    return scipy_to_torch_sparse(norm)


def csr_to_torch(csr, device):
    """scipy CSR -> (indptr, indices, data) tensors on ``device`` for :func:`gather_csr_rows`."""
    csr = sp.csr_matrix(csr)
    return (
        torch.from_numpy(csr.indptr.astype(np.int64)).to(device),
        torch.from_numpy(csr.indices.astype(np.int64)).to(device),
        torch.from_numpy(csr.data.astype(np.float32)).to(device),
    )


def gather_csr_rows(indptr, indices, data, rows):
    """Slice rows of a CSR matrix that lives on the GPU.

    Returns ``(cols, vals, offsets)`` -- the concatenated column indices / values of the selected rows
    (each row in its stored order) and each row's start offset, i.e. exactly the ``input`` /
    ``per_sample_weights`` / ``offsets`` triple that ``F.embedding_bag`` expects.
    """
    starts = indptr[rows]
    lens = indptr[rows + 1] - starts
    offsets = torch.cumsum(lens, 0) - lens
    total = int(lens.sum().item())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=indices.device)
        return empty, torch.empty(0, dtype=data.dtype, device=data.device), offsets
    pos = torch.arange(total, device=indices.device) - torch.repeat_interleave(offsets - starts, lens)
    return indices[pos], data[pos], offsets
