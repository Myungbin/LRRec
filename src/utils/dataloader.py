"""Data loaders.

Same interface as MMRec (``inter_matrix()``, ``pretrain_setup()``, iteration yields an ``interaction``),
with sampling vectorised on the device so that hundreds of negatives per sample stay cheap.

Training ``interaction`` = ``[users, pos_items, neg_items]``
    neg_items is (B,) when ``neg_sample_num == 1`` (MMRec/BPR convention), (B, N) when N > 1, and absent
    when ``neg_sample_num == 0`` (models then use in-batch negatives).

Evaluation ``interaction`` = ``[users, mask]``
    mask is a 2 x M tensor of (row in batch, item) pairs whose scores must be set to -inf before ranking.
"""

import math
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix


class AbstractDataLoader(object):
    def __init__(self, config, dataset, additional_dataset=None, batch_size=1, neg_sampling=False, shuffle=False):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.neg_sampling = neg_sampling
        self.device = config["device"]
        self.pr = 0
        self.inter_pr = 0

    def pretrain_setup(self):
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            self.inter_pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        raise NotImplementedError("Method [pr_end] should be implemented")

    def _shuffle(self):
        raise NotImplementedError("Method [shuffle] should be implemented.")

    def _next_batch_data(self):
        raise NotImplementedError("Method [next_batch_data] should be implemented.")


class TrainDataLoader(AbstractDataLoader):
    """Training sampler.

    ``train_sampling``:
        user        AlphaFree / AlphaRec protocol. An epoch has |train interactions| samples; sample k is the
                    user at position (perm[k] mod n_train_users), so every user is visited about equally
                    often, and its positive is drawn uniformly from that user's training items.
                    Incomplete last batch is dropped (as in the AlphaFree DataLoader).
        interaction MMRec protocol. Every training (user, item) pair once per epoch, in shuffled order.

    Negatives: ``neg_sample_num`` items drawn uniformly from all items, rejecting the user's training items.
    """

    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(
            config, dataset, additional_dataset=None, batch_size=batch_size, neg_sampling=True, shuffle=shuffle
        )
        self.n_users = dataset.get_user_num()
        self.n_items = dataset.get_item_num()
        self.sampling = (config["train_sampling"] or "user").lower()
        if self.sampling not in ("user", "interaction"):
            raise ValueError("train_sampling must be 'user' or 'interaction'")
        self.neg_num = int(config["neg_sample_num"] if config["neg_sample_num"] is not None else 1)

        users = dataset.df[dataset.uid_field].values.astype(np.int64)
        items = dataset.df[dataset.iid_field].values.astype(np.int64)
        self.n_obs = len(users)
        self.inter_users = torch.from_numpy(users).to(self.device)
        self.inter_items = torch.from_numpy(items).to(self.device)

        # per-user training history as CSR, in file order within each user (stable sort)
        order = np.argsort(users, kind="stable")
        deg = np.bincount(users, minlength=self.n_users)
        indptr = np.concatenate([[0], np.cumsum(deg)])
        self.hist_indptr = torch.from_numpy(indptr).to(self.device)
        self.hist_items = torch.from_numpy(items[order]).to(self.device)
        self.train_users = torch.from_numpy(np.flatnonzero(deg > 0)).to(self.device)
        # sorted (user * n_items + item) keys for O(log n) "is a training positive?" lookups
        self.train_keys = torch.unique(self.inter_users * self.n_items + self.inter_items)

        self.generator = None
        self.perm = None
        self.pretrain_setup()

    def pretrain_setup(self):
        """Re-seed the sampler; called before every hyper-parameter combination for reproducibility."""
        seed = self.config["seed"]
        seed = int(seed[0] if isinstance(seed, (list, tuple)) else (seed or 0))
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.perm = torch.arange(self.n_obs, device=self.device)
        self.pr = 0

    def inter_matrix(self, form="coo", value_field=None):
        return self._create_sparse_matrix(
            self.dataset.df, self.dataset.uid_field, self.dataset.iid_field, form, value_field
        )

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form="coo", value_field=None):
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError("value_field [{}] should be one of `df_feat`'s features.".format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(self.n_users, self.n_items))
        if form == "coo":
            return mat
        elif form == "csr":
            return mat.tocsr()
        raise NotImplementedError("sparse matrix format [{}] has not been implemented.".format(form))

    @property
    def pr_end(self):
        if self.sampling == "user":
            return (self.n_obs // self.step) * self.step
        return self.n_obs

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def _shuffle(self):
        self.perm = torch.randperm(self.n_obs, device=self.device, generator=self.generator)

    def _next_batch_data(self):
        idx = self.perm[self.pr : self.pr + self.step]
        self.pr += self.step
        if self.sampling == "user":
            users = self.train_users[idx % self.train_users.numel()]
            start = self.hist_indptr[users]
            deg = self.hist_indptr[users + 1] - start
            r = torch.rand(users.shape[0], device=self.device, generator=self.generator)
            offset = torch.minimum((r * deg).long(), deg - 1)
            pos = self.hist_items[start + offset]
        else:
            users = self.inter_users[idx]
            pos = self.inter_items[idx]
        if self.neg_num <= 0:
            return [users, pos]
        neg = self.sample_negatives(users, self.neg_num)
        if self.neg_num == 1:
            neg = neg.squeeze(1)
        return [users, pos, neg]

    def sample_negatives(self, users, num):
        """(B, num) items ~ U(all items) excluding each user's training items (rejection sampling)."""
        neg = torch.randint(0, self.n_items, (users.shape[0], num), device=self.device, generator=self.generator)
        base = users.unsqueeze(1) * self.n_items
        bad = self._is_train_pair(base + neg)
        while bool(bad.any()):
            n_bad = int(bad.sum().item())
            neg[bad] = torch.randint(0, self.n_items, (n_bad,), device=self.device, generator=self.generator)
            bad = self._is_train_pair(base + neg)
        return neg

    def _is_train_pair(self, keys):
        pos = torch.searchsorted(self.train_keys, keys).clamp_max(self.train_keys.numel() - 1)
        return self.train_keys[pos] == keys


class EvalDataLoader(AbstractDataLoader):
    """Full-ranking evaluation over every user of ``dataset`` (the valid or test split).

    Args:
        mask_datasets: splits whose positives are masked (score -inf) before ranking.
        exclude_eval_positives: if True, a pair that is also a positive of the evaluated split is never
            masked (AlphaFree evaluator: ``dump_items - test_items``); MMRec masks it.
    """

    def __init__(self, config, dataset, mask_datasets, batch_size=1, exclude_eval_positives=True):
        super().__init__(
            config, dataset, additional_dataset=None, batch_size=batch_size, neg_sampling=False, shuffle=False
        )
        uid, iid = dataset.uid_field, dataset.iid_field
        self.n_items = dataset.get_item_num()

        eval_df = dataset.df[[uid, iid]].drop_duplicates()
        self.eval_u_np = pd.unique(eval_df[uid].values)
        row_of = pd.Series(np.arange(len(self.eval_u_np)), index=self.eval_u_np)

        # ground truth, as sorted (row * n_items + item) keys
        gt_rows = row_of.loc[eval_df[uid].values].values.astype(np.int64)
        gt_items = eval_df[iid].values.astype(np.int64)
        self.eval_len_list = np.bincount(gt_rows, minlength=len(self.eval_u_np))
        self.gt_keys = torch.from_numpy(np.unique(gt_rows * self.n_items + gt_items)).to(self.device)

        mask_df = pd.concat([d.df[[uid, iid]] for d in mask_datasets], ignore_index=True).drop_duplicates()
        mask_df = mask_df[mask_df[uid].isin(row_of.index)]
        m_rows = row_of.loc[mask_df[uid].values].values.astype(np.int64)
        m_items = mask_df[iid].values.astype(np.int64)
        if exclude_eval_positives and len(m_rows):
            keep = ~np.isin(m_rows * self.n_items + m_items, self.gt_keys.cpu().numpy())
            m_rows, m_items = m_rows[keep], m_items[keep]
        order = np.argsort(m_rows, kind="stable")
        m_rows, m_items = m_rows[order], m_items[order]
        self.mask_indptr = np.concatenate([[0], np.cumsum(np.bincount(m_rows, minlength=len(self.eval_u_np)))])
        self.mask_rows = torch.from_numpy(m_rows).to(self.device)
        self.mask_items = torch.from_numpy(m_items).to(self.device)

        self.eval_u = torch.from_numpy(self.eval_u_np.astype(np.int64)).to(self.device)

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        pass

    def _next_batch_data(self):
        lo, hi = self.pr, min(self.pr + self.step, self.pr_end)
        m_lo, m_hi = int(self.mask_indptr[lo]), int(self.mask_indptr[hi])
        batch_users = self.eval_u[lo:hi]
        batch_mask = torch.stack([self.mask_rows[m_lo:m_hi] - lo, self.mask_items[m_lo:m_hi]])
        self.pr = hi
        return [batch_users, batch_mask]

    def batch_hits(self, row_start, topk_index):
        """(B, K) bool: is topk_index[b, k] a ground-truth item of evaluated user (row_start + b)?"""
        rows = torch.arange(row_start, row_start + topk_index.shape[0], device=topk_index.device).unsqueeze(1)
        keys = rows * self.n_items + topk_index
        pos = torch.searchsorted(self.gt_keys, keys).clamp_max(self.gt_keys.numel() - 1)
        return self.gt_keys[pos] == keys

    def get_eval_len_list(self):
        return self.eval_len_list

    def get_eval_users(self):
        return self.eval_u.cpu()
