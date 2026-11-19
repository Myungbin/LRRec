"""Dataset.

Reads the AlphaFree / AlphaRec on-disk format as-is (the files are never rewritten)::

    <data_path>/<dataset>/cf_data/train.txt     one line per user: "<user> <item> <item> ..."
    <data_path>/<dataset>/cf_data/valid.txt
    <data_path>/<dataset>/cf_data/test.txt
    <data_path>/<dataset>/cf_data/n_items.txt   (optional) pins the item universe, e.g. for cold-item splits
    <data_path>/<dataset>/item_info/...         item language representations (see GeneralRecommender)

Internally the three files become one MMRec-style interaction table ``[userID, itemID, x_label]`` with
``x_label`` 0/1/2 = train/valid/test, so splitting, data loaders and models work exactly as in MMRec.

With ``split_ratio`` (e.g. ``[0.8, 0.1, 0.1]``) the three files are merged and every user's interactions are
re-split at random in memory (:func:`resplit_per_user`); the files are still not touched.
"""

import os
from logging import getLogger

import numpy as np
import pandas as pd

SPLIT_FILES = ("train.txt", "valid.txt", "test.txt")


def split_ratio(config):
    """``split_ratio`` normalized to sum 1 as a (train, valid, test) tuple, or None for the split of the files."""
    ratio = config["split_ratio"]
    if ratio is None:
        return None
    if not isinstance(ratio, (list, tuple)) or len(ratio) != 3:
        raise ValueError("split_ratio must be ~ or a list [train, valid, test], e.g. [0.8, 0.1, 0.1]")
    ratio = np.asarray(ratio, dtype=np.float64)
    if (ratio < 0).any() or ratio[0] <= 0:
        raise ValueError("split_ratio needs a positive train share and non-negative valid / test shares")
    return tuple(float(r) for r in ratio / ratio.sum())


def split_tag(config):
    """'' for the split of the files, else '_split<train>-<valid>-<test>_seed<split_seed>'. Part of the name of
    every cache derived from the training split, so caches of different splits never mix."""
    ratio = split_ratio(config)
    if ratio is None:
        return ""
    return "_split{}_seed{}".format("-".join(f"{r:g}" for r in ratio), int(config["split_seed"]))


def _quota(share, counts):
    """Items per user for a valid/test share: round-half-up(share * n), at least 1 if share > 0."""
    if share <= 0:
        return np.zeros_like(counts)
    return np.maximum(np.floor(share * counts + 0.5 + 1e-9).astype(np.int64), 1)


def resplit_per_user(users, ratio, seed):
    """New split label (0/1/2 = train/valid/test) for every row of ``users``.

    Each user's interactions are shuffled (``numpy.random.default_rng(seed)``) and the first
    ``round(ratio_test * n)`` go to test, the next ``round(ratio_valid * n)`` to valid and the rest to train.
    valid and test get at least one item when their share is > 0; train always keeps at least one, so a user
    with 2 interactions gets 1 train + 1 test and a user with 1 interaction is train only.
    """
    users = np.asarray(users)
    n_rows = len(users)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n_rows), users))  # grouped by user, random order inside each user
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_users)) + 1]
    counts = np.diff(np.r_[starts, n_rows])

    k_test = _quota(ratio[2], counts)
    k_valid = _quota(ratio[1], counts)
    over = np.maximum(k_test + k_valid - (counts - 1), 0)  # keep >= 1 train item: shrink valid first, then test
    cut = np.minimum(over, k_valid)
    k_valid -= cut
    k_test -= np.minimum(over - cut, k_test)

    rank = np.arange(n_rows) - np.repeat(starts, counts)
    k_test_row, k_valid_row = np.repeat(k_test, counts), np.repeat(k_valid, counts)
    labels_sorted = np.where(rank < k_test_row, 2, np.where(rank < k_test_row + k_valid_row, 1, 0))
    labels = np.empty(n_rows, dtype=np.int8)
    labels[order] = labels_sorted
    return labels


def read_user_item_lists(path):
    """'<user> <item> <item> ...' per line -> (users, items) int64 arrays, in file order."""
    users, items = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:  # user with no items in this split
                continue
            ids = np.asarray(parts, dtype=np.int64)
            users.append(np.full(len(ids) - 1, ids[0], dtype=np.int64))
            items.append(ids[1:])
    if not users:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(users), np.concatenate(items)


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        self.dataset_name = config["dataset"]
        self.dataset_path = os.path.abspath(os.path.join(config["data_path"], self.dataset_name))
        self.cf_path = os.path.join(self.dataset_path, "cf_data")

        self.uid_field = config["USER_ID_FIELD"]
        self.iid_field = config["ITEM_ID_FIELD"]
        self.splitting_label = config["inter_splitting_label"]

        if df is not None:
            self.df = df
            return

        for name in SPLIT_FILES:
            if not os.path.isfile(os.path.join(self.cf_path, name)):
                raise ValueError("File {} not exist".format(os.path.join(self.cf_path, name)))
        self.load_inter_graph()

        self.user_num = int(self.df[self.uid_field].max()) + 1
        self.item_num = int(self.df[self.iid_field].max()) + 1
        n_items_file = os.path.join(self.cf_path, "n_items.txt")
        if os.path.isfile(n_items_file):
            with open(n_items_file) as f:
                self.item_num = max(self.item_num, int(f.read().split()[0]))

    def load_inter_graph(self):
        frames = []
        for label, name in enumerate(SPLIT_FILES):
            users, items = read_user_item_lists(os.path.join(self.cf_path, name))
            frames.append(
                pd.DataFrame(
                    {
                        self.uid_field: users,
                        self.iid_field: items,
                        self.splitting_label: np.full(len(users), label, dtype=np.int8),
                    }
                )
            )
        self.df = pd.concat(frames, ignore_index=True)

        ratio = split_ratio(self.config)
        if ratio is not None:
            seed = int(self.config["split_seed"])
            self.df[self.splitting_label] = resplit_per_user(self.df[self.uid_field].values, ratio, seed)
            sizes = np.bincount(self.df[self.splitting_label].values, minlength=3)
            self.logger.info(
                "Re-split per user: split_ratio={} split_seed={} -> train/valid/test = {}/{}/{} ({})".format(
                    "/".join(f"{r:g}" for r in ratio),
                    seed,
                    *sizes,
                    "/".join(f"{100 * s / sizes.sum():.1f}%" for s in sizes),
                )
            )

    def split(self):
        dfs = []
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)
            dfs.append(temp_df.reset_index(drop=True))
        if self.config["filter_out_cod_start_users"]:
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dfs[i] = dfs[i][dfs[i][self.uid_field].isin(train_u)].reset_index(drop=True)
        return [self.copy(_) for _ in dfs]

    def copy(self, new_df):
        """A new :class:`RecDataset` sharing everything but the interaction table."""
        nxt = RecDataset(self.config, new_df)
        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        n_u = self.df[self.uid_field].nunique()
        n_i = self.df[self.iid_field].nunique()
        info.append("The number of users: {}".format(n_u))
        info.append("Average actions of users: {}".format(self.inter_num / max(n_u, 1)))
        info.append("The number of items: {}".format(n_i))
        info.append("Average actions of items: {}".format(self.inter_num / max(n_i, 1)))
        info.append("The number of inters: {}".format(self.inter_num))
        if n_u and n_i:
            info.append("The sparsity of the dataset: {}%".format((1 - self.inter_num / n_u / n_i) * 100))
        return "\n".join(info)
