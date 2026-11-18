# coding: utf-8
"""
Dataset
##########################

Reads the AlphaFree / AlphaRec on-disk format as-is (the files are never rewritten)::

    <data_path>/<dataset>/cf_data/train.txt     one line per user: "<user> <item> <item> ..."
    <data_path>/<dataset>/cf_data/valid.txt
    <data_path>/<dataset>/cf_data/test.txt
    <data_path>/<dataset>/cf_data/n_items.txt   (optional) pins the item universe, e.g. for cold-item splits
    <data_path>/<dataset>/item_info/...         item language representations (see GeneralRecommender)

Internally the three files become one MMRec-style interaction table ``[userID, itemID, x_label]`` with
``x_label`` 0/1/2 = train/valid/test, so splitting, data loaders and models work exactly as in MMRec.
"""

import os
from logging import getLogger

import numpy as np
import pandas as pd

SPLIT_FILES = ("train.txt", "valid.txt", "test.txt")


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

    def split(self):
        dfs = []
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)
            dfs.append(temp_df.reset_index(drop=True))
        if self.config["filter_out_cod_start_users"]:
            # drop valid/test users that have no training interaction
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
