# coding: utf-8
"""
Base recommenders
################################
Every model subclasses :class:`GeneralRecommender` and implements

    __init__(self, config, dataloader)
    calculate_loss(self, interaction)        interaction = [users, pos_items, (neg_items)]
    full_sort_predict(self, interaction)     (B, n_items) scores for interaction[0]; masking is done by the Trainer

and optionally ``pre_epoch_processing()`` / ``post_epoch_processing()``.
"""

import os

import numpy as np
import torch
import torch.nn as nn

# Hidden width of the LR -> embedding MLP as a fraction of the LR dimension, per language model
# (AlphaRec / AlphaFree): text-embedding-3-large 3072 -> 1536, LLaMA-3 4096 -> 1152.
LM_MLP_RATIO = {"v3": 1 / 2, "llama": 9 / 32}


def lr_mlp_hidden(d_in, lm_model):
    return int(LM_MLP_RATIO.get(lm_model, 9 / 32) * d_in)


def load_lr_feature(config):
    """Item language representations of ``config['dataset']`` as a numpy array (original dtype)."""
    path = os.path.join(os.path.abspath(config["data_path"] + config["dataset"]), config["lr_feature_file"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"item language representations not found: {path}")
    return np.load(path)  # allow_pickle=False


class AbstractRecommender(nn.Module):
    r"""Base class for all models"""

    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        raise NotImplementedError

    def __str__(self):
        params = sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class GeneralRecommender(AbstractRecommender):
    """Loads dataset sizes and -- for language-representation models (``is_lr_model: True``) -- the frozen item
    LR matrix as ``self.lr_feat`` (n_items x d_LR, float32, on device)."""

    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        self.batch_size = config["train_batch_size"]
        self.device = config["device"]
        self.lm_model = config["lm_model"]

        self.lr_feat = None
        if config["is_lr_model"]:
            z = load_lr_feature(config)
            if z.shape[0] != self.n_items:
                raise ValueError(
                    f"LR matrix has {z.shape[0]} rows but the dataset has {self.n_items} items "
                    f"({config['lr_feature_file']})"
                )
            self.lr_feat = torch.from_numpy(z).float().to(self.device)
