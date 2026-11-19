"""Top-K evaluator.

User-averaged top-K metrics (MMRec ``utils/metrics.py``). Recall@K = hits@K / |ground truth|,
NDCG@K with IDCG over min(|ground truth|, K) -- the same definitions as the AlphaFree / AlphaRec C++ evaluator.
"""

import os

import numpy as np
import pandas as pd
import torch

from utils.metrics import metrics_dict
from utils.utils import get_local_time

topk_metrics = {metric.lower(): metric for metric in ["Recall", "Recall2", "Precision", "NDCG", "MAP"]}


class TopKEvaluator(object):
    def __init__(self, config):
        self.config = config
        self.metrics = config["metrics"]
        self.topk = config["topk"]
        self.save_recom_result = config["save_recommended_topk"]
        self._check_args()

    def evaluate(self, topk_list, hits_list, eval_data, is_test=False, idx=0):
        """
        Args:
            topk_list: per batch, (B, max K) recommended item ids (cpu)
            hits_list: per batch, (B, max K) bool, recommended item is a ground-truth item (cpu)
            eval_data (EvalDataLoader)
        Returns:
            dict: e.g. ``{'recall@20': 0.1267, 'ndcg@20': 0.1194, ...}``
        """
        pos_len_list = eval_data.get_eval_len_list()
        bool_rec_matrix = torch.cat(hits_list, dim=0).numpy()
        assert len(pos_len_list) == len(bool_rec_matrix)

        if self.save_recom_result and is_test:
            self._save_topk(torch.cat(topk_list, dim=0).numpy(), eval_data, idx)

        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                metric_dict["{}@{}".format(metric, k)] = round(float(value[k - 1]), 4)
        return metric_dict

    def _save_topk(self, topk_index, eval_data, idx):
        max_k = max(self.topk)
        dir_name = os.path.abspath(self.config["recommend_topk"])
        os.makedirs(dir_name, exist_ok=True)
        file_path = os.path.join(
            dir_name,
            "{}-{}-idx{}-top{}-{}.csv".format(
                self.config["model"], self.config["dataset"], idx, max_k, get_local_time()
            ),
        )
        x_df = pd.DataFrame(topk_index)
        x_df.insert(0, "id", eval_data.get_eval_users().numpy())
        x_df.columns = ["id"] + ["top_" + str(i) for i in range(max_k)]
        x_df.astype(int).to_csv(file_path, sep="\t", index=False)

    def _check_args(self):
        if isinstance(self.metrics, str):
            self.metrics = [self.metrics]
        elif not isinstance(self.metrics, list):
            raise TypeError("metrics must be str or list")
        for m in self.metrics:
            if m.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

        if isinstance(self.topk, int):
            self.topk = [self.topk]
        elif not isinstance(self.topk, list):
            raise TypeError("The topk must be a integer, list")
        for topk in self.topk:
            if topk <= 0:
                raise ValueError(
                    "topk must be a positive integer or a list of positive integers, but get `{}`".format(topk)
                )

    def _calculate_metrics(self, pos_len_list, topk_index):
        result_list = []
        for metric in self.metrics:
            result_list.append(metrics_dict[metric.lower()](topk_index, pos_len_list))
        return np.stack(result_list, axis=0)

    def __str__(self):
        return (
            "The TopK Evaluator Info:\n\tMetrics:["
            + ", ".join([topk_metrics[metric.lower()] for metric in self.metrics])
            + "], TopK:["
            + ", ".join(map(str, self.topk))
            + "]"
        )
