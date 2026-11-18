# coding: utf-8
r"""
Trainer
################################
Epoch loop with validation-based early stopping. Test metrics are logged at every evaluation and the
reported test result is the one at the best validation epoch.
"""

import itertools
import os
from logging import getLogger
from time import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from utils.topk_evaluator import TopKEvaluator
from utils.utils import dict2str, early_stopping, log_directory


class AbstractTrainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        raise NotImplementedError("Method [fit] should be implemented.")

    def evaluate(self, eval_data):
        raise NotImplementedError("Method [evaluate] should be implemented.")


class Trainer(AbstractTrainer):
    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.stopping_step = config["stopping_step"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.valid_metric = config["valid_metric"].lower()
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.device = config["device"]
        self.weight_decay = 0.0
        if config["weight_decay"] is not None:
            wd = config["weight_decay"]
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd
        self.req_training = config["req_training"]
        self.show_progress = bool(config["show_progress"])

        self.start_epoch = 0
        self.cur_step = 0
        self.best_epoch = -1

        tmp_dd = {}
        for j, k in list(itertools.product(config["metrics"], config["topk"])):
            tmp_dd[f"{j.lower()}@{k}"] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        lr_scheduler = config["learning_rate_scheduler"]
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)

        self.evaluator = TopKEvaluator(config)
        self.save_path = log_directory(self.config)

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        learner = self.learner.lower()
        if learner == "adam":
            return optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == "sgd":
            return optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == "adagrad":
            return optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == "rmsprop":
            return optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == "adamw":
            return optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.logger.warning("Received unrecognized optimizer, set default Adam optimizer")
        return optim.Adam(params, lr=self.learning_rate)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        if not self.req_training:
            return 0.0, None
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        for batch_idx, interaction in enumerate(
            tqdm(train_data, total=len(train_data), ascii=" >=", disable=not self.show_progress, mininterval=2)
        ):
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if torch.isnan(loss):
                self.logger.info("Loss is nan at epoch: {}, batch index: {}. Exiting.".format(epoch_idx, batch_idx))
                return loss, None
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
        n_batches = max(len(train_data), 1)
        if isinstance(total_loss, tuple):
            total_loss = tuple(x / n_batches for x in total_loss)
        elif total_loss is not None:
            total_loss = total_loss / n_batches
        return total_loss, None

    def _valid_epoch(self, valid_data):
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result["recall@20"]
        return valid_score, valid_result

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        out = "epoch %d training [time: %.2fs, " % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            out += ", ".join("train_loss%d: %.4f" % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            out += "train loss: %.4f" % losses
        return out + "]"

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):  # nan
                break
            self.lr_scheduler.step()
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(
                    self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
                )
                if post_info is not None:
                    self.logger.info(post_info)

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(
                        "epoch %d evaluating [time: %.2fs, valid_score: %f]"
                        % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                    )
                    self.logger.info("valid result: \n" + dict2str(valid_result))
                    self.logger.info("test result: \n" + dict2str(test_result))

                if update_flag:
                    self.best_epoch = epoch_idx
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result
                    if saved:
                        self._save_checkpoint(self.model)
                    if verbose:
                        self.logger.info(self.config["model"] + "--Best validation results updated!!!")

                if stop_flag:
                    if verbose:
                        self.logger.info("+++++Finished training, best eval result in epoch %d" % self.best_epoch)
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        """Full ranking over all items for every user of ``eval_data``."""
        self.model.eval()
        max_k = max(self.config["topk"])
        topk_list, hits_list = [], []
        row = 0
        for batched_data in eval_data:
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(scores, max_k, dim=-1)
            hits_list.append(eval_data.batch_hits(row, topk_index).cpu())
            topk_list.append(topk_index.cpu())
            row += topk_index.shape[0]
        return self.evaluator.evaluate(topk_list, hits_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        epochs = sorted(self.train_loss_dict.keys())
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

    def _checkpoint_path(self):
        hp_str = ""
        for param_name in self.config["hyper_parameters"] or []:
            hp_str += f"_{param_name}{self.config[param_name]}"
        return os.path.join(self.save_path, f"Best_{self.config['model']}{hp_str}.pth")

    def _save_checkpoint(self, model):
        torch.save(model.state_dict(), self._checkpoint_path())
