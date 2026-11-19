# coding: utf-8
"""
Hierarchical configuration
################################

Priority (later wins):

    configs/overall.yaml
      < configs/dataset/<dataset>.yaml
      < configs/model/<MODEL>.yaml
      < configs/model/<MODEL>.yaml :: dataset_params[<dataset>]   (per-dataset tuned values)
      < config_dict (main.py: CLI flags and ``--key value`` overrides)

Inside a model every value is read dict-style: ``config['key']`` (missing keys return None).
"""

import os
import re

import torch
import yaml


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader that also reads scientific notation without a dot ('1e-3') as a float; plain YAML 1.1 makes it
    a string. Used for the config files and for the command-line overrides of main.py."""


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
         [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_yaml(text):
    return yaml.load(text, Loader=_ConfigLoader)


class Config(object):
    def __init__(self, model=None, dataset=None, config_dict=None):
        if config_dict is None:
            config_dict = {}
        config_dict["model"] = model
        config_dict["dataset"] = dataset
        self.final_config_dict = self._load_dataset_model_config(config_dict)
        self.final_config_dict.update(config_dict)
        self._set_default_parameters()
        self._init_device()

    def _load_dataset_model_config(self, config_dict):
        file_config_dict = dict()
        cur_dir = os.path.join(os.getcwd(), "configs")
        file_list = [
            os.path.join(cur_dir, "overall.yaml"),
            os.path.join(cur_dir, "dataset", "{}.yaml".format(config_dict["dataset"])),
            os.path.join(cur_dir, "model", "{}.yaml".format(config_dict["model"])),
        ]
        if not os.path.isfile(file_list[0]):
            raise FileNotFoundError("configs/overall.yaml not found -- run from LRRec/src")
        hyper_parameters = []
        dataset_params = {}
        for file in file_list:
            if os.path.isfile(file):
                with open(file, "r", encoding="utf-8") as f:
                    fdata = load_yaml(f.read()) or {}
                if fdata.get("hyper_parameters"):
                    hyper_parameters.extend(fdata["hyper_parameters"])
                dataset_params = fdata.pop("dataset_params", None) or dataset_params
                file_config_dict.update(fdata)
        # per-dataset overrides declared inside the model yaml
        file_config_dict.update(dataset_params.get(config_dict["dataset"], {}) or {})
        file_config_dict["hyper_parameters"] = hyper_parameters
        return file_config_dict

    def _set_default_parameters(self):
        smaller_metric = ["rmse", "mae", "logloss"]
        valid_metric = self.final_config_dict["valid_metric"].split("@")[0]
        self.final_config_dict["valid_metric_bigger"] = valid_metric.lower() not in smaller_metric
        if "seed" not in self.final_config_dict["hyper_parameters"]:
            self.final_config_dict["hyper_parameters"] += ["seed"]

    def _init_device(self):
        use_gpu = self.final_config_dict["use_gpu"]
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict["gpu_id"])
        self.final_config_dict["device"] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        return self.final_config_dict.get(item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = "\n"
        args_info += "\n".join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()])
        args_info += "\n\n"
        return args_info

    def __repr__(self):
        return self.__str__()
