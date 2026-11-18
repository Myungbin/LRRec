# coding: utf-8
"""
Run application
##########################
main.py -> quick_start(): Config -> RecDataset -> loaders -> (for every hyper-parameter combination) model -> Trainer.fit
"""

import gc
import json
import os
import platform
from itertools import product
from logging import getLogger

import torch

from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.utils import dict2str, get_model, get_trainer, init_seed


def build_loaders(config, logger):
    dataset = RecDataset(config)
    logger.info(str(dataset))
    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info("\n====Training====\n" + str(train_dataset))
    logger.info("\n====Validation====\n" + str(valid_dataset))
    logger.info("\n====Testing====\n" + str(test_dataset))

    train_data = TrainDataLoader(config, train_dataset, batch_size=config["train_batch_size"], shuffle=True)

    protocol = (config["eval_protocol"] or "alphafree").lower()
    if protocol == "alphafree":
        # AlphaFree / AlphaRec evaluator: everything the user interacted with outside the evaluated split is
        # masked; a positive of the evaluated split itself is never masked
        valid_mask, test_mask, exclude = [train_dataset, test_dataset], [train_dataset, valid_dataset], True
        logger.info("Eval protocol: alphafree (valid masks train+test, test masks train+valid)")
    elif protocol == "mmrec":
        valid_mask, test_mask, exclude = [train_dataset], [train_dataset], False
        if config["mask_valid_for_test"]:
            test_mask = [train_dataset, valid_dataset]
        logger.info("Eval protocol: mmrec (train positives masked%s)"
                    % (", + valid for test" if config["mask_valid_for_test"] else ""))
    else:
        raise ValueError("eval_protocol must be 'alphafree' or 'mmrec'")

    bs = config["eval_batch_size"]
    valid_data = EvalDataLoader(config, valid_dataset, valid_mask, batch_size=bs, exclude_eval_positives=exclude)
    test_data = EvalDataLoader(config, test_dataset, test_mask, batch_size=bs, exclude_eval_positives=exclude)
    return train_data, valid_data, test_data


def _hyper_combinations(config):
    if "seed" not in config["hyper_parameters"]:
        config["hyper_parameters"] = ["seed"] + config["hyper_parameters"]
    hyper_ls = []
    for name in config["hyper_parameters"]:
        value = config[name]
        hyper_ls.append(value if isinstance(value, list) else [value])
    return list(product(*hyper_ls))


def quick_start(model, dataset, config_dict, save_model=True):
    config = Config(model, dataset, config_dict)
    log_dir = init_logger(config)
    logger = getLogger()
    logger.info("Server: \t" + platform.node())
    logger.info("Dir: \t" + os.getcwd() + "\n")
    logger.info(config)

    train_data, valid_data, test_data = build_loaders(config, logger)

    hyper_ret = []
    val_metric = config["valid_metric"].lower()
    best_valid_value, best_idx = None, 0
    combinators = _hyper_combinations(config)
    total_loops = len(combinators)
    logger.info("\n\n=================================\n\n")
    for idx, hyper_tuple in enumerate(combinators):
        for name, value in zip(config["hyper_parameters"], hyper_tuple):
            config[name] = value
        init_seed(config["seed"], config["deterministic"])
        logger.info(
            "========={}/{}: Parameters:{}={}=======".format(idx + 1, total_loops, config["hyper_parameters"], hyper_tuple)
        )

        train_data.pretrain_setup()
        model_obj = get_model(config["model"])(config, train_data).to(config["device"])
        logger.info(model_obj)

        trainer = get_trainer()(config, model_obj)
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data, valid_data=valid_data, test_data=test_data, saved=save_model
        )
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid, trainer.best_epoch))

        # select the best combination on VALIDATION (the test score is only reported)
        if best_valid_value is None or best_valid_result[val_metric] > best_valid_value:
            best_valid_value = best_valid_result[val_metric]
            best_idx = idx

        logger.info("best valid result: {}".format(dict2str(best_valid_result)))
        logger.info("test result: {}".format(dict2str(best_test_upon_valid)))
        logger.info(
            "████Current BEST████:\nParameters: {}={},\nValid: {},\nTest: {}\n\n\n".format(
                config["hyper_parameters"],
                hyper_ret[best_idx][0],
                dict2str(hyper_ret[best_idx][1]),
                dict2str(hyper_ret[best_idx][2]),
            )
        )

        if idx + 1 < total_loops:
            del model_obj, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    logger.info("\n============All Over=====================")
    for p, k, v, e in hyper_ret:
        logger.info(
            "Parameters: {}={} (best epoch {}),\n best valid: {},\n best test: {}".format(
                config["hyper_parameters"], p, e, dict2str(k), dict2str(v)
            )
        )
    logger.info("\n\n█████████████ BEST ████████████████")
    logger.info(
        "\tParameters: {}={},\nValid: {},\nTest: {}\n\n".format(
            config["hyper_parameters"],
            hyper_ret[best_idx][0],
            dict2str(hyper_ret[best_idx][1]),
            dict2str(hyper_ret[best_idx][2]),
        )
    )
    _write_summary(log_dir, config, hyper_ret, best_idx)
    return hyper_ret[best_idx]


def _write_summary(log_dir, config, hyper_ret, best_idx):
    """Machine-readable result: log/<model>/<dataset>/<time>/summary.json."""
    runs = [
        {"params": dict(zip(config["hyper_parameters"], map(_jsonable, p))), "best_epoch": e, "valid": k, "test": v}
        for p, k, v, e in hyper_ret
    ]
    summary = {"model": config["model"], "dataset": config["dataset"], "best": runs[best_idx], "runs": runs}
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _jsonable(x):
    return x if isinstance(x, (int, float, str, bool)) or x is None else str(x)


def quick_eval(model, dataset, config_dict, model_path):
    """Evaluate a saved state_dict (or an official ``{'epoch', 'state_dict'}`` checkpoint) without training."""
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    train_data, valid_data, test_data = build_loaders(config, logger)
    seed = config["seed"]
    init_seed(seed[0] if isinstance(seed, list) else seed, config["deterministic"])
    model_obj = get_model(config["model"])(config, train_data).to(config["device"])
    state = torch.load(model_path, map_location=config["device"], weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_obj.load_state_dict(state)
    logger.info(f"Loaded {model_path}")
    trainer = get_trainer()(config, model_obj)
    valid_result = trainer.evaluate(valid_data)
    test_result = trainer.evaluate(test_data, is_test=True)
    logger.info("valid result: \n" + dict2str(valid_result))
    logger.info("test result: \n" + dict2str(test_result))
    return valid_result, test_result
