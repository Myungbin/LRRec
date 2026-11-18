# coding: utf-8
"""
Main entry (run from LRRec/src)
##########################

    python main.py -m AlphaFree -d amazon_movie -g 0
    python main.py -m AlphaFree -d amazon_movie --tau_r 0.2 --K_c "[5,10]" --hyper_parameters "[K_c]"
    python main.py -m AlphaFree -d amazon_movie --eval_only --model_path ../data/pretrained/amazon_movie.pth.tar

Any ``--<key> <value>`` (or ``--<key>=<value>``) not listed below overrides that config key; values are parsed as
YAML, so ``0.2`` -> float, ``"[0.1,0.2]"`` -> list, ``true`` -> bool.
"""

import argparse
import os

import yaml

os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count()))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by deterministic cuBLAS (config: deterministic)


def parse_overrides(unknown):
    overrides, i = {}, 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            raise SystemExit(f"unexpected argument: {token}")
        key = token[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            i += 1
        elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            value = unknown[i + 1]
            i += 2
        else:
            value, i = "true", i + 1
        overrides[key] = yaml.safe_load(value)
    return overrides


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="AlphaFree", help="name of models")
    parser.add_argument("--dataset", "-d", type=str, default="amazon_movie", help="name of datasets")
    parser.add_argument("--gpu_id", "-g", type=int, default=0, help="GPU device id")
    parser.add_argument("--save_model", "-s", action="store_true", help="save the best checkpoint in the log dir")
    parser.add_argument("--tag", "-t", type=str, default=None,
                        help="optional run tag appended to the log dir: log/<model>/<dataset>/<time>_<tag>/")
    parser.add_argument("--eval_only", action="store_true", help="evaluate --model_path without training")
    parser.add_argument("--model_path", type=str, default=None, help="checkpoint for --eval_only")
    args, unknown = parser.parse_known_args()

    config_dict = {"gpu_id": args.gpu_id}
    if args.tag:
        config_dict["log_tag"] = args.tag
    config_dict.update(parse_overrides(unknown))

    from utils.quick_start import quick_eval, quick_start  # after CUDA_VISIBLE_DEVICES can still be set

    if args.eval_only:
        if not args.model_path:
            raise SystemExit("--eval_only needs --model_path")
        quick_eval(args.model, args.dataset, config_dict, args.model_path)
    else:
        quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=args.save_model)
