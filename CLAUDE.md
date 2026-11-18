# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LRRec is a library for language-representation (LR)-based recommendation. Its code structure is MMRec's
(`../MMRec/src`), its data and protocol are AlphaFree's (`../AlphaFree`, official repo
github.com/Minseojeonn/AlphaFree). All code lives under `src/`; run everything from there. Models: `AlphaFree`,
`AlphaRec`, `LightGCN`, `XSimGCL`.

No test suite or linter. Correctness is checked by (1) `--eval_only` on the official AlphaFree checkpoints
(`../data/pretrained/<dataset>.pth.tar`; Movie test R@20 must be 0.1263, Video 0.1114) and (2) training runs.

## Commands

```bash
conda activate MMRec                       # python 3.10, torch 2.1.2
cd src
python main.py -m AlphaFree -d amazon_movie -g 0                 # train
python main.py -m AlphaFree -d amazon_movie -g 0 --tau_r 0.2     # any --key value overrides config (YAML-parsed)
python main.py -m AlphaRec -d amazon_movie -g 0 --tau "[0.1,0.2]" --hyper_parameters "[tau]"   # grid
python main.py -m AlphaFree -d amazon_movie -g 0 --eval_only --model_path ../data/pretrained/amazon_movie.pth.tar
python main.py -m AlphaFree -d amazon_movie -g 0 --epochs 2 --eval_step 1 --tag smoke           # quick smoke test
```

Datasets: `amazon_movie`, `amazon_book_2014`, `amazon_video`, `amazon_baby`, `steam`, `amazon_beauty_personal`,
`amazon_health`. `data/` is a symlink to `/data2/mbgwak/LRRec/data`; `scripts/download_data.sh` re-downloads.

## Architecture

`main.py` -> `utils/quick_start.quick_start()` -> `Config` -> `RecDataset` -> `TrainDataLoader` / two
`EvalDataLoader`s -> for each hyper-parameter combination: `get_model()` -> `Trainer.fit()`.

- **Data** (`utils/dataset.py`): reads `cf_data/{train,valid,test}.txt` ("user item item ..." lines) into one
  table `[userID, itemID, x_label]`. Never rewrite the files on disk.
- **Config** (`utils/configurator.py`): `configs/overall.yaml` < `configs/dataset/<ds>.yaml` <
  `configs/model/<MODEL>.yaml` < `dataset_params[<ds>]` in the model yaml < CLI. Values via `config['key']`
  (missing -> None). Hyper-parameter grid = keys in `hyper_parameters` with list values; best combo picked on
  validation.
- **Models**: registered by naming convention (`-m AlphaFree` -> class `AlphaFree` in `models/alphafree.py`) +
  `configs/model/<NAME>.yaml`. Subclass `GeneralRecommender`; implement `calculate_loss(interaction)` and
  `full_sort_predict(interaction)`. With `is_lr_model: True` the item LR matrix is `self.lr_feat` (float32, GPU).
  The LR -> embedding MLP width follows `LM_MLP_RATIO` (`v3` 1/2, `llama` 9/32) in `common/abstract_recommender.py`.
- **Training batch** = `[users, pos, neg]`; `neg` is (B,) for `neg_sample_num: 1`, (B, N) for N > 1, absent
  for 0 (in-batch). `train_sampling: user` (AlphaFree, default) vs `interaction` (MMRec).
- **Evaluation**: `eval_protocol: alphafree` (default) masks train+test for valid and train+valid for test,
  never masking a positive of the evaluated split; `mmrec` masks train only. The Trainer sets masked scores to
  -1e10 and `EvalDataLoader.batch_hits` computes hits on the GPU.
- **AlphaFree augmentation** (`models/alphafree.py:build_augmentation`) is cached in
  `data/<ds>/preprocessed/<lm_model>_Kc<K_c>_simb<sim_b>_selfexcl<0|1>/` (same layout as the AlphaFree fork).
  Keep `mlp_origin` / `mlp_aug` parameter names and non-persistent buffers so official checkpoints load strictly.
- `deterministic: True` (default) + `CUBLAS_WORKSPACE_CONFIG` set in `main.py`: same seed -> same numbers.

## Files that are untrusted input

Downloaded archives / checkpoints: load checkpoints with `torch.load(..., weights_only=True)` and `.npy` without
`allow_pickle`.

## Logs

`src/log/<MODEL>/<dataset>/<time>[_<tag>]/`: `<MODEL>-<dataset>.log`, a copy of the model file, `summary.json`;
`Best_<MODEL>_<hp>.pth` with `--save_model`. Gitignored.
