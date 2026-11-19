#!/usr/bin/env python
"""Tune the ID-based baselines with the search ranges of their original papers.

One process (``src/main.py``) per configuration, several processes per GPU, selection on validation Recall@20.
Staged searches start their next stage as soon as the previous one has finished on that dataset. Finished
configurations are found again by their tag (``src/log/<MODEL>/<dataset>/*_<tag>/summary.json``), so an
interrupted run continues where it stopped when started again with the same --name.

Jobs run with ``deterministic: False`` (cuSPARSE CSR graph propagation, about 2.5x faster for the graph models)
unless --deterministic is given. One job already keeps a GPU busy; more slots per GPU add little throughput
(about +15% at 4-8 jobs on an RTX A6000) but let short jobs run next to long ones.

    python scripts/tune_baselines.py --gpus 0:4,1:4,2:2                    # run (or resume)
    python scripts/tune_baselines.py --gpus 0:4,1:4,2:2 --dry_run          # list the first-stage jobs
    python scripts/tune_baselines.py --report                              # best configuration per model/dataset

Search spaces (seed 2026; general settings in configs/model/<MODEL>.yaml):
  MF        reg_weight in {1e-6, ..., 1e-2}                         (the BPR paper has no grid; LightGCN L2 range)
  LightGCN  n_layers in {1, 2, 3, 4} x reg_weight in {1e-6, ..., 1e-2}           (LightGCN, Sec. 4.1.2)
  SimGCL    a: lambda_cl in {0.01, 0.05, 0.1, 0.2, 0.5, 1, 2}, eps 0.1, 2 layers  (SimGCL, Sec. 4.3.1)
            b: eps in {0, 0.01, 0.05, 0.1, 0.2, 0.5} at the best lambda_cl         (SimGCL, Sec. 4.3.2)
            c: n_layers in {1, 2, 3} at the best (lambda_cl, eps)                 (SimGCL, Table 3)
  XSimGCL   a: lambda_cl in {0.01, 0.05, 0.1, 0.2, 0.5, 1} x eps in {0, 0.01, 0.05, 0.1, 0.2, 0.5},
               2 layers, layer_cl 1                                               (XSimGCL, Sec. 4.3.1)
            b: n_layers in {1, ..., 5} x layer_cl in {1, ..., n_layers} at the best (lambda_cl, eps)
                                                                                  (XSimGCL, Table 4, Sec. 4.3.2)
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import deque

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
SRC = os.path.abspath(SRC)
METRIC = "recall@20"

REGS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
CL_LAMBDAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
CL_EPS = [0, 0.01, 0.05, 0.1, 0.2, 0.5]

# relative cost per epoch (dataset) and per run (model); only used to start long jobs first
DATASET_COST = {"amazon_movie": 1.0, "amazon_book_2014": 2.5, "amazon_video": 1.1}
MODEL_COST = {"MF": 1.0, "LightGCN": 3.0, "SimGCL": 1.5, "XSimGCL": 1.0}
STAGES_LEFT = {"SimGCL": 3, "XSimGCL": 2}  # staged models go first: their later stages wait for the first one


# ------------------------------------------------------------------------------------------- search spaces
def first_stage(model):
    if model == "MF":
        return [{"reg_weight": r} for r in REGS]
    if model == "LightGCN":
        return [{"n_layers": n, "reg_weight": r} for n in (1, 2, 3, 4) for r in REGS]
    if model == "SimGCL":
        return [{"n_layers": 2, "lambda_cl": lam, "eps": 0.1} for lam in CL_LAMBDAS + [2]]
    if model == "XSimGCL":
        return [{"n_layers": 2, "layer_cl": 1, "lambda_cl": lam, "eps": e} for lam in CL_LAMBDAS for e in CL_EPS]
    raise ValueError(model)


def next_stage(model, stage, best):
    """Parameter sets of the stage after ``stage`` given the best result so far, or None when done."""
    p = best["params"]
    if model == "SimGCL" and stage == 0:
        return [{"n_layers": 2, "lambda_cl": p["lambda_cl"], "eps": e} for e in CL_EPS]
    if model == "SimGCL" and stage == 1:
        return [{"n_layers": n, "lambda_cl": p["lambda_cl"], "eps": p["eps"]} for n in (1, 2, 3)]
    if model == "XSimGCL" and stage == 0:
        return [
            {"n_layers": n, "layer_cl": l, "lambda_cl": p["lambda_cl"], "eps": p["eps"]}
            for n in range(1, 6)
            for l in range(1, n + 1)
        ]
    return None


# ------------------------------------------------------------------------------------------- runs
def tag_of(name, params):
    return name + "_" + "_".join(f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in params.items())


def find_result(model, dataset, tag):
    """The newest finished run with this tag, as {'valid', 'test', 'best_epoch', 'dir'}, or None."""
    for d in sorted(glob.glob(os.path.join(SRC, "log", model, dataset, f"*_{tag}")), reverse=True):
        path = os.path.join(d, "summary.json")
        if os.path.isfile(path):
            with open(path) as f:
                best = json.load(f)["best"]
            return {"valid": best["valid"], "test": best["test"], "best_epoch": best["best_epoch"], "dir": d}
    return None


class Job:
    def __init__(self, name, model, dataset, params, stage):
        self.model, self.dataset, self.params, self.stage = model, dataset, params, stage
        self.tag = tag_of(name, params)
        self.cost = DATASET_COST.get(dataset, 1.0) * MODEL_COST[model] * STAGES_LEFT.get(model, 1)

    def command(self, gpu, seed, deterministic):
        cmd = [sys.executable, "main.py", "-m", self.model, "-d", self.dataset, "-g", str(gpu),
               "--tag", self.tag, "--seed", str(seed), "--deterministic", str(bool(deterministic))]
        for k, v in self.params.items():
            cmd += [f"--{k}", str(v)]
        return cmd


class Search:
    """Staged search of one (model, dataset)."""

    def __init__(self, name, model, dataset):
        self.name, self.model, self.dataset = name, model, dataset
        self.stage = 0
        self.results = []  # (params, result) of finished runs, failures excluded
        self.open = set()  # tags of the current stage not finished yet

    def start(self, params_list):
        jobs, seen = [], {tag_of(self.name, p) for p, _ in self.results}
        for params in params_list:
            job = Job(self.name, self.model, self.dataset, params, self.stage)
            if job.tag in seen:  # e.g. the best of stage a is part of stage b
                continue
            seen.add(job.tag)
            self.open.add(job.tag)
            jobs.append(job)
        return jobs

    def best(self):
        ok = [(p, r) for p, r in self.results if r is not None]
        if not ok:
            return None
        p, r = max(ok, key=lambda pr: pr[1]["valid"][METRIC])
        return {"params": p, **r}

    def finish(self, job, result):
        """Record a finished job; returns the jobs of the next stage when this stage is complete."""
        self.results.append((job.params, result))
        self.open.discard(job.tag)
        if self.open:
            return []
        best = self.best()
        nxt = next_stage(self.model, self.stage, best) if best else None
        if not nxt:
            return []
        self.stage += 1
        jobs = self.start(nxt)
        return jobs


def parse_gpus(spec):
    slots = {}
    for part in spec.split(","):
        gpu, _, n = part.partition(":")
        slots[int(gpu)] = int(n or 1)
    return slots


def run(args):
    out_dir = os.path.join(SRC, "log", "tuning", args.name)
    os.makedirs(os.path.join(out_dir, "jobs"), exist_ok=True)
    searches = {(m, d): Search(args.name, m, d) for m in args.models for d in args.datasets}
    pending = deque(sorted((j for s in searches.values() for j in s.start(first_stage(s.model))),
                           key=lambda j: -j.cost))
    if args.dry_run:
        for j in pending:
            print(j.model, j.dataset, j.tag)
        print(f"{len(pending)} first-stage jobs")
        return

    free = parse_gpus(args.gpus)
    running = []  # (proc, job, gpu, logfile)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    log = open(os.path.join(out_dir, "tune.log"), "a")

    def note(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def complete(job, result):
        with open(os.path.join(out_dir, "results.jsonl"), "a") as f:
            f.write(json.dumps({"model": job.model, "dataset": job.dataset, "params": job.params,
                                "stage": job.stage, "tag": job.tag, "result": result}) + "\n")
        follow = searches[(job.model, job.dataset)].finish(job, result)
        for j in reversed(follow):  # next stages jump the queue
            pending.appendleft(j)
        if follow:
            note(f"{job.model} {job.dataset}: stage {follow[0].stage} -> {len(follow)} jobs")

    note(f"start: {len(pending)} first-stage jobs, gpus {args.gpus}")
    while pending or running:
        # launch (or reuse a finished run with the same tag)
        while pending:
            job = pending[0]
            done = find_result(job.model, job.dataset, job.tag)
            if done is not None:
                pending.popleft()
                complete(job, done)
                continue
            gpu = max(free, key=lambda g: free[g])
            if free[gpu] == 0:
                break
            pending.popleft()
            free[gpu] -= 1
            logfile = open(os.path.join(out_dir, "jobs", f"{job.model}_{job.dataset}_{job.tag}.out"), "w")
            proc = subprocess.Popen(job.command(gpu, args.seed, args.deterministic), cwd=SRC, env=env, stdout=logfile,
                                    stderr=subprocess.STDOUT)
            running.append((proc, job, gpu, logfile))
        # collect
        still = []
        for proc, job, gpu, logfile in running:
            if proc.poll() is None:
                still.append((proc, job, gpu, logfile))
                continue
            logfile.close()
            free[gpu] += 1
            result = find_result(job.model, job.dataset, job.tag) if proc.returncode == 0 else None
            if result is None:
                note(f"FAILED (exit {proc.returncode}) {job.model} {job.dataset} {job.tag}: see {logfile.name}")
            else:
                note(f"done {job.model} {job.dataset} {job.tag}: valid {result['valid'][METRIC]:.4f} "
                     f"test {result['test'][METRIC]:.4f} (epoch {result['best_epoch']})")
            complete(job, result)
        running = still
        if running or pending:
            time.sleep(args.poll)
    note("all done")
    report(args, searches)


def report(args, searches=None):
    """Best configuration per (model, dataset) from the runs of this --name."""
    rows = []
    for m in args.models:
        for d in args.datasets:
            best = None
            for path in glob.glob(os.path.join(SRC, "log", m, d, f"*_{args.name}_*", "summary.json")):
                with open(path) as f:
                    s = json.load(f)
                v = s["best"]["valid"][METRIC]
                if best is None or v > best[0]:
                    best = (v, s, path)
            if best:
                rows.append((m, d, best[1], os.path.dirname(best[2])))
    lines = ["| Model | Dataset | Best configuration | Best epoch | Valid R@20 | Test R@20 | Test N@20 |",
             "|---|---|---|--:|--:|--:|--:|"]
    for m, d, s, path in rows:
        params = ", ".join(f"{k}={v}" for k, v in s.get("overrides", {}).items() if k != "seed")
        lines.append(f"| {m} | {d} | {params} | {s['best']['best_epoch']} | {s['best']['valid']['recall@20']:.4f} | "
                     f"{s['best']['test']['recall@20']:.4f} | {s['best']['test']['ndcg@20']:.4f} |")
    text = "\n".join(lines)
    print(text)
    out_dir = os.path.join(SRC, "log", "tuning", args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "best.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="tune2026", help="run group; part of every tag")
    parser.add_argument("--models", default="MF,LightGCN,SimGCL,XSimGCL")
    parser.add_argument("--datasets", default="amazon_movie,amazon_book_2014,amazon_video")
    parser.add_argument("--gpus", default="0:1", help="gpu:slots,... e.g. 0:6,1:6,2:3")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads per job")
    parser.add_argument("--deterministic", action="store_true",
                        help="bit-for-bit repeatable runs; the graph models are about 2.5x slower")
    parser.add_argument("--poll", type=float, default=15.0, help="seconds between checks")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    args.models = args.models.split(",")
    args.datasets = args.datasets.split(",")
    if args.report:
        report(args)
    else:
        run(args)
