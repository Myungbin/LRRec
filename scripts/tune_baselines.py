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
    python scripts/tune_baselines.py --merge     # result/<MODEL>/<dataset>/: MMRec-style log of all configs + code

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
import shutil
import subprocess
import sys
import time
from collections import deque

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
SRC = os.path.abspath(SRC)
RESULT = os.path.join(os.path.dirname(SRC), "result")  # merged logs: result/<MODEL>/<dataset>/
METRIC = "recall@20"

REGS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
CL_LAMBDAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
CL_EPS = [0, 0.01, 0.05, 0.1, 0.2, 0.5]

# relative cost per epoch (dataset) and per run (model); only used to start long jobs first
DATASET_COST = {"amazon_movie": 1.0, "amazon_book_2014": 2.5, "amazon_video": 1.1}
MODEL_COST = {"MF": 1.0, "LightGCN": 3.0, "SimGCL": 1.5, "XSimGCL": 1.0}
STAGES_LEFT = {"SimGCL": 3, "XSimGCL": 2}  # staged models go first: their later stages wait for the first one


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
        cmd = [
            sys.executable,
            "main.py",
            "-m",
            self.model,
            "-d",
            self.dataset,
            "-g",
            str(gpu),
            "--tag",
            self.tag,
            "--seed",
            str(seed),
            "--deterministic",
            str(bool(deterministic)),
        ]
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


MERGE_SEP = "████Current BEST████"


def _stamp(line):
    """'[2026.11.19 00:58:39] ...' -> '[2026.11.19 00:58:39] '"""
    return line[: line.index("]") + 2] if line.startswith("[") and "]" in line else ""


def _dict2str(d):
    return "    ".join(f"{k}: {v:.4f}" for k, v in d.items())


DATASET_NAMES = {"amazon_movie": "Movie", "amazon_book_2014": "Book", "amazon_video": "Video"}
SEARCH_TEXT = {
    "MF": "`reg_weight` in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}",
    "LightGCN": "`n_layers` in {1, 2, 3, 4} x `reg_weight` in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}",
    "SimGCL": "(a) `lambda_cl` in {0.01, 0.05, 0.1, 0.2, 0.5, 1, 2}, `eps` 0.1, 2 layers -> (b) `eps` in "
    "{0, 0.01, 0.05, 0.1, 0.2, 0.5} -> (c) `n_layers` in {1, 2, 3}",
    "XSimGCL": "(a) `lambda_cl` in {0.01, 0.05, 0.1, 0.2, 0.5, 1} x `eps` in {0, 0.01, 0.05, 0.1, 0.2, 0.5}, "
    "2 layers, `layer_cl` 1 -> (b) `n_layers` in {1, ..., 5} x `layer_cl` in {1, ..., `n_layers`}",
}
# Table 1 of the AlphaFree paper (WWW'26): test Recall@20 / NDCG@20, user-wise holdout split
ALPHAFREE_PAPER = {
    ("MF", "amazon_movie"): (0.0580, 0.0533),
    ("MF", "amazon_book_2014"): (0.0436, 0.0387),
    ("MF", "amazon_video"): (0.0177, 0.0095),
    ("LightGCN", "amazon_movie"): (0.0860, 0.0754),
    ("LightGCN", "amazon_book_2014"): (0.0712, 0.0597),
    ("LightGCN", "amazon_video"): (0.0732, 0.0409),
    ("XSimGCL", "amazon_movie"): (0.0967, 0.0866),
    ("XSimGCL", "amazon_book_2014"): (0.0818, 0.0690),
    ("XSimGCL", "amazon_video"): (0.0897, 0.0500),
    ("AlphaFree", "amazon_movie"): (0.1267, 0.1194),
    ("AlphaFree", "amazon_book_2014"): (0.1014, 0.0861),
    ("AlphaFree", "amazon_video"): (0.1111, 0.0615),
}
MD_METRICS = ["recall@10", "recall@20", "ndcg@10", "ndcg@20"]
MD_HEAD = "R@10 | R@20 | N@10 | N@20"


def _md_params(params):
    return ", ".join(f"`{k}` {v:g}" if isinstance(v, float) else f"`{k}` {v}" for k, v in params.items())


def _md_metrics(res):
    return " | ".join(f"{res[k]:.4f}" for k in MD_METRICS)


def write_markdown(args, done):
    """result/README.md (best configuration per dataset and model) and result/<MODEL>/<dataset>/<MODEL>-<dataset>.md
    (every configuration, best first)."""
    by_valid = lambda r: -r["result"]["valid"][METRIC]
    setting = (
        f"Tuning `{args.name}`: seed {args.seed}, split of the files (about 4:3:3), AlphaFree evaluation protocol, "
        f"selection on valid {METRIC.replace('recall', 'Recall')}; metrics at the best validation epoch."
    )
    for m, d, keys, runs in done:
        best = min(runs, key=by_valid)
        lines = [
            f"# {m} on {DATASET_NAMES.get(d, d)} (`{d}`)",
            "",
            setting,
            "",
            f"- Search: {SEARCH_TEXT.get(m, '')}",
            f"- Best: {_md_params(best['params'])} (best epoch {best['result']['best_epoch']})",
            f"- Log: [`{m}-{d}.log`]({m}-{d}.log) ({len(runs)} configurations, MMRec format)",
            "",
            f"| | Valid {MD_HEAD} | Test {MD_HEAD} |",
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|".replace("|--:|--:|--:|--:|--:|--:|--:|--:|", "|--:" * 8 + "|"),
            f"| Best | {_md_metrics(best['result']['valid'])} | {_md_metrics(best['result']['test'])} |",
            "",
            "## All configurations (best valid first)",
            "",
            f"| # | Stage | Configuration | Best epoch | Valid {MD_HEAD} | Test {MD_HEAD} |",
            "|--:|--:|---|--:" + "|--:" * 8 + "|",
        ]
        for i, r in enumerate(sorted(runs, key=by_valid), 1):
            row = (
                f"| {i} | {r['stage']} | {_md_params(r['params'])} | {r['result']['best_epoch']} | "
                f"{_md_metrics(r['result']['valid'])} | {_md_metrics(r['result']['test'])} |"
            )
            lines.append(row.replace(f"| {i} |", f"| **{i}** |", 1) if r is best else row)
        with open(os.path.join(RESULT, m, d, f"{m}-{d}.md"), "w") as f:
            f.write("\n".join(lines) + "\n")

    models = list(dict.fromkeys(m for m, _, _, _ in done))
    datasets = list(dict.fromkeys(d for _, d, _, _ in done))
    lines = [
        "# Baseline tuning results",
        "",
        setting,
        "",
        "Each `<MODEL>/<dataset>/` folder holds the merged log of all configurations (MMRec format), the model code "
        "of the best run and a table of every configuration. The paper column is Table 1 of the AlphaFree paper "
        "(test Recall@20 / NDCG@20).",
        "",
        "Search spaces (original papers):",
        "",
    ]
    lines += [f"- **{m}**: {SEARCH_TEXT.get(m, '')}" for m in models]
    for d in datasets:
        lines += [
            "",
            f"## {DATASET_NAMES.get(d, d)} (`{d}`)",
            "",
            f"| Model | Best configuration | Best epoch | Test {MD_HEAD} | Valid {MD_HEAD} | Paper R@20 / N@20 |",
            "|---|---|--:" + "|--:" * 8 + "|--:|",
        ]
        for m, dd, keys, runs in done:
            if dd != d:
                continue
            best = min(runs, key=by_valid)
            paper = ALPHAFREE_PAPER.get((m, d))
            lines.append(
                f"| [{m}]({m}/{d}/{m}-{d}.md) | {_md_params(best['params'])} | {best['result']['best_epoch']} | "
                f"{_md_metrics(best['result']['test'])} | {_md_metrics(best['result']['valid'])} | "
                + (f"{paper[0]:.4f} / {paper[1]:.4f} |" if paper else "– |")
            )
        paper = ALPHAFREE_PAPER.get(("AlphaFree", d))
        if paper:
            lines.append(
                f"| AlphaFree (paper) | | | | {paper[0]:.4f} | | {paper[1]:.4f} | | | | | {paper[0]:.4f} / {paper[1]:.4f} |"
            )
    with open(os.path.join(RESULT, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"markdown -> {os.path.join(RESULT, 'README.md')} and result/<MODEL>/<dataset>/<MODEL>-<dataset>.md")


def merge(args):
    """One MMRec-style log per (model, dataset): the runs of this --name in a single log, as if they had run
    as one hyper-parameter grid (``=========i/N: Parameters:[...]=(...)=======`` per run, then All Over and
    BEST), written to ``result/<MODEL>/<dataset>/<MODEL>-<dataset>.log`` next to the model code of the best run
    (the copy saved in its log directory)."""
    out_dir = os.path.join(SRC, "log", "tuning", args.name)
    rows = {}
    with open(os.path.join(out_dir, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["result"] is not None:
                rows[(r["model"], r["dataset"], r["tag"])] = r  # a resumed run is listed again; keep one
    done = []
    for m in args.models:
        for d in args.datasets:
            runs = [r for (rm, rd, _), r in rows.items() if (rm, rd) == (m, d)]
            if not runs:
                continue
            keys = list(runs[0]["params"])
            runs.sort(key=lambda r: (r["stage"], tuple(r["params"][k] for k in keys)))
            hyper = ["seed"] + keys
            planned = len(first_stage(m)) if m not in STAGES_LEFT else None
            logs = []
            for r in runs:
                with open(os.path.join(r["result"]["dir"], f"{m}-{d}.log")) as f:
                    logs.append(f.read().splitlines())

            first = logs[0]
            start = next(i for i, x in enumerate(first) if "=========1/1: Parameters:" in x)
            values = {k: sorted({r["params"][k] for r in runs}) for k in keys}
            header = []
            for x in first[:start]:
                k = x.split("=", 1)[0]
                if k == "hyper_parameters":
                    x = f"hyper_parameters={hyper}"
                elif k in values:
                    x = f"{k}={values[k]}"
                elif k == "log_tag":
                    x = f"log_tag={args.name}"
                header.append(x)
            status = f"{len(runs)}/{planned} configurations" if planned else f"{len(runs)} configurations (staged)"
            header.insert(
                0,
                f"{_stamp(first[0])}Merged log of tuning '{args.name}': {status} finished, each run as its own "
                f"process (timestamps overlap); best on valid {METRIC}",
            )

            out, best = header, None
            for i, (r, log) in enumerate(zip(runs, logs), 1):
                s = next(j for j, x in enumerate(log) if "=========1/1: Parameters:" in x)
                e = next(j for j, x in enumerate(log) if MERGE_SEP in x)
                params = tuple([args.seed] + [r["params"][k] for k in keys])
                out.append(f"{_stamp(log[s])}========={i}/{len(runs)}: Parameters:{hyper}={params}=======")
                out += log[s + 1 : e]
                res = r["result"]
                if best is None or res["valid"][METRIC] > best[1]["valid"][METRIC]:
                    best = (params, res)
                out += [
                    f"{_stamp(log[e])}{MERGE_SEP}:",
                    f"Parameters: {hyper}={best[0]},",
                    f"Valid: {_dict2str(best[1]['valid'])},",
                    f"Test: {_dict2str(best[1]['test'])}",
                    "",
                    "",
                    "",
                ]
            stamp = next(_stamp(x) for x in reversed(logs[-1]) if _stamp(x))
            out += [f"{stamp}", "============All Over====================="]
            for r in runs:
                params = tuple([args.seed] + [r["params"][k] for k in keys])
                res = r["result"]
                out += [
                    f"{stamp}Parameters: {hyper}={params} (best epoch {res['best_epoch']}),",
                    f" best valid: {_dict2str(res['valid'])},",
                    f" best test: {_dict2str(res['test'])}",
                ]
            out += [
                f"{stamp}",
                "",
                "█████████████ BEST ████████████████",
                f"{stamp}\tParameters: {hyper}={best[0]},",
                f"Valid: {_dict2str(best[1]['valid'])},",
                f"Test: {_dict2str(best[1]['test'])}",
                "",
            ]
            target = os.path.join(RESULT, m, d)
            os.makedirs(target, exist_ok=True)
            path = os.path.join(target, f"{m}-{d}.log")
            with open(path, "w") as f:
                f.write("\n".join(out) + "\n")
            for code in glob.glob(os.path.join(best[1]["dir"], "*.py")):
                shutil.copy2(code, target)
            print(f"{m} {d}: {len(runs)} runs -> {target}")
            done.append((m, d, keys, runs))
    write_markdown(args, done)


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
    pending = deque(
        sorted((j for s in searches.values() for j in s.start(first_stage(s.model))), key=lambda j: -j.cost)
    )
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
            f.write(
                json.dumps(
                    {
                        "model": job.model,
                        "dataset": job.dataset,
                        "params": job.params,
                        "stage": job.stage,
                        "tag": job.tag,
                        "result": result,
                    }
                )
                + "\n"
            )
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
            proc = subprocess.Popen(
                job.command(gpu, args.seed, args.deterministic),
                cwd=SRC,
                env=env,
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )
            running.append((proc, job, gpu, logfile))
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
                note(
                    f"done {job.model} {job.dataset} {job.tag}: valid {result['valid'][METRIC]:.4f} "
                    f"test {result['test'][METRIC]:.4f} (epoch {result['best_epoch']})"
                )
            complete(job, result)
        running = still
        if running or pending:
            time.sleep(args.poll)
    note("all done")
    report(args, searches)
    merge(args)


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
    lines = [
        "| Model | Dataset | Best configuration | Best epoch | Valid R@20 | Test R@20 | Test N@20 |",
        "|---|---|---|--:|--:|--:|--:|",
    ]
    for m, d, s, path in rows:
        skip = ("seed", "model", "dataset", "deterministic")
        params = ", ".join(f"{k}={v}" for k, v in s.get("overrides", {}).items() if k not in skip)
        lines.append(
            f"| {m} | {d} | {params} | {s['best']['best_epoch']} | {s['best']['valid']['recall@20']:.4f} | "
            f"{s['best']['test']['recall@20']:.4f} | {s['best']['test']['ndcg@20']:.4f} |"
        )
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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="bit-for-bit repeatable runs; the graph models are about 2.5x slower",
    )
    parser.add_argument("--poll", type=float, default=15.0, help="seconds between checks")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--merge", action="store_true", help="write one MMRec-style log per model/dataset")
    args = parser.parse_args()
    args.models = args.models.split(",")
    args.datasets = args.datasets.split(",")
    if args.report:
        report(args)
    elif args.merge:
        merge(args)
    else:
        run(args)
