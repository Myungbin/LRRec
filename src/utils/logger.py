"""Logging: stdout + ./log/<model>/<dataset>/<time>/<model>-<dataset>.log, plus a snapshot of the model file."""

import logging
import os
import shutil
from datetime import datetime

from utils.utils import KST, log_directory


def _kst_timetuple(timestamp):
    return datetime.fromtimestamp(timestamp, KST).timetuple()


def init_logger(config):
    log_root = log_directory(config)
    os.makedirs(log_root, exist_ok=True)
    logfilepath = os.path.join(log_root, "{}-{}.log".format(config["model"], config["dataset"]))

    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y.%m.%d %H:%M:%S")
    fmt.converter = _kst_timetuple  # same clock (KST) as the run directory name, whatever the server's timezone
    state = (config["state"] or "info").lower()
    level = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }.get(state, logging.INFO)

    fh = logging.FileHandler(logfilepath, "w", "utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[sh, fh], force=True)

    copy_model_file(config, log_root)
    return log_root


def copy_model_file(config, log_dir):
    """Snapshot models/<model>.py next to the log so a result can always be traced to its code."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    model_file = os.path.join(models_dir, f"{config['model'].lower()}.py")
    if os.path.exists(model_file):
        shutil.copy2(model_file, os.path.join(log_dir, os.path.basename(model_file)))
