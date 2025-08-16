"""Toolbox for handling experiment metadata."""

from __future__ import annotations

import abc
import hashlib
import logging
import multiprocessing
import pathlib
import pickle
import re
import signal
import subprocess
import sys
import time
import uuid
from typing import Any

import codecarbon
import pynvml

from t5.config import Config


MAXIMUM_FILENAME_LENGTH: int = 256

logger: logging.Logger = logging.getLogger(__name__)


class BaseExperiment(abc.ABC):
    """
    Base class for running an experiment.

    Calling run on an instance of this class will call the `init()` then `main()` functions.
    The sole purpose of this class is to make an experiment "prettier": it displays config values, store a diff of the repo in the experiment logdir, etc.

    Config
    ------
    log_level: str
        Set log level for stderr (one of DEBUG, INFO, WARNING, ERROR, CRITICAL)

    """

    def __init__(self: BaseExperiment) -> None:
        """Create a new experiment, reading the config from argv."""
        self.config = Config(sys.argv)
        self.version: str = self.get_repository_version()
        self.set_expdir_path()
        self.setup_logging()
        self.save_patch()
        self.save_raw_config()
        self.hook_signals()
        self.init()
        self.check_save_config()

    def __call__(self: BaseExperiment) -> None:
        """Run the experiment."""
        self.run()

    @abc.abstractmethod
    def init(self: BaseExperiment) -> None:
        """Initialize experiment (e.g. create model)."""

    @abc.abstractmethod
    def run(self: BaseExperiment) -> None:
        """Run experiment (e.g. train model)."""

    def set_expdir_path(self: BaseExperiment) -> None:
        """Set paths to the experiment log and model directories."""
        args: str = " ".join(sys.argv[1:])
        stime: str = time.strftime("%FT%H:%M:%S")
        directory_name: str = f"AEC {self.version} {args} {stime}".replace("/", "_")
        if len(directory_name) > MAXIMUM_FILENAME_LENGTH:
            unique_suffix: str = hashlib.sha256(directory_name.encode("utf-8")).hexdigest()[:8]
            directory_name = f"{directory_name[:MAXIMUM_FILENAME_LENGTH-len(unique_suffix)-1]} {unique_suffix}"
        self.logdir: pathlib.Path = self.config.LOG_PATH / directory_name
        self.modeldir: pathlib.Path = self.config.MODEL_PATH / directory_name

    @staticmethod
    def get_repository_version() -> str:
        """Get the git repository version."""
        source_directory = pathlib.Path(__file__).parents[0]
        result: subprocess.CompletedProcess[str] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8",
                cwd=source_directory, shell=False, check=False)

        if result.returncode != 0:
            return "release"
        commit_hash: str = result.stdout.strip()[:8]

        result = subprocess.run(
                ["git", "status", "--porcelain"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8",
                cwd=source_directory, shell=False, check=True)
        if re.search(r"^ M ", result.stdout, re.MULTILINE):
            return f"{commit_hash}+"
        return commit_hash

    def setup_logging(self: BaseExperiment) -> None:
        """Initialise the logging system."""
        print(f"Log directory is {self.logdir}", file=sys.stderr)
        self.logdir.mkdir(parents=True, exist_ok=False)
        root_logger: logging.Logger = logging.getLogger()
        root_logger.setLevel(logging.NOTSET)

        stderr_handler: logging.Handler = logging.StreamHandler()
        self.config.declare_option("log_level", str, "Set log level for stderr (one of DEBUG, INFO, WARNING, ERROR, CRITICAL)", "WARNING")
        stderr_handler.setLevel(getattr(logging, self.config.log_level))
        stderr_handler.setFormatter(logging.Formatter("%(levelname)s (%(name)s) %(message)s"))
        stderr_handler.addFilter(logging.Filter("t5"))
        root_logger.addHandler(stderr_handler)

        logfile_handler: logging.Handler = logging.FileHandler(self.logdir / "log")
        logfile_handler.setLevel(logging.INFO)
        logfile_handler.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s:%(name)s:%(message)s"))
        root_logger.addHandler(logfile_handler)

    def save_patch(self: BaseExperiment) -> None:
        """Save the diff between the current code and the last commit."""
        if not self.version.endswith("+"):
            # The code is a release or wasn't modified
            return

        logging.warning("Uncommited changes detected, saving patch to logdir.")
        source_directory = pathlib.Path(__file__).parents[0]
        output_path: pathlib.Path = self.logdir / "uncommited.patch"
        with output_path.open("w") as outfile:
            subprocess.run(
                    ["git", "diff", "HEAD"],
                    stdout=outfile, stderr=subprocess.DEVNULL, encoding="utf-8",
                    cwd=source_directory, shell=False, check=True)

    def save_raw_config(self: BaseExperiment) -> None:
        """Save the unchecked configuration in a machine format."""
        with (self.logdir / "config.pkl").open("wb") as file:
            pickle.dump(self.config, file)

    def check_save_config(self: BaseExperiment) -> None:
        """Check the config is valid and save it in a human-readable format."""
        self.config.check_schema()
        self.config.log()
        self.config.save(self.logdir / "config.py")

    def hook_signals(self: BaseExperiment) -> None:
        """Change the behavior of SIGINT (^C) to change a variable `self.interrupted' before killing the process."""
        self.interrupted: bool = False

        def handler(*_: Any) -> None:
            if multiprocessing.current_process().name != "MainProcess":
                return

            logging.critical("Interrupted, attempting to stop execution gracefully.")
            logging.critical("NEXT ^C WILL KILL THE PROCESS!")
            self.interrupted = True
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        signal.signal(signal.SIGINT, handler)
