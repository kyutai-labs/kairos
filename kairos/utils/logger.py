import csv
import json
import logging
import os
import sys
from abc import ABC
from pathlib import Path
from typing import Any


# Logger API
class BaseLogger(ABC):
    def log(self, *args, **kwargs) -> None:
        pass

    def log_config(self, config: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def end_run(self) -> None:
        pass


# Console and file logger
class ConsoleFileLogger(BaseLogger, logging.LoggerAdapter):
    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = True,
        extra: dict[str, object] | None = None,
        log_file: str | None = None,
        level: int = logging.INFO,
    ) -> None:
        logger = self._init_logger(name=name, log_file=log_file, level=level)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def _init_logger(
        self, name: str, log_file: str | None = None, level: int = logging.INFO
    ) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        logger.handlers = []

        LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=LOGGING_FORMAT, datefmt=DATE_FORMAT)

        # Handler for stdout (for INFO and WARNING messages)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(
            lambda record: record.levelno < logging.ERROR
        )  # Filter out ERROR
        logger.addHandler(stdout_handler)

        # Handler for stderr (ERROR messages)
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(logging.ERROR)
        logger.addHandler(stderr_handler)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        current_rank = self._get_rank()
        msg = f"[ConsoleFileLogger] Initialized. Level: {logging.getLevelName(level)}"
        logger.info(self._rank_prefixed_message(msg, current_rank))
        return logger

    @staticmethod
    def _rank_prefixed_message(msg: str, rank: int) -> str:
        return f"[RANK {rank}] {msg}"

    @staticmethod
    def _get_rank() -> int:
        rank = os.environ.get("SLURM_PROCID", "0")
        return int(rank)

    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        if self.isEnabledFor(level):
            msg, processed_kwargs = self.process(msg, kwargs)
            current_rank = self._get_rank()
            msg = self._rank_prefixed_message(str(msg), current_rank)
            if not self.rank_zero_only or current_rank == 0:
                self.logger.log(level, msg, *args, **processed_kwargs)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        metrics_msg = f"Step {step}: " if step is not None else ""
        metrics_msg += ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        return self.info(f"{metrics_msg}")

    def log_config(self, config: dict) -> None:
        current_rank = self._get_rank()
        if not self.rank_zero_only or current_rank == 0:
            config_str = json.dumps(config, indent=2, default=str)
            self.info(f"{config_str}")


# CSV logger
class CSVLogger(BaseLogger):
    """Per-run CSV logger in long format (metric,value,step).

    Writes metrics to `<results_dir>/<run_name>.csv` and config to
    `<results_dir>/<run_name>_config.json`. Only rank-zero processes write.
    """

    def __init__(
        self,
        console_logger: ConsoleFileLogger,
        results_dir: Path,
        run_name: str,
    ):
        self.console_logger = console_logger
        self.csv_path = results_dir / f"{run_name}.csv"
        self.config_path = results_dir / f"{run_name}_config.json"
        if self._should_log:
            results_dir.mkdir(parents=True, exist_ok=True)
            # Atomic create-or-skip so concurrent SLURM array tasks sharing the
            # same run_name don't all race to write the header row.
            try:
                with self.csv_path.open("x", newline="") as f:
                    csv.writer(f).writerow(["metric", "value", "step"])
            except FileExistsError:
                pass
            self.console_logger.info(f"[CSVLogger] Writing metrics to {self.csv_path}")

    @property
    def _should_log(self) -> bool:
        return int(os.environ.get("SLURM_PROCID", "0")) == 0

    def log(self, *args, **kwargs) -> None:
        pass

    def log_config(self, config: dict[str, Any]) -> None:
        if not self._should_log:
            return
        with self.config_path.open("w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._should_log:
            return
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            for name, value in metrics.items():
                writer.writerow([name, value, step if step is not None else ""])

    def end_run(self) -> None:
        pass


class MultiLogger:
    """High-level logger that combines multiple loggers"""

    def __init__(self, *loggers: BaseLogger) -> None:
        self.loggers = loggers

    def __getattr__(self, name: str) -> Any:
        """Forward method call to underlying loggers"""

        def wrapping_method(*args, **kwargs):
            for logger in self.loggers:
                if hasattr(logger, name):
                    getattr(logger, name)(*args, **kwargs)

        return wrapping_method

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_run()


def create_logger(
    name: str = __name__,
    rank_zero_only: bool = True,
    log_file: str | None = None,
    log_level: int = logging.INFO,
    results_dir: str | os.PathLike | None = None,
    run_name: str | None = None,
) -> MultiLogger:
    console_logger = ConsoleFileLogger(
        name=name, rank_zero_only=rank_zero_only, log_file=log_file, level=log_level
    )

    if results_dir is not None and run_name is not None:
        csv_logger = CSVLogger(
            console_logger=console_logger,
            results_dir=Path(results_dir),
            run_name=run_name,
        )
        return MultiLogger(console_logger, csv_logger)

    return MultiLogger(console_logger)
