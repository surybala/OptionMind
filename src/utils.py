import json
import logging
import os


def load_config(path="config.json"):
    with open(path, 'r') as f:
        return json.load(f)


def get_logger(
    name="optionwheel",
    level=logging.INFO,
    log_file=None,
    max_bytes=10 * 1024 * 1024,  # 10 MB per file
    backup_count=5,               # keep 5 rotated files
):
    """Return (or configure) the named logger.

    When *log_file* is given, a ``RotatingFileHandler`` is attached so the
    process writes to a size-capped file instead of (or in addition to) stdout.
    Safe to call multiple times with the same name — handlers are not duplicated.
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if log_file:
        from logging.handlers import RotatingFileHandler
        log_file = os.path.abspath(log_file)
        # Avoid adding a duplicate handler on repeated calls (e.g. tests)
        if not any(
            isinstance(h, RotatingFileHandler) and h.baseFilename == log_file
            for h in logger.handlers
        ):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(fh)

    return logger
