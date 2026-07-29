import logging
import sys

def configure_logging(name: str) -> logging.Logger:
    """Configure root logging and return a named logger for a script.

    Installs a root handler (via ``logging.basicConfig``) that emits INFO-level
    records as ``<timestamp> <LEVEL> <name>: <message>`` to stdout. Calling this
    from more than one module is safe: ``basicConfig`` only configures the root
    handler on the first call and is a no-op afterwards, so the format is set once
    and shared across the pipeline.

    Args:
        name: Logger name, conventionally the calling script (e.g.
            ``"train_consensus"``); appears in each line and lets callers filter
            or set per-logger levels.

    Returns:
        The ``logging.Logger`` for `name`, ready to use.
    """
    logger = logging.getLogger(name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,          # default is stderr; send logs to stdout (SLURM .out)
    )
    return logger
