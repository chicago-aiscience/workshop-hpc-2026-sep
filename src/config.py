#!/usr/bin/env python3
"""Shared experiment configuration for the learned-consensus pipeline.

One place that defines every tunable "knob" (model + training hyperparameters and
data selection), loaded from a YAML experiment file with CLI-flag overrides.
Committing the YAML makes each run reproducible and diffable in Git; the CLI
overrides let you tweak a single value for a one-off without editing the file.

Precedence (lowest to highest):  DEFAULTS  <  YAML config file  <  explicit CLI flag.

    from config import load_config, apply_overrides
    cfg = load_config(args.config)                       # DEFAULTS <- YAML
    apply_overrides(cfg, [("train.lr", args.lr)])        # <- CLI (when not None)
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

# Canonical defaults -- the single source of truth for every knob's fallback value.
# A YAML config overrides these; an explicit CLI flag overrides the config.
DEFAULTS: dict = {
    "data": {
        "basins": ["2322", "2326"],     # SWORD basin prefixes to include
    },
    "model": {
        "hidden": 64,                    # width of the MLP's two hidden layers
    },
    "train": {
        "epochs": 50,
        "batch_size": 256,
        "lr": 1e-3,
        "checkpoint_interval": 50,       # steps between rolling checkpoints (0 disables)
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (in place); nested dicts are merged."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None) -> dict:
    """Return `DEFAULTS` with an optional YAML experiment file merged on top.

    Args:
        path: Path to a YAML experiment config, or None to use `DEFAULTS` alone.

    Returns:
        A fresh, fully-populated config dict (a deep copy -- never mutates DEFAULTS).
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        _deep_merge(cfg, user)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Write the fully-resolved `cfg` to `path` as YAML.

    Dropped into a run's output directory so each run records exactly the knobs it
    used -- reproducible and diff-able next to its results (Lesson 2: track configs).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def apply_overrides(cfg: dict, overrides: list[tuple[str, object]]) -> dict:
    """Apply CLI overrides in place; a None value means "flag not passed, keep config".

    Args:
        cfg: The config dict from `load_config`.
        overrides: `(dotted_key, value)` pairs, e.g. `("train.lr", args.lr)`. The
            dotted key addresses a nested entry; the value is set only when not None.

    Returns:
        The same `cfg`, mutated.
    """
    for dotted, value in overrides:
        if value is None:
            continue
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg
