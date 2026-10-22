#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal configuration loader shared by all scripts in this repository.

Resolution order (highest precedence first):
    1. environment variable            e.g. DOGCT_FEATURES_XLSX
    2. ./config.yaml                   (git-ignored; copy from config.example.yaml)
    3. ./config.example.yaml           (checked-in defaults)
    4. the `default` argument passed to get()

This keeps machine-specific absolute paths out of the published code while
leaving every script runnable without arguments once config.yaml exists.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_CACHE: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    """Read config.yaml, falling back to config.example.yaml."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    cfg: dict[str, Any] = {}
    for name in ("config.example.yaml", "config.yaml"):  # later wins
        path = _REPO_ROOT / name
        if not path.exists():
            continue
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to read config files. "
                "Install it with `pip install PyYAML`, or set the "
                "DOGCT_* environment variables instead."
            ) from exc
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        cfg.update(loaded)

    _CACHE = cfg
    return cfg


def get(key: str, default: Any = None) -> Any:
    """Return config value for `key`, honouring the DOGCT_<KEY> env var."""
    env_val = os.environ.get(f"DOGCT_{key.upper()}")
    if env_val is not None:
        return env_val
    return _load().get(key, default)


def get_path(key: str, default: str | None = None) -> Path:
    """Return a config value as a Path, resolved relative to the repo root."""
    raw = get(key, default)
    if raw is None:
        raise KeyError(
            f"Required path '{key}' is not configured. Set it in config.yaml "
            f"or export DOGCT_{key.upper()}."
        )
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path)


def require_path(key: str, default: str | None = None) -> Path:
    """Like get_path(), but fail immediately with a helpful message if missing."""
    path = get_path(key, default)
    if not path.exists():
        raise FileNotFoundError(
            f"Configured path for '{key}' does not exist: {path}\n"
            f"Edit config.yaml, or export DOGCT_{key.upper()}=/your/path.\n"
            f"To run on synthetic data instead, see examples/make_synthetic_example.py."
        )
    return path
