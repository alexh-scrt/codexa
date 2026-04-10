"""Configuration loading and validation for Codexa.

Loads ``.codexa.toml`` from the project root (or an explicit path) and
merges it with sensible defaults.  Validates each setting and raises
Descriptive errors for invalid values.

Configuration file example::

    [codexa]
    model = "gpt-4o-mini"
    api_key = "sk-..."
    base_url = "https://api.openai.com/v1"
    max_tokens = 1024
    max_depth = 5
    ignore = [".git", "node_modules", "__pycache__", "*.pyc"]
    template = ""  # Optional path to a custom .j2 template
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "gpt-4o-mini",
    "api_key": "",           # Falls back to OPENAI_API_KEY env var
    "base_url": "",          # Empty means use the default OpenAI endpoint
    "max_tokens": 1024,
    "max_depth": None,       # None means unlimited
    "ignore": [
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "*.egg-info",
    ],
    "template": "",          # Empty means use the bundled template
}

CONFIG_FILENAME = ".codexa.toml"


class ConfigError(Exception):
    """Raised when the configuration is missing or invalid."""


class CodexaConfig:
    """Validated, immutable configuration for a Codexa run.

    Attributes:
        model: LLM model identifier.
        api_key: API key for the LLM backend.
        base_url: Optional custom endpoint URL.
        max_tokens: Maximum tokens per LLM completion request.
        max_depth: Maximum directory recursion depth (None = unlimited).
        ignore: List of pathspec patterns to ignore.
        template: Path to a custom Jinja2 template, or empty string for default.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        max_depth: Optional[int],
        ignore: List[str],
        template: str,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.max_depth = max_depth
        self.ignore = ignore
        self.template = template

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CodexaConfig(model={self.model!r}, max_tokens={self.max_tokens}, "
            f"max_depth={self.max_depth}, ignore_count={len(self.ignore)})"
        )

    @property
    def effective_api_key(self) -> str:
        """Return the API key, preferring the env var over the config file."""
        env_key = os.environ.get("OPENAI_API_KEY", "")
        return env_key or self.api_key

    @property
    def effective_base_url(self) -> Optional[str]:
        """Return the base URL, or None if not set."""
        return self.base_url or os.environ.get("OPENAI_BASE_URL") or None


def _load_toml(path: Path) -> Dict[str, Any]:
    """Read and parse a TOML file, returning the parsed dict.

    Uses ``tomllib`` (Python ≥ 3.11) or the ``tomli`` back-port.

    Args:
        path: Path to the .toml file.

    Returns:
        Parsed TOML data as a dict.

    Raises:
        ConfigError: If the file cannot be read or parsed.
    """
    try:
        import sys  # noqa: PLC0415

        if sys.version_info >= (3, 11):
            import tomllib  # type: ignore[import]  # noqa: PLC0415
        else:
            import tomli as tomllib  # type: ignore[no-redef]  # noqa: PLC0415

        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except ImportError as exc:
        raise ConfigError(
            "TOML parsing requires 'tomli' on Python < 3.11.  "
            "Install it with: pip install tomli"
        ) from exc
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Failed to parse {path}: {exc}") from exc


def _merge_with_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the ``[codexa]`` section of a TOML dict with defaults.

    Args:
        raw: The top-level parsed TOML dict (may contain a ``codexa`` key).

    Returns:
        A flat dict with all known config keys resolved.
    """
    section: Dict[str, Any] = raw.get("codexa", {})
    merged: Dict[str, Any] = {**DEFAULT_CONFIG}
    for key, value in section.items():
        if key in merged:
            merged[key] = value
        else:
            logger.warning("Unknown config key 'codexa.%s' — ignoring.", key)
    return merged


def _validate(cfg: Dict[str, Any]) -> None:
    """Validate the merged config dict in-place.

    Raises:
        ConfigError: On any validation failure.
    """
    if not isinstance(cfg["model"], str) or not cfg["model"].strip():
        raise ConfigError("'model' must be a non-empty string.")

    if not isinstance(cfg["max_tokens"], int) or cfg["max_tokens"] < 64:
        raise ConfigError("'max_tokens' must be an integer >= 64.")

    if cfg["max_depth"] is not None:
        if not isinstance(cfg["max_depth"], int) or cfg["max_depth"] < 0:
            raise ConfigError("'max_depth' must be a non-negative integer or null.")

    if not isinstance(cfg["ignore"], list):
        raise ConfigError("'ignore' must be a list of strings.")
    for pattern in cfg["ignore"]:
        if not isinstance(pattern, str):
            raise ConfigError(
                f"Each entry in 'ignore' must be a string, got {type(pattern).__name__}."
            )

    if not isinstance(cfg["template"], str):
        raise ConfigError("'template' must be a string (path or empty).")

    if cfg["template"] and not Path(cfg["template"]).exists():
        raise ConfigError(
            f"Custom template not found: {cfg['template']}"
        )


def load_config(
    config_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> CodexaConfig:
    """Load and validate codexa configuration.

    Resolution order:
      1. Explicit *config_path* if provided.
      2. ``.codexa.toml`` in *root* (or ``Path(".")`` if *root* is None).
      3. Pure defaults if no config file exists.

    Args:
        config_path: Explicit path to a ``.codexa.toml`` file.
        root: Project root directory to search for ``.codexa.toml``.

    Returns:
        A validated :class:`CodexaConfig` instance.

    Raises:
        ConfigError: If the file is found but invalid.
    """
    if config_path is not None:
        raw = _load_toml(config_path)
    else:
        search_dir = (root or Path(".")).resolve()
        candidate = search_dir / CONFIG_FILENAME
        if candidate.exists():
            logger.debug("Loading config from %s", candidate)
            raw = _load_toml(candidate)
        else:
            logger.debug("No .codexa.toml found; using defaults.")
            raw = {}

    merged = _merge_with_defaults(raw)
    _validate(merged)

    return CodexaConfig(
        model=merged["model"].strip(),
        api_key=merged["api_key"],
        base_url=merged["base_url"],
        max_tokens=int(merged["max_tokens"]),
        max_depth=merged["max_depth"],
        ignore=list(merged["ignore"]),
        template=merged["template"],
    )
