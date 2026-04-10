"""Configuration loading and validation for Codexa.

Loads ``.codexa.toml`` from the project root (or an explicit path) and
merges it with sensible defaults.  Validates each setting and raises
descriptive errors for invalid values.

Configuration file example::

    [codexa]
    model = "gpt-4o-mini"
    api_key = "sk-..."
    base_url = "https://api.openai.com/v1"
    max_tokens = 1024
    max_depth = 5
    ignore = [".git", "node_modules", "__pycache__", "*.pyc"]
    template = ""  # Optional path to a custom .j2 template

Environment variable overrides::

    OPENAI_API_KEY   — overrides the ``api_key`` config value
    OPENAI_BASE_URL  — overrides the ``base_url`` config value
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

#: Default configuration values applied when a key is absent from the file.
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
        "CODEXA.md",
    ],
    "template": "",          # Empty means use the bundled template
}

#: The expected filename for the project-level config file.
CONFIG_FILENAME = ".codexa.toml"

#: Minimum allowed value for ``max_tokens``.
_MIN_MAX_TOKENS = 64

#: Maximum allowed value for ``max_tokens`` (hard safety cap).
_MAX_MAX_TOKENS = 128_000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when the configuration is missing, malformed, or invalid.

    Attributes:
        message: Human-readable description of the problem.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # pragma: no cover
        return self.message


# ---------------------------------------------------------------------------
# CodexaConfig
# ---------------------------------------------------------------------------


class CodexaConfig:
    """Validated, immutable-ish configuration for a Codexa run.

    Construct instances via :func:`load_config` rather than directly; the
    constructor does *not* re-validate its arguments.

    Attributes:
        model: LLM model identifier (e.g. ``"gpt-4o-mini"``).
        api_key: Raw API key from the config file (may be empty).
        base_url: Custom endpoint URL, or empty string for the default.
        max_tokens: Maximum tokens per LLM completion request.
        max_depth: Maximum directory recursion depth; ``None`` = unlimited.
        ignore: List of pathspec-compatible glob patterns to ignore.
        template: Absolute/relative path to a custom Jinja2 template, or
            empty string to use the bundled template.
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
        self.ignore = list(ignore)
        self.template = template

    # ------------------------------------------------------------------
    # Environment-aware properties
    # ------------------------------------------------------------------

    @property
    def effective_api_key(self) -> str:
        """Return the API key, preferring the ``OPENAI_API_KEY`` env var.

        Resolution order:
          1. ``OPENAI_API_KEY`` environment variable (if non-empty).
          2. ``api_key`` field from the config file.

        Returns:
            The resolved API key string (may be empty if neither is set).
        """
        env_key = os.environ.get("OPENAI_API_KEY", "").strip()
        return env_key if env_key else self.api_key

    @property
    def effective_base_url(self) -> Optional[str]:
        """Return the effective base URL, or ``None`` when not configured.

        Resolution order:
          1. ``base_url`` field from the config file (if non-empty).
          2. ``OPENAI_BASE_URL`` environment variable (if non-empty).
          3. ``None`` — the OpenAI SDK default endpoint will be used.

        Returns:
            URL string or ``None``.
        """
        if self.base_url and self.base_url.strip():
            return self.base_url.strip()
        env_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        return env_url if env_url else None

    @property
    def template_path(self) -> Optional[Path]:
        """Return the custom template path as a :class:`~pathlib.Path`, or ``None``.

        Returns:
            A :class:`~pathlib.Path` if ``template`` is non-empty, else ``None``.
        """
        return Path(self.template) if self.template else None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CodexaConfig("
            f"model={self.model!r}, "
            f"max_tokens={self.max_tokens}, "
            f"max_depth={self.max_depth!r}, "
            f"ignore_count={len(self.ignore)}, "
            f"template={self.template!r}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodexaConfig):
            return NotImplemented
        return (
            self.model == other.model
            and self.api_key == other.api_key
            and self.base_url == other.base_url
            and self.max_tokens == other.max_tokens
            and self.max_depth == other.max_depth
            and self.ignore == other.ignore
            and self.template == other.template
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict snapshot of this configuration.

        The ``api_key`` is intentionally included so that the dict can be
        used for serialization; callers that log this dict should redact
        the key themselves.

        Returns:
            A dict with one key per config attribute.
        """
        return {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "max_depth": self.max_depth,
            "ignore": list(self.ignore),
            "template": self.template,
        }


# ---------------------------------------------------------------------------
# TOML loading helpers
# ---------------------------------------------------------------------------


def _load_toml(path: Path) -> Dict[str, Any]:
    """Read and parse a TOML file, returning the top-level dict.

    Uses ``tomllib`` (Python ≥ 3.11) or the ``tomli`` back-port for older
    interpreters.

    Args:
        path: Path to the ``.toml`` file to read.

    Returns:
        The top-level parsed TOML data as a plain dict.

    Raises:
        ConfigError: If the file cannot be found, read, or parsed.
    """
    try:
        if sys.version_info >= (3, 11):
            import tomllib  # type: ignore[import]
        else:
            import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as exc:
        raise ConfigError(
            "TOML parsing requires the 'tomli' package on Python < 3.11.  "
            "Install it with: pip install tomli"
        ) from exc

    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except PermissionError as exc:
        raise ConfigError(f"Permission denied reading config file: {path}") from exc
    except Exception as exc:  # covers tomllib.TOMLDecodeError and similar
        raise ConfigError(f"Failed to parse config file {path}: {exc}") from exc


def _merge_with_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the ``[codexa]`` section and merge it with :data:`DEFAULT_CONFIG`.

    Keys present in the TOML file override the defaults; unknown keys produce
    a warning and are discarded so they don't pollute the validated config.

    Args:
        raw: The top-level parsed TOML dict (may or may not contain a
            ``"codexa"`` key).

    Returns:
        A flat dict with every key from :data:`DEFAULT_CONFIG` resolved to
        either the user-supplied value or the default.
    """
    section: Dict[str, Any] = raw.get("codexa", {})
    if not isinstance(section, dict):
        raise ConfigError(
            "Expected [codexa] to be a TOML table, got "
            f"{type(section).__name__}."
        )

    merged: Dict[str, Any] = {**DEFAULT_CONFIG}
    for key, value in section.items():
        if key in merged:
            merged[key] = value
        else:
            logger.warning(
                "Unknown config key 'codexa.%s' — ignoring.", key
            )
    return merged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(cfg: Dict[str, Any]) -> None:
    """Validate the merged configuration dict, raising on the first error found.

    Checks performed:
      * ``model`` — non-empty string.
      * ``api_key`` — string (may be empty; resolved later via env var).
      * ``base_url`` — string.
      * ``max_tokens`` — integer in [``_MIN_MAX_TOKENS``, ``_MAX_MAX_TOKENS``].
      * ``max_depth`` — ``None`` or a non-negative integer.
      * ``ignore`` — list of strings.
      * ``template`` — string; if non-empty the path must exist on disk.

    Args:
        cfg: The merged configuration dict to validate.

    Raises:
        ConfigError: Describing the first validation failure encountered.
    """
    # --- model ---
    if not isinstance(cfg["model"], str) or not cfg["model"].strip():
        raise ConfigError(
            "'codexa.model' must be a non-empty string, "
            f"got {cfg['model']!r}."
        )

    # --- api_key ---
    if not isinstance(cfg["api_key"], str):
        raise ConfigError(
            "'codexa.api_key' must be a string, "
            f"got {type(cfg['api_key']).__name__}."
        )

    # --- base_url ---
    if not isinstance(cfg["base_url"], str):
        raise ConfigError(
            "'codexa.base_url' must be a string, "
            f"got {type(cfg['base_url']).__name__}."
        )

    # --- max_tokens ---
    if not isinstance(cfg["max_tokens"], int) or isinstance(cfg["max_tokens"], bool):
        raise ConfigError(
            "'codexa.max_tokens' must be an integer, "
            f"got {type(cfg['max_tokens']).__name__}."
        )
    if cfg["max_tokens"] < _MIN_MAX_TOKENS:
        raise ConfigError(
            f"'codexa.max_tokens' must be >= {_MIN_MAX_TOKENS}, "
            f"got {cfg['max_tokens']}."
        )
    if cfg["max_tokens"] > _MAX_MAX_TOKENS:
        raise ConfigError(
            f"'codexa.max_tokens' must be <= {_MAX_MAX_TOKENS}, "
            f"got {cfg['max_tokens']}."
        )

    # --- max_depth ---
    if cfg["max_depth"] is not None:
        if isinstance(cfg["max_depth"], bool) or not isinstance(cfg["max_depth"], int):
            raise ConfigError(
                "'codexa.max_depth' must be a non-negative integer or null/None, "
                f"got {type(cfg['max_depth']).__name__}."
            )
        if cfg["max_depth"] < 0:
            raise ConfigError(
                "'codexa.max_depth' must be >= 0, "
                f"got {cfg['max_depth']}."
            )

    # --- ignore ---
    if not isinstance(cfg["ignore"], list):
        raise ConfigError(
            "'codexa.ignore' must be a list of strings, "
            f"got {type(cfg['ignore']).__name__}."
        )
    for idx, pattern in enumerate(cfg["ignore"]):
        if not isinstance(pattern, str):
            raise ConfigError(
                f"Each entry in 'codexa.ignore' must be a string; "
                f"entry {idx} has type {type(pattern).__name__}."
            )

    # --- template ---
    if not isinstance(cfg["template"], str):
        raise ConfigError(
            "'codexa.template' must be a string (path or empty), "
            f"got {type(cfg['template']).__name__}."
        )
    if cfg["template"].strip():
        template_path = Path(cfg["template"].strip())
        if not template_path.exists():
            raise ConfigError(
                f"Custom template file not found: {template_path}"
            )
        if not template_path.is_file():
            raise ConfigError(
                f"'codexa.template' must point to a file, not a directory: "
                f"{template_path}"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    config_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> "CodexaConfig":
    """Load, merge, validate, and return a :class:`CodexaConfig`.

    Resolution order for the config file:

      1. *config_path* — if explicitly supplied, load only that file.
      2. ``<root>/.codexa.toml`` — auto-discovered if *root* is given or
         defaults to ``Path(".")``.  If the file does not exist here,
         fall through to defaults.
      3. Pure defaults — no config file is required; all keys have safe
         default values.

    Args:
        config_path: Explicit path to a ``.codexa.toml`` file.  When
            provided, the file *must* exist (a :class:`ConfigError` is
            raised if it does not).
        root: Project root directory used for auto-discovery.  Defaults to
            the current working directory if *None*.

    Returns:
        A fully validated :class:`CodexaConfig` instance.

    Raises:
        ConfigError: If an explicit *config_path* is not found, if the
            TOML cannot be parsed, or if any value fails validation.
    """
    raw: Dict[str, Any] = {}

    if config_path is not None:
        # Explicit path — must exist.
        logger.debug("Loading config from explicit path: %s", config_path)
        raw = _load_toml(config_path)
    else:
        search_dir = (root if root is not None else Path(".")).resolve()
        candidate = search_dir / CONFIG_FILENAME
        if candidate.exists():
            logger.debug("Auto-discovered config at %s", candidate)
            raw = _load_toml(candidate)
        else:
            logger.debug(
                "No %s found in %s; using built-in defaults.",
                CONFIG_FILENAME,
                search_dir,
            )

    merged = _merge_with_defaults(raw)
    _validate(merged)

    return CodexaConfig(
        model=merged["model"].strip(),
        api_key=str(merged["api_key"]),
        base_url=str(merged["base_url"]),
        max_tokens=int(merged["max_tokens"]),
        max_depth=merged["max_depth"],
        ignore=list(merged["ignore"]),
        template=str(merged["template"]).strip(),
    )


def build_ignore_spec(ignore_patterns: List[str]) -> Any:
    """Build a :class:`pathspec.PathSpec` from a list of glob patterns.

    This convenience function wraps ``pathspec.PathSpec.from_lines`` so that
    the rest of the codebase can remain decoupled from the ``pathspec`` API.

    Args:
        ignore_patterns: A list of gitignore-style glob pattern strings.

    Returns:
        A ``pathspec.PathSpec`` instance that can be used to test file paths.

    Raises:
        ConfigError: If the ``pathspec`` package is not available.
    """
    try:
        import pathspec  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "The 'pathspec' package is required for ignore pattern matching.  "
            "Install it with: pip install pathspec"
        ) from exc

    return pathspec.PathSpec.from_lines("gitwildmatch", ignore_patterns)
