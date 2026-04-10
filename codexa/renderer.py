"""Jinja2-based renderer for CODEXA.md files.

This module is responsible for:
  - Loading the bundled ``CODEXA.md.j2`` Jinja2 template (or a user-supplied
    override path).
  - Accepting a populated :class:`~codexa.models.DirContext` (with an optional
    :class:`~codexa.models.ModuleSummary`) and rendering it to markdown.
  - Performing incremental skip logic by comparing a stored content hash
    embedded in an existing CODEXA.md file against the current directory hash
    before (re)writing the file.
  - Writing the rendered markdown to ``<directory>/CODEXA.md``.

The hash is embedded in the first line of every generated file as an HTML
comment so it survives round-trips through editors:

    <!-- codexa-hash: <64-char-hex> -->

Design notes:
  - The Jinja2 :class:`~jinja2.Environment` is created lazily on the first
    call to :meth:`Renderer.render` so that the class can be instantiated
    without importing Jinja2 at module load time.
  - :class:`~jinja2.StrictUndefined` is used so that typos in template
    variable names surface as errors rather than silently rendering as empty
    strings.
  - The template receives the full :class:`~codexa.models.DirContext` dict
    plus a flattened set of convenience variables (see :func:`build_template_context`).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from codexa.models import DirContext, ModuleSummary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Directory containing the bundled Jinja2 template.
_TEMPLATE_DIR = Path(__file__).parent / "templates"

#: Filename of the default template within ``_TEMPLATE_DIR``.
_DEFAULT_TEMPLATE_NAME = "CODEXA.md.j2"

#: Prefix of the hash comment embedded in the first line of generated files.
_HASH_COMMENT_PREFIX = "<!-- codexa-hash:"

#: Suffix of the hash comment.
_HASH_COMMENT_SUFFIX = "-->"

#: Regex for extracting the hash from the first line.
_HASH_LINE_RE = re.compile(
    r"^<!--\s*codexa-hash:\s*([0-9a-f]{64})\s*-->$"
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RendererError(Exception):
    """Raised when the renderer encounters a fatal error.

    Attributes:
        message: Human-readable description of the problem.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # pragma: no cover
        return self.message


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def build_template_context(dir_context: DirContext) -> Dict[str, Any]:
    """Build a flat template context dict from a :class:`~codexa.models.DirContext`.

    Provides both the raw ``dir_context`` object and a set of pre-computed
    convenience variables so that the Jinja2 template can remain readable
    without complex filter chains.

    Args:
        dir_context: The populated directory context.  The ``summary`` field
            may be ``None`` if LLM summarization was skipped.

    Returns:
        A dict containing all variables available to the template:

        ``dir_context``
            The original :class:`~codexa.models.DirContext` instance.
        ``path``
            Absolute path string of the directory.
        ``dir_name``
            Bare directory name (last path component).
        ``content_hash``
            64-char hex hash of the directory's source files.
        ``files``
            List of :class:`~codexa.models.FileInfo` instances.
        ``file_count``
            Number of source files.
        ``subdirectories``
            Sorted list of immediate child directory names.
        ``all_functions``
            Deduplicated list of function names across all files.
        ``all_classes``
            Deduplicated list of class names across all files.
        ``all_imports``
            Deduplicated list of imported module names across all files.
        ``summary``
            :class:`~codexa.models.ModuleSummary` or ``None``.
        ``overview``
            Overview string (empty string if no summary).
        ``key_symbols``
            List of key symbol strings (empty list if no summary).
        ``patterns``
            List of pattern strings (empty list if no summary).
        ``tribal_knowledge``
            List of tribal knowledge strings (empty list if no summary).
        ``has_summary``
            Boolean: ``True`` when a non-empty summary is attached.
        ``generated_at``
            ISO-8601 UTC timestamp string for the generation time.
    """
    summary: Optional[ModuleSummary] = dir_context.summary
    has_summary = summary is not None and summary.has_content

    return {
        # Raw object — available for advanced template use
        "dir_context": dir_context,
        # Directory basics
        "path": str(dir_context.path),
        "dir_name": dir_context.name,
        "content_hash": dir_context.content_hash,
        # File lists
        "files": dir_context.files,
        "file_count": dir_context.file_count,
        "subdirectories": list(dir_context.subdirectories),
        # Aggregate symbol lists
        "all_functions": dir_context.all_functions,
        "all_classes": dir_context.all_classes,
        "all_imports": dir_context.all_imports,
        # LLM summary fields (safely defaulted)
        "summary": summary,
        "overview": summary.overview if summary else "",
        "key_symbols": list(summary.key_symbols) if summary else [],
        "patterns": list(summary.patterns) if summary else [],
        "tribal_knowledge": list(summary.tribal_knowledge) if summary else [],
        "has_summary": has_summary,
        # Metadata
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------


def read_stored_hash(codexa_md_path: Path) -> Optional[str]:
    """Extract the content hash embedded in an existing CODEXA.md file.

    The hash is stored in an HTML comment on the **first line** of the file::

        <!-- codexa-hash: <64-hex-chars> -->

    Args:
        codexa_md_path: Path to an existing ``CODEXA.md`` file.

    Returns:
        The 64-character hex hash string, or ``None`` if the first line does
        not match the expected format or the file cannot be read.
    """
    try:
        with open(codexa_md_path, "r", encoding="utf-8") as fh:
            first_line = fh.readline().rstrip("\n\r")
    except OSError:
        return None

    match = _HASH_LINE_RE.match(first_line.strip())
    if match:
        return match.group(1)
    return None


def compute_context_hash(context: Dict[str, Any]) -> str:
    """Produce a deterministic SHA-256 hash for a rendering context dict.

    Uses the ``content_hash`` key from *context* when available (this is the
    hash computed by the analyzer from file contents).  Falls back to hashing
    the string representation of the entire context dict when that key is
    absent.

    Args:
        context: The template rendering context as produced by
            :func:`build_template_context`.

    Returns:
        A 64-character lowercase hex digest.
    """
    if "content_hash" in context and context["content_hash"]:
        return str(context["content_hash"])
    raw = str(sorted((k, str(v)) for k, v in context.items() if isinstance(v, (str, int, float, bool, type(None)))))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class Renderer:
    """Renders CODEXA.md files from a Jinja2 template.

    Encapsulates the Jinja2 environment, template loading, rendering, and
    incremental write logic.  Instances are safe to reuse across multiple
    directories in a single run.

    Args:
        template_path: Optional path to a custom Jinja2 template file.
            When ``None`` (the default), the bundled
            ``codexa/templates/CODEXA.md.j2`` template is used.
    """

    def __init__(self, template_path: Optional[Path] = None) -> None:
        self._custom_template_path = template_path
        self._env: Any = None  # Lazily initialised Jinja2 Environment

    # ------------------------------------------------------------------
    # Lazy Jinja2 environment
    # ------------------------------------------------------------------

    def _get_env(self) -> Any:
        """Return a lazily-initialised Jinja2 :class:`~jinja2.Environment`.

        Raises:
            RendererError: If Jinja2 is not installed or the template
                directory cannot be located.
        """
        if self._env is not None:
            return self._env

        try:
            from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: PLC0415
        except ImportError as exc:
            raise RendererError(
                "Jinja2 is required for rendering. "
                "Install it with: pip install jinja2"
            ) from exc

        if self._custom_template_path is not None:
            template_dir = self._custom_template_path.parent
        else:
            template_dir = _TEMPLATE_DIR

        if not template_dir.is_dir():
            raise RendererError(
                f"Template directory not found: {template_dir}"
            )

        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
            trim_blocks=False,
            lstrip_blocks=False,
        )
        return self._env

    # ------------------------------------------------------------------
    # Template name resolution
    # ------------------------------------------------------------------

    @property
    def _template_name(self) -> str:
        """Return the template filename to load from the Jinja2 environment."""
        if self._custom_template_path is not None:
            return self._custom_template_path.name
        return _DEFAULT_TEMPLATE_NAME

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------

    def render(self, context: Dict[str, Any]) -> str:
        """Render the CODEXA.md template with the given *context*.

        The rendered string always starts with the hash comment line so that
        :func:`read_stored_hash` can detect unchanged files on the next run.

        Args:
            context: A template rendering context, typically produced by
                :func:`build_template_context`.  Must contain a
                ``content_hash`` key.

        Returns:
            The fully rendered markdown string, beginning with the hash
            comment line.

        Raises:
            RendererError: If the template cannot be loaded or rendered.
        """
        env = self._get_env()
        try:
            template = env.get_template(self._template_name)
        except Exception as exc:
            raise RendererError(
                f"Cannot load template '{self._template_name}': {exc}"
            ) from exc

        # Inject the hash comment as a template variable so the template
        # can place it on the first line.
        content_hash = compute_context_hash(context)
        context = dict(context)  # shallow copy to avoid mutating the caller's dict
        context["hash_comment"] = (
            f"{_HASH_COMMENT_PREFIX} {content_hash} {_HASH_COMMENT_SUFFIX}"
        )
        context["content_hash"] = content_hash

        try:
            rendered = template.render(**context)
        except Exception as exc:
            raise RendererError(f"Template rendering failed: {exc}") from exc

        return rendered

    def render_dir_context(self, dir_context: DirContext) -> str:
        """Convenience method: build context and render in one call.

        Args:
            dir_context: The populated directory context to render.

        Returns:
            Rendered markdown string.

        Raises:
            RendererError: If rendering fails.
        """
        context = build_template_context(dir_context)
        return self.render(context)

    # ------------------------------------------------------------------
    # Write with incremental skip logic
    # ------------------------------------------------------------------

    def write(
        self,
        directory: Path,
        context: Dict[str, Any],
        force: bool = False,
    ) -> bool:
        """Render and write ``CODEXA.md`` to *directory*.

        Compares the ``content_hash`` in *context* against the hash stored
        in any existing ``CODEXA.md``.  Skips writing when they match
        (incremental mode), unless *force* is ``True``.

        Args:
            directory: The directory in which to write ``CODEXA.md``.
            context: Template rendering context (from
                :func:`build_template_context`).
            force: When ``True``, always overwrite even if the hash is
                unchanged.

        Returns:
            ``True`` if the file was written, ``False`` if it was skipped
            because the content hash matched.

        Raises:
            RendererError: On template rendering or file I/O errors.
        """
        output_path = directory / "CODEXA.md"
        content_hash = compute_context_hash(context)

        if not force and output_path.exists():
            stored = read_stored_hash(output_path)
            if stored and stored == content_hash:
                logger.debug(
                    "Skipping %s — hash unchanged (%s…).",
                    output_path,
                    content_hash[:12],
                )
                return False

        rendered = self.render(context)

        try:
            output_path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            raise RendererError(f"Cannot write {output_path}: {exc}") from exc

        logger.info("Written %s", output_path)
        return True

    def write_dir_context(
        self,
        dir_context: DirContext,
        force: bool = False,
    ) -> bool:
        """Convenience method: build context and write CODEXA.md in one call.

        Args:
            dir_context: The populated directory context to render and write.
            force: When ``True``, always overwrite.

        Returns:
            ``True`` if the file was written, ``False`` if skipped.

        Raises:
            RendererError: On rendering or I/O errors.
        """
        context = build_template_context(dir_context)
        return self.write(dir_context.path, context, force=force)
