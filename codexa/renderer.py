"""Jinja2-based renderer for CODEXA.md files.

This module is responsible for:
  - Loading the bundled ``CODEXA.md.j2`` Jinja2 template (or a user-supplied
    override path).
  - Accepting a populated context dict (derived from ``DirContext`` and an
    LLM-generated ``ModuleSummary``) and rendering it to markdown.
  - Performing incremental skip logic by comparing a stored content hash
    against the current directory hash before (re)writing the file.
  - Writing the rendered markdown to ``<directory>/CODEXA.md``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Path to the bundled Jinja2 template, relative to this file.
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DEFAULT_TEMPLATE = "CODEXA.md.j2"


class RendererError(Exception):
    """Raised when the renderer encounters a fatal error."""


class Renderer:
    """Renders CODEXA.md files from Jinja2 templates.

    Args:
        template_path: Optional path to a custom Jinja2 template file.
            Defaults to the bundled ``codexa/templates/CODEXA.md.j2``.
    """

    def __init__(self, template_path: Optional[Path] = None) -> None:
        self._template_path = template_path
        self._env: Any = None  # Lazily initialised

    def _get_env(self) -> Any:
        """Return a lazily-initialised Jinja2 Environment."""
        if self._env is None:
            try:
                from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: PLC0415

                if self._template_path is not None:
                    template_dir = self._template_path.parent
                else:
                    template_dir = _TEMPLATE_DIR

                self._env = Environment(
                    loader=FileSystemLoader(str(template_dir)),
                    undefined=StrictUndefined,
                    autoescape=False,
                    keep_trailing_newline=True,
                )
            except ImportError as exc:
                raise RendererError(
                    "Jinja2 is required for rendering.  "
                    "Install it with: pip install jinja2"
                ) from exc
        return self._env

    def render(self, context: Dict[str, Any]) -> str:
        """Render the CODEXA.md template with the given *context*.

        Args:
            context: A dict containing all template variables.  Expected keys
                match the fields of ``DirContext`` and ``ModuleSummary``.

        Returns:
            Rendered markdown as a string.

        Raises:
            RendererError: If the template cannot be loaded or rendered.
        """
        env = self._get_env()
        template_name = (
            self._template_path.name
            if self._template_path is not None
            else _DEFAULT_TEMPLATE
        )
        try:
            template = env.get_template(template_name)
            return template.render(**context)
        except Exception as exc:
            raise RendererError(f"Template rendering failed: {exc}") from exc

    def write(
        self,
        directory: Path,
        context: Dict[str, Any],
        force: bool = False,
    ) -> bool:
        """Render and write ``CODEXA.md`` to *directory*.

        Skips writing if an existing CODEXA.md was produced from the same
        content hash (incremental mode), unless *force* is True.

        Args:
            directory: The directory in which to write ``CODEXA.md``.
            context: Template rendering context.
            force: If True, always overwrite even if hash matches.

        Returns:
            True if the file was written, False if skipped.

        Raises:
            RendererError: On template or I/O errors.
        """
        output_path = directory / "CODEXA.md"
        content_hash = context.get("content_hash", "")

        if not force and output_path.exists():
            stored_hash = _read_stored_hash(output_path)
            if stored_hash and stored_hash == content_hash:
                logger.debug(
                    "Skipping %s — content hash unchanged (%s).",
                    output_path,
                    content_hash[:12],
                )
                return False

        rendered = self.render(context)

        try:
            output_path.write_text(rendered, encoding="utf-8")
            logger.info("Written %s", output_path)
        except OSError as exc:
            raise RendererError(f"Cannot write {output_path}: {exc}") from exc

        return True


def _read_stored_hash(codexa_md_path: Path) -> Optional[str]:
    """Extract the content hash embedded in an existing CODEXA.md file.

    The hash is stored in an HTML comment on the first line::

        <!-- codexa-hash: <hex> -->

    Args:
        codexa_md_path: Path to an existing CODEXA.md file.

    Returns:
        The hex hash string, or None if not found or unreadable.
    """
    try:
        first_line = ""
        with open(codexa_md_path, "r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
    except OSError:
        return None

    prefix = "<!-- codexa-hash:"
    suffix = "-->"
    if first_line.startswith(prefix) and first_line.endswith(suffix):
        return first_line[len(prefix) : -len(suffix)].strip()
    return None


def compute_context_hash(context: Dict[str, Any]) -> str:
    """Produce a deterministic hash for a rendering context dict.

    Uses the string representation of the context dict via SHA-256 to
    detect changes between runs.

    Args:
        context: The template rendering context.

    Returns:
        A 64-character hex digest.
    """
    raw = str(sorted(context.items()))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
