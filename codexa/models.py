"""Data models for Codexa's analysis and summarization pipeline.

Defines three core dataclasses that serve as the shared data contract
across the analyzer, LLM client, and renderer:

  - :class:`FileInfo`: Metadata extracted from a single source file.
  - :class:`ModuleSummary`: LLM-generated narrative for a directory.
  - :class:`DirContext`: Aggregate context for an entire directory,
    combining FileInfo instances with an optional ModuleSummary.

All dataclasses expose ``to_dict()`` and ``from_dict()`` helpers so they
can be serialised to/from plain Python dicts for JSON logging, caching,
and prompt construction without introducing a heavy serialization library.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# FileInfo
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FileInfo:
    """Metadata extracted from a single source file.

    Attributes:
        path: Absolute path to the file on disk.
        relative_path: Path relative to the containing directory.  Set to
            ``None`` when the containing directory is not yet known.
        size_bytes: File size in bytes at analysis time.
        content_hash: SHA-256 hex digest of the raw file bytes, used for
            incremental-regeneration comparisons.
        source: Raw text contents of the file.  May be ``None`` when the
            file could not be decoded (e.g. binary or unknown encoding).
        module_docstring: The module-level docstring extracted by the AST
            parser, or ``None`` if the file has no module docstring.
        functions: Names of all top-level (module-scope) function
            definitions found in the file, in source order, deduplicated.
        classes: Names of all top-level class definitions, in source order,
            deduplicated.
        imports: Imported module names (both ``import X`` and
            ``from X import …`` forms), in source order, deduplicated.
    """

    path: Path
    relative_path: Optional[Path]
    size_bytes: int
    content_hash: str
    source: Optional[str]
    module_docstring: Optional[str]
    functions: List[str]
    classes: List[str]
    imports: List[str]

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the bare filename (e.g. ``'models.py'``)."""
        return self.path.name

    @property
    def is_empty(self) -> bool:
        """Return ``True`` when the file has no symbols and no docstring."""
        return (
            not self.functions
            and not self.classes
            and self.module_docstring is None
        )

    @property
    def symbol_count(self) -> int:
        """Total number of top-level functions plus classes."""
        return len(self.functions) + len(self.classes)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this instance.

        All :class:`~pathlib.Path` fields are converted to strings so the
        result can be passed directly to ``json.dumps``.

        Returns:
            A plain dict with the same field names as the dataclass.
        """
        return {
            "path": str(self.path),
            "relative_path": (
                str(self.relative_path) if self.relative_path is not None else None
            ),
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
            "source": self.source,
            "module_docstring": self.module_docstring,
            "functions": list(self.functions),
            "classes": list(self.classes),
            "imports": list(self.imports),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileInfo":
        """Reconstruct a :class:`FileInfo` from a dict produced by :meth:`to_dict`.

        Missing keys fall back to safe defaults so that partial dicts
        (e.g. from older cache files) are handled gracefully.

        Args:
            data: A mapping with the same keys as returned by :meth:`to_dict`.

        Returns:
            A new :class:`FileInfo` instance.

        Raises:
            KeyError: If the mandatory ``'path'`` key is absent.
        """
        raw_rel = data.get("relative_path")
        return cls(
            path=Path(data["path"]),
            relative_path=Path(raw_rel) if raw_rel is not None else None,
            size_bytes=int(data.get("size_bytes", 0)),
            content_hash=str(data.get("content_hash", "")),
            source=data.get("source"),
            module_docstring=data.get("module_docstring"),
            functions=list(data.get("functions", [])),
            classes=list(data.get("classes", [])),
            imports=list(data.get("imports", [])),
        )

    @classmethod
    def empty(cls, path: Path, relative_path: Optional[Path] = None) -> "FileInfo":
        """Create a blank :class:`FileInfo` for *path* with zeroed-out fields.

        Useful as a placeholder when a file cannot be analysed.

        Args:
            path: Absolute path to the file.
            relative_path: Optional relative path within its directory.

        Returns:
            A :class:`FileInfo` with empty metadata fields.
        """
        return cls(
            path=path,
            relative_path=relative_path,
            size_bytes=0,
            content_hash="",
            source=None,
            module_docstring=None,
            functions=[],
            classes=[],
            imports=[],
        )


# ---------------------------------------------------------------------------
# ModuleSummary
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModuleSummary:
    """LLM-generated narrative summary for a single directory / module.

    This dataclass is populated by the LLM client after it receives a
    :class:`DirContext` and returns structured JSON.  All fields default to
    empty values so that a summary can be constructed incrementally.

    Attributes:
        overview: A 2–4 sentence plain-English description of the module's
            purpose and responsibilities.
        key_symbols: The most important function and class names that a new
            developer should know about first.
        patterns: Non-obvious design patterns, idioms, or gotchas that are
            not immediately apparent from reading the code.
        tribal_knowledge: Contextual hints, historical decisions, or
            "why is it done this way?" explanations for new contributors.
    """

    overview: str
    key_symbols: List[str]
    patterns: List[str]
    tribal_knowledge: List[str]

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def has_content(self) -> bool:
        """Return ``True`` when at least one field has non-empty content."""
        return bool(
            self.overview
            or self.key_symbols
            or self.patterns
            or self.tribal_knowledge
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Returns:
            A plain dict with keys ``overview``, ``key_symbols``,
            ``patterns``, and ``tribal_knowledge``.
        """
        return {
            "overview": self.overview,
            "key_symbols": list(self.key_symbols),
            "patterns": list(self.patterns),
            "tribal_knowledge": list(self.tribal_knowledge),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleSummary":
        """Construct a :class:`ModuleSummary` from a plain dict.

        Unknown keys are silently ignored; missing keys fall back to
        empty defaults.

        Args:
            data: Dict with keys matching :class:`ModuleSummary` attributes.

        Returns:
            A new :class:`ModuleSummary` instance.
        """
        return cls(
            overview=str(data.get("overview", "")),
            key_symbols=_coerce_str_list(data.get("key_symbols", [])),
            patterns=_coerce_str_list(data.get("patterns", [])),
            tribal_knowledge=_coerce_str_list(data.get("tribal_knowledge", [])),
        )

    @classmethod
    def empty(cls) -> "ModuleSummary":
        """Return a blank :class:`ModuleSummary` used as a placeholder.

        Returns:
            A :class:`ModuleSummary` with all fields empty.
        """
        return cls(
            overview="",
            key_symbols=[],
            patterns=[],
            tribal_knowledge=[],
        )


# ---------------------------------------------------------------------------
# DirContext
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DirContext:
    """Aggregate context for a single directory in the codebase.

    :class:`DirContext` is the central data structure in Codexa.  It is
    produced by the analyzer, enriched by the LLM client, and consumed by
    the renderer to produce a ``CODEXA.md`` file.

    Attributes:
        path: Absolute path to the directory.
        files: :class:`FileInfo` instances for each source file directly
            within this directory (not recursed into sub-directories).
        subdirectories: Names of the immediate child directories that are
            not excluded by the ignore spec, in sorted order.
        content_hash: A combined SHA-256 digest computed from the individual
            :attr:`FileInfo.content_hash` values of all *files*.  Used by
            the renderer to detect unchanged directories and skip
            regeneration.
        summary: The :class:`ModuleSummary` generated by the LLM client for
            this directory.  ``None`` until the LLM step has been executed.
    """

    path: Path
    files: List[FileInfo]
    subdirectories: List[str]
    content_hash: str
    summary: Optional[ModuleSummary] = None

    # ------------------------------------------------------------------
    # Derived / aggregate properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the bare directory name (last path component)."""
        return self.path.name

    @property
    def file_count(self) -> int:
        """Number of source files directly inside this directory."""
        return len(self.files)

    @property
    def all_functions(self) -> List[str]:
        """Deduplicated list of every function name across all files.

        Preserves first-occurrence order.

        Returns:
            A list of function name strings.
        """
        seen: List[str] = []
        for file_info in self.files:
            for fn in file_info.functions:
                if fn not in seen:
                    seen.append(fn)
        return seen

    @property
    def all_classes(self) -> List[str]:
        """Deduplicated list of every class name across all files.

        Preserves first-occurrence order.

        Returns:
            A list of class name strings.
        """
        seen: List[str] = []
        for file_info in self.files:
            for cls_name in file_info.classes:
                if cls_name not in seen:
                    seen.append(cls_name)
        return seen

    @property
    def all_imports(self) -> List[str]:
        """Deduplicated list of every imported module name across all files.

        Preserves first-occurrence order.

        Returns:
            A list of module name strings.
        """
        seen: List[str] = []
        for file_info in self.files:
            for imp in file_info.imports:
                if imp not in seen:
                    seen.append(imp)
        return seen

    @property
    def total_size_bytes(self) -> int:
        """Sum of :attr:`FileInfo.size_bytes` across all files."""
        return sum(f.size_bytes for f in self.files)

    @property
    def is_empty(self) -> bool:
        """Return ``True`` when this directory contains no source files."""
        return len(self.files) == 0

    # ------------------------------------------------------------------
    # Hash computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(files: List[FileInfo]) -> str:
        """Compute a combined SHA-256 digest from a list of :class:`FileInfo` objects.

        The hash is derived from the concatenated :attr:`FileInfo.content_hash`
        values (joined by ``'|'``) so that any change to any file in the
        directory produces a different hash.

        Args:
            files: The list of :class:`FileInfo` objects to include.

        Returns:
            A 64-character lowercase hex digest string.  Returns the hash of
            an empty string when *files* is empty.
        """
        raw = "|".join(f.content_hash for f in files)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def refresh_hash(self) -> None:
        """Recompute and update :attr:`content_hash` from the current files list.

        Call this after modifying :attr:`files` to keep the hash consistent.
        """
        self.content_hash = DirContext.compute_hash(self.files)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this instance.

        The ``summary`` field is ``None`` in the output dict when no
        :class:`ModuleSummary` has been attached yet.

        Returns:
            A plain dict with keys matching the dataclass fields.
        """
        return {
            "path": str(self.path),
            "files": [f.to_dict() for f in self.files],
            "subdirectories": list(self.subdirectories),
            "content_hash": self.content_hash,
            "summary": self.summary.to_dict() if self.summary is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DirContext":
        """Reconstruct a :class:`DirContext` from a dict produced by :meth:`to_dict`.

        Args:
            data: A mapping with the same keys as returned by :meth:`to_dict`.

        Returns:
            A new :class:`DirContext` instance.

        Raises:
            KeyError: If the mandatory ``'path'`` key is absent.
        """
        summary_data: Optional[Dict[str, Any]] = data.get("summary")
        return cls(
            path=Path(data["path"]),
            files=[FileInfo.from_dict(f) for f in data.get("files", [])],
            subdirectories=list(data.get("subdirectories", [])),
            content_hash=str(data.get("content_hash", "")),
            summary=(
                ModuleSummary.from_dict(summary_data)
                if summary_data is not None
                else None
            ),
        )

    @classmethod
    def empty(cls, path: Path) -> "DirContext":
        """Create a blank :class:`DirContext` for *path* with no files or summary.

        Args:
            path: Absolute path to the directory.

        Returns:
            A :class:`DirContext` with empty ``files``, ``subdirectories``,
            a zeroed hash, and no summary.
        """
        return cls(
            path=path,
            files=[],
            subdirectories=[],
            content_hash=DirContext.compute_hash([]),
            summary=None,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _coerce_str_list(value: Any) -> List[str]:
    """Coerce *value* to a list of strings, ignoring non-string items.

    If *value* is already a list, each element is converted via ``str()``.
    Any other type results in an empty list.

    Args:
        value: The raw value to coerce (typically from a JSON-parsed dict).

    Returns:
        A list of strings.
    """
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]
