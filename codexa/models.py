"""Data models for Codexa's analysis and summarization pipeline.

Defines three core dataclasses that serve as the shared data contract
across the analyzer, LLM client, and renderer:

  - :class:`FileInfo`: Metadata extracted from a single source file.
  - :class:`ModuleSummary`: LLM-generated narrative for a directory.
  - :class:`DirContext`: Aggregate context for an entire directory,
    combining FileInfo instances with an optional ModuleSummary.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional


@dataclasses.dataclass
class FileInfo:
    """Metadata extracted from a single source file.

    Attributes:
        path: Absolute path to the file.
        relative_path: Path relative to the containing directory.
        size_bytes: File size in bytes.
        content_hash: SHA-256 hex digest of the file contents.
        source: Raw file contents (may be None if unreadable).
        module_docstring: Module-level docstring, or None.
        functions: Names of top-level function definitions.
        classes: Names of top-level class definitions.
        imports: Names of imported modules.
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

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "path": str(self.path),
            "relative_path": str(self.relative_path) if self.relative_path else None,
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
            "source": self.source,
            "module_docstring": self.module_docstring,
            "functions": list(self.functions),
            "classes": list(self.classes),
            "imports": list(self.imports),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileInfo":
        """Construct a :class:`FileInfo` from a dict produced by :meth:`to_dict`.

        Args:
            data: Dict with the same keys as returned by :meth:`to_dict`.

        Returns:
            A new :class:`FileInfo` instance.
        """
        return cls(
            path=Path(data["path"]),
            relative_path=Path(data["relative_path"]) if data.get("relative_path") else None,
            size_bytes=int(data.get("size_bytes", 0)),
            content_hash=str(data.get("content_hash", "")),
            source=data.get("source"),
            module_docstring=data.get("module_docstring"),
            functions=list(data.get("functions", [])),
            classes=list(data.get("classes", [])),
            imports=list(data.get("imports", [])),
        )


@dataclasses.dataclass
class ModuleSummary:
    """LLM-generated narrative summary for a directory.

    Attributes:
        overview: A 2-4 sentence plain-text description of the module's purpose.
        key_symbols: The most important function and class names worth highlighting.
        patterns: Non-obvious patterns, idioms, or gotchas a reader should know.
        tribal_knowledge: Contextual hints for new developers.
    """

    overview: str
    key_symbols: List[str]
    patterns: List[str]
    tribal_knowledge: List[str]

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "overview": self.overview,
            "key_symbols": list(self.key_symbols),
            "patterns": list(self.patterns),
            "tribal_knowledge": list(self.tribal_knowledge),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModuleSummary":
        """Construct a :class:`ModuleSummary` from a dict.

        Args:
            data: Dict with keys matching :class:`ModuleSummary` attributes.

        Returns:
            A new :class:`ModuleSummary` instance.
        """
        return cls(
            overview=str(data.get("overview", "")),
            key_symbols=list(data.get("key_symbols", [])),
            patterns=list(data.get("patterns", [])),
            tribal_knowledge=list(data.get("tribal_knowledge", [])),
        )

    @classmethod
    def empty(cls) -> "ModuleSummary":
        """Return a blank :class:`ModuleSummary` used as a placeholder."""
        return cls(
            overview="",
            key_symbols=[],
            patterns=[],
            tribal_knowledge=[],
        )


@dataclasses.dataclass
class DirContext:
    """Aggregate context for a single directory in the codebase.

    Combines the per-file metadata gathered by the analyzer with the
    LLM-generated narrative produced by the summarizer.

    Attributes:
        path: Absolute path to the directory.
        files: :class:`FileInfo` instances for each source file directly
            within this directory.
        subdirectories: Names of immediate child directories (not ignored).
        content_hash: Combined SHA-256 hash across all file content hashes,
            used for incremental regeneration.
        summary: LLM-generated :class:`ModuleSummary`, or None if not yet
            computed.
    """

    path: Path
    files: List[FileInfo]
    subdirectories: List[str]
    content_hash: str
    summary: Optional[ModuleSummary] = None

    @property
    def all_functions(self) -> List[str]:
        """Return a deduplicated list of function names across all files."""
        seen: List[str] = []
        for f in self.files:
            for fn in f.functions:
                if fn not in seen:
                    seen.append(fn)
        return seen

    @property
    def all_classes(self) -> List[str]:
        """Return a deduplicated list of class names across all files."""
        seen: List[str] = []
        for f in self.files:
            for cls in f.classes:
                if cls not in seen:
                    seen.append(cls)
        return seen

    @property
    def all_imports(self) -> List[str]:
        """Return a deduplicated list of imported module names across all files."""
        seen: List[str] = []
        for f in self.files:
            for imp in f.imports:
                if imp not in seen:
                    seen.append(imp)
        return seen

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "path": str(self.path),
            "files": [f.to_dict() for f in self.files],
            "subdirectories": list(self.subdirectories),
            "content_hash": self.content_hash,
            "summary": self.summary.to_dict() if self.summary else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DirContext":
        """Construct a :class:`DirContext` from a dict.

        Args:
            data: Dict with the same keys as returned by :meth:`to_dict`.

        Returns:
            A new :class:`DirContext` instance.
        """
        summary_data = data.get("summary")
        return cls(
            path=Path(data["path"]),
            files=[FileInfo.from_dict(f) for f in data.get("files", [])],
            subdirectories=list(data.get("subdirectories", [])),
            content_hash=str(data.get("content_hash", "")),
            summary=ModuleSummary.from_dict(summary_data) if summary_data else None,
        )
