"""Directory tree walker and AST-based source code analyzer.

This module is responsible for:
  - Recursively walking a directory tree while respecting pathspec-based
    ignore patterns loaded from .codexa.toml or .gitignore files.
  - Reading Python source files and extracting structural metadata
    (functions, classes, imports, module-level docstrings) using the
    built-in ``ast`` module.
  - Building :class:`~codexa.models.FileInfo` and
    :class:`~codexa.models.DirContext` model instances that serve as the
    shared data contract consumed by the LLM summarizer and renderer.

Design notes:
  - Only top-level (module-scope) function and class definitions are
    extracted; nested definitions are intentionally excluded to keep the
    metadata concise.
  - Import deduplication preserves first-occurrence order.
  - All file I/O errors are handled gracefully; unreadable files produce
    a warning and are represented with empty metadata rather than raising.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from codexa.models import DirContext, FileInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AnalyzerError(Exception):
    """Raised when the analyzer encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Low-level file utilities
# ---------------------------------------------------------------------------


def compute_file_hash(path: Path) -> str:
    """Compute a stable SHA-256 hex digest for the contents of *path*.

    Reads the file in 64 KiB chunks to keep memory usage bounded even for
    very large files.

    Args:
        path: Absolute or relative path to an existing file.

    Returns:
        A 64-character lowercase hex string.

    Raises:
        AnalyzerError: If the file cannot be read.
    """
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise AnalyzerError(f"Cannot read file for hashing: {path}") from exc
    return hasher.hexdigest()


def read_source_file(path: Path) -> Optional[str]:
    """Read a text file, returning its contents or ``None`` on error.

    Attempts UTF-8 decoding first, then falls back to latin-1 so that
    files with non-ASCII bytes (e.g. legacy Latin-encoded source) are
    still processed rather than silently dropped.

    Args:
        path: Path to the file to read.

    Returns:
        File contents as a string, or ``None`` if the file cannot be
        decoded or accessed.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return None
    logger.warning("Cannot decode %s with any supported encoding.", path)
    return None


def is_python_file(path: Path) -> bool:
    """Return ``True`` if *path* has a ``.py`` suffix (case-insensitive).

    Args:
        path: Any file path.

    Returns:
        Boolean indicating whether the file is a Python source file.
    """
    return path.suffix.lower() == ".py"


# ---------------------------------------------------------------------------
# AST-based metadata extraction
# ---------------------------------------------------------------------------


def extract_python_metadata(source: str, filepath: Path) -> dict:
    """Parse *source* as Python and return a structural metadata dict.

    Only **top-level** (module-scope) definitions are extracted so that
    the metadata remains a concise representation of the public surface
    area.  Nested functions or classes defined inside other functions or
    classes are omitted.

    Extracted fields:

    * ``module_docstring`` — The first string expression in the module
      body if it is a :class:`ast.Constant`, otherwise ``None``.
    * ``functions`` — Top-level :class:`ast.FunctionDef` and
      :class:`ast.AsyncFunctionDef` names, deduplicated in source order.
    * ``classes`` — Top-level :class:`ast.ClassDef` names, deduplicated
      in source order.
    * ``imports`` — Module names from ``import X`` and ``from X import …``
      statements at any nesting level, deduplicated in source order.
      Empty-string module names (e.g. relative imports without a base
      module) are excluded.

    Args:
        source: Raw Python source text.
        filepath: Path used only in diagnostic messages on parse error.

    Returns:
        A dict with keys ``module_docstring``, ``functions``, ``classes``,
        and ``imports``.
    """
    metadata: dict = {
        "module_docstring": None,
        "functions": [],
        "classes": [],
        "imports": [],
    }

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as exc:
        logger.warning("Syntax error in %s: %s", filepath, exc)
        return metadata
    except ValueError as exc:
        # ast.parse can raise ValueError for null bytes
        logger.warning("Cannot parse %s: %s", filepath, exc)
        return metadata

    # Module-level docstring
    metadata["module_docstring"] = ast.get_docstring(tree)

    # Top-level function and class names come from direct children of
    # the Module node only, so we iterate tree.body rather than
    # ast.walk (which would also visit nested definitions).
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            metadata["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            metadata["classes"].append(node.name)

    # Imports can appear anywhere (inside if __name__ == '__main__' blocks
    # etc.) so we use ast.walk for completeness.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    metadata["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                metadata["imports"].append(module)

    # De-duplicate while preserving first-occurrence order
    metadata["functions"] = list(dict.fromkeys(metadata["functions"]))
    metadata["classes"] = list(dict.fromkeys(metadata["classes"]))
    metadata["imports"] = list(dict.fromkeys(filter(None, metadata["imports"])))

    return metadata


# ---------------------------------------------------------------------------
# Single-file analysis
# ---------------------------------------------------------------------------


def analyze_file(path: Path, relative_to: Optional[Path] = None) -> FileInfo:
    """Analyze a single Python source file and return a :class:`~codexa.models.FileInfo`.

    Computes the content hash, reads the source text, and runs AST
    extraction.  If any step fails the corresponding fields are left at
    their empty defaults and a warning is logged rather than raising so
    that a single problematic file does not abort a full directory scan.

    Args:
        path: Absolute (or resolvable) path to a Python file.
        relative_to: If provided, :attr:`~codexa.models.FileInfo.relative_path`
            is set to ``path.relative_to(relative_to)``.

    Returns:
        A populated :class:`~codexa.models.FileInfo` instance.

    Raises:
        AnalyzerError: If the file does not exist and therefore cannot be
            hashed at all.
    """
    path = path.resolve()

    # File must exist to compute a hash.
    if not path.exists():
        raise AnalyzerError(f"File does not exist: {path}")

    content_hash = compute_file_hash(path)

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        logger.warning("Cannot stat %s: %s", path, exc)
        size_bytes = 0

    source = read_source_file(path)

    if source is not None:
        ast_meta = extract_python_metadata(source, path)
    else:
        ast_meta = {
            "module_docstring": None,
            "functions": [],
            "classes": [],
            "imports": [],
        }

    relative_path: Optional[Path] = None
    if relative_to is not None:
        try:
            relative_path = path.relative_to(relative_to.resolve())
        except ValueError:
            logger.warning(
                "Cannot make %s relative to %s; using None.", path, relative_to
            )

    return FileInfo(
        path=path,
        relative_path=relative_path,
        size_bytes=size_bytes,
        content_hash=content_hash,
        source=source,
        module_docstring=ast_meta["module_docstring"],
        functions=ast_meta["functions"],
        classes=ast_meta["classes"],
        imports=ast_meta["imports"],
    )


# ---------------------------------------------------------------------------
# Directory walking
# ---------------------------------------------------------------------------


def walk_directory(
    root: Path,
    ignore_spec: Any = None,
    max_depth: Optional[int] = None,
) -> List[Path]:
    """Recursively collect Python source files under *root*.

    Directories and files whose paths (relative to *root*) match
    *ignore_spec* are skipped.  Depth is measured from *root*: depth 0
    means only files immediately inside *root*; depth 1 includes one
    level of subdirectories, and so on.  ``None`` means unlimited depth.

    Args:
        root: The top-level directory to start from.  Resolved to an
            absolute path internally.
        ignore_spec: An optional ``pathspec.PathSpec`` instance.  Any
            file or directory whose path relative to *root* matches the
            spec is excluded from the results.
        max_depth: Maximum directory depth to recurse into.

    Returns:
        A sorted list of absolute :class:`~pathlib.Path` objects for all
        Python source files found.
    """
    root = root.resolve()
    collected: List[Path] = []

    for dirpath_str, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath_str)
        # Compute depth relative to root
        try:
            relative_dir = current.relative_to(root)
        except ValueError:
            # Should not happen because os.walk starts at root
            relative_dir = Path(".")
        depth = len(relative_dir.parts)

        # Prune recursion when max_depth is reached
        if max_depth is not None and depth >= max_depth:
            dirnames.clear()

        # Filter ignored directories in-place so os.walk skips them
        if ignore_spec is not None:
            kept: List[str] = []
            for dirname in dirnames:
                # Append trailing slash so gitignore-style dir patterns match
                rel_dir_str = str((relative_dir / dirname).as_posix()) + "/"
                if not ignore_spec.match_file(rel_dir_str):
                    kept.append(dirname)
            dirnames[:] = kept

        # Collect Python files, skipping ignored ones
        for filename in filenames:
            filepath = current / filename
            if not is_python_file(filepath):
                continue
            if ignore_spec is not None:
                rel_file_str = str((relative_dir / filename).as_posix())
                if ignore_spec.match_file(rel_file_str):
                    continue
            collected.append(filepath)

    return sorted(collected)


# ---------------------------------------------------------------------------
# Directory-level analysis
# ---------------------------------------------------------------------------


def analyze_directory(
    directory: Path,
    ignore_spec: Any = None,
    max_depth: Optional[int] = None,
) -> DirContext:
    """Collect analysis data for all Python files **directly** in *directory*.

    Unlike :func:`walk_directory` (which recurses), this function only
    analyses files that are immediate children of *directory*.  The
    *ignore_spec* and *max_depth* arguments are forwarded to
    :func:`walk_directory` so that the caller can retrieve all files
    across the subtree when needed, but only direct children are
    packaged into the returned :class:`~codexa.models.DirContext`.

    The :attr:`~codexa.models.DirContext.content_hash` is computed from
    the individual :attr:`~codexa.models.FileInfo.content_hash` values of
    the direct-child files via
    :meth:`~codexa.models.DirContext.compute_hash`.

    Args:
        directory: The directory to analyse.  Resolved to an absolute path
            internally.
        ignore_spec: Optional ``pathspec.PathSpec`` for filtering.
        max_depth: Maximum recursion depth (forwarded to
            :func:`walk_directory`).

    Returns:
        A populated :class:`~codexa.models.DirContext` instance with
        ``summary`` left as ``None`` (to be filled in by the LLM step).
    """
    directory = directory.resolve()

    # Collect only files that are direct children of this directory
    file_infos: List[FileInfo] = []
    try:
        for entry in sorted(directory.iterdir()):
            if not entry.is_file() or not is_python_file(entry):
                continue
            # Check ignore spec for this specific file
            if ignore_spec is not None:
                rel_str = entry.name
                if ignore_spec.match_file(rel_str):
                    logger.debug("Ignoring file %s (matched ignore spec)", entry)
                    continue
            try:
                fi = analyze_file(entry, relative_to=directory)
                file_infos.append(fi)
            except AnalyzerError as exc:
                logger.warning("Skipping %s: %s", entry, exc)
    except PermissionError as exc:
        logger.warning("Cannot list directory %s: %s", directory, exc)

    # Collect immediate subdirectory names that are not ignored
    subdirs: List[str] = []
    try:
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            if ignore_spec is not None:
                rel_dir_str = entry.name + "/"
                if ignore_spec.match_file(rel_dir_str):
                    logger.debug(
                        "Ignoring subdirectory %s (matched ignore spec)", entry
                    )
                    continue
            subdirs.append(entry.name)
    except PermissionError as exc:
        logger.warning("Cannot list directory %s: %s", directory, exc)

    content_hash = DirContext.compute_hash(file_infos)

    return DirContext(
        path=directory,
        files=file_infos,
        subdirectories=subdirs,
        content_hash=content_hash,
        summary=None,
    )


def analyze_tree(
    root: Path,
    ignore_spec: Any = None,
    max_depth: Optional[int] = None,
) -> List[DirContext]:
    """Recursively analyse every directory under *root* and return a list of
    :class:`~codexa.models.DirContext` objects, one per directory that
    contains at least one Python source file (directly, not recursively).

    Directories excluded by *ignore_spec* or beyond *max_depth* are
    skipped entirely.

    Args:
        root: The root directory from which to start the walk.
        ignore_spec: Optional ``pathspec.PathSpec`` for filtering.
        max_depth: Maximum directory depth to recurse into.

    Returns:
        A list of :class:`~codexa.models.DirContext` objects in
        top-down order (breadth-first by ``os.walk`` traversal order).
    """
    root = root.resolve()
    contexts: List[DirContext] = []

    # Determine which directories to visit using the same logic as
    # walk_directory so ignore and depth rules are consistent.
    for dirpath_str, dirnames, _filenames in os.walk(root, topdown=True):
        current = Path(dirpath_str)
        try:
            relative_dir = current.relative_to(root)
        except ValueError:
            relative_dir = Path(".")
        depth = len(relative_dir.parts)

        # Prune recursion when max_depth is reached
        if max_depth is not None and depth >= max_depth:
            dirnames.clear()

        # Filter ignored directories
        if ignore_spec is not None:
            kept_dirs: List[str] = []
            for dirname in dirnames:
                rel_dir_str = str((relative_dir / dirname).as_posix()) + "/"
                if not ignore_spec.match_file(rel_dir_str):
                    kept_dirs.append(dirname)
            dirnames[:] = kept_dirs

        # Analyse this directory (only direct-child files)
        ctx = analyze_directory(current, ignore_spec=ignore_spec, max_depth=None)
        # Include directories that have Python files OR are the root
        if not ctx.is_empty or current == root:
            contexts.append(ctx)

    return contexts
