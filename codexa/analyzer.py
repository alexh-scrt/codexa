"""Directory tree walker and AST-based source code analyzer.

This module is responsible for:
  - Recursively walking a directory tree while respecting pathspec-based
    ignore patterns loaded from .codexa.toml or .gitignore files.
  - Reading Python source files and extracting structural metadata
    (functions, classes, imports, module-level docstrings) using the
    built-in ``ast`` module.
  - Building FileInfo and DirContext model instances that serve as the
    shared data contract consumed by the LLM summarizer and renderer.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class AnalyzerError(Exception):
    """Raised when the analyzer encounters an unrecoverable error."""


def compute_file_hash(path: Path) -> str:
    """Compute a stable SHA-256 hex digest for the contents of *path*.

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


def extract_python_metadata(
    source: str,
    filepath: Path,
) -> dict:
    """Parse *source* as Python and return a metadata dict.

    Extracts:
      - ``module_docstring``: The top-level docstring, if present.
      - ``functions``: List of top-level function names.
      - ``classes``: List of top-level class names.
      - ``imports``: List of imported module names.

    Args:
        source: Raw Python source text.
        filepath: Path used only for diagnostic messages on parse error.

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

    # Module-level docstring
    metadata["module_docstring"] = ast.get_docstring(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(
            node, ast.AsyncFunctionDef
        ):
            # Only top-level functions (parent is Module)
            metadata["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            metadata["classes"].append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                metadata["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            metadata["imports"].append(module)

    # De-duplicate while preserving order
    metadata["functions"] = list(dict.fromkeys(metadata["functions"]))
    metadata["classes"] = list(dict.fromkeys(metadata["classes"]))
    metadata["imports"] = list(dict.fromkeys(filter(None, metadata["imports"])))

    return metadata


def read_source_file(path: Path) -> Optional[str]:
    """Read a text file, returning its contents or None on error.

    Attempts UTF-8 decoding first, then falls back to latin-1.

    Args:
        path: Path to the file to read.

    Returns:
        File contents as a string, or None if the file cannot be decoded.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return None
    logger.warning("Cannot decode %s with supported encodings.", path)
    return None


def is_python_file(path: Path) -> bool:
    """Return True if *path* has a .py suffix."""
    return path.suffix.lower() == ".py"


def walk_directory(
    root: Path,
    ignore_spec=None,
    max_depth: Optional[int] = None,
) -> List[Path]:
    """Yield Python source files under *root*, respecting ignore rules.

    Args:
        root: The top-level directory to start from.
        ignore_spec: An optional ``pathspec.PathSpec`` instance.  Files or
            directories whose paths (relative to *root*) match the spec are
            skipped.
        max_depth: Maximum directory depth to recurse into.  Depth 0 means
            only files directly inside *root*.  None means unlimited.

    Returns:
        A sorted list of absolute Path objects for matching files.
    """
    collected: List[Path] = []
    root = root.resolve()

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)
        relative_dir = current.relative_to(root)
        depth = len(relative_dir.parts)

        # Enforce max_depth: prune further recursion
        if max_depth is not None and depth >= max_depth:
            dirnames.clear()

        # Filter out ignored directories in-place so os.walk skips them
        if ignore_spec is not None:
            filtered = []
            for d in dirnames:
                rel = str((relative_dir / d).as_posix()) + "/"
                if not ignore_spec.match_file(rel):
                    filtered.append(d)
            dirnames[:] = filtered

        for filename in filenames:
            filepath = current / filename
            if ignore_spec is not None:
                rel = str((relative_dir / filename).as_posix())
                if ignore_spec.match_file(rel):
                    continue
            if is_python_file(filepath):
                collected.append(filepath)

    return sorted(collected)


def analyze_file(path: Path) -> dict:
    """Analyze a single Python source file and return a metadata dict.

    The returned dict contains keys compatible with ``FileInfo``:
      - ``path``: The absolute path.
      - ``relative_path``: Set to None here; callers should populate it.
      - ``size_bytes``: File size in bytes.
      - ``content_hash``: SHA-256 of the file contents.
      - ``source``: Raw file contents.
      - ``module_docstring``: Extracted module docstring or None.
      - ``functions``: List of function names.
      - ``classes``: List of class names.
      - ``imports``: List of imported module names.

    Args:
        path: Absolute path to a Python file.

    Returns:
        Metadata dictionary.

    Raises:
        AnalyzerError: If the file cannot be hashed.
    """
    content_hash = compute_file_hash(path)
    source = read_source_file(path)
    size_bytes = path.stat().st_size if path.exists() else 0

    ast_meta: dict = {
        "module_docstring": None,
        "functions": [],
        "classes": [],
        "imports": [],
    }
    if source is not None:
        ast_meta = extract_python_metadata(source, path)

    return {
        "path": path,
        "relative_path": None,
        "size_bytes": size_bytes,
        "content_hash": content_hash,
        "source": source,
        "module_docstring": ast_meta["module_docstring"],
        "functions": ast_meta["functions"],
        "classes": ast_meta["classes"],
        "imports": ast_meta["imports"],
    }


def analyze_directory(
    directory: Path,
    ignore_spec=None,
    max_depth: Optional[int] = None,
) -> dict:
    """Collect analysis data for all Python files in *directory*.

    Returns a dict compatible with ``DirContext`` containing:
      - ``path``: Absolute path of the directory.
      - ``files``: List of per-file metadata dicts from :func:`analyze_file`.
      - ``subdirectories``: Immediate subdirectory names (not ignored).
      - ``content_hash``: Combined hash across all file hashes.

    Args:
        directory: The directory to analyze.
        ignore_spec: Optional pathspec for filtering.
        max_depth: Maximum recursion depth.

    Returns:
        A dict suitable for constructing a ``DirContext`` instance.
    """
    directory = directory.resolve()
    python_files = walk_directory(directory, ignore_spec=ignore_spec, max_depth=max_depth)

    # Only files directly in this directory for DirContext
    direct_files = [f for f in python_files if f.parent == directory]

    file_metas = []
    for fp in direct_files:
        try:
            meta = analyze_file(fp)
            meta["relative_path"] = fp.relative_to(directory)
            file_metas.append(meta)
        except AnalyzerError as exc:
            logger.warning("Skipping %s: %s", fp, exc)

    # Collect immediate subdirectory names that are not ignored
    subdirs: List[str] = []
    try:
        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                if ignore_spec is not None:
                    rel = str(entry.name) + "/"
                    if ignore_spec.match_file(rel):
                        continue
                subdirs.append(entry.name)
    except PermissionError as exc:
        logger.warning("Cannot list directory %s: %s", directory, exc)

    # Combined content hash
    combined = hashlib.sha256(
        "|".join(m["content_hash"] for m in file_metas).encode()
    ).hexdigest()

    return {
        "path": directory,
        "files": file_metas,
        "subdirectories": subdirs,
        "content_hash": combined,
    }
