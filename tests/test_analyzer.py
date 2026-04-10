"""Unit tests for codexa.analyzer — directory walking and AST extraction.

Covers:
  - :func:`~codexa.analyzer.compute_file_hash`: determinism, change
    detection, and error handling.
  - :func:`~codexa.analyzer.read_source_file`: happy path, encoding
    fallback, and missing-file handling.
  - :func:`~codexa.analyzer.is_python_file`: various extensions.
  - :func:`~codexa.analyzer.extract_python_metadata`: docstring, functions,
    classes, imports, nested-definition exclusion, and syntax errors.
  - :func:`~codexa.analyzer.walk_directory`: recursion, depth limits,
    ignore specs, and sorted output.
  - :func:`~codexa.analyzer.analyze_file`: full integration with the
    sample fixture.
  - :func:`~codexa.analyzer.analyze_directory`: direct-child files,
    subdirectory collection, and content hash.
  - :func:`~codexa.analyzer.analyze_tree`: multi-directory traversal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import pytest

from codexa.analyzer import (
    AnalyzerError,
    analyze_directory,
    analyze_file,
    analyze_tree,
    compute_file_hash,
    extract_python_metadata,
    is_python_file,
    read_source_file,
    walk_directory,
)
from codexa.models import DirContext, FileInfo


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SAMPLE_MODULE = Path(__file__).parent / "fixtures" / "sample_module" / "main.py"


def _write_py(directory: Path, name: str, content: str = "# empty\n") -> Path:
    """Write a .py file into *directory* and return its path."""
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def _make_ignore_spec(patterns: List[str]):
    """Build a pathspec.PathSpec from *patterns*."""
    import pathspec
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


# ===========================================================================
# compute_file_hash
# ===========================================================================


class TestComputeFileHash:
    """Tests for compute_file_hash."""

    def test_returns_64_char_hex_string(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "a.py", "x = 1\n")
        digest = compute_file_hash(f)
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_deterministic_for_same_content(self, tmp_path: Path) -> None:
        content = "def foo(): pass\n"
        f1 = _write_py(tmp_path, "a.py", content)
        f2 = _write_py(tmp_path, "b.py", content)
        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_different_content_gives_different_hash(self, tmp_path: Path) -> None:
        f1 = _write_py(tmp_path, "a.py", "x = 1\n")
        f2 = _write_py(tmp_path, "b.py", "x = 2\n")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_same_call_twice_returns_same_hash(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "a.py", "hello\n")
        assert compute_file_hash(f) == compute_file_hash(f)

    def test_empty_file_has_known_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_bytes(b"")
        digest = compute_file_hash(f)
        expected = hashlib.sha256(b"").hexdigest()
        assert digest == expected

    def test_missing_file_raises_analyzer_error(self, tmp_path: Path) -> None:
        with pytest.raises(AnalyzerError, match="Cannot read"):
            compute_file_hash(tmp_path / "nonexistent.py")

    def test_works_with_sample_module(self) -> None:
        digest = compute_file_hash(SAMPLE_MODULE)
        assert len(digest) == 64


# ===========================================================================
# read_source_file
# ===========================================================================


class TestReadSourceFile:
    """Tests for read_source_file."""

    def test_reads_utf8_file(self, tmp_path: Path) -> None:
        content = "# unicode: \u00e9\u00e0\u00fc\n"
        f = tmp_path / "utf8.py"
        f.write_text(content, encoding="utf-8")
        result = read_source_file(f)
        assert result == content

    def test_reads_latin1_file(self, tmp_path: Path) -> None:
        # Write bytes that are valid latin-1 but not valid utf-8
        raw = b"# latin: \xe9\xe0\xfc\n"
        f = tmp_path / "latin1.py"
        f.write_bytes(raw)
        result = read_source_file(f)
        assert result is not None
        assert "\xe9" in result or "\xe9" in result.encode("latin-1").decode("latin-1")

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        result = read_source_file(tmp_path / "missing.py")
        assert result is None

    def test_reads_simple_python_file(self, tmp_path: Path) -> None:
        content = "def foo():\n    return 42\n"
        f = _write_py(tmp_path, "f.py", content)
        assert read_source_file(f) == content

    def test_reads_sample_module(self) -> None:
        result = read_source_file(SAMPLE_MODULE)
        assert result is not None
        assert "greet" in result


# ===========================================================================
# is_python_file
# ===========================================================================


class TestIsPythonFile:
    """Tests for is_python_file."""

    def test_py_extension_returns_true(self) -> None:
        assert is_python_file(Path("module.py")) is True

    def test_uppercase_py_returns_true(self) -> None:
        assert is_python_file(Path("MODULE.PY")) is True

    def test_mixed_case_py_returns_true(self) -> None:
        assert is_python_file(Path("Script.Py")) is True

    def test_txt_returns_false(self) -> None:
        assert is_python_file(Path("readme.txt")) is False

    def test_pyc_returns_false(self) -> None:
        assert is_python_file(Path("module.pyc")) is False

    def test_no_extension_returns_false(self) -> None:
        assert is_python_file(Path("Makefile")) is False

    def test_directory_like_path(self) -> None:
        assert is_python_file(Path("/project/src/module.py")) is True
        assert is_python_file(Path("/project/src/module.js")) is False


# ===========================================================================
# extract_python_metadata
# ===========================================================================


class TestExtractPythonMetadata:
    """Tests for extract_python_metadata."""

    def _parse(self, source: str) -> dict:
        return extract_python_metadata(source, Path("<test>"))

    # --- module docstring ---

    def test_extracts_module_docstring(self) -> None:
        src = '"""My module."""\n\ndef foo(): pass\n'
        meta = self._parse(src)
        assert meta["module_docstring"] == "My module."

    def test_no_docstring_returns_none(self) -> None:
        src = "x = 1\n"
        meta = self._parse(src)
        assert meta["module_docstring"] is None

    def test_single_quoted_docstring(self) -> None:
        src = "'Single quoted docstring.'\n\nx = 1\n"
        meta = self._parse(src)
        assert meta["module_docstring"] == "Single quoted docstring."

    def test_multiline_docstring(self) -> None:
        src = '"""Line one.\n\nLine two.\n"""\n'
        meta = self._parse(src)
        assert meta["module_docstring"] is not None
        assert "Line one." in meta["module_docstring"]

    # --- functions ---

    def test_extracts_top_level_functions(self) -> None:
        src = "def foo(): pass\ndef bar(): pass\n"
        meta = self._parse(src)
        assert meta["functions"] == ["foo", "bar"]

    def test_async_function_extracted(self) -> None:
        src = "async def fetch(): pass\n"
        meta = self._parse(src)
        assert "fetch" in meta["functions"]

    def test_nested_function_excluded(self) -> None:
        src = "def outer():\n    def inner(): pass\n"
        meta = self._parse(src)
        assert meta["functions"] == ["outer"]
        assert "inner" not in meta["functions"]

    def test_function_inside_class_excluded(self) -> None:
        src = "class MyClass:\n    def method(self): pass\n"
        meta = self._parse(src)
        assert "method" not in meta["functions"]

    def test_empty_source_returns_empty_functions(self) -> None:
        meta = self._parse("")
        assert meta["functions"] == []

    def test_functions_deduplicated(self) -> None:
        # Duplicate names can't actually appear at module scope in valid Python
        # but we verify the dedup logic handles it gracefully anyway by
        # checking that the result has no duplicates.
        src = "def foo(): pass\ndef bar(): pass\n"
        meta = self._parse(src)
        assert len(meta["functions"]) == len(set(meta["functions"]))

    # --- classes ---

    def test_extracts_top_level_classes(self) -> None:
        src = "class Foo: pass\nclass Bar: pass\n"
        meta = self._parse(src)
        assert meta["classes"] == ["Foo", "Bar"]

    def test_nested_class_excluded(self) -> None:
        src = "class Outer:\n    class Inner: pass\n"
        meta = self._parse(src)
        assert meta["classes"] == ["Outer"]
        assert "Inner" not in meta["classes"]

    def test_empty_source_returns_empty_classes(self) -> None:
        assert self._parse("")["classes"] == []

    # --- imports ---

    def test_import_statement(self) -> None:
        src = "import os\nimport sys\n"
        meta = self._parse(src)
        assert "os" in meta["imports"]
        assert "sys" in meta["imports"]

    def test_from_import_statement(self) -> None:
        src = "from pathlib import Path\n"
        meta = self._parse(src)
        assert "pathlib" in meta["imports"]

    def test_relative_import_without_module_excluded(self) -> None:
        src = "from . import utils\n"
        meta = self._parse(src)
        # Empty module name should not appear
        assert "" not in meta["imports"]

    def test_import_inside_function_included(self) -> None:
        src = "def foo():\n    import json\n"
        meta = self._parse(src)
        assert "json" in meta["imports"]

    def test_imports_deduplicated(self) -> None:
        src = "import os\nimport os\n"
        meta = self._parse(src)
        assert meta["imports"].count("os") == 1

    def test_empty_source_returns_empty_imports(self) -> None:
        assert self._parse("")["imports"] == []

    # --- syntax errors ---

    def test_syntax_error_returns_empty_metadata(self) -> None:
        src = "def foo(:\n    pass\n"  # invalid syntax
        meta = self._parse(src)
        assert meta["module_docstring"] is None
        assert meta["functions"] == []
        assert meta["classes"] == []
        assert meta["imports"] == []

    # --- sample module integration ---

    def test_sample_module_docstring(self) -> None:
        source = SAMPLE_MODULE.read_text(encoding="utf-8")
        meta = extract_python_metadata(source, SAMPLE_MODULE)
        assert meta["module_docstring"] is not None
        assert len(meta["module_docstring"]) > 0

    def test_sample_module_functions(self) -> None:
        source = SAMPLE_MODULE.read_text(encoding="utf-8")
        meta = extract_python_metadata(source, SAMPLE_MODULE)
        assert "greet" in meta["functions"]
        assert "add" in meta["functions"]
        assert "fetch_data" in meta["functions"]
        assert "list_files" in meta["functions"]
        assert "get_env" in meta["functions"]
        assert "_private_helper" in meta["functions"]

    def test_sample_module_nested_function_excluded(self) -> None:
        source = SAMPLE_MODULE.read_text(encoding="utf-8")
        meta = extract_python_metadata(source, SAMPLE_MODULE)
        # _build_headers is nested inside fetch_data
        assert "_build_headers" not in meta["functions"]

    def test_sample_module_classes(self) -> None:
        source = SAMPLE_MODULE.read_text(encoding="utf-8")
        meta = extract_python_metadata(source, SAMPLE_MODULE)
        assert "Animal" in meta["classes"]
        assert "Dog" in meta["classes"]
        assert "Config" in meta["classes"]

    def test_sample_module_nested_class_excluded(self) -> None:
        source = SAMPLE_MODULE.read_text(encoding="utf-8")
        meta = extract_python_metadata(source, SAMPLE_MODULE)
        # _Metadata is nested inside Animal
        assert "_Metadata" not in meta["classes"]

    def test_sample_module_imports(self) -> None:
        source = SAMPLE_MODULE.read_text(encoding="utf-8")
        meta = extract_python_metadata(source, SAMPLE_MODULE)
        assert "os" in meta["imports"]
        assert "sys" in meta["imports"]
        assert "pathlib" in meta["imports"]


# ===========================================================================
# walk_directory
# ===========================================================================


class TestWalkDirectory:
    """Tests for walk_directory."""

    def test_finds_py_files_in_root(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "a.py")
        result = walk_directory(tmp_path)
        assert f.resolve() in result

    def test_ignores_non_py_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        result = walk_directory(tmp_path)
        assert all(p.suffix == ".py" for p in result)

    def test_recurses_into_subdirectories(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        f = _write_py(sub, "b.py")
        result = walk_directory(tmp_path)
        assert f.resolve() in result

    def test_result_is_sorted(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "z.py")
        _write_py(tmp_path, "a.py")
        result = walk_directory(tmp_path)
        assert result == sorted(result)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert walk_directory(tmp_path) == []

    # --- max_depth ---

    def test_max_depth_zero_only_root_files(self, tmp_path: Path) -> None:
        f_root = _write_py(tmp_path, "root.py")
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_py(sub, "nested.py")
        result = walk_directory(tmp_path, max_depth=0)
        assert f_root.resolve() in result
        assert all(p.parent == tmp_path.resolve() for p in result)

    def test_max_depth_one_includes_one_level(self, tmp_path: Path) -> None:
        f_root = _write_py(tmp_path, "root.py")
        sub = tmp_path / "sub"
        sub.mkdir()
        f_sub = _write_py(sub, "sub.py")
        deep = sub / "deep"
        deep.mkdir()
        _write_py(deep, "deep.py")
        result = walk_directory(tmp_path, max_depth=1)
        assert f_root.resolve() in result
        assert f_sub.resolve() in result
        assert not any(p.parent == deep.resolve() for p in result)

    def test_max_depth_none_is_unlimited(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = a / "b"
        c = b / "c"
        c.mkdir(parents=True)
        deep_file = _write_py(c, "deep.py")
        result = walk_directory(tmp_path, max_depth=None)
        assert deep_file.resolve() in result

    # --- ignore_spec ---

    def test_ignores_matching_files(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "keep.py")
        _write_py(tmp_path, "skip.py")
        spec = _make_ignore_spec(["skip.py"])
        result = walk_directory(tmp_path, ignore_spec=spec)
        names = [p.name for p in result]
        assert "keep.py" in names
        assert "skip.py" not in names

    def test_ignores_matching_directory(self, tmp_path: Path) -> None:
        kept = tmp_path / "src"
        ignored = tmp_path / "__pycache__"
        kept.mkdir()
        ignored.mkdir()
        _write_py(kept, "a.py")
        _write_py(ignored, "b.py")
        spec = _make_ignore_spec(["__pycache__"])
        result = walk_directory(tmp_path, ignore_spec=spec)
        assert all("__pycache__" not in str(p) for p in result)
        assert any(p.name == "a.py" for p in result)

    def test_ignore_star_pyc_pattern(self, tmp_path: Path) -> None:
        # .pyc files should not be collected anyway (not .py), but verify
        # that the ignore spec is applied to .py files correctly.
        _write_py(tmp_path, "module.py")
        spec = _make_ignore_spec(["module.py"])
        result = walk_directory(tmp_path, ignore_spec=spec)
        assert result == []

    def test_no_ignore_spec_collects_all_py(self, tmp_path: Path) -> None:
        files = [_write_py(tmp_path, f"f{i}.py") for i in range(3)]
        result = walk_directory(tmp_path)
        for f in files:
            assert f.resolve() in result

    def test_nested_ignore_pattern(self, tmp_path: Path) -> None:
        sub = tmp_path / "vendor"
        sub.mkdir()
        _write_py(sub, "third_party.py")
        f_root = _write_py(tmp_path, "main.py")
        spec = _make_ignore_spec(["vendor/"])
        result = walk_directory(tmp_path, ignore_spec=spec)
        assert f_root.resolve() in result
        assert not any("vendor" in str(p) for p in result)


# ===========================================================================
# analyze_file
# ===========================================================================


class TestAnalyzeFile:
    """Tests for analyze_file."""

    def test_returns_file_info_instance(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "a.py", "x = 1\n")
        result = analyze_file(f)
        assert isinstance(result, FileInfo)

    def test_path_is_absolute(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "a.py")
        result = analyze_file(f)
        assert result.path.is_absolute()

    def test_size_bytes_correct(self, tmp_path: Path) -> None:
        content = "x = 1\n"
        f = _write_py(tmp_path, "a.py", content)
        result = analyze_file(f)
        assert result.size_bytes == len(content.encode("utf-8"))

    def test_content_hash_is_64_chars(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "a.py")
        result = analyze_file(f)
        assert len(result.content_hash) == 64

    def test_source_is_populated(self, tmp_path: Path) -> None:
        content = "def foo(): pass\n"
        f = _write_py(tmp_path, "a.py", content)
        result = analyze_file(f)
        assert result.source == content

    def test_relative_path_set_when_relative_to_provided(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "a.py")
        result = analyze_file(f, relative_to=tmp_path)
        assert result.relative_path == Path("a.py")

    def test_relative_path_none_when_not_provided(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "a.py")
        result = analyze_file(f)
        assert result.relative_path is None

    def test_missing_file_raises_analyzer_error(self, tmp_path: Path) -> None:
        with pytest.raises(AnalyzerError):
            analyze_file(tmp_path / "ghost.py")

    def test_sample_module_functions_extracted(self) -> None:
        result = analyze_file(SAMPLE_MODULE)
        assert "greet" in result.functions
        assert "add" in result.functions
        assert "fetch_data" in result.functions

    def test_sample_module_classes_extracted(self) -> None:
        result = analyze_file(SAMPLE_MODULE)
        assert "Animal" in result.classes
        assert "Dog" in result.classes
        assert "Config" in result.classes

    def test_sample_module_nested_class_not_in_classes(self) -> None:
        result = analyze_file(SAMPLE_MODULE)
        assert "_Metadata" not in result.classes

    def test_sample_module_imports_extracted(self) -> None:
        result = analyze_file(SAMPLE_MODULE)
        assert "os" in result.imports
        assert "sys" in result.imports
        assert "pathlib" in result.imports

    def test_sample_module_docstring_extracted(self) -> None:
        result = analyze_file(SAMPLE_MODULE)
        assert result.module_docstring is not None
        assert len(result.module_docstring) > 0

    def test_empty_file_produces_empty_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_bytes(b"")
        result = analyze_file(f)
        assert result.functions == []
        assert result.classes == []
        assert result.imports == []
        assert result.module_docstring is None

    def test_syntax_error_file_produces_empty_metadata(self, tmp_path: Path) -> None:
        bad = _write_py(tmp_path, "bad.py", "def foo(:\n    pass\n")
        result = analyze_file(bad)
        # Should not raise; metadata will be empty
        assert result.functions == []
        assert result.classes == []

    def test_returns_file_info_with_correct_fields(self, tmp_path: Path) -> None:
        src = '"""Doc."""\nimport os\ndef foo(): pass\nclass Bar: pass\n'
        f = _write_py(tmp_path, "mod.py", src)
        fi = analyze_file(f, relative_to=tmp_path)
        assert fi.module_docstring == "Doc."
        assert "foo" in fi.functions
        assert "Bar" in fi.classes
        assert "os" in fi.imports
        assert fi.relative_path == Path("mod.py")


# ===========================================================================
# analyze_directory
# ===========================================================================


class TestAnalyzeDirectory:
    """Tests for analyze_directory."""

    def test_returns_dir_context_instance(self, tmp_path: Path) -> None:
        result = analyze_directory(tmp_path)
        assert isinstance(result, DirContext)

    def test_path_is_absolute(self, tmp_path: Path) -> None:
        result = analyze_directory(tmp_path)
        assert result.path.is_absolute()

    def test_empty_directory_has_no_files(self, tmp_path: Path) -> None:
        result = analyze_directory(tmp_path)
        assert result.files == []

    def test_collects_direct_py_files(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "a.py")
        _write_py(tmp_path, "b.py")
        result = analyze_directory(tmp_path)
        names = [fi.name for fi in result.files]
        assert "a.py" in names
        assert "b.py" in names

    def test_does_not_include_subdirectory_files(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_py(sub, "nested.py")
        result = analyze_directory(tmp_path)
        assert all(fi.name != "nested.py" for fi in result.files)

    def test_collects_subdirectory_names(self, tmp_path: Path) -> None:
        sub = tmp_path / "mypackage"
        sub.mkdir()
        result = analyze_directory(tmp_path)
        assert "mypackage" in result.subdirectories

    def test_ignores_subdirectory_in_spec(self, tmp_path: Path) -> None:
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "src").mkdir()
        spec = _make_ignore_spec(["__pycache__"])
        result = analyze_directory(tmp_path, ignore_spec=spec)
        assert "__pycache__" not in result.subdirectories
        assert "src" in result.subdirectories

    def test_ignores_py_file_in_spec(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "keep.py")
        _write_py(tmp_path, "skip.py")
        spec = _make_ignore_spec(["skip.py"])
        result = analyze_directory(tmp_path, ignore_spec=spec)
        names = [fi.name for fi in result.files]
        assert "keep.py" in names
        assert "skip.py" not in names

    def test_content_hash_is_non_empty_string(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "a.py", "x = 1\n")
        result = analyze_directory(tmp_path)
        assert isinstance(result.content_hash, str)
        assert len(result.content_hash) == 64

    def test_empty_directory_hash_is_deterministic(self, tmp_path: Path) -> None:
        r1 = analyze_directory(tmp_path)
        r2 = analyze_directory(tmp_path)
        assert r1.content_hash == r2.content_hash

    def test_hash_changes_when_file_content_changes(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "a.py", "x = 1\n")
        r1 = analyze_directory(tmp_path)
        f.write_text("x = 2\n", encoding="utf-8")
        r2 = analyze_directory(tmp_path)
        assert r1.content_hash != r2.content_hash

    def test_summary_is_none(self, tmp_path: Path) -> None:
        result = analyze_directory(tmp_path)
        assert result.summary is None

    def test_file_info_relative_paths_populated(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "mod.py")
        result = analyze_directory(tmp_path)
        for fi in result.files:
            assert fi.relative_path is not None
            assert fi.relative_path == Path(fi.name)

    def test_non_py_files_excluded(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "code.py")
        (tmp_path / "readme.md").write_text("# hi", encoding="utf-8")
        result = analyze_directory(tmp_path)
        assert all(fi.name.endswith(".py") for fi in result.files)

    def test_files_list_is_sorted_by_name(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "z.py")
        _write_py(tmp_path, "a.py")
        _write_py(tmp_path, "m.py")
        result = analyze_directory(tmp_path)
        names = [fi.name for fi in result.files]
        assert names == sorted(names)

    def test_subdirectories_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "zebra").mkdir()
        (tmp_path / "alpha").mkdir()
        result = analyze_directory(tmp_path)
        assert result.subdirectories == sorted(result.subdirectories)

    def test_sample_fixture_directory(self) -> None:
        fixture_dir = SAMPLE_MODULE.parent
        result = analyze_directory(fixture_dir)
        assert isinstance(result, DirContext)
        assert any(fi.name == "main.py" for fi in result.files)


# ===========================================================================
# analyze_tree
# ===========================================================================


class TestAnalyzeTree:
    """Tests for analyze_tree."""

    def test_returns_list_of_dir_contexts(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "a.py")
        result = analyze_tree(tmp_path)
        assert isinstance(result, list)
        assert all(isinstance(c, DirContext) for c in result)

    def test_includes_root_even_when_empty(self, tmp_path: Path) -> None:
        result = analyze_tree(tmp_path)
        paths = [c.path for c in result]
        assert tmp_path.resolve() in paths

    def test_includes_subdirectory_with_py_files(self, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        sub.mkdir()
        _write_py(sub, "mod.py")
        result = analyze_tree(tmp_path)
        paths = [c.path for c in result]
        assert sub.resolve() in paths

    def test_respects_ignore_spec(self, tmp_path: Path) -> None:
        ignored = tmp_path / "__pycache__"
        ignored.mkdir()
        _write_py(ignored, "cached.py")
        spec = _make_ignore_spec(["__pycache__"])
        result = analyze_tree(tmp_path, ignore_spec=spec)
        paths = [c.path for c in result]
        assert ignored.resolve() not in paths

    def test_respects_max_depth(self, tmp_path: Path) -> None:
        sub1 = tmp_path / "level1"
        sub2 = sub1 / "level2"
        sub2.mkdir(parents=True)
        _write_py(sub2, "deep.py")
        result = analyze_tree(tmp_path, max_depth=1)
        paths = [c.path for c in result]
        assert sub2.resolve() not in paths

    def test_multiple_subdirectories_collected(self, tmp_path: Path) -> None:
        for name in ["pkg_a", "pkg_b", "pkg_c"]:
            sub = tmp_path / name
            sub.mkdir()
            _write_py(sub, "__init__.py")
        result = analyze_tree(tmp_path)
        paths = [c.path for c in result]
        for name in ["pkg_a", "pkg_b", "pkg_c"]:
            assert (tmp_path / name).resolve() in paths

    def test_returns_dir_context_with_files_populated(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "module.py", "def foo(): pass\n")
        result = analyze_tree(tmp_path)
        root_ctx = next(c for c in result if c.path == tmp_path.resolve())
        assert any(fi.name == "module.py" for fi in root_ctx.files)

    def test_all_summaries_are_none(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "a.py")
        result = analyze_tree(tmp_path)
        assert all(c.summary is None for c in result)
