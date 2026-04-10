"""Unit tests for codexa.models — FileInfo, ModuleSummary, and DirContext.

Covers construction, property helpers, serialisation round-trips, and
edge-case handling such as empty files and missing dict keys.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from codexa.models import DirContext, FileInfo, ModuleSummary, _coerce_str_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file_info(
    path: str = "/project/module.py",
    relative_path: str = "module.py",
    size_bytes: int = 512,
    content_hash: str = "abc123",
    source: str = "def foo(): pass\n",
    module_docstring: str = "A test module.",
    functions: list = None,
    classes: list = None,
    imports: list = None,
) -> FileInfo:
    return FileInfo(
        path=Path(path),
        relative_path=Path(relative_path),
        size_bytes=size_bytes,
        content_hash=content_hash,
        source=source,
        module_docstring=module_docstring,
        functions=functions if functions is not None else ["foo"],
        classes=classes if classes is not None else [],
        imports=imports if imports is not None else ["os"],
    )


def _make_summary(
    overview: str = "This module does X.",
    key_symbols: list = None,
    patterns: list = None,
    tribal_knowledge: list = None,
) -> ModuleSummary:
    return ModuleSummary(
        overview=overview,
        key_symbols=key_symbols if key_symbols is not None else ["foo"],
        patterns=patterns if patterns is not None else ["Singleton pattern"],
        tribal_knowledge=tribal_knowledge if tribal_knowledge is not None else ["Use X, not Y"],
    )


def _make_dir_context(
    path: str = "/project",
    files: list = None,
    subdirectories: list = None,
    content_hash: str = "deadbeef",
    summary: ModuleSummary = None,
) -> DirContext:
    return DirContext(
        path=Path(path),
        files=files if files is not None else [],
        subdirectories=subdirectories if subdirectories is not None else ["sub"],
        content_hash=content_hash,
        summary=summary,
    )


# ===========================================================================
# FileInfo tests
# ===========================================================================


class TestFileInfo:
    """Tests for the FileInfo dataclass."""

    def test_basic_construction(self) -> None:
        fi = _make_file_info()
        assert fi.path == Path("/project/module.py")
        assert fi.relative_path == Path("module.py")
        assert fi.size_bytes == 512
        assert fi.content_hash == "abc123"
        assert fi.source == "def foo(): pass\n"
        assert fi.module_docstring == "A test module."
        assert fi.functions == ["foo"]
        assert fi.classes == []
        assert fi.imports == ["os"]

    def test_name_property(self) -> None:
        fi = _make_file_info(path="/project/utils/helpers.py")
        assert fi.name == "helpers.py"

    def test_is_empty_true(self) -> None:
        fi = _make_file_info(
            functions=[],
            classes=[],
            module_docstring=None,
        )
        assert fi.is_empty is True

    def test_is_empty_false_with_function(self) -> None:
        fi = _make_file_info(functions=["foo"], classes=[], module_docstring=None)
        assert fi.is_empty is False

    def test_is_empty_false_with_docstring(self) -> None:
        fi = _make_file_info(functions=[], classes=[], module_docstring="Hello")
        assert fi.is_empty is False

    def test_symbol_count(self) -> None:
        fi = _make_file_info(functions=["a", "b"], classes=["C"])
        assert fi.symbol_count == 3

    def test_symbol_count_empty(self) -> None:
        fi = _make_file_info(functions=[], classes=[])
        assert fi.symbol_count == 0

    def test_to_dict_keys(self) -> None:
        fi = _make_file_info()
        d = fi.to_dict()
        expected_keys = {
            "path", "relative_path", "size_bytes", "content_hash",
            "source", "module_docstring", "functions", "classes", "imports",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_path_is_string(self) -> None:
        fi = _make_file_info()
        d = fi.to_dict()
        assert isinstance(d["path"], str)
        assert isinstance(d["relative_path"], str)

    def test_to_dict_none_relative_path(self) -> None:
        fi = FileInfo(
            path=Path("/project/foo.py"),
            relative_path=None,
            size_bytes=0,
            content_hash="",
            source=None,
            module_docstring=None,
            functions=[],
            classes=[],
            imports=[],
        )
        d = fi.to_dict()
        assert d["relative_path"] is None

    def test_round_trip(self) -> None:
        fi = _make_file_info(
            functions=["foo", "bar"],
            classes=["MyClass"],
            imports=["os", "sys"],
        )
        restored = FileInfo.from_dict(fi.to_dict())
        assert restored.path == fi.path
        assert restored.relative_path == fi.relative_path
        assert restored.size_bytes == fi.size_bytes
        assert restored.content_hash == fi.content_hash
        assert restored.source == fi.source
        assert restored.module_docstring == fi.module_docstring
        assert restored.functions == fi.functions
        assert restored.classes == fi.classes
        assert restored.imports == fi.imports

    def test_from_dict_missing_optional_keys(self) -> None:
        """from_dict should tolerate missing optional keys."""
        fi = FileInfo.from_dict({"path": "/foo/bar.py"})
        assert fi.path == Path("/foo/bar.py")
        assert fi.relative_path is None
        assert fi.size_bytes == 0
        assert fi.content_hash == ""
        assert fi.source is None
        assert fi.module_docstring is None
        assert fi.functions == []
        assert fi.classes == []
        assert fi.imports == []

    def test_from_dict_requires_path(self) -> None:
        with pytest.raises(KeyError):
            FileInfo.from_dict({"size_bytes": 0})

    def test_empty_classmethod(self) -> None:
        path = Path("/project/empty.py")
        fi = FileInfo.empty(path)
        assert fi.path == path
        assert fi.relative_path is None
        assert fi.size_bytes == 0
        assert fi.content_hash == ""
        assert fi.source is None
        assert fi.functions == []
        assert fi.classes == []
        assert fi.imports == []

    def test_empty_classmethod_with_relative_path(self) -> None:
        path = Path("/project/empty.py")
        rel = Path("empty.py")
        fi = FileInfo.empty(path, relative_path=rel)
        assert fi.relative_path == rel

    def test_lists_are_independent_copies(self) -> None:
        """to_dict should return new list objects, not references."""
        fi = _make_file_info(functions=["a"])
        d = fi.to_dict()
        d["functions"].append("b")
        assert fi.functions == ["a"], "Mutation of to_dict() output should not affect FileInfo"


# ===========================================================================
# ModuleSummary tests
# ===========================================================================


class TestModuleSummary:
    """Tests for the ModuleSummary dataclass."""

    def test_basic_construction(self) -> None:
        ms = _make_summary()
        assert ms.overview == "This module does X."
        assert ms.key_symbols == ["foo"]
        assert ms.patterns == ["Singleton pattern"]
        assert ms.tribal_knowledge == ["Use X, not Y"]

    def test_has_content_true(self) -> None:
        ms = _make_summary()
        assert ms.has_content is True

    def test_has_content_false(self) -> None:
        ms = ModuleSummary.empty()
        assert ms.has_content is False

    def test_has_content_overview_only(self) -> None:
        ms = ModuleSummary(overview="Hello", key_symbols=[], patterns=[], tribal_knowledge=[])
        assert ms.has_content is True

    def test_empty_classmethod(self) -> None:
        ms = ModuleSummary.empty()
        assert ms.overview == ""
        assert ms.key_symbols == []
        assert ms.patterns == []
        assert ms.tribal_knowledge == []

    def test_to_dict_keys(self) -> None:
        ms = _make_summary()
        d = ms.to_dict()
        assert set(d.keys()) == {"overview", "key_symbols", "patterns", "tribal_knowledge"}

    def test_to_dict_values(self) -> None:
        ms = _make_summary(
            overview="Overview text",
            key_symbols=["A", "B"],
            patterns=["P1"],
            tribal_knowledge=["T1", "T2"],
        )
        d = ms.to_dict()
        assert d["overview"] == "Overview text"
        assert d["key_symbols"] == ["A", "B"]
        assert d["patterns"] == ["P1"]
        assert d["tribal_knowledge"] == ["T1", "T2"]

    def test_round_trip(self) -> None:
        ms = _make_summary()
        restored = ModuleSummary.from_dict(ms.to_dict())
        assert restored.overview == ms.overview
        assert restored.key_symbols == ms.key_symbols
        assert restored.patterns == ms.patterns
        assert restored.tribal_knowledge == ms.tribal_knowledge

    def test_from_dict_missing_keys(self) -> None:
        ms = ModuleSummary.from_dict({})
        assert ms.overview == ""
        assert ms.key_symbols == []
        assert ms.patterns == []
        assert ms.tribal_knowledge == []

    def test_from_dict_ignores_extra_keys(self) -> None:
        ms = ModuleSummary.from_dict({"overview": "X", "unknown_key": "ignored"})
        assert ms.overview == "X"

    def test_from_dict_coerces_list_items_to_str(self) -> None:
        ms = ModuleSummary.from_dict({"key_symbols": [1, 2, 3]})
        assert ms.key_symbols == ["1", "2", "3"]

    def test_from_dict_non_list_becomes_empty(self) -> None:
        ms = ModuleSummary.from_dict({"patterns": "not-a-list"})
        assert ms.patterns == []

    def test_lists_are_independent_copies(self) -> None:
        ms = _make_summary(key_symbols=["X"])
        d = ms.to_dict()
        d["key_symbols"].append("Y")
        assert ms.key_symbols == ["X"]


# ===========================================================================
# DirContext tests
# ===========================================================================


class TestDirContext:
    """Tests for the DirContext dataclass."""

    def test_basic_construction(self) -> None:
        dc = _make_dir_context()
        assert dc.path == Path("/project")
        assert dc.files == []
        assert dc.subdirectories == ["sub"]
        assert dc.content_hash == "deadbeef"
        assert dc.summary is None

    def test_name_property(self) -> None:
        dc = _make_dir_context(path="/project/src")
        assert dc.name == "src"

    def test_file_count_empty(self) -> None:
        dc = _make_dir_context(files=[])
        assert dc.file_count == 0

    def test_file_count_non_empty(self) -> None:
        fi1 = _make_file_info(path="/p/a.py", relative_path="a.py")
        fi2 = _make_file_info(path="/p/b.py", relative_path="b.py")
        dc = _make_dir_context(files=[fi1, fi2])
        assert dc.file_count == 2

    def test_is_empty_true(self) -> None:
        dc = _make_dir_context(files=[])
        assert dc.is_empty is True

    def test_is_empty_false(self) -> None:
        fi = _make_file_info()
        dc = _make_dir_context(files=[fi])
        assert dc.is_empty is False

    def test_total_size_bytes(self) -> None:
        fi1 = _make_file_info(path="/p/a.py", relative_path="a.py", size_bytes=100)
        fi2 = _make_file_info(path="/p/b.py", relative_path="b.py", size_bytes=200)
        dc = _make_dir_context(files=[fi1, fi2])
        assert dc.total_size_bytes == 300

    def test_total_size_bytes_empty(self) -> None:
        dc = _make_dir_context(files=[])
        assert dc.total_size_bytes == 0

    # ------------------------------------------------------------------
    # Aggregate property tests
    # ------------------------------------------------------------------

    def test_all_functions_deduplication(self) -> None:
        fi1 = _make_file_info(path="/p/a.py", relative_path="a.py", functions=["foo", "bar"])
        fi2 = _make_file_info(path="/p/b.py", relative_path="b.py", functions=["bar", "baz"])
        dc = _make_dir_context(files=[fi1, fi2])
        result = dc.all_functions
        assert result == ["foo", "bar", "baz"]

    def test_all_functions_empty(self) -> None:
        dc = _make_dir_context(files=[])
        assert dc.all_functions == []

    def test_all_classes_deduplication(self) -> None:
        fi1 = _make_file_info(path="/p/a.py", relative_path="a.py", classes=["A", "B"])
        fi2 = _make_file_info(path="/p/b.py", relative_path="b.py", classes=["B", "C"])
        dc = _make_dir_context(files=[fi1, fi2])
        assert dc.all_classes == ["A", "B", "C"]

    def test_all_imports_deduplication(self) -> None:
        fi1 = _make_file_info(path="/p/a.py", relative_path="a.py", imports=["os", "sys"])
        fi2 = _make_file_info(path="/p/b.py", relative_path="b.py", imports=["sys", "re"])
        dc = _make_dir_context(files=[fi1, fi2])
        assert dc.all_imports == ["os", "sys", "re"]

    def test_all_functions_preserves_order(self) -> None:
        fi1 = _make_file_info(path="/p/a.py", relative_path="a.py", functions=["z", "a"])
        fi2 = _make_file_info(path="/p/b.py", relative_path="b.py", functions=["m"])
        dc = _make_dir_context(files=[fi1, fi2])
        assert dc.all_functions == ["z", "a", "m"]

    # ------------------------------------------------------------------
    # Hash tests
    # ------------------------------------------------------------------

    def test_compute_hash_deterministic(self) -> None:
        fi1 = _make_file_info(content_hash="aaa")
        fi2 = _make_file_info(content_hash="bbb")
        h1 = DirContext.compute_hash([fi1, fi2])
        h2 = DirContext.compute_hash([fi1, fi2])
        assert h1 == h2

    def test_compute_hash_changes_with_content(self) -> None:
        fi1 = _make_file_info(content_hash="aaa")
        fi2 = _make_file_info(content_hash="bbb")
        fi3 = _make_file_info(content_hash="ccc")
        h1 = DirContext.compute_hash([fi1, fi2])
        h2 = DirContext.compute_hash([fi1, fi3])
        assert h1 != h2

    def test_compute_hash_empty_list(self) -> None:
        h = DirContext.compute_hash([])
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected

    def test_refresh_hash_updates_content_hash(self) -> None:
        fi = _make_file_info(content_hash="abc")
        dc = _make_dir_context(files=[fi], content_hash="old_hash")
        dc.refresh_hash()
        expected = DirContext.compute_hash([fi])
        assert dc.content_hash == expected
        assert dc.content_hash != "old_hash"

    def test_refresh_hash_empty_files(self) -> None:
        dc = _make_dir_context(files=[], content_hash="old")
        dc.refresh_hash()
        assert dc.content_hash == hashlib.sha256(b"").hexdigest()

    # ------------------------------------------------------------------
    # Serialisation tests
    # ------------------------------------------------------------------

    def test_to_dict_keys(self) -> None:
        dc = _make_dir_context()
        d = dc.to_dict()
        assert set(d.keys()) == {"path", "files", "subdirectories", "content_hash", "summary"}

    def test_to_dict_path_is_string(self) -> None:
        dc = _make_dir_context()
        d = dc.to_dict()
        assert isinstance(d["path"], str)

    def test_to_dict_summary_none(self) -> None:
        dc = _make_dir_context(summary=None)
        assert dc.to_dict()["summary"] is None

    def test_to_dict_summary_populated(self) -> None:
        ms = _make_summary()
        dc = _make_dir_context(summary=ms)
        d = dc.to_dict()
        assert d["summary"] is not None
        assert d["summary"]["overview"] == ms.overview

    def test_to_dict_files_are_dicts(self) -> None:
        fi = _make_file_info()
        dc = _make_dir_context(files=[fi])
        d = dc.to_dict()
        assert isinstance(d["files"], list)
        assert isinstance(d["files"][0], dict)

    def test_round_trip_no_summary(self) -> None:
        fi = _make_file_info(
            path="/project/foo.py",
            relative_path="foo.py",
            functions=["a"],
            classes=["B"],
        )
        dc = _make_dir_context(
            path="/project",
            files=[fi],
            subdirectories=["sub1", "sub2"],
            content_hash="cafebabe",
            summary=None,
        )
        restored = DirContext.from_dict(dc.to_dict())
        assert restored.path == dc.path
        assert len(restored.files) == 1
        assert restored.files[0].path == fi.path
        assert restored.subdirectories == dc.subdirectories
        assert restored.content_hash == dc.content_hash
        assert restored.summary is None

    def test_round_trip_with_summary(self) -> None:
        ms = _make_summary()
        dc = _make_dir_context(summary=ms)
        restored = DirContext.from_dict(dc.to_dict())
        assert restored.summary is not None
        assert restored.summary.overview == ms.overview
        assert restored.summary.key_symbols == ms.key_symbols

    def test_from_dict_missing_optional_keys(self) -> None:
        dc = DirContext.from_dict({"path": "/some/dir"})
        assert dc.path == Path("/some/dir")
        assert dc.files == []
        assert dc.subdirectories == []
        assert dc.content_hash == ""
        assert dc.summary is None

    def test_from_dict_requires_path(self) -> None:
        with pytest.raises(KeyError):
            DirContext.from_dict({"files": []})

    def test_empty_classmethod(self) -> None:
        path = Path("/some/directory")
        dc = DirContext.empty(path)
        assert dc.path == path
        assert dc.files == []
        assert dc.subdirectories == []
        assert dc.summary is None
        expected_hash = hashlib.sha256(b"").hexdigest()
        assert dc.content_hash == expected_hash


# ===========================================================================
# _coerce_str_list helper tests
# ===========================================================================


class TestCoerceStrList:
    """Tests for the private _coerce_str_list helper."""

    def test_string_list_unchanged(self) -> None:
        assert _coerce_str_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_int_items_coerced(self) -> None:
        assert _coerce_str_list([1, 2, 3]) == ["1", "2", "3"]

    def test_mixed_types_coerced(self) -> None:
        result = _coerce_str_list(["hello", 42, 3.14])
        assert result == ["hello", "42", "3.14"]

    def test_none_items_excluded(self) -> None:
        result = _coerce_str_list(["a", None, "b"])
        assert result == ["a", "b"]

    def test_empty_list(self) -> None:
        assert _coerce_str_list([]) == []

    def test_non_list_returns_empty(self) -> None:
        assert _coerce_str_list("not a list") == []
        assert _coerce_str_list(42) == []
        assert _coerce_str_list(None) == []
        assert _coerce_str_list({"a": 1}) == []
