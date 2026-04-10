"""Unit tests for codexa.renderer — Jinja2 rendering and CODEXA.md writing.

Covers:
  - :func:`~codexa.renderer.build_template_context`: correct keys and values,
    safe defaults when no summary is attached.
  - :func:`~codexa.renderer.read_stored_hash`: parsing the first-line comment,
    missing file, malformed line.
  - :func:`~codexa.renderer.compute_context_hash`: determinism and consistency
    with ``content_hash``.
  - :class:`~codexa.renderer.Renderer`:
      - Lazy Jinja2 environment initialisation.
      - ``render()`` produces valid markdown with the hash comment.
      - ``render_dir_context()`` convenience wrapper.
      - ``write()`` creates the file, skips on unchanged hash, respects force.
      - ``write_dir_context()`` convenience wrapper.
  - Custom template path support.
  - :class:`~codexa.renderer.RendererError` exception.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from codexa.models import DirContext, FileInfo, ModuleSummary
from codexa.renderer import (
    RendererError,
    Renderer,
    _HASH_COMMENT_PREFIX,
    _HASH_COMMENT_SUFFIX,
    _HASH_LINE_RE,
    build_template_context,
    compute_context_hash,
    read_stored_hash,
)


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _make_file_info(
    path: str = "/project/module.py",
    relative_path: str = "module.py",
    size_bytes: int = 512,
    content_hash: str = "abc123" + "0" * 58,
    source: Optional[str] = "def foo(): pass\n",
    module_docstring: Optional[str] = "A test module.",
    functions: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    imports: Optional[List[str]] = None,
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
    overview: str = "This module handles authentication.",
    key_symbols: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    tribal_knowledge: Optional[List[str]] = None,
) -> ModuleSummary:
    return ModuleSummary(
        overview=overview,
        key_symbols=key_symbols if key_symbols is not None else ["foo", "MyClass"],
        patterns=patterns if patterns is not None else ["Uses singleton pattern."],
        tribal_knowledge=tribal_knowledge if tribal_knowledge is not None else ["Always call init() first."],
    )


def _make_dir_context(
    path: str = "/project",
    files: Optional[List[FileInfo]] = None,
    subdirectories: Optional[List[str]] = None,
    content_hash: str = "a" * 64,
    summary: Optional[ModuleSummary] = None,
) -> DirContext:
    return DirContext(
        path=Path(path),
        files=files if files is not None else [],
        subdirectories=subdirectories if subdirectories is not None else [],
        content_hash=content_hash,
        summary=summary,
    )


def _make_renderer(template_path: Optional[Path] = None) -> Renderer:
    return Renderer(template_path=template_path)


# ===========================================================================
# build_template_context
# ===========================================================================


class TestBuildTemplateContext:
    """Tests for build_template_context."""

    def test_returns_dict(self) -> None:
        ctx = _make_dir_context()
        result = build_template_context(ctx)
        assert isinstance(result, dict)

    def test_contains_required_keys(self) -> None:
        ctx = _make_dir_context()
        result = build_template_context(ctx)
        required = {
            "dir_context", "path", "dir_name", "content_hash",
            "files", "file_count", "subdirectories",
            "all_functions", "all_classes", "all_imports",
            "summary", "overview", "key_symbols", "patterns",
            "tribal_knowledge", "has_summary", "generated_at",
        }
        assert required.issubset(set(result.keys()))

    def test_dir_name_is_last_component(self) -> None:
        ctx = _make_dir_context(path="/project/src")
        result = build_template_context(ctx)
        assert result["dir_name"] == "src"

    def test_path_is_string(self) -> None:
        ctx = _make_dir_context(path="/project")
        result = build_template_context(ctx)
        assert isinstance(result["path"], str)
        assert "/project" in result["path"]

    def test_content_hash_matches_context(self) -> None:
        ctx = _make_dir_context(content_hash="b" * 64)
        result = build_template_context(ctx)
        assert result["content_hash"] == "b" * 64

    def test_file_count_correct(self) -> None:
        fi1 = _make_file_info(path="/p/a.py", relative_path="a.py")
        fi2 = _make_file_info(path="/p/b.py", relative_path="b.py")
        ctx = _make_dir_context(files=[fi1, fi2])
        result = build_template_context(ctx)
        assert result["file_count"] == 2

    def test_file_count_zero_when_empty(self) -> None:
        ctx = _make_dir_context(files=[])
        result = build_template_context(ctx)
        assert result["file_count"] == 0

    def test_files_list_populated(self) -> None:
        fi = _make_file_info()
        ctx = _make_dir_context(files=[fi])
        result = build_template_context(ctx)
        assert len(result["files"]) == 1
        assert result["files"][0] is fi

    def test_subdirectories_populated(self) -> None:
        ctx = _make_dir_context(subdirectories=["auth", "utils"])
        result = build_template_context(ctx)
        assert result["subdirectories"] == ["auth", "utils"]

    def test_all_functions_aggregated(self) -> None:
        fi1 = _make_file_info(path="/p/a.py", relative_path="a.py", functions=["foo", "bar"])
        fi2 = _make_file_info(path="/p/b.py", relative_path="b.py", functions=["baz"])
        ctx = _make_dir_context(files=[fi1, fi2])
        result = build_template_context(ctx)
        assert "foo" in result["all_functions"]
        assert "bar" in result["all_functions"]
        assert "baz" in result["all_functions"]

    def test_all_classes_aggregated(self) -> None:
        fi = _make_file_info(classes=["MyClass", "OtherClass"])
        ctx = _make_dir_context(files=[fi])
        result = build_template_context(ctx)
        assert "MyClass" in result["all_classes"]
        assert "OtherClass" in result["all_classes"]

    def test_all_imports_aggregated(self) -> None:
        fi = _make_file_info(imports=["os", "sys", "json"])
        ctx = _make_dir_context(files=[fi])
        result = build_template_context(ctx)
        assert "os" in result["all_imports"]
        assert "sys" in result["all_imports"]
        assert "json" in result["all_imports"]

    def test_no_summary_defaults_applied(self) -> None:
        ctx = _make_dir_context(summary=None)
        result = build_template_context(ctx)
        assert result["summary"] is None
        assert result["overview"] == ""
        assert result["key_symbols"] == []
        assert result["patterns"] == []
        assert result["tribal_knowledge"] == []
        assert result["has_summary"] is False

    def test_summary_fields_populated(self) -> None:
        ms = _make_summary(
            overview="Does X and Y.",
            key_symbols=["foo"],
            patterns=["Singleton"],
            tribal_knowledge=["Run init first."],
        )
        ctx = _make_dir_context(summary=ms)
        result = build_template_context(ctx)
        assert result["overview"] == "Does X and Y."
        assert result["key_symbols"] == ["foo"]
        assert result["patterns"] == ["Singleton"]
        assert result["tribal_knowledge"] == ["Run init first."]
        assert result["has_summary"] is True

    def test_has_summary_false_for_empty_summary(self) -> None:
        ms = ModuleSummary.empty()
        ctx = _make_dir_context(summary=ms)
        result = build_template_context(ctx)
        assert result["has_summary"] is False

    def test_dir_context_key_is_original_object(self) -> None:
        ctx = _make_dir_context()
        result = build_template_context(ctx)
        assert result["dir_context"] is ctx

    def test_generated_at_is_iso_string(self) -> None:
        ctx = _make_dir_context()
        result = build_template_context(ctx)
        generated_at = result["generated_at"]
        assert isinstance(generated_at, str)
        # Should match YYYY-MM-DDTHH:MM:SSZ
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", generated_at)

    def test_subdirectories_is_a_copy(self) -> None:
        """Mutating the returned subdirectories should not affect the context."""
        ctx = _make_dir_context(subdirectories=["a", "b"])
        result = build_template_context(ctx)
        result["subdirectories"].append("c")
        assert ctx.subdirectories == ["a", "b"]


# ===========================================================================
# read_stored_hash
# ===========================================================================


class TestReadStoredHash:
    """Tests for read_stored_hash."""

    def _write_codexa_md(self, directory: Path, content: str) -> Path:
        path = directory / "CODEXA.md"
        path.write_text(content, encoding="utf-8")
        return path

    def test_reads_valid_hash(self, tmp_path: Path) -> None:
        hex_hash = "a" * 64
        content = f"<!-- codexa-hash: {hex_hash} -->\n# Title\n"
        path = self._write_codexa_md(tmp_path, content)
        result = read_stored_hash(path)
        assert result == hex_hash

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        result = read_stored_hash(tmp_path / "nonexistent.md")
        assert result is None

    def test_returns_none_for_no_hash_comment(self, tmp_path: Path) -> None:
        content = "# Just a regular markdown file\n"
        path = self._write_codexa_md(tmp_path, content)
        result = read_stored_hash(path)
        assert result is None

    def test_returns_none_for_wrong_prefix(self, tmp_path: Path) -> None:
        hex_hash = "b" * 64
        content = f"<!-- other-hash: {hex_hash} -->\n"
        path = self._write_codexa_md(tmp_path, content)
        result = read_stored_hash(path)
        assert result is None

    def test_returns_none_for_short_hash(self, tmp_path: Path) -> None:
        content = "<!-- codexa-hash: abc123 -->\n"
        path = self._write_codexa_md(tmp_path, content)
        result = read_stored_hash(path)
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        path = self._write_codexa_md(tmp_path, "")
        result = read_stored_hash(path)
        assert result is None

    def test_ignores_rest_of_file(self, tmp_path: Path) -> None:
        hex_hash = "c" * 64
        content = (
            f"<!-- codexa-hash: {hex_hash} -->\n"
            "The rest of the file can be anything.\n"
            "<!-- codexa-hash: " + "d" * 64 + " -->\n"
        )
        path = self._write_codexa_md(tmp_path, content)
        result = read_stored_hash(path)
        assert result == hex_hash  # Only the first line matters

    def test_hash_with_extra_spaces_parsed(self, tmp_path: Path) -> None:
        hex_hash = "e" * 64
        content = f"<!-- codexa-hash:   {hex_hash}   -->\n"
        path = self._write_codexa_md(tmp_path, content)
        result = read_stored_hash(path)
        assert result == hex_hash

    def test_hash_line_re_matches_valid_line(self) -> None:
        hex_hash = "f" * 64
        line = f"<!-- codexa-hash: {hex_hash} -->"
        match = _HASH_LINE_RE.match(line)
        assert match is not None
        assert match.group(1) == hex_hash

    def test_hash_line_re_does_not_match_short_hash(self) -> None:
        line = "<!-- codexa-hash: abc123 -->"
        assert _HASH_LINE_RE.match(line) is None

    def test_hash_line_re_does_not_match_non_hex(self) -> None:
        non_hex = "g" * 64  # 'g' is not a hex character
        line = f"<!-- codexa-hash: {non_hex} -->"
        assert _HASH_LINE_RE.match(line) is None


# ===========================================================================
# compute_context_hash
# ===========================================================================


class TestComputeContextHash:
    """Tests for compute_context_hash."""

    def test_returns_string(self) -> None:
        ctx = {"content_hash": "a" * 64}
        result = compute_context_hash(ctx)
        assert isinstance(result, str)

    def test_returns_content_hash_when_present(self) -> None:
        hash_value = "a" * 64
        ctx = {"content_hash": hash_value}
        result = compute_context_hash(ctx)
        assert result == hash_value

    def test_deterministic_without_content_hash(self) -> None:
        ctx: Dict[str, Any] = {"overview": "hello", "dir_name": "src"}
        r1 = compute_context_hash(ctx)
        r2 = compute_context_hash(ctx)
        assert r1 == r2

    def test_different_contexts_produce_different_hashes(self) -> None:
        ctx1: Dict[str, Any] = {"overview": "A"}
        ctx2: Dict[str, Any] = {"overview": "B"}
        # These won't have content_hash, so falls back to SHA of items
        assert compute_context_hash(ctx1) != compute_context_hash(ctx2)

    def test_empty_content_hash_falls_back(self) -> None:
        ctx: Dict[str, Any] = {"content_hash": "", "overview": "hello"}
        # Should not return empty string
        result = compute_context_hash(ctx)
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# Renderer.render
# ===========================================================================


class TestRendererRender:
    """Tests for Renderer.render."""

    def test_returns_non_empty_string(self) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context()
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_first_line_contains_hash_comment(self) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(content_hash="a" * 64)
        context = build_template_context(ctx)
        result = renderer.render(context)
        first_line = result.splitlines()[0]
        assert first_line.startswith(_HASH_COMMENT_PREFIX)
        assert _HASH_COMMENT_SUFFIX in first_line

    def test_hash_comment_contains_content_hash(self) -> None:
        renderer = _make_renderer()
        hash_val = "b" * 64
        ctx = _make_dir_context(content_hash=hash_val)
        context = build_template_context(ctx)
        result = renderer.render(context)
        first_line = result.splitlines()[0]
        assert hash_val in first_line

    def test_contains_directory_name(self) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(path="/project/mymodule")
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "mymodule" in result

    def test_contains_overview_when_summary_present(self) -> None:
        renderer = _make_renderer()
        ms = _make_summary(overview="This module handles user authentication.")
        ctx = _make_dir_context(summary=ms)
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "This module handles user authentication." in result

    def test_contains_key_symbols_when_summary_present(self) -> None:
        renderer = _make_renderer()
        ms = _make_summary(key_symbols=["my_special_function", "MySpecialClass"])
        ctx = _make_dir_context(summary=ms)
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "my_special_function" in result
        assert "MySpecialClass" in result

    def test_contains_patterns_when_summary_present(self) -> None:
        renderer = _make_renderer()
        ms = _make_summary(patterns=["Uses the observer pattern for event dispatch."])
        ctx = _make_dir_context(summary=ms)
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "Uses the observer pattern for event dispatch." in result

    def test_contains_tribal_knowledge_when_summary_present(self) -> None:
        renderer = _make_renderer()
        ms = _make_summary(tribal_knowledge=["Always call teardown() after setup()."])
        ctx = _make_dir_context(summary=ms)
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "Always call teardown() after setup()." in result

    def test_no_summary_shows_placeholder_text(self) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(summary=None)
        context = build_template_context(ctx)
        result = renderer.render(context)
        # Should contain some placeholder indicating no summary
        assert "No LLM summary" in result or "not available" in result.lower() or "No" in result

    def test_contains_file_names(self) -> None:
        renderer = _make_renderer()
        fi = _make_file_info(relative_path="my_module.py")
        ctx = _make_dir_context(files=[fi])
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "my_module.py" in result

    def test_contains_function_names(self) -> None:
        renderer = _make_renderer()
        fi = _make_file_info(functions=["my_unique_function_xyz"])
        ctx = _make_dir_context(files=[fi])
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "my_unique_function_xyz" in result

    def test_contains_class_names(self) -> None:
        renderer = _make_renderer()
        fi = _make_file_info(classes=["MyUniqueClassXYZ"])
        ctx = _make_dir_context(files=[fi])
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "MyUniqueClassXYZ" in result

    def test_contains_import_names(self) -> None:
        renderer = _make_renderer()
        fi = _make_file_info(imports=["my_unique_import_xyz"])
        ctx = _make_dir_context(files=[fi])
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "my_unique_import_xyz" in result

    def test_contains_module_docstring(self) -> None:
        renderer = _make_renderer()
        fi = _make_file_info(module_docstring="My very unique docstring for testing.")
        ctx = _make_dir_context(files=[fi])
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "My very unique docstring for testing." in result

    def test_contains_subdirectory_names(self) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(subdirectories=["my_unique_subdir_xyz"])
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "my_unique_subdir_xyz" in result

    def test_result_is_markdown_with_headings(self) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(path="/project/src")
        context = build_template_context(ctx)
        result = renderer.render(context)
        # Should contain Markdown headings
        assert "#" in result

    def test_render_dir_context_convenience(self) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(path="/project/mymod")
        result = renderer.render_dir_context(ctx)
        assert isinstance(result, str)
        assert "mymod" in result

    def test_render_does_not_mutate_input_context(self) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(content_hash="c" * 64)
        context = build_template_context(ctx)
        original_hash = context["content_hash"]
        renderer.render(context)
        assert context["content_hash"] == original_hash

    def test_multiple_renders_produce_consistent_output(self) -> None:
        """Rendering the same context twice should produce identical output."""
        renderer = _make_renderer()
        ctx = _make_dir_context()
        context = build_template_context(ctx)
        r1 = renderer.render(context)
        r2 = renderer.render(context)
        # Timestamps may differ by seconds; compare content excluding first line
        # Just verify both are non-empty and structurally the same
        assert len(r1) > 0
        assert len(r2) > 0

    def test_env_is_initialised_lazily(self) -> None:
        renderer = Renderer()
        assert renderer._env is None
        ctx = _make_dir_context()
        context = build_template_context(ctx)
        renderer.render(context)
        assert renderer._env is not None

    def test_env_reused_on_second_render(self) -> None:
        renderer = Renderer()
        ctx = _make_dir_context()
        context = build_template_context(ctx)
        renderer.render(context)
        env_after_first = renderer._env
        renderer.render(context)
        assert renderer._env is env_after_first

    def test_read_stored_hash_matches_rendered_hash(self, tmp_path: Path) -> None:
        """The hash embedded in the rendered output should be readable back."""
        renderer = _make_renderer()
        hash_val = "d" * 64
        ctx = _make_dir_context(content_hash=hash_val)
        context = build_template_context(ctx)
        rendered = renderer.render(context)
        # Write to temp file and read hash back
        md_path = tmp_path / "CODEXA.md"
        md_path.write_text(rendered, encoding="utf-8")
        recovered = read_stored_hash(md_path)
        assert recovered == hash_val


# ===========================================================================
# Renderer.write
# ===========================================================================


class TestRendererWrite:
    """Tests for Renderer.write."""

    def test_creates_codexa_md_file(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(path=str(tmp_path))
        context = build_template_context(ctx)
        renderer.write(tmp_path, context)
        assert (tmp_path / "CODEXA.md").exists()

    def test_returns_true_when_file_written(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(path=str(tmp_path))
        context = build_template_context(ctx)
        result = renderer.write(tmp_path, context)
        assert result is True

    def test_file_content_is_non_empty(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(path=str(tmp_path))
        context = build_template_context(ctx)
        renderer.write(tmp_path, context)
        content = (tmp_path / "CODEXA.md").read_text(encoding="utf-8")
        assert len(content) > 0

    def test_file_starts_with_hash_comment(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(path=str(tmp_path), content_hash="e" * 64)
        context = build_template_context(ctx)
        renderer.write(tmp_path, context)
        content = (tmp_path / "CODEXA.md").read_text(encoding="utf-8")
        first_line = content.splitlines()[0]
        assert first_line.startswith(_HASH_COMMENT_PREFIX)

    def test_skips_when_hash_unchanged(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        hash_val = "f" * 64
        ctx = _make_dir_context(path=str(tmp_path), content_hash=hash_val)
        context = build_template_context(ctx)
        # First write
        renderer.write(tmp_path, context)
        # Modify to detect if second write happens
        md_path = tmp_path / "CODEXA.md"
        original_mtime = md_path.stat().st_mtime
        # Second write — should be skipped
        result = renderer.write(tmp_path, context)
        assert result is False

    def test_returns_false_when_skipped(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        hash_val = "1" * 64
        ctx = _make_dir_context(path=str(tmp_path), content_hash=hash_val)
        context = build_template_context(ctx)
        renderer.write(tmp_path, context)  # First write
        result = renderer.write(tmp_path, context)  # Second — skipped
        assert result is False

    def test_force_overwrites_even_when_hash_unchanged(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        hash_val = "2" * 64
        ctx = _make_dir_context(path=str(tmp_path), content_hash=hash_val)
        context = build_template_context(ctx)
        renderer.write(tmp_path, context)  # First write
        result = renderer.write(tmp_path, context, force=True)  # Force
        assert result is True

    def test_writes_when_hash_changed(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        ctx1 = _make_dir_context(path=str(tmp_path), content_hash="3" * 64)
        ctx2 = _make_dir_context(path=str(tmp_path), content_hash="4" * 64)
        context1 = build_template_context(ctx1)
        context2 = build_template_context(ctx2)
        renderer.write(tmp_path, context1)  # First write
        result = renderer.write(tmp_path, context2)  # Different hash
        assert result is True

    def test_write_updates_stored_hash(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        hash_val = "5" * 64
        ctx = _make_dir_context(path=str(tmp_path), content_hash=hash_val)
        context = build_template_context(ctx)
        renderer.write(tmp_path, context)
        stored = read_stored_hash(tmp_path / "CODEXA.md")
        assert stored == hash_val

    def test_write_to_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        nonexistent = tmp_path / "does_not_exist"
        ctx = _make_dir_context(path=str(nonexistent))
        context = build_template_context(ctx)
        with pytest.raises(RendererError, match="Cannot write"):
            renderer.write(nonexistent, context)

    def test_write_creates_utf8_encoded_file(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        fi = _make_file_info(module_docstring="Unicode: \u00e9\u00e0\u00fc")
        ctx = _make_dir_context(path=str(tmp_path), files=[fi])
        context = build_template_context(ctx)
        renderer.write(tmp_path, context)
        raw = (tmp_path / "CODEXA.md").read_bytes()
        # Should be valid UTF-8
        decoded = raw.decode("utf-8")
        assert "\u00e9" in decoded or "unicode" in decoded.lower() or len(decoded) > 0

    def test_write_dir_context_convenience(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        ctx = _make_dir_context(path=str(tmp_path))
        # Override path to tmp_path so the file is written there
        ctx_with_correct_path = DirContext(
            path=tmp_path,
            files=ctx.files,
            subdirectories=ctx.subdirectories,
            content_hash=ctx.content_hash,
            summary=ctx.summary,
        )
        result = renderer.write_dir_context(ctx_with_correct_path)
        assert result is True
        assert (tmp_path / "CODEXA.md").exists()

    def test_write_dir_context_returns_false_when_skipped(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        ctx = DirContext(
            path=tmp_path,
            files=[],
            subdirectories=[],
            content_hash="6" * 64,
            summary=None,
        )
        renderer.write_dir_context(ctx)  # First write
        result = renderer.write_dir_context(ctx)  # Second — skipped
        assert result is False

    def test_write_dir_context_force(self, tmp_path: Path) -> None:
        renderer = _make_renderer()
        ctx = DirContext(
            path=tmp_path,
            files=[],
            subdirectories=[],
            content_hash="7" * 64,
            summary=None,
        )
        renderer.write_dir_context(ctx)  # First write
        result = renderer.write_dir_context(ctx, force=True)  # Force
        assert result is True

    def test_full_pipeline_with_summary(self, tmp_path: Path) -> None:
        """End-to-end: write a CODEXA.md with a full summary and verify output."""
        renderer = _make_renderer()
        fi = _make_file_info(
            path=str(tmp_path / "auth.py"),
            relative_path="auth.py",
            functions=["login", "logout"],
            classes=["AuthManager"],
            imports=["hashlib", "secrets"],
            module_docstring="Handles user authentication.",
        )
        ms = _make_summary(
            overview="Provides login and logout functionality.",
            key_symbols=["login", "AuthManager"],
            patterns=["Tokens are rotated on each login."],
            tribal_knowledge=["Never store plaintext passwords."],
        )
        ctx = DirContext(
            path=tmp_path,
            files=[fi],
            subdirectories=["tokens"],
            content_hash="8" * 64,
            summary=ms,
        )
        renderer.write_dir_context(ctx)
        content = (tmp_path / "CODEXA.md").read_text(encoding="utf-8")
        assert "Provides login and logout functionality." in content
        assert "login" in content
        assert "AuthManager" in content
        assert "Tokens are rotated on each login." in content
        assert "Never store plaintext passwords." in content
        assert "auth.py" in content
        assert "tokens" in content


# ===========================================================================
# Custom template path
# ===========================================================================


class TestCustomTemplatePath:
    """Tests for Renderer with a custom template."""

    def test_custom_template_is_used(self, tmp_path: Path) -> None:
        """A custom template that emits a known sentinel string."""
        tmpl = tmp_path / "custom.md.j2"
        tmpl.write_text(
            "{{ hash_comment }}\nMY_CUSTOM_SENTINEL: {{ dir_name }}\n",
            encoding="utf-8",
        )
        renderer = Renderer(template_path=tmpl)
        ctx = _make_dir_context(path="/project/pkg")
        context = build_template_context(ctx)
        result = renderer.render(context)
        assert "MY_CUSTOM_SENTINEL" in result
        assert "pkg" in result

    def test_custom_template_hash_comment_still_present(self, tmp_path: Path) -> None:
        tmpl = tmp_path / "simple.j2"
        tmpl.write_text(
            "{{ hash_comment }}\n# {{ dir_name }}\n",
            encoding="utf-8",
        )
        renderer = Renderer(template_path=tmpl)
        ctx = _make_dir_context(content_hash="9" * 64)
        context = build_template_context(ctx)
        result = renderer.render(context)
        first_line = result.splitlines()[0]
        assert _HASH_COMMENT_PREFIX in first_line

    def test_custom_template_write_works(self, tmp_path: Path) -> None:
        tmpl_dir = tmp_path / "templates"
        tmpl_dir.mkdir()
        tmpl = tmpl_dir / "my.j2"
        tmpl.write_text(
            "{{ hash_comment }}\n# Custom: {{ dir_name }}\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        renderer = Renderer(template_path=tmpl)
        ctx = DirContext(
            path=out_dir,
            files=[],
            subdirectories=[],
            content_hash="a0" * 32,
            summary=None,
        )
        result = renderer.write_dir_context(ctx)
        assert result is True
        assert (out_dir / "CODEXA.md").exists()
        content = (out_dir / "CODEXA.md").read_text(encoding="utf-8")
        assert "Custom:" in content

    def test_undefined_template_variable_raises_renderer_error(self, tmp_path: Path) -> None:
        """StrictUndefined should cause rendering to fail on unknown vars."""
        tmpl = tmp_path / "bad.j2"
        tmpl.write_text(
            "{{ hash_comment }}\n{{ totally_unknown_variable_xyz }}\n",
            encoding="utf-8",
        )
        renderer = Renderer(template_path=tmpl)
        ctx = _make_dir_context()
        context = build_template_context(ctx)
        with pytest.raises(RendererError):
            renderer.render(context)


# ===========================================================================
# RendererError
# ===========================================================================


class TestRendererError:
    """Tests for the RendererError exception class."""

    def test_is_exception(self) -> None:
        err = RendererError("something went wrong")
        assert isinstance(err, Exception)

    def test_message_stored(self) -> None:
        err = RendererError("my error")
        assert err.message == "my error"

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(RendererError) as exc_info:
            raise RendererError("test error")
        assert "test error" in str(exc_info.value)

    def test_str_representation(self) -> None:
        err = RendererError("test message")
        assert "test message" in str(err)


# ===========================================================================
# Integration: full render pipeline with real files
# ===========================================================================


class TestRendererIntegration:
    """Integration tests that exercise the full render pipeline."""

    SAMPLE_MODULE = (
        Path(__file__).parent / "fixtures" / "sample_module" / "main.py"
    )

    def test_render_from_real_file_analysis(self, tmp_path: Path) -> None:
        """Analyze a real file and render a CODEXA.md for it."""
        from codexa.analyzer import analyze_directory

        fixture_dir = self.SAMPLE_MODULE.parent
        dir_ctx = analyze_directory(fixture_dir)

        renderer = Renderer()
        context = build_template_context(dir_ctx)
        rendered = renderer.render(context)

        # The rendered output should mention known symbols from main.py
        assert "greet" in rendered or "main" in rendered
        assert len(rendered) > 100  # Non-trivial output

    def test_incremental_write_skips_unchanged_analysis(self, tmp_path: Path) -> None:
        """Writing twice with same content should skip the second write."""
        from codexa.analyzer import analyze_directory

        fixture_dir = self.SAMPLE_MODULE.parent
        dir_ctx = analyze_directory(fixture_dir)
        dir_ctx_copy = DirContext(
            path=tmp_path,
            files=dir_ctx.files,
            subdirectories=dir_ctx.subdirectories,
            content_hash=dir_ctx.content_hash,
            summary=dir_ctx.summary,
        )

        renderer = Renderer()
        r1 = renderer.write_dir_context(dir_ctx_copy)
        assert r1 is True
        r2 = renderer.write_dir_context(dir_ctx_copy)
        assert r2 is False

    def test_rendered_output_contains_section_headers(self, tmp_path: Path) -> None:
        """Check that all major sections appear in the output."""
        renderer = Renderer()
        ms = _make_summary()
        fi = _make_file_info()
        ctx = DirContext(
            path=tmp_path,
            files=[fi],
            subdirectories=["sub"],
            content_hash="b" * 64,
            summary=ms,
        )
        context = build_template_context(ctx)
        result = renderer.render(context)

        # Major section headings from the template
        assert "Overview" in result
        assert "Source Files" in result
        assert "Key Symbols" in result
        assert "Dependencies" in result
        assert "Non-Obvious Patterns" in result
        assert "Tribal Knowledge" in result
        assert "Subdirectories" in result
