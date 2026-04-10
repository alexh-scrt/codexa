"""Unit tests for codexa.llm — LLM client, prompt builder, and response parser.

Covers:
  - :func:`~codexa.llm._build_summarization_prompt`: prompt structure and
    content for various directory contexts.
  - :func:`~codexa.llm._parse_summary_response`: valid JSON, malformed JSON,
    missing keys, wrong types, and markdown-fenced responses.
  - :class:`~codexa.llm.MockLLMClient`: deterministic summarization without
    API calls.
  - :class:`~codexa.llm.LLMClient`: lazy client initialization, retry logic
    (using a patched openai module), and error propagation.
  - :func:`~codexa.llm.create_llm_client`: factory function routing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from codexa.llm import (
    LLMClient,
    LLMError,
    MockLLMClient,
    BaseLLMClient,
    _build_summarization_prompt,
    _parse_summary_response,
    create_llm_client,
)
from codexa.models import DirContext, FileInfo, ModuleSummary


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_file_info(
    path: str = "/project/module.py",
    relative_path: str = "module.py",
    size_bytes: int = 512,
    content_hash: str = "abc123",
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


def _make_dir_context(
    path: str = "/project",
    files: Optional[List[FileInfo]] = None,
    subdirectories: Optional[List[str]] = None,
    content_hash: str = "deadbeef",
    summary: Optional[ModuleSummary] = None,
) -> DirContext:
    return DirContext(
        path=Path(path),
        files=files if files is not None else [],
        subdirectories=subdirectories if subdirectories is not None else [],
        content_hash=content_hash,
        summary=summary,
    )


def _valid_summary_json(
    overview: str = "This module handles X.",
    key_symbols: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    tribal_knowledge: Optional[List[str]] = None,
) -> str:
    return json.dumps({
        "overview": overview,
        "key_symbols": key_symbols if key_symbols is not None else ["foo", "Bar"],
        "patterns": patterns if patterns is not None else ["Uses singleton pattern."],
        "tribal_knowledge": tribal_knowledge if tribal_knowledge is not None else ["Always call init() first."],
    })


# ===========================================================================
# _build_summarization_prompt
# ===========================================================================


class TestBuildSummarizationPrompt:
    """Tests for _build_summarization_prompt."""

    def test_returns_non_empty_string(self) -> None:
        ctx = _make_dir_context()
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_includes_directory_path(self) -> None:
        ctx = _make_dir_context(path="/my/project/src")
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "/my/project/src" in prompt

    def test_includes_file_names(self) -> None:
        fi = _make_file_info(relative_path="utils.py")
        ctx = _make_dir_context(files=[fi])
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "utils.py" in prompt

    def test_includes_function_names(self) -> None:
        fi = _make_file_info(functions=["compute_hash", "walk_dir"])
        ctx = _make_dir_context(files=[fi])
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "compute_hash" in prompt
        assert "walk_dir" in prompt

    def test_includes_class_names(self) -> None:
        fi = _make_file_info(classes=["MyClient", "Config"])
        ctx = _make_dir_context(files=[fi])
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "MyClient" in prompt
        assert "Config" in prompt

    def test_includes_import_names(self) -> None:
        fi = _make_file_info(imports=["pathlib", "json"])
        ctx = _make_dir_context(files=[fi])
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "pathlib" in prompt
        assert "json" in prompt

    def test_includes_module_docstring(self) -> None:
        fi = _make_file_info(module_docstring="Handles authentication logic.")
        ctx = _make_dir_context(files=[fi])
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "Handles authentication logic." in prompt

    def test_no_docstring_shows_placeholder(self) -> None:
        fi = _make_file_info(module_docstring=None)
        ctx = _make_dir_context(files=[fi])
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "no docstring" in prompt

    def test_includes_subdirectories(self) -> None:
        ctx = _make_dir_context(subdirectories=["models", "views", "tests"])
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "models" in prompt
        assert "views" in prompt

    def test_empty_directory_mentions_no_files(self) -> None:
        ctx = _make_dir_context(files=[])
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "no Python" in prompt.lower() or "no python" in prompt.lower()

    def test_mentions_required_output_keys(self) -> None:
        ctx = _make_dir_context()
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "overview" in prompt
        assert "key_symbols" in prompt
        assert "patterns" in prompt
        assert "tribal_knowledge" in prompt

    def test_multiple_files_all_included(self) -> None:
        fi1 = _make_file_info(
            path="/p/a.py", relative_path="a.py", functions=["alpha"]
        )
        fi2 = _make_file_info(
            path="/p/b.py", relative_path="b.py", functions=["beta"]
        )
        ctx = _make_dir_context(files=[fi1, fi2])
        prompt = _build_summarization_prompt(ctx.to_dict())
        assert "a.py" in prompt
        assert "b.py" in prompt
        assert "alpha" in prompt
        assert "beta" in prompt

    def test_handles_raw_dict_directly(self) -> None:
        raw: Dict[str, Any] = {
            "path": "/raw/path",
            "files": [],
            "subdirectories": [],
            "content_hash": "abc",
            "summary": None,
        }
        prompt = _build_summarization_prompt(raw)
        assert "/raw/path" in prompt

    def test_unknown_path_fallback(self) -> None:
        prompt = _build_summarization_prompt({})
        assert "<unknown>" in prompt


# ===========================================================================
# _parse_summary_response
# ===========================================================================


class TestParseSummaryResponse:
    """Tests for _parse_summary_response."""

    def test_parses_valid_json(self) -> None:
        raw = _valid_summary_json()
        result = _parse_summary_response(raw)
        assert isinstance(result, dict)
        assert result["overview"] == "This module handles X."
        assert result["key_symbols"] == ["foo", "Bar"]
        assert result["patterns"] == ["Uses singleton pattern."]
        assert result["tribal_knowledge"] == ["Always call init() first."]

    def test_returns_all_four_keys(self) -> None:
        raw = _valid_summary_json()
        result = _parse_summary_response(raw)
        assert set(result.keys()) == {"overview", "key_symbols", "patterns", "tribal_knowledge"}

    def test_invalid_json_returns_defaults(self) -> None:
        result = _parse_summary_response("not valid json {{{")
        assert result["overview"] == ""
        assert result["key_symbols"] == []
        assert result["patterns"] == []
        assert result["tribal_knowledge"] == []

    def test_empty_string_returns_defaults(self) -> None:
        result = _parse_summary_response("")
        assert result["overview"] == ""
        assert result["key_symbols"] == []

    def test_json_array_returns_defaults(self) -> None:
        result = _parse_summary_response("[1, 2, 3]")
        assert result["overview"] == ""
        assert result["key_symbols"] == []

    def test_missing_overview_defaults_to_empty_string(self) -> None:
        raw = json.dumps({
            "key_symbols": ["foo"],
            "patterns": [],
            "tribal_knowledge": [],
        })
        result = _parse_summary_response(raw)
        assert result["overview"] == ""

    def test_missing_key_symbols_defaults_to_empty_list(self) -> None:
        raw = json.dumps({
            "overview": "Hello",
            "patterns": [],
            "tribal_knowledge": [],
        })
        result = _parse_summary_response(raw)
        assert result["key_symbols"] == []

    def test_missing_patterns_defaults_to_empty_list(self) -> None:
        raw = json.dumps({
            "overview": "Hello",
            "key_symbols": [],
            "tribal_knowledge": [],
        })
        result = _parse_summary_response(raw)
        assert result["patterns"] == []

    def test_missing_tribal_knowledge_defaults_to_empty_list(self) -> None:
        raw = json.dumps({
            "overview": "Hello",
            "key_symbols": [],
            "patterns": [],
        })
        result = _parse_summary_response(raw)
        assert result["tribal_knowledge"] == []

    def test_all_keys_missing_returns_all_defaults(self) -> None:
        result = _parse_summary_response("{}")
        assert result["overview"] == ""
        assert result["key_symbols"] == []
        assert result["patterns"] == []
        assert result["tribal_knowledge"] == []

    def test_non_list_key_symbols_becomes_empty_list(self) -> None:
        raw = json.dumps({
            "overview": "X",
            "key_symbols": "should-be-a-list",
            "patterns": [],
            "tribal_knowledge": [],
        })
        result = _parse_summary_response(raw)
        assert result["key_symbols"] == []

    def test_non_string_overview_coerced_to_string(self) -> None:
        raw = json.dumps({
            "overview": 42,
            "key_symbols": [],
            "patterns": [],
            "tribal_knowledge": [],
        })
        result = _parse_summary_response(raw)
        assert result["overview"] == "42"

    def test_list_items_coerced_to_strings(self) -> None:
        raw = json.dumps({
            "overview": "",
            "key_symbols": [1, 2, 3],
            "patterns": [],
            "tribal_knowledge": [],
        })
        result = _parse_summary_response(raw)
        assert result["key_symbols"] == ["1", "2", "3"]

    def test_none_items_in_list_excluded(self) -> None:
        raw = json.dumps({
            "overview": "",
            "key_symbols": ["foo", None, "bar"],
            "patterns": [],
            "tribal_knowledge": [],
        })
        result = _parse_summary_response(raw)
        assert None not in result["key_symbols"]
        assert "foo" in result["key_symbols"]
        assert "bar" in result["key_symbols"]

    def test_strips_markdown_code_fence_json(self) -> None:
        raw = "```json\n" + _valid_summary_json() + "\n```"
        result = _parse_summary_response(raw)
        assert result["overview"] == "This module handles X."

    def test_strips_plain_code_fence(self) -> None:
        raw = "```\n" + _valid_summary_json() + "\n```"
        result = _parse_summary_response(raw)
        assert result["overview"] == "This module handles X."

    def test_extra_keys_in_response_ignored(self) -> None:
        raw = json.dumps({
            "overview": "OK",
            "key_symbols": [],
            "patterns": [],
            "tribal_knowledge": [],
            "extra_future_key": "ignored",
        })
        result = _parse_summary_response(raw)
        assert "extra_future_key" not in result

    def test_empty_lists_are_preserved(self) -> None:
        raw = json.dumps({
            "overview": "Overview.",
            "key_symbols": [],
            "patterns": [],
            "tribal_knowledge": [],
        })
        result = _parse_summary_response(raw)
        assert result["key_symbols"] == []
        assert result["patterns"] == []
        assert result["tribal_knowledge"] == []

    def test_whitespace_around_json_is_tolerated(self) -> None:
        raw = "   " + _valid_summary_json() + "   "
        result = _parse_summary_response(raw)
        assert result["overview"] == "This module handles X."


# ===========================================================================
# MockLLMClient
# ===========================================================================


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    def test_is_subclass_of_base(self) -> None:
        client = MockLLMClient()
        assert isinstance(client, BaseLLMClient)

    def test_returns_module_summary(self) -> None:
        client = MockLLMClient()
        ctx = _make_dir_context()
        result = client.summarize_directory(ctx)
        assert isinstance(result, ModuleSummary)

    def test_overview_is_non_empty_string(self) -> None:
        client = MockLLMClient()
        ctx = _make_dir_context(path="/my/pkg")
        result = client.summarize_directory(ctx)
        assert isinstance(result.overview, str)
        assert len(result.overview) > 0

    def test_overview_contains_prefix(self) -> None:
        client = MockLLMClient(overview_prefix="[TEST] ")
        ctx = _make_dir_context()
        result = client.summarize_directory(ctx)
        assert result.overview.startswith("[TEST] ")

    def test_overview_contains_directory_name(self) -> None:
        client = MockLLMClient()
        ctx = _make_dir_context(path="/project/analytics")
        result = client.summarize_directory(ctx)
        assert "analytics" in result.overview

    def test_overview_mentions_file_count(self) -> None:
        fi1 = _make_file_info(path="/p/a.py", relative_path="a.py")
        fi2 = _make_file_info(path="/p/b.py", relative_path="b.py")
        client = MockLLMClient()
        ctx = _make_dir_context(files=[fi1, fi2])
        result = client.summarize_directory(ctx)
        assert "2" in result.overview

    def test_key_symbols_derived_from_context(self) -> None:
        fi = _make_file_info(
            functions=["process", "validate"],
            classes=["Processor"],
        )
        client = MockLLMClient()
        ctx = _make_dir_context(files=[fi])
        result = client.summarize_directory(ctx)
        for name in ["process", "validate", "Processor"]:
            assert name in result.key_symbols

    def test_key_symbols_capped_at_ten(self) -> None:
        fi = _make_file_info(
            functions=[f"fn{i}" for i in range(8)],
            classes=[f"Cls{i}" for i in range(8)],
        )
        client = MockLLMClient()
        ctx = _make_dir_context(files=[fi])
        result = client.summarize_directory(ctx)
        assert len(result.key_symbols) <= 10

    def test_patterns_is_empty_list(self) -> None:
        client = MockLLMClient()
        ctx = _make_dir_context()
        result = client.summarize_directory(ctx)
        assert result.patterns == []

    def test_tribal_knowledge_is_empty_list(self) -> None:
        client = MockLLMClient()
        ctx = _make_dir_context()
        result = client.summarize_directory(ctx)
        assert result.tribal_knowledge == []

    def test_call_count_increments(self) -> None:
        client = MockLLMClient()
        ctx = _make_dir_context()
        assert client.call_count == 0
        client.summarize_directory(ctx)
        assert client.call_count == 1
        client.summarize_directory(ctx)
        assert client.call_count == 2

    def test_last_context_updated(self) -> None:
        client = MockLLMClient()
        ctx1 = _make_dir_context(path="/a")
        ctx2 = _make_dir_context(path="/b")
        assert client.last_context is None
        client.summarize_directory(ctx1)
        assert client.last_context is ctx1
        client.summarize_directory(ctx2)
        assert client.last_context is ctx2

    def test_summarize_directory_dict_returns_dict(self) -> None:
        client = MockLLMClient()
        ctx = _make_dir_context()
        result = client.summarize_directory_dict(ctx.to_dict())
        assert isinstance(result, dict)
        assert "overview" in result
        assert "key_symbols" in result
        assert "patterns" in result
        assert "tribal_knowledge" in result

    def test_empty_directory_does_not_raise(self) -> None:
        client = MockLLMClient()
        ctx = _make_dir_context(files=[], subdirectories=[])
        result = client.summarize_directory(ctx)
        assert isinstance(result, ModuleSummary)

    def test_default_overview_prefix(self) -> None:
        client = MockLLMClient()
        ctx = _make_dir_context()
        result = client.summarize_directory(ctx)
        assert result.overview.startswith("[MOCK] ")


# ===========================================================================
# LLMClient — unit tests with mocked openai
# ===========================================================================


class TestLLMClient:
    """Tests for LLMClient with openai patched out."""

    def _make_mock_response(self, content: str) -> MagicMock:
        """Build a fake openai.ChatCompletion response object."""
        choice = MagicMock()
        choice.message.content = content
        response = MagicMock()
        response.choices = [choice]
        return response

    def _make_client(
        self,
        api_key: str = "sk-test",
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        max_retries: int = 0,
        retry_delay: float = 0.0,
    ) -> LLMClient:
        return LLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def test_is_subclass_of_base(self) -> None:
        client = self._make_client()
        assert isinstance(client, BaseLLMClient)

    def test_lazy_init_client_is_none_before_first_call(self) -> None:
        client = self._make_client()
        assert client._client is None

    def test_summarize_directory_returns_module_summary(self) -> None:
        client = self._make_client(max_retries=0)
        ctx = _make_dir_context()
        fake_response = self._make_mock_response(_valid_summary_json())

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = fake_response

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            result = client.summarize_directory(ctx)

        assert isinstance(result, ModuleSummary)
        assert result.overview == "This module handles X."
        assert result.key_symbols == ["foo", "Bar"]

    def test_summarize_directory_uses_correct_model(self) -> None:
        client = self._make_client(model="gpt-4o", max_retries=0)
        ctx = _make_dir_context()
        fake_response = self._make_mock_response(_valid_summary_json())

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = fake_response

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            client.summarize_directory(ctx)

        call_kwargs = mock_openai_instance.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("model") == "gpt-4o" or call_kwargs.args[0] == "gpt-4o"

    def test_summarize_directory_passes_max_tokens(self) -> None:
        client = self._make_client(max_tokens=256, max_retries=0)
        ctx = _make_dir_context()
        fake_response = self._make_mock_response(_valid_summary_json())

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = fake_response

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            client.summarize_directory(ctx)

        call_kwargs = mock_openai_instance.chat.completions.create.call_args
        # max_tokens can be in kwargs
        assert call_kwargs.kwargs.get("max_tokens") == 256

    def test_summarize_directory_handles_malformed_response(self) -> None:
        client = self._make_client(max_retries=0)
        ctx = _make_dir_context()
        fake_response = self._make_mock_response("not json")

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = fake_response

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            result = client.summarize_directory(ctx)

        # Should return an empty-default summary rather than raising
        assert isinstance(result, ModuleSummary)
        assert result.overview == ""
        assert result.key_symbols == []

    def test_api_failure_raises_llm_error(self) -> None:
        client = self._make_client(max_retries=0)
        ctx = _make_dir_context()

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.side_effect = RuntimeError("Network error")

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            with pytest.raises(LLMError, match="failed"):
                client.summarize_directory(ctx)

    def test_retry_logic_succeeds_on_second_attempt(self) -> None:
        client = self._make_client(max_retries=1, retry_delay=0.0)
        ctx = _make_dir_context()
        fake_response = self._make_mock_response(_valid_summary_json())

        mock_openai_instance = MagicMock()
        # First call fails, second succeeds
        mock_openai_instance.chat.completions.create.side_effect = [
            RuntimeError("Transient error"),
            fake_response,
        ]

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            with patch("codexa.llm.time.sleep"):  # Don't actually sleep
                result = client.summarize_directory(ctx)

        assert isinstance(result, ModuleSummary)
        assert result.overview == "This module handles X."
        assert mock_openai_instance.chat.completions.create.call_count == 2

    def test_all_retries_exhausted_raises_llm_error(self) -> None:
        client = self._make_client(max_retries=2, retry_delay=0.0)
        ctx = _make_dir_context()

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.side_effect = RuntimeError("Always fails")

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            with patch("codexa.llm.time.sleep"):
                with pytest.raises(LLMError):
                    client.summarize_directory(ctx)

        assert mock_openai_instance.chat.completions.create.call_count == 3  # 1 + 2 retries

    def test_get_client_raises_llm_error_when_openai_missing(self) -> None:
        client = self._make_client()
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises((LLMError, ImportError)):
                client._get_client()

    def test_client_reuses_instance_across_calls(self) -> None:
        client = self._make_client(max_retries=0)
        ctx = _make_dir_context()
        fake_response = self._make_mock_response(_valid_summary_json())

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = fake_response

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            client.summarize_directory(ctx)
            client.summarize_directory(ctx)

        # Both calls use the same mocked instance (no second initialization)
        assert mock_openai_instance.chat.completions.create.call_count == 2

    def test_none_message_content_returns_empty_summary(self) -> None:
        client = self._make_client(max_retries=0)
        ctx = _make_dir_context()

        choice = MagicMock()
        choice.message.content = None
        fake_response = MagicMock()
        fake_response.choices = [choice]

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = fake_response

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            result = client.summarize_directory(ctx)

        assert isinstance(result, ModuleSummary)

    def test_summarize_directory_dict_works_via_base(self) -> None:
        client = self._make_client(max_retries=0)
        ctx = _make_dir_context()
        fake_response = self._make_mock_response(_valid_summary_json())

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = fake_response

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            result = client.summarize_directory_dict(ctx.to_dict())

        assert isinstance(result, dict)
        assert result["overview"] == "This module handles X."

    def test_messages_include_system_and_user(self) -> None:
        client = self._make_client(max_retries=0)
        ctx = _make_dir_context()
        fake_response = self._make_mock_response(_valid_summary_json())

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = fake_response

        with patch("codexa.llm.LLMClient._get_client", return_value=mock_openai_instance):
            client.summarize_directory(ctx)

        call_kwargs = mock_openai_instance.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_base_url_stored_correctly(self) -> None:
        client = self._make_client(base_url="https://my.endpoint/v1")
        assert client.base_url == "https://my.endpoint/v1"

    def test_temperature_stored_correctly(self) -> None:
        client = self._make_client(temperature=0.5)
        assert client.temperature == 0.5


# ===========================================================================
# LLMError
# ===========================================================================


class TestLLMError:
    """Tests for the LLMError exception class."""

    def test_is_exception(self) -> None:
        err = LLMError("something went wrong")
        assert isinstance(err, Exception)

    def test_message_stored(self) -> None:
        err = LLMError("my error")
        assert err.message == "my error"

    def test_cause_stored(self) -> None:
        cause = ValueError("root cause")
        err = LLMError("wrapper", cause=cause)
        assert err.cause is cause

    def test_cause_defaults_to_none(self) -> None:
        err = LLMError("no cause")
        assert err.cause is None

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(LLMError) as exc_info:
            raise LLMError("test error")
        assert "test error" in str(exc_info.value)


# ===========================================================================
# create_llm_client factory
# ===========================================================================


class TestCreateLLMClient:
    """Tests for the create_llm_client factory function."""

    def test_returns_llm_client_by_default(self) -> None:
        client = create_llm_client(api_key="sk-test")
        assert isinstance(client, LLMClient)

    def test_returns_mock_client_when_mock_true(self) -> None:
        client = create_llm_client(mock=True)
        assert isinstance(client, MockLLMClient)

    def test_llm_client_has_correct_model(self) -> None:
        client = create_llm_client(api_key="sk-test", model="gpt-4o")
        assert isinstance(client, LLMClient)
        assert client.model == "gpt-4o"

    def test_llm_client_has_correct_max_tokens(self) -> None:
        client = create_llm_client(api_key="sk-test", max_tokens=2048)
        assert isinstance(client, LLMClient)
        assert client.max_tokens == 2048

    def test_llm_client_has_correct_base_url(self) -> None:
        client = create_llm_client(
            api_key="sk-test", base_url="https://custom.endpoint/v1"
        )
        assert isinstance(client, LLMClient)
        assert client.base_url == "https://custom.endpoint/v1"

    def test_llm_client_base_url_none_by_default(self) -> None:
        client = create_llm_client(api_key="sk-test")
        assert isinstance(client, LLMClient)
        assert client.base_url is None

    def test_mock_client_is_base_llm_client_subclass(self) -> None:
        client = create_llm_client(mock=True)
        assert isinstance(client, BaseLLMClient)

    def test_real_client_is_base_llm_client_subclass(self) -> None:
        client = create_llm_client(api_key="sk-test")
        assert isinstance(client, BaseLLMClient)

    def test_temperature_forwarded(self) -> None:
        client = create_llm_client(api_key="sk-test", temperature=0.7)
        assert isinstance(client, LLMClient)
        assert client.temperature == 0.7
