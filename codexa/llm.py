"""OpenAI-compatible LLM client for module summarization.

This module wraps the OpenAI Python SDK to provide codexa-specific
summarization functionality:

  - Given a populated :class:`~codexa.models.DirContext` (or its dict
    equivalent), build a structured prompt and call the chat completions
    API.
  - Parse the response into a :class:`~codexa.models.ModuleSummary` with
    a plain-text overview, a list of key symbols, detected patterns, and
    tribal knowledge hints.
  - Support any OpenAI-compatible endpoint via a configurable ``base_url``.
  - Provide a ``MockLLMClient`` for use in tests that avoids real API calls.

Design notes:
  - The OpenAI client is instantiated lazily so that import-time errors
    are avoided when the package is used without an API key (e.g. during
    ``codexa clean`` or ``codexa preview`` with a cached result).
  - All JSON parsing is defensive: missing keys fall back to safe defaults
    so a malformed LLM response never crashes the pipeline.
  - Token budgeting is the caller's responsibility via ``max_tokens``;
    this module just forwards the value to the API.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from codexa.models import DirContext, ModuleSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Raised when an LLM API call fails in an unrecoverable way.

    Attributes:
        message: Human-readable description of the failure.
        cause: The underlying exception, if any.
    """

    def __init__(self, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:  # pragma: no cover
        if self.cause is not None:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior software engineer helping to document a codebase for "
    "AI coding agents and new developers. Your goal is to produce concise, "
    "accurate, and actionable documentation.\n"
    "\n"
    "Rules:\n"
    "  1. Respond ONLY with a single valid JSON object — no markdown, no prose "
    "outside the JSON.\n"
    "  2. All string values must be plain text (no markdown formatting inside "
    "JSON strings).\n"
    "  3. Keep the 'overview' to 2-4 sentences maximum.\n"
    "  4. List only the most important symbols in 'key_symbols' (max 10).\n"
    "  5. Focus 'patterns' on non-obvious design decisions, not obvious ones.\n"
    "  6. 'tribal_knowledge' entries should answer 'why' questions a new "
    "developer would have."
)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_summarization_prompt(dir_context_dict: Dict[str, Any]) -> str:
    """Build a structured summarization prompt from a directory context dict.

    The prompt is designed to elicit a JSON response with four keys:
    ``overview``, ``key_symbols``, ``patterns``, and ``tribal_knowledge``.

    Args:
        dir_context_dict: A plain-dict representation of a
            :class:`~codexa.models.DirContext` (e.g. from
            :meth:`~codexa.models.DirContext.to_dict`).

    Returns:
        A prompt string ready to send as the user message to the LLM.
    """
    path = dir_context_dict.get("path", "<unknown>")
    files: List[Dict[str, Any]] = dir_context_dict.get("files", [])
    subdirectories: List[str] = dir_context_dict.get("subdirectories", [])

    # Build per-file summaries
    file_blocks: List[str] = []
    for f in files:
        fname = str(f.get("relative_path") or f.get("path") or "<unknown>")
        docstring = (f.get("module_docstring") or "").strip() or "(no docstring)"
        functions = f.get("functions", []) or []
        classes = f.get("classes", []) or []
        imports = f.get("imports", []) or []
        size = f.get("size_bytes", 0)

        # Truncate long lists to keep the prompt manageable
        fn_str = ", ".join(functions[:15]) or "none"
        cls_str = ", ".join(classes[:10]) or "none"
        imp_str = ", ".join(imports[:15]) or "none"

        block = (
            f"  File: {fname} ({size} bytes)\n"
            f"    Docstring  : {docstring}\n"
            f"    Functions  : {fn_str}\n"
            f"    Classes    : {cls_str}\n"
            f"    Imports    : {imp_str}"
        )
        file_blocks.append(block)

    files_section = (
        "\n".join(file_blocks)
        if file_blocks
        else "  (no Python source files in this directory)"
    )

    subdirs_section = (
        ", ".join(subdirectories[:20]) if subdirectories else "(none)"
    )

    prompt = (
        f"Analyze the following Python module / directory and produce structured "
        f"documentation for it.\n"
        f"\n"
        f"Directory path : {path}\n"
        f"Subdirectories : {subdirs_section}\n"
        f"\n"
        f"Source files:\n"
        f"{files_section}\n"
        f"\n"
        f"Return a JSON object with EXACTLY these four keys:\n"
        f'  "overview"          : string  — 2-4 sentence plain-English description '
        f"of this module's overall purpose and responsibilities.\n"
        f'  "key_symbols"       : array of strings — the most important function / '
        f"class names a developer should know (max 10 items).\n"
        f'  "patterns"          : array of strings — non-obvious design patterns, '
        f"idioms, or gotchas that are not apparent from reading the names alone.\n"
        f'  "tribal_knowledge"  : array of strings — contextual hints, historical '
        f"decisions, or \"why is it done this way?\" explanations for new "
        f"contributors.\n"
        f"\n"
        f"If a section has nothing meaningful to report, use an empty array [] or "
        f"an empty string \"\"."
    )
    return prompt


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_summary_response(raw: str) -> Dict[str, Any]:
    """Parse a raw JSON string from the LLM into a validated summary dict.

    Applies defensive defaults for any missing or incorrectly-typed keys so
    that a malformed response never propagates as an exception.

    Args:
        raw: Raw text returned by the LLM (expected to be a JSON object).

    Returns:
        A dict with keys ``overview`` (str), ``key_symbols`` (List[str]),
        ``patterns`` (List[str]), and ``tribal_knowledge`` (List[str]).
    """
    defaults: Dict[str, Any] = {
        "overview": "",
        "key_symbols": [],
        "patterns": [],
        "tribal_knowledge": [],
    }

    # Strip potential markdown code fences that some models add
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")].rstrip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM response is not valid JSON (error: %s). Using empty defaults.", exc
        )
        logger.debug("Raw LLM response was: %r", raw[:500])
        return dict(defaults)

    if not isinstance(data, dict):
        logger.warning(
            "LLM returned a JSON value that is not an object (%s). "
            "Using empty defaults.",
            type(data).__name__,
        )
        return dict(defaults)

    result: Dict[str, Any] = {}

    # overview — must be a string
    overview = data.get("overview", "")
    result["overview"] = str(overview) if overview is not None else ""

    # key_symbols, patterns, tribal_knowledge — must be lists of strings
    for list_key in ("key_symbols", "patterns", "tribal_knowledge"):
        raw_value = data.get(list_key, [])
        if isinstance(raw_value, list):
            result[list_key] = [
                str(item) for item in raw_value if item is not None
            ]
        else:
            logger.warning(
                "LLM returned non-list value for '%s' (%s); using []",
                list_key,
                type(raw_value).__name__,
            )
            result[list_key] = []

    return result


# ---------------------------------------------------------------------------
# Base protocol / interface
# ---------------------------------------------------------------------------


class BaseLLMClient:
    """Abstract base for LLM client implementations.

    Subclasses must implement :meth:`summarize_directory`.
    """

    def summarize_directory(
        self,
        dir_context: DirContext,
    ) -> ModuleSummary:
        """Summarize a directory context and return a :class:`~codexa.models.ModuleSummary`.

        Args:
            dir_context: The populated directory context to summarize.

        Returns:
            A :class:`~codexa.models.ModuleSummary` with all four fields
            populated (some may be empty if the LLM has nothing to report).

        Raises:
            LLMError: If the API call fails.
            NotImplementedError: If the subclass has not overridden this method.
        """
        raise NotImplementedError  # pragma: no cover

    def summarize_directory_dict(
        self,
        dir_context_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize a directory context supplied as a plain dict.

        Convenience wrapper that constructs a :class:`~codexa.models.DirContext`
        from *dir_context_dict*, calls :meth:`summarize_directory`, and returns
        the summary as a dict.

        Args:
            dir_context_dict: A dict representation of a
                :class:`~codexa.models.DirContext`.

        Returns:
            A plain dict with keys ``overview``, ``key_symbols``,
            ``patterns``, and ``tribal_knowledge``.

        Raises:
            LLMError: If the API call fails.
        """
        ctx = DirContext.from_dict(dir_context_dict)
        summary = self.summarize_directory(ctx)
        return summary.to_dict()


# ---------------------------------------------------------------------------
# Real OpenAI-backed client
# ---------------------------------------------------------------------------


class LLMClient(BaseLLMClient):
    """OpenAI-compatible LLM client for codexa module summarization.

    Wraps the ``openai`` Python SDK and communicates with any
    OpenAI-compatible chat completions endpoint.  The underlying
    ``openai.OpenAI`` instance is created lazily on the first API call so
    that the class can be instantiated without a network connection.

    Args:
        api_key: API key for the LLM backend.  If empty, the OpenAI SDK
            will fall back to the ``OPENAI_API_KEY`` environment variable.
        model: Model identifier, e.g. ``"gpt-4o-mini"``.
        base_url: Optional custom endpoint for OpenAI-compatible servers
            (e.g. a local Ollama instance or Azure OpenAI).  ``None`` means
            use the default OpenAI endpoint.
        max_tokens: Maximum tokens to request in each completion response.
        temperature: Sampling temperature (0.0 – 2.0).  Lower values
            produce more deterministic output; recommended ≤ 0.3 for
            documentation tasks.
        max_retries: Number of times to retry on transient API errors
            (rate limits, server errors).  Defaults to 2.
        retry_delay: Base delay in seconds between retries.  Actual delay
            uses exponential back-off: ``retry_delay * (2 ** attempt)``.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Any = None  # Lazily initialised

    # ------------------------------------------------------------------
    # Lazy client initialisation
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return a lazily-initialised ``openai.OpenAI`` client instance.

        Raises:
            LLMError: If the ``openai`` package is not installed.
        """
        if self._client is None:
            try:
                import openai  # noqa: PLC0415
            except ImportError as exc:
                raise LLMError(
                    "The 'openai' package is required for LLM summarization. "
                    "Install it with: pip install openai",
                    cause=exc,
                ) from exc

            kwargs: Dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url

            try:
                self._client = openai.OpenAI(**kwargs)
            except Exception as exc:
                raise LLMError(
                    f"Failed to initialise the OpenAI client: {exc}",
                    cause=exc,
                ) from exc

        return self._client

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def _call_api(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Make a chat completions API call with retry logic.

        Args:
            messages: A list of message dicts with ``role`` and ``content``
                keys, as expected by the OpenAI chat completions API.

        Returns:
            The raw string content of the first choice's message.

        Raises:
            LLMError: If all retry attempts fail.
        """
        client = self._get_client()
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                return content or "{}"

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                is_last_attempt = attempt == self.max_retries
                if is_last_attempt:
                    break

                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "LLM API call failed (attempt %d/%d): %s. "
                    "Retrying in %.1f seconds…",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise LLMError(
            f"LLM API call failed after {self.max_retries + 1} attempt(s): "
            f"{last_exc}",
            cause=last_exc,
        ) from last_exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def summarize_directory(
        self,
        dir_context: DirContext,
    ) -> ModuleSummary:
        """Call the LLM to summarize a :class:`~codexa.models.DirContext`.

        Builds a structured prompt from *dir_context*, sends it to the
        configured model, parses the JSON response, and returns a
        :class:`~codexa.models.ModuleSummary`.

        Args:
            dir_context: The populated directory context to summarize.
                The ``summary`` field is ignored (it will be overwritten).

        Returns:
            A fully-populated :class:`~codexa.models.ModuleSummary`.

        Raises:
            LLMError: If the API call fails after all retries.
        """
        logger.debug(
            "Summarizing directory: %s (%d file(s))",
            dir_context.path,
            dir_context.file_count,
        )

        prompt = _build_summarization_prompt(dir_context.to_dict())

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw = self._call_api(messages)
        logger.debug("Raw LLM response for %s: %r", dir_context.path, raw[:200])

        parsed = _parse_summary_response(raw)
        summary = ModuleSummary.from_dict(parsed)

        logger.info(
            "Summarized %s: overview=%d chars, key_symbols=%d, "
            "patterns=%d, tribal_knowledge=%d",
            dir_context.path,
            len(summary.overview),
            len(summary.key_symbols),
            len(summary.patterns),
            len(summary.tribal_knowledge),
        )
        return summary


# ---------------------------------------------------------------------------
# Mock client for testing / dry-run
# ---------------------------------------------------------------------------


class MockLLMClient(BaseLLMClient):
    """A deterministic mock LLM client that returns canned summaries.

    Useful for unit tests and ``--dry-run`` execution paths where real API
    calls must be avoided.  Returns a predictable :class:`~codexa.models.ModuleSummary`
    derived from the directory's own metadata without making any network
    requests.

    Args:
        overview_prefix: Optional string prepended to the generated overview
            to make mock output identifiable in test assertions.
    """

    def __init__(self, overview_prefix: str = "[MOCK] ") -> None:
        self.overview_prefix = overview_prefix
        # Track calls for assertion in tests
        self.call_count: int = 0
        self.last_context: Optional[DirContext] = None

    def summarize_directory(
        self,
        dir_context: DirContext,
    ) -> ModuleSummary:
        """Return a deterministic mock summary derived from *dir_context*.

        The summary contains:
        - An ``overview`` listing the directory name and file count.
        - ``key_symbols`` taken directly from :attr:`~codexa.models.DirContext.all_functions`
          and :attr:`~codexa.models.DirContext.all_classes` (up to 10 each).
        - Empty ``patterns`` and ``tribal_knowledge`` lists.

        Args:
            dir_context: The directory context to summarize (read-only).

        Returns:
            A :class:`~codexa.models.ModuleSummary` instance.
        """
        self.call_count += 1
        self.last_context = dir_context

        file_count = dir_context.file_count
        dir_name = dir_context.name or str(dir_context.path)

        overview = (
            f"{self.overview_prefix}Directory '{dir_name}' contains "
            f"{file_count} Python source file(s). "
            f"This is a mock summary generated without an LLM API call."
        )

        key_symbols = (
            dir_context.all_functions[:10] + dir_context.all_classes[:10]
        )[:10]

        return ModuleSummary(
            overview=overview,
            key_symbols=key_symbols,
            patterns=[],
            tribal_knowledge=[],
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def create_llm_client(
    api_key: str = "",
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    mock: bool = False,
) -> BaseLLMClient:
    """Construct and return an appropriate LLM client.

    This factory function centralises client creation and makes it easy for
    callers to swap between the real and mock implementations based on
    configuration flags (e.g. ``--dry-run``).

    Args:
        api_key: API key for the LLM backend.
        model: Model identifier string.
        base_url: Optional custom endpoint URL.
        max_tokens: Maximum tokens per completion.
        temperature: Sampling temperature.
        mock: If ``True``, return a :class:`MockLLMClient` instead of a
            real :class:`LLMClient`.

    Returns:
        A :class:`BaseLLMClient` subclass instance ready for use.
    """
    if mock:
        logger.debug("Using MockLLMClient (mock=True).")
        return MockLLMClient()

    logger.debug("Creating LLMClient with model=%r, base_url=%r.", model, base_url)
    return LLMClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )
