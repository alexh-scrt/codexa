"""OpenAI-compatible LLM client for module summarization.

This module wraps the OpenAI Python SDK to provide codexa-specific
summarization functionality:
  - Given a populated DirContext (or its raw dict equivalent), build a
    structured prompt and call the chat completions API.
  - Parse the response into a ModuleSummary with a plain-text overview,
    a list of key symbols, detected patterns, and tribal knowledge hints.
  - Support any OpenAI-compatible endpoint via configurable ``base_url``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel used when the LLM backend is not available / not configured.
_UNAVAILABLE = "[LLM summarization not yet wired — implement in phase 5]"


class LLMError(Exception):
    """Raised when an LLM API call fails in an unrecoverable way."""


class LLMClient:
    """Thin wrapper around the OpenAI chat completions API.

    Args:
        api_key: OpenAI-compatible API key.
        model: Model identifier, e.g. ``"gpt-4o"``.
        base_url: Optional custom endpoint for OpenAI-compatible servers.
        max_tokens: Maximum tokens to request in each completion.
        temperature: Sampling temperature (0.0 – 2.0).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Any = None  # Lazily initialised

    def _get_client(self) -> Any:
        """Return a lazily-initialised OpenAI client instance."""
        if self._client is None:
            try:
                import openai  # noqa: PLC0415

                kwargs: Dict[str, Any] = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = openai.OpenAI(**kwargs)
            except ImportError as exc:
                raise LLMError(
                    "The 'openai' package is required for LLM summarization. "
                    "Install it with: pip install openai"
                ) from exc
        return self._client

    def summarize_directory(
        self,
        dir_context_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call the LLM to summarize a directory context.

        Args:
            dir_context_dict: A dict representation of a ``DirContext``
                (produced by ``dataclasses.asdict`` or equivalent).

        Returns:
            A dict with keys:
              - ``overview``: str — High-level purpose description.
              - ``key_symbols``: List[str] — Notable functions/classes.
              - ``patterns``: List[str] — Non-obvious patterns or gotchas.
              - ``tribal_knowledge``: List[str] — Contextual hints.

        Raises:
            LLMError: On API or parsing failure.
        """
        prompt = _build_summarization_prompt(dir_context_dict)
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior software engineer helping to document "
                            "a codebase. Respond only with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise LLMError(f"LLM API call failed: {exc}") from exc

        raw_content = response.choices[0].message.content or "{}"
        return _parse_summary_response(raw_content)


def _build_summarization_prompt(dir_context_dict: Dict[str, Any]) -> str:
    """Build a structured summarization prompt from a directory context dict.

    Args:
        dir_context_dict: Directory context data.

    Returns:
        A prompt string ready to send to the LLM.
    """
    path = dir_context_dict.get("path", "<unknown>")
    files = dir_context_dict.get("files", [])

    file_summaries: List[str] = []
    for f in files:
        fname = str(f.get("relative_path") or f.get("path", "<unknown>"))
        docstring = f.get("module_docstring") or "(no docstring)"
        functions = ", ".join(f.get("functions", [])[:10]) or "none"
        classes = ", ".join(f.get("classes", [])[:10]) or "none"
        imports = ", ".join(f.get("imports", [])[:10]) or "none"
        file_summaries.append(
            f"File: {fname}\n"
            f"  Docstring: {docstring}\n"
            f"  Functions: {functions}\n"
            f"  Classes: {classes}\n"
            f"  Imports: {imports}"
        )

    files_block = "\n\n".join(file_summaries) if file_summaries else "(no Python files)"

    return (
        f"Analyze the following directory and its Python source files.\n"
        f"Directory: {path}\n\n"
        f"{files_block}\n\n"
        f"Return a JSON object with these keys:\n"
        f'  "overview": string — 2-4 sentence description of this module\'s purpose.\n'
        f'  "key_symbols": array of strings — the most important functions/classes.\n'
        f'  "patterns": array of strings — non-obvious patterns, idioms, or gotchas.\n'
        f'  "tribal_knowledge": array of strings — hints a new developer should know.\n'
    )


def _parse_summary_response(raw: str) -> Dict[str, Any]:
    """Parse a raw JSON string from the LLM into a summary dict.

    Falls back to empty-list defaults for any missing keys.

    Args:
        raw: Raw JSON string from the LLM.

    Returns:
        A dict with keys ``overview``, ``key_symbols``, ``patterns``,
        and ``tribal_knowledge``.
    """
    defaults: Dict[str, Any] = {
        "overview": "",
        "key_symbols": [],
        "patterns": [],
        "tribal_knowledge": [],
    }
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.warning("LLM returned non-dict JSON; using defaults.")
            return defaults
        for key, default_val in defaults.items():
            if key not in data:
                data[key] = default_val
            elif not isinstance(data[key], type(default_val)) and key != "overview":
                data[key] = default_val
        return {k: data[k] for k in defaults}
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM response as JSON: %s", exc)
        return defaults
