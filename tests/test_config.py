"""Unit tests for codexa.config — configuration loading and validation.

Covers:
  - Default config values when no file is present.
  - Successful loading from a valid .codexa.toml file.
  - The ``[codexa]`` section merge/override logic.
  - Validation errors for every checked field.
  - Environment variable overrides via ``effective_api_key`` and
    ``effective_base_url``.
  - The ``build_ignore_spec`` helper.
  - ``CodexaConfig`` helpers (``to_dict``, ``template_path``, ``__eq__``).
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from codexa.config import (
    CONFIG_FILENAME,
    DEFAULT_CONFIG,
    CodexaConfig,
    ConfigError,
    _load_toml,
    _merge_with_defaults,
    _validate,
    build_ignore_spec,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(directory: Path, content: str) -> Path:
    """Write *content* to ``.codexa.toml`` inside *directory* and return its path."""
    path = directory / CONFIG_FILENAME
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def _minimal_valid_config() -> Dict[str, Any]:
    """Return a copy of the merged defaults that passes validation."""
    return {
        "model": "gpt-4o-mini",
        "api_key": "",
        "base_url": "",
        "max_tokens": 1024,
        "max_depth": None,
        "ignore": [".git", "__pycache__"],
        "template": "",
    }


# ===========================================================================
# DEFAULT_CONFIG sanity checks
# ===========================================================================


class TestDefaultConfig:
    """Verify the module-level DEFAULT_CONFIG constant."""

    def test_all_required_keys_present(self) -> None:
        required = {"model", "api_key", "base_url", "max_tokens", "max_depth", "ignore", "template"}
        assert required.issubset(set(DEFAULT_CONFIG.keys()))

    def test_default_model_is_string(self) -> None:
        assert isinstance(DEFAULT_CONFIG["model"], str)
        assert DEFAULT_CONFIG["model"]

    def test_default_max_tokens_in_range(self) -> None:
        assert DEFAULT_CONFIG["max_tokens"] >= 64

    def test_default_max_depth_is_none(self) -> None:
        assert DEFAULT_CONFIG["max_depth"] is None

    def test_default_ignore_is_list_of_strings(self) -> None:
        assert isinstance(DEFAULT_CONFIG["ignore"], list)
        for p in DEFAULT_CONFIG["ignore"]:
            assert isinstance(p, str)

    def test_default_ignore_contains_git(self) -> None:
        assert ".git" in DEFAULT_CONFIG["ignore"]


# ===========================================================================
# _load_toml
# ===========================================================================


class TestLoadToml:
    """Tests for the internal TOML loader."""

    def test_load_valid_file(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, """
            [codexa]
            model = "gpt-4o"
            max_tokens = 2048
        """)
        data = _load_toml(path)
        assert data["codexa"]["model"] == "gpt-4o"
        assert data["codexa"]["max_tokens"] == 2048

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            _load_toml(tmp_path / "nonexistent.toml")

    def test_load_invalid_toml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / ".codexa.toml"
        bad.write_text("[unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            _load_toml(bad)

    def test_load_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        empty = tmp_path / ".codexa.toml"
        empty.write_text("", encoding="utf-8")
        data = _load_toml(empty)
        assert isinstance(data, dict)

    def test_load_file_without_codexa_section(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, """
            [other]
            key = "value"
        """)
        data = _load_toml(path)
        assert "other" in data
        assert "codexa" not in data


# ===========================================================================
# _merge_with_defaults
# ===========================================================================


class TestMergeWithDefaults:
    """Tests for the default-merging logic."""

    def test_empty_raw_returns_all_defaults(self) -> None:
        merged = _merge_with_defaults({})
        for key in DEFAULT_CONFIG:
            assert key in merged

    def test_values_from_section_override_defaults(self) -> None:
        raw = {"codexa": {"model": "gpt-4o", "max_tokens": 512}}
        merged = _merge_with_defaults(raw)
        assert merged["model"] == "gpt-4o"
        assert merged["max_tokens"] == 512

    def test_unset_keys_retain_defaults(self) -> None:
        raw = {"codexa": {"model": "gpt-4o"}}
        merged = _merge_with_defaults(raw)
        assert merged["max_tokens"] == DEFAULT_CONFIG["max_tokens"]
        assert merged["ignore"] == DEFAULT_CONFIG["ignore"]

    def test_unknown_keys_are_ignored(self) -> None:
        raw = {"codexa": {"nonexistent_key": True}}
        merged = _merge_with_defaults(raw)
        assert "nonexistent_key" not in merged

    def test_missing_codexa_section_uses_defaults(self) -> None:
        merged = _merge_with_defaults({"other": {"x": 1}})
        assert merged["model"] == DEFAULT_CONFIG["model"]

    def test_ignore_override_replaces_not_extends(self) -> None:
        custom_ignore = ["*.log"]
        raw = {"codexa": {"ignore": custom_ignore}}
        merged = _merge_with_defaults(raw)
        assert merged["ignore"] == custom_ignore

    def test_non_dict_codexa_section_raises(self) -> None:
        with pytest.raises(ConfigError):
            _merge_with_defaults({"codexa": "not-a-table"})


# ===========================================================================
# _validate
# ===========================================================================


class TestValidate:
    """Tests for the _validate function."""

    def test_valid_config_passes(self) -> None:
        cfg = _minimal_valid_config()
        _validate(cfg)  # should not raise

    # --- model ---

    def test_empty_model_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"] = ""
        with pytest.raises(ConfigError, match="model"):
            _validate(cfg)

    def test_whitespace_only_model_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"] = "   "
        with pytest.raises(ConfigError, match="model"):
            _validate(cfg)

    def test_non_string_model_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["model"] = 42
        with pytest.raises(ConfigError, match="model"):
            _validate(cfg)

    # --- api_key ---

    def test_non_string_api_key_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["api_key"] = 12345
        with pytest.raises(ConfigError, match="api_key"):
            _validate(cfg)

    def test_empty_api_key_is_allowed(self) -> None:
        cfg = _minimal_valid_config()
        cfg["api_key"] = ""
        _validate(cfg)  # should not raise

    # --- base_url ---

    def test_non_string_base_url_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["base_url"] = ["http://example.com"]
        with pytest.raises(ConfigError, match="base_url"):
            _validate(cfg)

    def test_empty_base_url_is_allowed(self) -> None:
        cfg = _minimal_valid_config()
        cfg["base_url"] = ""
        _validate(cfg)  # should not raise

    # --- max_tokens ---

    def test_max_tokens_below_minimum_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_tokens"] = 63
        with pytest.raises(ConfigError, match="max_tokens"):
            _validate(cfg)

    def test_max_tokens_at_minimum_passes(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_tokens"] = 64
        _validate(cfg)  # should not raise

    def test_max_tokens_above_maximum_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_tokens"] = 200_000
        with pytest.raises(ConfigError, match="max_tokens"):
            _validate(cfg)

    def test_max_tokens_float_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_tokens"] = 1024.0
        with pytest.raises(ConfigError, match="max_tokens"):
            _validate(cfg)

    def test_max_tokens_bool_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_tokens"] = True  # bool is subclass of int; must be rejected
        with pytest.raises(ConfigError, match="max_tokens"):
            _validate(cfg)

    def test_max_tokens_string_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_tokens"] = "1024"
        with pytest.raises(ConfigError, match="max_tokens"):
            _validate(cfg)

    # --- max_depth ---

    def test_max_depth_none_is_valid(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_depth"] = None
        _validate(cfg)  # should not raise

    def test_max_depth_zero_is_valid(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_depth"] = 0
        _validate(cfg)  # should not raise

    def test_max_depth_positive_is_valid(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_depth"] = 10
        _validate(cfg)  # should not raise

    def test_max_depth_negative_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_depth"] = -1
        with pytest.raises(ConfigError, match="max_depth"):
            _validate(cfg)

    def test_max_depth_float_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_depth"] = 3.0
        with pytest.raises(ConfigError, match="max_depth"):
            _validate(cfg)

    def test_max_depth_bool_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_depth"] = True
        with pytest.raises(ConfigError, match="max_depth"):
            _validate(cfg)

    def test_max_depth_string_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["max_depth"] = "5"
        with pytest.raises(ConfigError, match="max_depth"):
            _validate(cfg)

    # --- ignore ---

    def test_ignore_non_list_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["ignore"] = "*.pyc"
        with pytest.raises(ConfigError, match="ignore"):
            _validate(cfg)

    def test_ignore_list_with_non_string_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["ignore"] = [".git", 42]
        with pytest.raises(ConfigError, match="ignore"):
            _validate(cfg)

    def test_ignore_empty_list_is_valid(self) -> None:
        cfg = _minimal_valid_config()
        cfg["ignore"] = []
        _validate(cfg)  # should not raise

    # --- template ---

    def test_template_non_string_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["template"] = 123
        with pytest.raises(ConfigError, match="template"):
            _validate(cfg)

    def test_template_empty_string_is_valid(self) -> None:
        cfg = _minimal_valid_config()
        cfg["template"] = ""
        _validate(cfg)  # should not raise

    def test_template_nonexistent_path_raises(self) -> None:
        cfg = _minimal_valid_config()
        cfg["template"] = "/nonexistent/path/template.j2"
        with pytest.raises(ConfigError, match="template"):
            _validate(cfg)

    def test_template_existing_file_is_valid(self, tmp_path: Path) -> None:
        tmpl = tmp_path / "my_template.j2"
        tmpl.write_text("# Hello", encoding="utf-8")
        cfg = _minimal_valid_config()
        cfg["template"] = str(tmpl)
        _validate(cfg)  # should not raise

    def test_template_existing_directory_raises(self, tmp_path: Path) -> None:
        cfg = _minimal_valid_config()
        cfg["template"] = str(tmp_path)  # tmp_path is a directory
        with pytest.raises(ConfigError, match="template"):
            _validate(cfg)


# ===========================================================================
# load_config — integration-level tests
# ===========================================================================


class TestLoadConfig:
    """Integration tests for load_config."""

    def test_no_file_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_config(root=tmp_path)
        assert cfg.model == DEFAULT_CONFIG["model"]
        assert cfg.max_tokens == DEFAULT_CONFIG["max_tokens"]
        assert cfg.max_depth == DEFAULT_CONFIG["max_depth"]
        assert cfg.ignore == DEFAULT_CONFIG["ignore"]
        assert cfg.template == DEFAULT_CONFIG["template"]

    def test_auto_discovers_config_in_root(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, """
            [codexa]
            model = "gpt-4-turbo"
            max_tokens = 2048
        """)
        cfg = load_config(root=tmp_path)
        assert cfg.model == "gpt-4-turbo"
        assert cfg.max_tokens == 2048

    def test_explicit_config_path_overrides_root(self, tmp_path: Path) -> None:
        explicit = tmp_path / "custom.toml"
        explicit.write_text(
            textwrap.dedent("""
                [codexa]
                model = "gpt-3.5-turbo"
            """),
            encoding="utf-8",
        )
        # Also create a root-level .codexa.toml that should NOT be used
        _write_toml(tmp_path, """
            [codexa]
            model = "gpt-4o"
        """)
        cfg = load_config(config_path=explicit)
        assert cfg.model == "gpt-3.5-turbo"

    def test_explicit_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(config_path=tmp_path / "missing.toml")

    def test_returns_codexa_config_instance(self, tmp_path: Path) -> None:
        cfg = load_config(root=tmp_path)
        assert isinstance(cfg, CodexaConfig)

    def test_model_is_stripped(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, """
            [codexa]
            model = "  gpt-4o  "
        """)
        cfg = load_config(root=tmp_path)
        assert cfg.model == "gpt-4o"

    def test_ignore_patterns_loaded(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, """
            [codexa]
            ignore = ["*.log", "tmp/"]
        """)
        cfg = load_config(root=tmp_path)
        assert "*.log" in cfg.ignore
        assert "tmp/" in cfg.ignore

    def test_max_depth_none_loaded(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, """
            [codexa]
            max_depth = 5
        """)
        cfg = load_config(root=tmp_path)
        assert cfg.max_depth == 5

    def test_invalid_config_raises_config_error(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, """
            [codexa]
            max_tokens = 10
        """)
        with pytest.raises(ConfigError, match="max_tokens"):
            load_config(root=tmp_path)

    def test_unknown_keys_in_file_do_not_raise(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, """
            [codexa]
            model = "gpt-4o"
            future_option = true
        """)
        cfg = load_config(root=tmp_path)  # should not raise
        assert cfg.model == "gpt-4o"

    def test_default_root_uses_cwd(self, tmp_path: Path, monkeypatch: Any) -> None:
        _write_toml(tmp_path, """
            [codexa]
            model = "gpt-4o-mini"
        """)
        monkeypatch.chdir(tmp_path)
        cfg = load_config()  # root defaults to Path(".")
        assert cfg.model == "gpt-4o-mini"

    def test_api_key_loaded_from_file(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, """
            [codexa]
            api_key = "sk-from-file"
        """)
        cfg = load_config(root=tmp_path)
        assert cfg.api_key == "sk-from-file"

    def test_base_url_loaded_from_file(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, """
            [codexa]
            base_url = "https://my.custom.endpoint/v1"
        """)
        cfg = load_config(root=tmp_path)
        assert cfg.base_url == "https://my.custom.endpoint/v1"


# ===========================================================================
# CodexaConfig — property and helper tests
# ===========================================================================


class TestCodexaConfigProperties:
    """Tests for CodexaConfig properties and helpers."""

    def _make_config(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "",
        max_tokens: int = 1024,
        max_depth: Any = None,
        ignore: list = None,
        template: str = "",
    ) -> CodexaConfig:
        return CodexaConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            max_depth=max_depth,
            ignore=ignore if ignore is not None else [],
            template=template,
        )

    # --- effective_api_key ---

    def test_effective_api_key_uses_env_var(self) -> None:
        cfg = self._make_config(api_key="from-file")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "from-env"}):
            assert cfg.effective_api_key == "from-env"

    def test_effective_api_key_falls_back_to_config(self) -> None:
        cfg = self._make_config(api_key="from-file")
        # Make sure OPENAI_API_KEY is not set
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            assert cfg.effective_api_key == "from-file"

    def test_effective_api_key_empty_when_neither_set(self) -> None:
        cfg = self._make_config(api_key="")
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            assert cfg.effective_api_key == ""

    def test_effective_api_key_env_whitespace_uses_file(self) -> None:
        cfg = self._make_config(api_key="from-file")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "   "}):
            assert cfg.effective_api_key == "from-file"

    # --- effective_base_url ---

    def test_effective_base_url_uses_config_file(self) -> None:
        cfg = self._make_config(base_url="https://config.example.com/v1")
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_BASE_URL"}
        with patch.dict(os.environ, env, clear=True):
            assert cfg.effective_base_url == "https://config.example.com/v1"

    def test_effective_base_url_env_var_used_when_config_empty(self) -> None:
        cfg = self._make_config(base_url="")
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.example.com/v1"}):
            assert cfg.effective_base_url == "https://env.example.com/v1"

    def test_effective_base_url_none_when_neither_set(self) -> None:
        cfg = self._make_config(base_url="")
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_BASE_URL"}
        with patch.dict(os.environ, env, clear=True):
            assert cfg.effective_base_url is None

    def test_effective_base_url_config_takes_precedence_over_env(self) -> None:
        cfg = self._make_config(base_url="https://config.example.com/v1")
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.example.com/v1"}):
            assert cfg.effective_base_url == "https://config.example.com/v1"

    # --- template_path ---

    def test_template_path_returns_none_when_empty(self) -> None:
        cfg = self._make_config(template="")
        assert cfg.template_path is None

    def test_template_path_returns_path_object(self) -> None:
        cfg = self._make_config(template="/some/template.j2")
        assert cfg.template_path == Path("/some/template.j2")

    # --- to_dict ---

    def test_to_dict_contains_all_keys(self) -> None:
        cfg = self._make_config()
        d = cfg.to_dict()
        expected = {"model", "api_key", "base_url", "max_tokens", "max_depth", "ignore", "template"}
        assert set(d.keys()) == expected

    def test_to_dict_values_match(self) -> None:
        cfg = self._make_config(
            model="gpt-4o",
            api_key="sk-test",
            max_tokens=512,
            max_depth=3,
            ignore=[".git"],
        )
        d = cfg.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["api_key"] == "sk-test"
        assert d["max_tokens"] == 512
        assert d["max_depth"] == 3
        assert d["ignore"] == [".git"]

    def test_to_dict_ignore_is_independent_copy(self) -> None:
        cfg = self._make_config(ignore=[".git"])
        d = cfg.to_dict()
        d["ignore"].append("extra")
        assert cfg.ignore == [".git"]

    # --- __eq__ ---

    def test_equal_configs_are_equal(self) -> None:
        a = self._make_config(model="gpt-4o", max_tokens=512)
        b = self._make_config(model="gpt-4o", max_tokens=512)
        assert a == b

    def test_different_model_not_equal(self) -> None:
        a = self._make_config(model="gpt-4o")
        b = self._make_config(model="gpt-3.5")
        assert a != b

    def test_different_max_tokens_not_equal(self) -> None:
        a = self._make_config(max_tokens=512)
        b = self._make_config(max_tokens=1024)
        assert a != b

    def test_not_equal_to_non_config(self) -> None:
        cfg = self._make_config()
        assert cfg != "not a config"
        assert cfg != 42
        assert cfg != None  # noqa: E711

    # --- __repr__ ---

    def test_repr_contains_model(self) -> None:
        cfg = self._make_config(model="gpt-4o")
        assert "gpt-4o" in repr(cfg)

    def test_repr_contains_max_tokens(self) -> None:
        cfg = self._make_config(max_tokens=2048)
        assert "2048" in repr(cfg)


# ===========================================================================
# build_ignore_spec
# ===========================================================================


class TestBuildIgnoreSpec:
    """Tests for the build_ignore_spec helper."""

    def test_returns_pathspec_instance(self) -> None:
        import pathspec  # type: ignore
        spec = build_ignore_spec([".git", "__pycache__"])
        assert isinstance(spec, pathspec.PathSpec)

    def test_matches_simple_pattern(self) -> None:
        spec = build_ignore_spec(["*.pyc"])
        assert spec.match_file("module.pyc")
        assert not spec.match_file("module.py")

    def test_matches_directory_pattern(self) -> None:
        spec = build_ignore_spec([".git"])
        assert spec.match_file(".git/config")

    def test_matches_nested_pattern(self) -> None:
        spec = build_ignore_spec(["__pycache__"])
        assert spec.match_file("src/__pycache__/module.cpython-39.pyc")

    def test_empty_patterns_matches_nothing(self) -> None:
        spec = build_ignore_spec([])
        assert not spec.match_file("anything.py")
        assert not spec.match_file(".git/HEAD")

    def test_multiple_patterns(self) -> None:
        spec = build_ignore_spec([".git", "node_modules", "*.log"])
        assert spec.match_file(".git/HEAD")
        assert spec.match_file("node_modules/lodash/index.js")
        assert spec.match_file("debug.log")
        assert not spec.match_file("src/main.py")

    def test_glob_star_star_pattern(self) -> None:
        spec = build_ignore_spec(["**/__pycache__/**"])
        assert spec.match_file("src/__pycache__/module.cpython-39.pyc")

    def test_default_ignore_list_from_config(self, tmp_path: Path) -> None:
        """build_ignore_spec should work with the default ignore list."""
        cfg = load_config(root=tmp_path)
        spec = build_ignore_spec(cfg.ignore)
        import pathspec  # type: ignore
        assert isinstance(spec, pathspec.PathSpec)
        assert spec.match_file(".git/HEAD")
        assert spec.match_file("__pycache__/foo.pyc")
