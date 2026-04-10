# Codexa

> **Auto-generate structured CODEXA.md context files for AI coding agents.**

Codexa walks your codebase and writes a `CODEXA.md` file to every directory that contains Python source files. Each file contains:

- A high-level **overview** of the module's purpose (LLM-generated)
- A **file index** with extracted functions, classes, and imports
- **Key symbols** — the most important names a developer should know
- **Dependencies** — all imported modules in the directory
- **Non-obvious patterns** — design decisions and gotchas
- **Tribal knowledge** — contextual hints for new contributors
- **Subdirectory links** — navigation to child CODEXA.md files

The result is a persistent, human-readable knowledge layer that sits alongside your source code and helps AI coding agents (and humans) navigate large codebases with far fewer tool calls.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [generate](#generate)
  - [preview](#preview)
  - [clean](#clean)
- [Configuration Reference](#configuration-reference)
- [Example Output](#example-output)
- [How It Works](#how-it-works)
- [Incremental Updates](#incremental-updates)
- [Using a Custom Template](#using-a-custom-template)
- [OpenAI-Compatible Endpoints](#openai-compatible-endpoints)
- [Development](#development)
- [License](#license)

---

## Installation

**Requirements:** Python 3.8+

```bash
pip install codexa
```

Or install from source:

```bash
git clone https://github.com/codexa/codexa.git
cd codexa
pip install -e .
```

---

## Quick Start

1. **Set your OpenAI API key:**

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **Generate CODEXA.md files for your project:**

   ```bash
   codexa generate
   ```

3. **Preview what would be generated (no files written):**

   ```bash
   codexa preview ./src/mymodule
   ```

4. **Clean up all generated files:**

   ```bash
   codexa clean --yes
   ```

---

## Commands

### `generate`

Analyze a directory tree and write `CODEXA.md` files.

```
codexa generate [ROOT] [OPTIONS]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `ROOT` | Root directory to analyze | `.` (current directory) |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--config`, `-c` | Path to `.codexa.toml` config file | Auto-detected |
| `--dry-run` | Print previews without writing files | `False` |
| `--force`, `-f` | Regenerate even if content hash is unchanged | `False` |
| `--depth`, `-d` | Maximum recursion depth (0 = root only) | Config / unlimited |
| `--mock-llm` | Use deterministic mock LLM (no API calls) | `False` |
| `--verbose` | Enable verbose logging | `False` |
| `--version`, `-v` | Print version and exit | — |

**Examples:**

```bash
# Analyze the current directory
codexa generate

# Analyze a specific directory, limit to 3 levels deep
codexa generate ./src --depth 3

# Dry-run with mock LLM (no API key needed)
codexa generate . --dry-run --mock-llm

# Force regeneration of all files
codexa generate . --force

# Use a custom config file
codexa generate ./monorepo --config ./configs/codexa-prod.toml
```

---

### `preview`

Preview the `CODEXA.md` that would be generated for a single directory. Prints to stdout without writing any files.

```
codexa preview [DIRECTORY] [OPTIONS]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `DIRECTORY` | Directory to preview | `.` (current directory) |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--config`, `-c` | Path to `.codexa.toml` config file | Auto-detected |
| `--mock-llm` | Use deterministic mock LLM (no API calls) | `False` |
| `--no-llm` | Skip LLM entirely; render with extracted metadata only | `False` |

**Examples:**

```bash
# Preview the current directory with full LLM summarization
codexa preview

# Preview a specific module without making any API calls
codexa preview ./src/auth --no-llm

# Preview with mock LLM output
codexa preview ./src --mock-llm

# Pipe the output to a file
codexa preview ./src/utils > CODEXA_preview.md
```

---

### `clean`

Remove all `CODEXA.md` files from a directory tree.

```
codexa clean [ROOT] [OPTIONS]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `ROOT` | Root directory to search | `.` (current directory) |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--yes`, `-y` | Skip confirmation prompt | `False` |

**Examples:**

```bash
# Interactive cleanup (will prompt before deleting)
codexa clean

# Non-interactive cleanup (CI-safe)
codexa clean --yes

# Clean a specific subdirectory
codexa clean ./src --yes
```

---

## Configuration Reference

Create a `.codexa.toml` file in your project root to customize Codexa's behavior:

```toml
[codexa]
# LLM model to use (any OpenAI-compatible model identifier)
model = "gpt-4o-mini"

# API key (prefer OPENAI_API_KEY env var instead)
api_key = ""

# Custom endpoint for OpenAI-compatible servers (leave empty for OpenAI)
# Examples: local Ollama, Azure OpenAI, Groq, Anthropic (via proxy)
base_url = ""

# Maximum tokens per LLM completion request
max_tokens = 1024

# Maximum directory depth to recurse into (null = unlimited)
max_depth = null

# Gitignore-style patterns to exclude from analysis
ignore = [
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    "*.egg-info",
    "CODEXA.md",
]

# Path to a custom Jinja2 template (empty = use bundled template)
template = ""
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for the LLM backend (takes precedence over `api_key` in config) |
| `OPENAI_BASE_URL` | Custom endpoint URL (overridden by `base_url` in config if set) |

### Configuration Resolution Order

1. Explicit `--config` path (if provided on CLI)
2. `.codexa.toml` in the root directory being analyzed
3. Built-in defaults (all keys have safe default values)

---

## Example Output

Here is an example `CODEXA.md` generated for a hypothetical `auth/` directory:

```markdown
<!-- codexa-hash: a3f2c1d4e5b6... -->
# CODEXA: auth

> **Generated by Codexa** on 2024-01-15T10:30:00Z
> Directory: `/project/src/auth`

---

## Overview

The `auth` module implements JWT-based authentication and session management
for the API layer. It exposes login, logout, and token refresh endpoints and
delegates password hashing to the `secrets` and `hashlib` standard library
modules. All tokens are stored in Redis via the `SessionStore` class.

---

## Source Files

| File | Size | Functions | Classes |
|------|------|-----------|--------|
| `models.py` | 2,048 B | `hash_password`, `verify_password` | `User`, `Session` |
| `handlers.py` | 4,096 B | `login`, `logout`, `refresh` | `AuthHandler` |
| `middleware.py` | 1,024 B | `require_auth` | `JWTMiddleware` |

...

## Key Symbols

- `AuthHandler`
- `JWTMiddleware`
- `login`
- `require_auth`

## Dependencies

- `hashlib`
- `secrets`
- `jwt`
- `redis`

## Non-Obvious Patterns

- Tokens are rotated on every successful login to prevent session fixation.
- The `require_auth` decorator must be applied after the rate-limit decorator.

## Tribal Knowledge

- The JWT secret is loaded from `AUTH_SECRET` env var; never hardcode it.
- Redis session TTL is set to 7 days; changing it requires a cache flush.
```

---

## How It Works

Codexa operates in four stages:

1. **Analysis** (`codexa/analyzer.py`)
   - Walks the directory tree using `os.walk`, filtered by your ignore patterns (via `pathspec`).
   - Reads each `.py` file and extracts functions, classes, imports, and module docstrings using Python's built-in `ast` module.
   - Assembles `FileInfo` and `DirContext` data structures for each directory.

2. **Summarization** (`codexa/llm.py`)
   - For each directory context, builds a structured prompt containing file metadata.
   - Calls the configured LLM (default: `gpt-4o-mini`) via the OpenAI Python SDK.
   - Parses the JSON response into a `ModuleSummary` with overview, key symbols, patterns, and tribal knowledge.

3. **Rendering** (`codexa/renderer.py`)
   - Merges the `DirContext` and `ModuleSummary` into a template context dict.
   - Renders the bundled Jinja2 template (`codexa/templates/CODEXA.md.j2`) to markdown.
   - Embeds a SHA-256 content hash in the first line as an HTML comment.

4. **Writing**
   - Compares the embedded hash against the newly computed hash.
   - Skips writing if the hash is unchanged (incremental mode).
   - Writes `CODEXA.md` to each directory otherwise.

---

## Incremental Updates

Codexa is designed to be fast on large, frequently-changing codebases:

- Every generated `CODEXA.md` contains a content hash on its first line:
  ```
  <!-- codexa-hash: a3f2c1d4e5b6... -->
  ```
- On subsequent runs, Codexa re-computes the hash for each directory and **skips** the LLM call and file write if nothing has changed.
- Use `--force` to regenerate all files regardless of hash state.

This makes it practical to run `codexa generate` in CI or as a pre-commit hook.

---

## Using a Custom Template

You can provide your own Jinja2 template to change the CODEXA.md format:

1. Copy the bundled template as a starting point:
   ```bash
   cp $(python -c "import codexa; print(codexa.__file__.replace('__init__.py', 'templates/CODEXA.md.j2'))") my_template.j2
   ```

2. Edit `my_template.j2` to your liking.

3. Reference it in `.codexa.toml`:
   ```toml
   [codexa]
   template = "./my_template.j2"
   ```

### Available Template Variables

| Variable | Type | Description |
|----------|------|-------------|
| `dir_context` | `DirContext` | The full directory context object |
| `path` | `str` | Absolute directory path |
| `dir_name` | `str` | Directory name (last component) |
| `content_hash` | `str` | 64-char hex content hash |
| `hash_comment` | `str` | Full `<!-- codexa-hash: ... -->` string (put on line 1) |
| `files` | `List[FileInfo]` | Source files in this directory |
| `file_count` | `int` | Number of source files |
| `subdirectories` | `List[str]` | Immediate subdirectory names |
| `all_functions` | `List[str]` | All function names (deduplicated) |
| `all_classes` | `List[str]` | All class names (deduplicated) |
| `all_imports` | `List[str]` | All import names (deduplicated) |
| `summary` | `ModuleSummary \| None` | LLM-generated summary object |
| `overview` | `str` | Overview text (empty string if no summary) |
| `key_symbols` | `List[str]` | Key symbol names |
| `patterns` | `List[str]` | Non-obvious patterns |
| `tribal_knowledge` | `List[str]` | Tribal knowledge hints |
| `has_summary` | `bool` | True when a non-empty summary exists |
| `generated_at` | `str` | ISO-8601 UTC timestamp |

> **Important:** Your custom template must place `{{ hash_comment }}` on the very first line so that incremental hash detection works correctly.

---

## OpenAI-Compatible Endpoints

Codexa works with any OpenAI-compatible API server. Set `base_url` in your config:

### Local Ollama

```toml
[codexa]
model = "llama3.1:8b"
base_url = "http://localhost:11434/v1"
api_key = "ollama"  # Required by the SDK; value doesn't matter for Ollama
```

### Azure OpenAI

```toml
[codexa]
model = "gpt-4o-mini"
base_url = "https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT"
api_key = "your-azure-api-key"
```

### Groq

```toml
[codexa]
model = "llama-3.1-8b-instant"
base_url = "https://api.groq.com/openai/v1"
api_key = "gsk_..."
```

### Environment Variable Override

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your.custom.endpoint/v1"
codexa generate
```

---

## Development

### Setup

```bash
git clone https://github.com/codexa/codexa.git
cd codexa
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=codexa --cov-report=term-missing
```

### Project Structure

```
codexa/
├── __init__.py        # Package version and exports
├── cli.py             # Typer CLI entry point (generate, preview, clean)
├── analyzer.py        # Directory walker and AST metadata extractor
├── llm.py             # OpenAI-compatible LLM client and mock client
├── renderer.py        # Jinja2 renderer with incremental hash logic
├── config.py          # .codexa.toml loader and validator
├── models.py          # FileInfo, ModuleSummary, DirContext dataclasses
└── templates/
    └── CODEXA.md.j2   # Default Jinja2 template
tests/
├── test_analyzer.py
├── test_config.py
├── test_renderer.py
├── test_llm.py        # (if present)
└── fixtures/
    └── sample_module/
        └── main.py
```

### Adding a New LLM Backend

1. Subclass `BaseLLMClient` in `codexa/llm.py`.
2. Implement `summarize_directory(self, dir_context: DirContext) -> ModuleSummary`.
3. Update `create_llm_client()` to route to your new class.

### Contributing

Pull requests are welcome! Please:

- Write tests for new functionality.
- Follow the existing code style (PEP 8, type hints everywhere).
- Run `pytest` before submitting.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Generated context files are most useful when committed alongside your source code and regenerated in CI on each merge to the main branch.*
