"""Sample module used as a fixture in analyzer and renderer tests.

This file provides a variety of Python constructs so that the test suite
can verify AST extraction of:
  - Module-level docstrings.
  - Top-level function definitions (sync and async).
  - Top-level class definitions.
  - Import statements (``import X`` and ``from X import Y`` forms).
  - Nested definitions that should *not* appear in top-level lists.

The module is intentionally self-contained and has no side effects when
imported so it is safe to import during test runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# A module-level constant
VERSION: str = "1.0.0"

_PRIVATE_CONSTANT: int = 42


def greet(name: str) -> str:
    """Return a greeting string for *name*.

    Args:
        name: The name of the person to greet.

    Returns:
        A formatted greeting string.
    """
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Return the sum of *a* and *b*.

    Args:
        a: First integer operand.
        b: Second integer operand.

    Returns:
        The integer sum.
    """
    return a + b


async def fetch_data(url: str, timeout: int = 30) -> Optional[str]:
    """Simulate an async HTTP fetch (stub for testing).

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Simulated response body or ``None`` on failure.
    """
    # Nested helper — should NOT appear in top-level function list
    def _build_headers() -> Dict[str, str]:
        return {"User-Agent": "codexa/1.0"}

    _ = _build_headers()
    _ = timeout
    return f"data from {url}"


class Animal:
    """Base class representing a generic animal.

    Attributes:
        name: The animal's name.
        sound: The sound the animal makes.
    """

    def __init__(self, name: str, sound: str) -> None:
        self.name = name
        self.sound = sound

    def speak(self) -> str:
        """Return the animal's vocalisation."""
        return f"{self.name} says {self.sound}!"

    # Nested class — should NOT appear in top-level class list
    class _Metadata:
        """Internal metadata container."""

        kingdom: str = "Animalia"


class Dog(Animal):
    """A subclass of :class:`Animal` representing a dog."""

    def __init__(self, name: str) -> None:
        super().__init__(name, "woof")

    def fetch(self, item: str) -> str:
        """Fetch an item and return a description."""
        return f"{self.name} fetched the {item}!"


def list_files(directory: Path) -> List[Path]:
    """Return a sorted list of files in *directory*.

    Args:
        directory: The directory to list.

    Returns:
        A sorted list of :class:`~pathlib.Path` objects for each file
        found directly in *directory*.
    """
    try:
        return sorted(p for p in directory.iterdir() if p.is_file())
    except PermissionError:
        return []


def get_env(key: str, default: str = "") -> str:
    """Retrieve an environment variable with a fallback default.

    Args:
        key: Name of the environment variable.
        default: Value to return if the variable is not set.

    Returns:
        The environment variable value or *default*.
    """
    return os.environ.get(key, default)


class Config:
    """Simple configuration container.

    Holds key-value settings loaded from the environment or a dict.

    Attributes:
        settings: Internal settings dictionary.
    """

    def __init__(self, settings: Optional[Dict[str, str]] = None) -> None:
        self.settings: Dict[str, str] = settings or {}

    def get(self, key: str, default: str = "") -> str:
        """Return the setting for *key* or *default*."""
        return self.settings.get(key, default)

    def set(self, key: str, value: str) -> None:
        """Set *key* to *value*."""
        self.settings[key] = value

    @classmethod
    def from_env(cls, prefix: str = "") -> "Config":
        """Build a :class:`Config` from environment variables.

        Args:
            prefix: Only include variables whose names start with *prefix*.

        Returns:
            A new :class:`Config` populated from matching env vars.
        """
        settings = {
            k[len(prefix):]: v
            for k, v in os.environ.items()
            if k.startswith(prefix)
        }
        return cls(settings)


def _private_helper(value: int) -> int:
    """Private helper function — included to verify private names are extracted."""
    return value * 2


if __name__ == "__main__":
    # Entry-point guard — functions defined here should not appear in the
    # top-level list because they're inside an if-block body.  However the
    # AST walker will see any imports or simple assignments.
    print(greet("world"))
    cfg = Config.from_env()
    sys.exit(0)
