"""Shared helpers for robust downloader writes."""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar


T = TypeVar("T")
# Non-deterministic (OS entropy) jitter source for retry backoff delays.
_JITTER_RNG = random.SystemRandom()
RETRYABLE_NETWORK_MARKERS = (
    "Could not resolve host",
    "Connection reset",
    "Recv failure",
    "Operation timed out",
    "UNEXPECTED_EOF_WHILE_READING",
    "SSL connection timeout",
    "Failed to connect",
    "Temporary failure",
    "Name or service not known",
    "Connection aborted",
    "Connection refused",
    "timed out",
    "EOFError",
    "Missing temp download file",
)


def load_repo_env(env_path: str | Path | None = None) -> None:
    """Load repo-local .env entries into os.environ when not already set."""
    path = Path(env_path) if env_path else Path(__file__).resolve().parents[3] / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_repo_env()


def is_retryable_network_error(exc: Exception) -> bool:
    """Best-effort classification for transient network failures."""
    text = str(exc)
    return any(marker in text for marker in RETRYABLE_NETWORK_MARKERS)


def retry_with_backoff(
    func: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay_sec: float = 2.0,
    max_delay_sec: float = 30.0,
    label: str = "operation",
    verbose: bool = True,
) -> T:
    """Retry transient network work with exponential backoff and small jitter."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            if attempt >= attempts or not is_retryable_network_error(exc):
                raise
            delay = min(max_delay_sec, base_delay_sec * (2 ** (attempt - 1)))
            delay += _JITTER_RNG.uniform(0, min(1.0, delay * 0.1))
            if verbose:
                print(f"  Retryable network error during {label}, attempt {attempt}/{attempts}: {exc}")
                print(f"  Sleeping {delay:.1f}s before retry")
            time.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{label} failed without raising an exception")


def temp_path_for(path: str | Path) -> str:
    """Return the temporary path used for atomic downloads."""
    path = str(path)
    return f"{path}.part.{os.getpid()}"


def cleanup_temp_file(path: str | Path) -> None:
    """Remove this process's temporary download file for the target path if it exists."""
    temp_path = temp_path_for(path)
    if os.path.exists(temp_path):
        os.remove(temp_path)


@contextmanager
def atomic_output_file(path: str | Path, mode: str = "wb") -> Iterator[tuple[object, str]]:
    """Write to a temporary file and atomically replace the target on success."""
    final_path = str(path)
    temp_path = temp_path_for(final_path)
    cleanup_temp_file(final_path)
    handle = open(temp_path, mode)
    try:
        yield handle, temp_path
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temp_path, final_path)
    except Exception:
        try:
            handle.close()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        raise
