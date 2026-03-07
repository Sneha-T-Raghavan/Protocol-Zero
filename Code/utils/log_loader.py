"""
utils/log_loader.py
-------------------
Handles loading log files efficiently.
Supports large files via generator-based streaming.
"""

import os
from typing import Generator, List


def stream_logs(filepath: str) -> Generator[str, None, None]:
    """
    Stream log lines one at a time — memory-efficient for large files.
    Yields each non-empty line stripped of whitespace.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def load_logs(filepath: str, max_lines: int = None) -> List[str]:
    """
    Load all (or up to max_lines) log lines from a file into a list.

    Args:
        filepath:  Path to the log file.
        max_lines: Optional cap on lines to load (useful for testing).

    Returns:
        List of raw log strings.
    """
    logs = []
    for i, line in enumerate(stream_logs(filepath)):
        if max_lines and i >= max_lines:
            break
        logs.append(line)

    print(f"[LogLoader] Loaded {len(logs)} log lines from '{filepath}'")
    return logs