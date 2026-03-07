"""
utils/parser.py
---------------
Converts raw log strings into structured dicts.

Supports common log formats:
  - Standard: YYYY-MM-DD HH:MM:SS LEVEL  message
  - Syslog-style: LEVEL: message
  - Fallback: raw line as message
"""

import re
from typing import Dict, List, Optional

# Pattern covers:  2024-01-15 08:04:01 ERROR  some message text
_STANDARD_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\s+"
    r"(?P<message>.+)$",
    re.IGNORECASE,
)

# Pattern covers:  ERROR: some message  or  [ERROR] some message
_SHORT_PATTERN = re.compile(
    r"^\[?(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\]?[:\s]+(?P<message>.+)$",
    re.IGNORECASE,
)

# Normalise level aliases to canonical form
_LEVEL_MAP = {
    "WARN": "WARNING",
    "FATAL": "CRITICAL",
}


def _normalise_level(raw: str) -> str:
    upper = raw.upper()
    return _LEVEL_MAP.get(upper, upper)


def parse_line(raw: str) -> Dict:
    """
    Parse a single log line into a structured dict.

    Returns a dict with keys:
        raw        – original log string
        timestamp  – ISO-style timestamp string or None
        level      – normalised log level string or "UNKNOWN"
        message    – log message body
    """
    m = _STANDARD_PATTERN.match(raw)
    if m:
        return {
            "raw": raw,
            "timestamp": m.group("timestamp"),
            "level": _normalise_level(m.group("level")),
            "message": m.group("message").strip(),
        }

    m = _SHORT_PATTERN.match(raw)
    if m:
        return {
            "raw": raw,
            "timestamp": None,
            "level": _normalise_level(m.group("level")),
            "message": m.group("message").strip(),
        }

    # Fallback — keep raw as message, level unknown
    return {
        "raw": raw,
        "timestamp": None,
        "level": "UNKNOWN",
        "message": raw,
    }


def parse_logs(raw_lines: List[str]) -> List[Dict]:
    """
    Parse a list of raw log strings into structured dicts.

    Args:
        raw_lines: List of raw log strings from the loader.

    Returns:
        List of parsed log dicts.
    """
    parsed = [parse_line(line) for line in raw_lines]
    level_counts = {}
    for entry in parsed:
        level_counts[entry["level"]] = level_counts.get(entry["level"], 0) + 1

    print(f"[Parser] Parsed {len(parsed)} entries — level distribution: {level_counts}")
    return parsed