"""
utils/bgl_parser.py
--------------------
Parser adapter for the BGL (BlueGene/L) log format from LogHub.

Real BGL line format (10 space-separated fields):
  LABEL  UNIX_TS  DATE  NODE  DATETIME  NODE_REPEAT  RAS  COMPONENT  LEVEL  MESSAGE...

Field index:
  0  → label         ('-' = normal, anything else = anomaly label/level)
  1  → unix_timestamp
  2  → date          (YYYY.MM.DD)
  3  → node          (R02-M1-N0-C:J12-U11)
  4  → datetime      (YYYY-MM-DD-HH.MM.SS.microseconds)
  5  → node_repeat
  6  → event_type    (always 'RAS')
  7  → component     (KERNEL, APP, MMCS, …)
  8  → level         (INFO, WARNING, ERROR, FATAL, SEVERE, …)
  9+ → message       (rest of the line)

This module provides:
  parse_bgl_line()   — parse a single raw BGL line → structured dict
  parse_bgl_logs()   — batch parse a list of raw lines
  level_to_standard()— map BGL levels → standard protocol-zero levels
"""

import re
from typing import Dict, List, Optional

# BGL levels → canonical protocol-zero levels
_BGL_LEVEL_MAP = {
    "INFO":    "INFO",
    "WARNING": "WARNING",
    "WARN":    "WARNING",
    "ERROR":   "ERROR",
    "SEVERE":  "ERROR",
    "FATAL":   "CRITICAL",
    "FAILURE": "ERROR",
    "ALERT":   "ERROR",
}

# Minimum number of fields a valid BGL line must have
_MIN_FIELDS = 9

# BGL datetime format: 2005-06-03-15.42.50.675872
_BGL_DT_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})-(\d{2})\.(\d{2})\.(\d{2})\.(\d+)"
)


def _parse_bgl_datetime(raw: str) -> Optional[str]:
    """Convert BGL datetime string to ISO-8601 format, or None on failure."""
    m = _BGL_DT_RE.match(raw)
    if not m:
        return None
    year, month, day, hour, minute, sec, micro = m.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{sec}.{micro}"


def level_to_standard(bgl_level: str) -> str:
    """Map a raw BGL level string to a canonical protocol-zero level."""
    return _BGL_LEVEL_MAP.get(bgl_level.upper(), "UNKNOWN")


def parse_bgl_line(raw: str) -> Dict:
    """
    Parse a single raw BGL log line into a structured dict.

    Returns a dict with keys:
        raw         – original line
        timestamp   – ISO-8601 string or None
        level       – canonical level (INFO/WARNING/ERROR/CRITICAL/UNKNOWN)
        bgl_level   – original BGL level string
        label       – BGL anomaly label ('-' = normal)
        is_anomaly  – True if label != '-'
        node        – node identifier string
        component   – BGL component (KERNEL, APP, …)
        message     – log message content
    """
    parts = raw.strip().split(None, 9)   # split on whitespace, max 10 parts

    if len(parts) < _MIN_FIELDS:
        # Malformed line — return a safe fallback
        return {
            "raw":        raw,
            "timestamp":  None,
            "level":      "UNKNOWN",
            "bgl_level":  "UNKNOWN",
            "label":      "?",
            "is_anomaly": False,
            "node":       "unknown",
            "component":  "unknown",
            "message":    raw,
        }

    label     = parts[0]
    node      = parts[3]
    datetime_ = _parse_bgl_datetime(parts[4]) if len(parts) > 4 else None
    component = parts[7] if len(parts) > 7 else "UNKNOWN"
    bgl_level = parts[8] if len(parts) > 8 else "UNKNOWN"
    message   = parts[9].strip() if len(parts) > 9 else ""

    return {
        "raw":        raw,
        "timestamp":  datetime_,
        "level":      level_to_standard(bgl_level),
        "bgl_level":  bgl_level,
        "label":      label,
        "is_anomaly": label != "-",
        "node":       node,
        "component":  component,
        "message":    message,
    }


def parse_bgl_logs(raw_lines: List[str]) -> List[Dict]:
    """
    Batch-parse a list of raw BGL log lines into structured dicts.

    Prints a summary of parsed level and anomaly distributions.
    """
    parsed = [parse_bgl_line(line) for line in raw_lines]

    # Build summary stats
    level_counts: Dict[str, int] = {}
    anomaly_count = 0
    for entry in parsed:
        lv = entry["level"]
        level_counts[lv] = level_counts.get(lv, 0) + 1
        if entry["is_anomaly"]:
            anomaly_count += 1

    print(
        f"[BGLParser] Parsed {len(parsed):,} entries | "
        f"levels: {level_counts} | "
        f"labelled anomalies: {anomaly_count:,} ({anomaly_count/max(len(parsed),1)*100:.1f}%)"
    )
    return parsed