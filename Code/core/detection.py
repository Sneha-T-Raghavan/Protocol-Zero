"""
core/detection.py
-----------------
Anomaly detection logic for Protocol Zero.

Implements two detection strategies:

  1. GlobalErrorSpikeDetector  — counts ERROR/CRITICAL logs across the full
     log batch and fires if the total exceeds a configurable threshold.

  2. SlidingWindowDetector     — scans logs in a rolling window (by list index,
     not wall-clock time) and fires when error density within any window
     exceeds a configurable threshold.

Both detectors return a list of Incident objects (empty list = no anomaly).
"""

from typing import List, Dict, Optional
from core.incident import Incident, severity_from_count

# Log levels that count as "errors" for detection purposes
ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sample_errors(entries: List[Dict], n: int = 5) -> List[str]:
    """Return up to n error messages from a list of parsed log entries."""
    errors = [e["message"] for e in entries if e["level"] in ERROR_LEVELS]
    return errors[:n]


# ---------------------------------------------------------------------------
# Detector 1 — Global Error Spike
# ---------------------------------------------------------------------------

class GlobalErrorSpikeDetector:
    """
    Fires a single incident when the total number of ERROR/CRITICAL log
    entries across the entire batch exceeds `threshold`.

    Configuration
    -------------
    threshold : int  — minimum error count to trigger (default: 5)
    source    : str  — label attached to generated incidents
    """

    def __init__(self, threshold: int = 5, source: str = "system-logs"):
        self.threshold = threshold
        self.source = source

    def detect(self, parsed_logs: List[Dict]) -> List[Incident]:
        """
        Scan all parsed log entries and return incidents.

        Returns a list with zero or one Incident.
        """
        error_entries = [e for e in parsed_logs if e["level"] in ERROR_LEVELS]
        error_count = len(error_entries)

        print(
            f"[GlobalErrorSpikeDetector] "
            f"Found {error_count} error-level entries (threshold={self.threshold})"
        )

        if error_count < self.threshold:
            print("  → No incident triggered.")
            return []

        severity = severity_from_count(error_count)
        incident = Incident(
            anomaly_type="ERROR_SPIKE",
            severity=severity,
            source=self.source,
            error_count=error_count,
            details=(
                f"Total error count {error_count} exceeded threshold {self.threshold}. "
                f"Severity assessed as {severity}."
            ),
            sample_errors=_sample_errors(error_entries),
        )

        print(f"  → Incident created: {incident}")
        return [incident]


# ---------------------------------------------------------------------------
# Detector 2 — Sliding Window Spike
# ---------------------------------------------------------------------------

class SlidingWindowDetector:
    """
    Scans logs in a rolling window of `window_size` entries.
    Fires an incident for each window where the error count
    exceeds `error_threshold`.

    To avoid duplicate incidents for overlapping windows,
    results are de-duplicated: a new incident is only raised when the
    window start index is at least `window_size // 2` entries beyond the
    previous incident's window start.

    Configuration
    -------------
    window_size     : int — number of log entries per window (default: 10)
    error_threshold : int — errors within a window to trigger (default: 3)
    source          : str — label attached to generated incidents
    """

    def __init__(
        self,
        window_size: int = 10,
        error_threshold: int = 3,
        source: str = "system-logs",
    ):
        self.window_size = window_size
        self.error_threshold = error_threshold
        self.source = source

    def detect(self, parsed_logs: List[Dict]) -> List[Incident]:
        """
        Slide a window over parsed_logs and return one Incident per
        distinct error burst detected.
        """
        incidents: List[Incident] = []
        last_incident_start: Optional[int] = None
        cooldown = self.window_size // 2  # minimum gap before next incident

        print(
            f"[SlidingWindowDetector] "
            f"window_size={self.window_size}, error_threshold={self.error_threshold}, "
            f"total_entries={len(parsed_logs)}"
        )

        for start in range(len(parsed_logs) - self.window_size + 1):
            window = parsed_logs[start : start + self.window_size]
            error_entries = [e for e in window if e["level"] in ERROR_LEVELS]
            error_count = len(error_entries)

            if error_count < self.error_threshold:
                continue

            # Cooldown check — suppress if too close to the last incident
            if last_incident_start is not None:
                if start - last_incident_start < cooldown:
                    continue

            last_incident_start = start
            severity = severity_from_count(error_count)

            # Try to extract a timestamp from the first log in the window
            window_ts = next(
                (e["timestamp"] for e in window if e.get("timestamp")), None
            )

            incident = Incident(
                anomaly_type="SLIDING_WINDOW_SPIKE",
                severity=severity,
                source=self.source,
                error_count=error_count,
                window_seconds=self.window_size,  # semantically "window entries" here
                details=(
                    f"Error burst detected: {error_count} errors in a window of "
                    f"{self.window_size} log entries starting at index {start}"
                    + (f" (approx. {window_ts})" if window_ts else "") + "."
                ),
                sample_errors=_sample_errors(error_entries),
            )

            print(f"  → Incident created: {incident}")
            incidents.append(incident)

        if not incidents:
            print("  → No sliding-window incidents triggered.")

        return incidents


# ---------------------------------------------------------------------------
# Composite pipeline helper
# ---------------------------------------------------------------------------

def run_all_detectors(
    parsed_logs: List[Dict],
    source: str = "system-logs",
    global_threshold: int = 5,
    window_size: int = 10,
    window_error_threshold: int = 3,
) -> List[Incident]:
    """
    Convenience function: run both detectors and return combined incident list.
    De-duplicates incidents that share the same anomaly_type and error_count.
    """
    detectors = [
        GlobalErrorSpikeDetector(threshold=global_threshold, source=source),
        SlidingWindowDetector(
            window_size=window_size,
            error_threshold=window_error_threshold,
            source=source,
        ),
    ]

    all_incidents: List[Incident] = []
    for detector in detectors:
        all_incidents.extend(detector.detect(parsed_logs))

    return all_incidents