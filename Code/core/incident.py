"""
core/incident.py
----------------
Defines the Incident data model used throughout Protocol Zero.

Incidents are the primary output artifact of the detection pipeline.
Each incident captures a detected anomaly with full context.
"""

import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# Severity levels in ascending order of impact
SEVERITY_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass
class Incident:
    """
    Structured representation of a detected system incident.

    Fields
    ------
    incident_id   : Unique identifier (UUID4 by default).
    timestamp     : ISO-8601 UTC timestamp of when the incident was created.
    anomaly_type  : Short label describing the detection method that fired
                    (e.g. "ERROR_SPIKE", "SLIDING_WINDOW_SPIKE").
    severity      : One of LOW / MEDIUM / HIGH / CRITICAL.
    source        : Origin of the log data (file path, service name, etc.).
    error_count   : Number of error-level log entries that triggered the incident.
    window_seconds: Size of the sliding window used (None for global detections).
    details       : Free-form human-readable description of the anomaly.
    sample_errors : Up to five representative error messages from the window.
    """

    # --- Required fields -------------------------------------------------------
    anomaly_type: str
    severity: str
    source: str
    error_count: int

    # --- Auto-populated fields -------------------------------------------------
    incident_id: str = field(default="")  # set in __post_init__ if empty
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- Optional context fields -----------------------------------------------
    window_seconds: Optional[int] = None
    details: str = ""
    sample_errors: list = field(default_factory=list)

    def __post_init__(self):
        if self.severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"Invalid severity '{self.severity}'. Must be one of {SEVERITY_LEVELS}"
            )
        # Generate deterministic ID from content so reruns produce the same ID
        # and the persistence layer can deduplicate across runs.
        if not self.incident_id:
            first_sample = self.sample_errors[0] if self.sample_errors else ""
            fingerprint = (
                f"{self.anomaly_type}|{self.source}|{self.error_count}|"
                f"{self.severity}|{self.details}|{first_sample}")
            self.incident_id = hashlib.md5(fingerprint.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Return incident as a plain dictionary (JSON-serialisable)."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[Incident {self.incident_id[:8]}] "
            f"{self.anomaly_type} | {self.severity} | "
            f"errors={self.error_count} | source={self.source}"
        )


def severity_from_count(error_count: int) -> str:
    """
    Map an error count to a severity label.

    Thresholds (configurable):
        1–4   → LOW
        5–9   → MEDIUM
        10–19 → HIGH
        20+   → CRITICAL
    """
    if error_count >= 20:
        return "CRITICAL"
    if error_count >= 10:
        return "HIGH"
    if error_count >= 5:
        return "MEDIUM"
    return "LOW"