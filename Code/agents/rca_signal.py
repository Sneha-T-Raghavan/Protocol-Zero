"""
agents/rca_signal.py
---------------------
Defines RCASignal — the structured output contract for all Phase 3 agents.

Every agent produces one RCASignal per incident it investigates.
The agent_runner collects all signals and merges them into a final
InvestigationReport per incident.

Design principles:
  - RCASignal is a pure data container (dataclass), not logic
  - All fields are JSON-serialisable
  - agent_name identifies which agent produced the signal
  - confidence is a float 0.0–1.0 (agents self-report their certainty)
  - findings is a list of discrete, human-readable observations
  - root_cause_hypothesis is the agent's best single explanation
  - recommendations is a list of actionable next steps
"""

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any


@dataclass
class RCASignal:
    """
    Structured Root Cause Analysis signal produced by a single agent
    for a single incident.

    Fields
    ------
    signal_id           : Deterministic MD5 hash of agent+incident_id
    agent_name          : Name of the agent that produced this signal
    incident_id         : ID of the Incident this signal is for
    timestamp           : UTC ISO-8601 time of signal creation
    confidence          : Agent's self-reported confidence (0.0 – 1.0)
    findings            : List of discrete observations (strings)
    root_cause_hypothesis: Best single explanation this agent can offer
    recommendations     : List of actionable remediation steps
    supporting_data     : Dict of agent-specific structured evidence
                          (e.g. cluster counts, burst timelines, node maps)
    """

    # Required
    agent_name:            str
    incident_id:           str
    confidence:            float
    root_cause_hypothesis: str

    # Auto-populated
    signal_id:  str = field(default="")
    timestamp:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Rich evidence fields
    findings:         List[str]        = field(default_factory=list)
    recommendations:  List[str]        = field(default_factory=list)
    supporting_data:  Dict[str, Any]   = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not self.signal_id:
            fp = f"{self.agent_name}|{self.incident_id}|{self.root_cause_hypothesis[:60]}"
            self.signal_id = hashlib.md5(fp.encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[RCASignal/{self.agent_name}] "
            f"incident={self.incident_id[:8]} | "
            f"confidence={self.confidence:.0%} | "
            f"{self.root_cause_hypothesis[:60]}"
        )


@dataclass
class InvestigationReport:
    """
    Merged output for one incident, combining signals from all agents.
    This is the final Phase 3 output artifact.

    Fields
    ------
    report_id       : Deterministic hash of incident_id
    incident_id     : The incident this report covers
    incident_summary: One-line summary of the incident
    timestamp       : UTC ISO-8601 time of report creation
    signals         : All RCASignals from all agents
    consensus_hypothesis : The highest-confidence hypothesis across agents
    overall_confidence   : Max confidence across all signals
    top_components  : Most affected components/services (from InfraAgent)
    top_error_patterns   : Most common error clusters (from LogInvestigationAgent)
    timeline        : Ordered list of key events extracted from logs
    action_items    : Deduplicated union of all agent recommendations
    """

    # Required
    incident_id:      str
    incident_summary: str

    # Auto-populated
    report_id:  str = field(default="")
    timestamp:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Agent outputs
    signals:              List[RCASignal]  = field(default_factory=list)

    # Merged / synthesised fields
    consensus_hypothesis: str             = ""
    overall_confidence:   float           = 0.0
    top_components:       List[str]       = field(default_factory=list)
    top_error_patterns:   List[str]       = field(default_factory=list)
    timeline:             List[str]       = field(default_factory=list)
    action_items:         List[str]       = field(default_factory=list)

    def __post_init__(self):
        if not self.report_id:
            fp = f"report|{self.incident_id}"
            self.report_id = hashlib.md5(fp.encode()).hexdigest()

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def __str__(self) -> str:
        return (
            f"[InvestigationReport] incident={self.incident_id[:8]} | "
            f"agents={len(self.signals)} | "
            f"confidence={self.overall_confidence:.0%} | "
            f"{self.consensus_hypothesis[:70]}"
        )