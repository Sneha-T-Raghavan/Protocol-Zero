"""
agents/rca_signal.py
---------------------
Protocol Zero — Phase 3.5 upgraded output contracts.

Changes from Phase 3:
    InvestigationReport gains:
        - ranked_hypotheses       : Ordered list of all agent hypotheses with scores
        - cascade_candidates      : Ordered list of likely cascade origin components
        - evidence_logs           : Representative error log lines (evidence)
        - evidence_metrics        : Key numeric metrics from agent supporting_data
        - agent_contributions     : Per-agent summary of what each contributed
        - conflict_notes          : Where agents disagreed and how it was resolved
        - agreement_pairs         : Where agents agreed (cross-validation signal)

    RCASignal is unchanged — agents do not need modification.

Backward compatibility:
    All new fields default to empty list/dict so existing JSON deserialisation
    and correlation code (Phase 4) is unaffected.
"""

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RCASignal:
    """
    Structured Root Cause Analysis signal produced by a single agent
    for a single incident. Unchanged from Phase 3.
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
    findings:         List[str]      = field(default_factory=list)
    recommendations:  List[str]      = field(default_factory=list)
    supporting_data:  Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0–1.0, got {self.confidence}")
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
    Merged, fused output for one incident — Phase 3.5 upgrade.

    Core fields (Phase 3, unchanged)
    ---------------------------------
    report_id            : Deterministic MD5 hash of incident_id
    incident_id          : The incident this report covers
    incident_summary     : One-line summary of the incident
    timestamp            : UTC ISO-8601 time of report creation
    signals              : All RCASignals from all agents
    consensus_hypothesis : Fused best hypothesis (from signal_fusion)
    overall_confidence   : Fused final confidence (from signal_fusion)
    top_components       : Most affected components (from InfraAgent)
    top_error_patterns   : Most common error templates (from LogAgent)
    timeline             : Ordered first-occurrence events
    action_items         : Deduplicated union of all recommendations

    Phase 3.5 additions
    --------------------
    ranked_hypotheses    : All agent hypotheses ranked by fused weight score
                           [{"rank", "agent", "hypothesis", "confidence", "score"}]
    cascade_candidates   : Ordered list of likely cascade-origin components
                           ["KERNEL", "TORUS", "LINKCARD"]
    evidence_logs        : Representative raw error log lines used as evidence
                           [{"template", "count", "examples", "pattern_id"}]
    evidence_metrics     : Key numeric metrics extracted from agent data
                           {"peak_density_pct": 52, "error_velocity": 19.7, ...}
    agent_contributions  : Per-agent summary of what each contributed to the report
                           {"LogInvestigationAgent": {"weight": 0.41, "key_finding": ...}}
    conflict_notes       : Human-readable descriptions of agent disagreements
    agreement_pairs      : Agent pairs that agreed (cross-validation)
    """

    # ── Core (Phase 3) ────────────────────────────────────────────────────────
    incident_id:      str
    incident_summary: str

    report_id:  str = field(default="")
    timestamp:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    signals:              List[RCASignal] = field(default_factory=list)

    consensus_hypothesis: str            = ""
    overall_confidence:   float          = 0.0
    top_components:       List[str]      = field(default_factory=list)
    top_error_patterns:   List[str]      = field(default_factory=list)
    timeline:             List[str]      = field(default_factory=list)
    action_items:         List[str]      = field(default_factory=list)

    # ── Phase 3.5 additions ───────────────────────────────────────────────────
    ranked_hypotheses:    List[Dict]     = field(default_factory=list)
    cascade_candidates:   List[str]      = field(default_factory=list)
    evidence_logs:        List[Dict]     = field(default_factory=list)
    evidence_metrics:     Dict[str, Any] = field(default_factory=dict)
    agent_contributions:  Dict[str, Any] = field(default_factory=dict)
    conflict_notes:       List[str]      = field(default_factory=list)
    agreement_pairs:      List[str]      = field(default_factory=list)

    def __post_init__(self):
        if not self.report_id:
            fp = f"report|{self.incident_id}"
            self.report_id = hashlib.md5(fp.encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[InvestigationReport] incident={self.incident_id[:8]} | "
            f"agents={len(self.signals)} | "
            f"confidence={self.overall_confidence:.0%} | "
            f"{self.consensus_hypothesis[:70]}"
        )