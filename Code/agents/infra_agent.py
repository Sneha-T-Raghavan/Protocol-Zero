"""
agents/infra_agent.py
----------------------
Infrastructure Agent — Phase 3, Protocol Zero.

Responsibility:
    Analyses the infrastructure-level signals embedded in the log corpus:
    error density over time, burst characterisation, node failure patterns,
    and component health topology. Operates on structured fields (node,
    component, timestamp) rather than free-text messages.

Focus areas:
    1. Burst analysis — width, peak, decay shape of error bursts
    2. Node concentration — are errors spread or isolated to specific nodes?
    3. Component health map — error rate per component
    4. Error velocity — rate of errors per unit of log-index time
    5. Cascade detection — does a failure in one component trigger others?
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

from core.incident import Incident
from core.incident import Incident
from agents.base_agent import BaseAgent
from agents.rca_signal import RCASignal

ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}


class InfraAgent(BaseAgent):
    """
    Infrastructure-level investigation agent.

    Analyses node, component, and temporal structure of errors to
    characterise burst shape, identify hotspot nodes/components, and
    detect cascade propagation patterns.
    """

    name        = "InfraAgent"
    description = "Error density, burst analysis, node/component topology mapping"

    def __init__(self, bucket_size: int = 50):
        """
        Args:
            bucket_size: Number of log entries per time bucket for
                         density analysis. Default 50 entries/bucket.
        """
        self.bucket_size = bucket_size

    # ── Public interface ──────────────────────────────────────────────────────

    def investigate(self, incident: Incident, parsed_logs: List[Dict]) -> RCASignal:
        print(f"  [{self.name}] Investigating incident {incident.incident_id[:8]} ...")
        scoped_logs = self._scope_logs(incident, parsed_logs)

        error_entries = [e for e in scoped_logs if e["level"] in ERROR_LEVELS]


        if not error_entries:
            return self._empty_signal(incident, "No error entries found for infra analysis.")

        # Core analyses
        density_profile  = self._compute_density_profile(scoped_logs)
        burst_stats      = self._characterise_burst(density_profile, incident)
        node_counts      = self._count_nodes(error_entries)
        component_counts = self._count_components(error_entries)
        cascade_signal   = self._detect_cascade(error_entries, component_counts)
        velocity         = self._compute_error_velocity(error_entries, scoped_logs)

        # Synthesise
        top_nodes = node_counts.most_common(5)
        top_comps = component_counts.most_common(5)

        hypothesis    = self._build_hypothesis(
            burst_stats, top_nodes, top_comps, cascade_signal, velocity
        )
        findings      = self._build_findings(
            error_entries, density_profile, burst_stats,
            top_nodes, top_comps, cascade_signal, velocity
        )
        recommendations = self._build_recommendations(
            burst_stats, top_nodes, top_comps, cascade_signal
        )
        confidence    = self._score_confidence(error_entries, burst_stats, top_comps)

        supporting = {
            "total_error_entries":   len(error_entries),
            "error_velocity_per_100": velocity,
            "burst_stats":           burst_stats,
            "node_error_counts":     [{"node": n, "errors": c} for n, c in top_nodes],
            "component_error_counts":[{"component": c, "errors": n} for c, n in top_comps],
            "cascade_detected":      cascade_signal["detected"],
            "cascade_sequence":      cascade_signal.get("sequence", []),
            "density_buckets":       density_profile[:20],  # cap for readability
        }

        signal = RCASignal(
            agent_name=self.name,
            incident_id=incident.incident_id,
            confidence=confidence,
            root_cause_hypothesis=hypothesis,
            findings=findings,
            recommendations=recommendations,
            supporting_data=supporting,
        )

        print(f"    → {signal}")
        return signal

    # ── Analysis methods ──────────────────────────────────────────────────────
    def _scope_logs(self, incident: Incident, parsed_logs: List[Dict]) -> List[Dict]:
        if incident.anomaly_type != "SLIDING_WINDOW_SPIKE" or not incident.sample_errors:
            return parsed_logs

        # Find the anchor: first sample error in the log list
        anchor_msg = incident.sample_errors[0]
        anchor_idx = None
        for i, entry in enumerate(parsed_logs):
            if anchor_msg in entry.get("message", "") or anchor_msg in entry.get("raw", ""):
                anchor_idx = i
                break

        if anchor_idx is None:
            return parsed_logs  # fallback: can't find anchor, use full corpus

        # Use a generous window around the anchor (context_window on each side)
        half = 75
        start = max(0, anchor_idx - half)
        end   = min(len(parsed_logs), anchor_idx + half)
        return parsed_logs[start:end]

    def _compute_density_profile(self, parsed_logs: List[Dict]) -> List[Dict]:
        """
        Divide all logs into fixed-size buckets and compute error density
        per bucket. Returns list of {bucket, total, errors, density_pct}.
        """
        profile = []
        for bucket_start in range(0, len(parsed_logs), self.bucket_size):
            bucket = parsed_logs[bucket_start: bucket_start + self.bucket_size]
            errors = sum(1 for e in bucket if e["level"] in ERROR_LEVELS)
            total  = len(bucket)
            profile.append({
                "bucket":      bucket_start // self.bucket_size,
                "start_index": bucket_start,
                "total":       total,
                "errors":      errors,
                "density_pct": round(errors / max(total, 1) * 100, 1),
            })
        return profile

    def _characterise_burst(
        self, density_profile: List[Dict], incident: Incident
    ) -> Dict:
        """
        Characterise the burst shape:
          - peak_density: maximum error density in any single bucket
          - peak_bucket: which bucket hit peak density
          - sustained_buckets: how many buckets have >20% error density
          - shape: "spike" (1 bucket), "sustained" (2–4), "flood" (5+)
        """
        if not density_profile:
            return {"peak_density": 0, "peak_bucket": 0, "sustained_buckets": 0, "shape": "unknown"}

        peak      = max(density_profile, key=lambda b: b["density_pct"])
        sustained = sum(1 for b in density_profile if b["density_pct"] > 20)

        shape = "spike"
        if sustained >= 5:
            shape = "flood"
        elif sustained >= 2:
            shape = "sustained"

        return {
            "peak_density_pct": peak["density_pct"],
            "peak_bucket":      peak["bucket"],
            "peak_start_index": peak["start_index"],
            "sustained_buckets": sustained,
            "total_buckets":    len(density_profile),
            "shape":            shape,
        }

    def _count_nodes(self, error_entries: List[Dict]) -> Counter:
        """Count errors per node (BGL-specific field; absent = no node data)."""
        c = Counter()
        for e in error_entries:
            node = e.get("node")
            if node:
                c[node] += 1
        return c

    def _count_components(self, error_entries: List[Dict]) -> Counter:
        c = Counter()
        for e in error_entries:
            comp = e.get("component")
            if comp:
                c[comp] += 1
        return c

    def _detect_cascade(
        self, error_entries: List[Dict], component_counts: Counter
    ) -> Dict:
        """
        Detect cascade propagation: does failure in component A precede
        failures in component B?

        Strategy: look at the first appearance order of each component's
        errors. If errors spread across 3+ components in sequence, flag as
        cascade.
        """
        first_seen: Dict[str, int] = {}
        for i, e in enumerate(error_entries):
            comp = e.get("component")
            if comp and comp not in first_seen:
                first_seen[comp] = i

        if len(first_seen) < 2:
            return {"detected": False, "sequence": []}

        ordered = sorted(first_seen.items(), key=lambda x: x[1])
        sequence = [comp for comp, _ in ordered]
        detected = len(sequence) >= 3

        return {
            "detected": detected,
            "sequence": sequence,
            "description": (
                f"Errors spread across {len(sequence)} components in sequence: "
                + " → ".join(sequence[:6])
                if detected else "No cascade detected"
            ),
        }

    def _compute_error_velocity(
        self, error_entries: List[Dict], parsed_logs: List[Dict]
    ) -> float:
        """
        Error velocity = errors per 100 log entries.
        """
        total = len(parsed_logs)
        errors = len(error_entries)
        if total == 0:
            return 0.0
        return round(errors / total * 100, 2)

    def _build_hypothesis(
        self,
        burst_stats:    Dict,
        top_nodes:      List[Tuple],
        top_comps:      List[Tuple],
        cascade_signal: Dict,
        velocity:       float,
    ) -> str:
        parts = []

        shape = burst_stats.get("shape", "unknown")
        peak  = burst_stats.get("peak_density_pct", 0)
        parts.append(f"Burst shape: {shape.upper()} (peak density {peak}%)")

        if top_comps:
            parts.append(f"Hotspot component: {top_comps[0][0]} ({top_comps[0][1]} errors)")

        if top_nodes and top_nodes[0][1] > 1:
            parts.append(f"Most affected node: {top_nodes[0][0]} ({top_nodes[0][1]} errors)")

        if cascade_signal["detected"]:
            seq = " → ".join(cascade_signal.get("sequence", [])[:4])
            parts.append(f"Cascade detected: {seq}")

        parts.append(f"Error velocity: {velocity} errors/100 log entries")

        return " | ".join(parts)

    def _build_findings(
        self,
        error_entries:  List[Dict],
        density_profile: List[Dict],
        burst_stats:    Dict,
        top_nodes:      List[Tuple],
        top_comps:      List[Tuple],
        cascade_signal: Dict,
        velocity:       float,
    ) -> List[str]:
        findings = []
        findings.append(
            f"Total error entries: {len(error_entries)} across "
            f"{len(density_profile)} log buckets of {self.bucket_size} entries each."
        )
        findings.append(
            f"Burst shape: {burst_stats['shape'].upper()} — "
            f"peak error density {burst_stats['peak_density_pct']}% "
            f"in bucket {burst_stats['peak_bucket']} "
            f"({burst_stats['sustained_buckets']} buckets above 20% threshold)."
        )
        findings.append(f"Error velocity: {velocity} errors per 100 log entries.")

        if top_comps:
            comp_str = ", ".join(f"{c}({n})" for c, n in top_comps[:5])
            findings.append(f"Component error distribution: {comp_str}")

        if top_nodes:
            node_str = ", ".join(f"{n}({c})" for n, c in top_nodes[:3])
            findings.append(f"Top error-generating nodes: {node_str}")
        else:
            findings.append("No node metadata available (non-BGL dataset or node field absent).")

        if cascade_signal["detected"]:
            findings.append(f"CASCADE DETECTED: {cascade_signal['description']}")
        else:
            findings.append("No clear cascade propagation pattern detected.")

        return findings

    def _build_recommendations(
        self,
        burst_stats:    Dict,
        top_nodes:      List[Tuple],
        top_comps:      List[Tuple],
        cascade_signal: Dict,
    ) -> List[str]:
        recs = []
        shape = burst_stats.get("shape", "unknown")

        if shape == "flood":
            recs.append("CRITICAL: Flood-level error burst — consider immediate service isolation.")
        elif shape == "sustained":
            recs.append("Sustained error burst — investigate root cause before errors escalate further.")
        elif shape == "spike":
            recs.append("Short error spike detected — investigate triggering event at peak timestamp.")

        if top_comps:
            recs.append(
                f"Prioritise investigation of component '{top_comps[0][0]}' "
                f"— highest error count ({top_comps[0][1]})."
            )
        if top_nodes:
            recs.append(
                f"Inspect node '{top_nodes[0][0]}' directly — "
                f"responsible for most errors in this incident."
            )
        if cascade_signal["detected"]:
            seq = cascade_signal.get("sequence", [])
            if seq:
                recs.append(
                    f"Cascade origin likely in first component: '{seq[0]}'. "
                    f"Isolate it to stop propagation."
                )
        recs.append("Review infrastructure change log for events preceding the burst start index.")
        return recs

    def _score_confidence(
        self,
        error_entries: List[Dict],
        burst_stats:   Dict,
        top_comps:     List[Tuple],
    ) -> float:
        """
        Confidence based on:
          - Volume of data (more = better)
          - Availability of component metadata (richer analysis)
          - Clarity of burst shape (non-unknown)
        """
        volume_score   = min(len(error_entries) / 100.0, 1.0)
        has_components = 1.0 if top_comps else 0.3
        shape_known    = 0.9 if burst_stats.get("shape") != "unknown" else 0.4
        return round((volume_score * 0.4 + has_components * 0.3 + shape_known * 0.3), 2)

    def _empty_signal(self, incident: Incident, reason: str) -> RCASignal:
        return RCASignal(
            agent_name=self.name,
            incident_id=incident.incident_id,
            confidence=0.0,
            root_cause_hypothesis=reason,
            findings=[reason],
            recommendations=["Verify that parsed logs contain node/component fields."],
        )