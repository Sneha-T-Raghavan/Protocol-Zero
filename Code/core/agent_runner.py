"""
core/agent_runner.py
---------------------
Protocol Zero — Phase 3.5 upgrade.

Changes from Phase 3:
    _merge_signals() now routes through the signal_fusion pipeline:
        normalize → weight → resolve conflicts → rank hypotheses

    _build_evidence_metrics()   — extracts key numeric metrics from agent data
    _build_agent_contributions() — per-agent summary for explainability
    _build_cascade_candidates()  — ordered cascade origin list
    _build_evidence_logs()       — pulls structured templates from LogAgent

    All other orchestration logic (agent execution, save/load, DEFAULT_AGENTS)
    is unchanged from Phase 3.

Architecture:
    - AgentRunner is still the ONLY orchestration point.
    - Agents do NOT call each other.
    - signal_fusion is called inside _merge_signals() only.
"""

import json
import os
import time
from typing import Dict, List, Optional

from core.incident       import Incident
from agents.base_agent   import BaseAgent
from agents.rca_signal   import RCASignal, InvestigationReport
from core.signal_fusion  import combine_signals

from agents.log_investigation_agent import LogInvestigationAgent
from agents.infra_agent             import InfraAgent
from agents.context_agent           import ContextAgent

DEFAULT_AGENTS: List[BaseAgent] = [
    LogInvestigationAgent(context_window=50),
    InfraAgent(bucket_size=50),
    ContextAgent(),
]


class AgentRunner:
    """
    Orchestrates multi-agent investigation for a list of incidents.

    Usage:
        runner  = AgentRunner()
        reports = runner.run(incidents, parsed_logs)
        runner.save_reports(reports, "outputs/investigation_reports.json")
    """

    def __init__(self, agents: Optional[List[BaseAgent]] = None):
        self.agents = agents if agents is not None else DEFAULT_AGENTS
        print(
            f"[AgentRunner] Initialised with {len(self.agents)} agent(s): "
            + ", ".join(a.name for a in self.agents)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        incidents:   List[Incident],
        parsed_logs: List[Dict],
    ) -> List[InvestigationReport]:
        if not incidents:
            print("[AgentRunner] No incidents to investigate.")
            return []

        t_start = time.time()
        print(f"\n[AgentRunner] Starting investigation of {len(incidents)} incident(s) ...")
        print(f"[AgentRunner] Log corpus: {len(parsed_logs):,} entries\n")

        reports = []
        for idx, incident in enumerate(incidents, 1):
            print(f"  ── Incident {idx}/{len(incidents)}: {incident} ──")
            report = self._investigate_incident(incident, parsed_logs)
            reports.append(report)
            print(f"  ✓ Report: {report}\n")

        elapsed = time.time() - t_start
        print(
            f"[AgentRunner] Investigation complete. "
            f"{len(reports)} report(s) in {elapsed:.2f}s."
        )
        return reports

    def save_reports(
        self,
        reports:   List[InvestigationReport],
        filepath:  str,
        overwrite: bool = True,
    ) -> None:
        if not reports:
            print("[AgentRunner] No reports to save.")
            return

        new_dicts = [r.to_dict() for r in reports]

        if overwrite or not os.path.exists(filepath):
            merged = new_dicts
        else:
            existing = []
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, ValueError):
                existing = []

            existing_ids = {r["report_id"] for r in existing}
            truly_new = [d for d in new_dicts if d["report_id"] not in existing_ids]
            merged = existing + truly_new

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        print(
            f"[AgentRunner] Saved {len(reports)} report(s) "
            f"(total in file: {len(merged)}) → '{filepath}'"
        )

    # ── Internal: orchestration ───────────────────────────────────────────────

    def _investigate_incident(
        self,
        incident:    Incident,
        parsed_logs: List[Dict],
    ) -> InvestigationReport:
        """Run all agents on a single incident and merge into a report."""
        signals: List[RCASignal] = []

        for agent in self.agents:
            try:
                signal = agent.investigate(incident, parsed_logs)
                signals.append(signal)
            except Exception as exc:
                print(f"    [WARNING] {agent.name} raised exception: {exc}")
                signals.append(RCASignal(
                    agent_name=agent.name,
                    incident_id=incident.incident_id,
                    confidence=0.0,
                    root_cause_hypothesis=f"Agent failed with error: {exc}",
                    findings=[f"Agent {agent.name} encountered an error: {exc}"],
                    recommendations=["Review agent code for the above exception."],
                ))

        return self._merge_signals(incident, signals)

    # ── Internal: Phase 3.5 merge pipeline ───────────────────────────────────

    def _merge_signals(
        self,
        incident: Incident,
        signals:  List[RCASignal],
    ) -> InvestigationReport:
        """
        Merge agent signals into an InvestigationReport using the
        Phase 3.5 signal fusion pipeline.

        Replaces the naïve "take highest confidence" logic with:
            normalize → weight → conflict resolve → rank
        """
        if not signals:
            return InvestigationReport(
                incident_id=incident.incident_id,
                incident_summary=str(incident),
            )

        # ── Phase 3.5: signal fusion ──────────────────────────────────────────
        fused = combine_signals(signals)

        # ── Extract structured fields from agent supporting_data ──────────────
        top_components    = self._extract_top_components(signals)
        top_patterns      = self._extract_top_patterns(signals)
        timeline          = self._extract_timeline(signals)
        action_items      = self._deduplicate_recommendations(signals, fused.agent_weights)
        cascade_candidates = self._build_cascade_candidates(signals)
        evidence_logs     = self._build_evidence_logs(signals)
        evidence_metrics  = self._build_evidence_metrics(signals)
        agent_contribs    = self._build_agent_contributions(signals, fused)

        return InvestigationReport(
            incident_id=incident.incident_id,
            incident_summary=(
                f"{incident.anomaly_type} | {incident.severity} | "
                f"errors={incident.error_count} | source={incident.source}"
            ),
            signals=signals,
            # Core hypothesis + confidence now come from fusion engine
            consensus_hypothesis=fused.weighted_hypothesis,
            overall_confidence=fused.final_confidence,
            # Structured fields
            top_components=top_components,
            top_error_patterns=top_patterns,
            timeline=timeline,
            action_items=action_items,
            # Phase 3.5 additions
            ranked_hypotheses=fused.ranked_hypotheses,
            cascade_candidates=cascade_candidates,
            evidence_logs=evidence_logs,
            evidence_metrics=evidence_metrics,
            agent_contributions=agent_contribs,
            conflict_notes=fused.conflict_notes,
            agreement_pairs=fused.agreement_pairs,
        )

    # ── Field extractors (Phase 3 — unchanged) ────────────────────────────────

    def _extract_top_components(self, signals: List[RCASignal]) -> List[str]:
        for sig in signals:
            if sig.agent_name == "InfraAgent":
                comps = sig.supporting_data.get("component_error_counts", [])
                return [c["component"] for c in comps[:5]]
        return []

    def _extract_top_patterns(self, signals: List[RCASignal]) -> List[str]:
        for sig in signals:
            if sig.agent_name == "LogInvestigationAgent":
                templates = sig.supporting_data.get("top_templates", [])
                return [t["template"][:80] for t in templates[:5]]
        return []

    def _extract_timeline(self, signals: List[RCASignal]) -> List[str]:
        for sig in signals:
            if sig.agent_name == "LogInvestigationAgent":
                return sig.supporting_data.get("timeline_events", [])
        return []

    def _deduplicate_recommendations(
        self,
        signals: List[RCASignal],
        agent_weights: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Collect all recommendations, deduplicating and ordering by agent weight.
        Phase 3.5: uses fused weights if available, otherwise falls back to confidence.
        """
        if agent_weights:
            sorted_signals = sorted(
                signals,
                key=lambda s: -(agent_weights.get(s.agent_name, 0.0)),
            )
        else:
            sorted_signals = sorted(signals, key=lambda s: -s.confidence)

        seen: set = set()
        result = []
        for sig in sorted_signals:
            for rec in sig.recommendations:
                norm = rec.strip().lower()
                if norm not in seen:
                    seen.add(norm)
                    result.append(rec)
        return result

    # ── Phase 3.5 new extractors ──────────────────────────────────────────────

    def _build_cascade_candidates(self, signals: List[RCASignal]) -> List[str]:
        """
        Build an ordered list of cascade-origin component candidates.

        Sources (in priority order):
            1. InfraAgent cascade_sequence (already ordered by first-error appearance)
            2. ContextAgent component_cofailure pairs (most co-occurring pair first)
            3. InfraAgent component_error_counts order (fallback)
        """
        # Source 1: InfraAgent cascade sequence
        for sig in signals:
            if sig.agent_name == "InfraAgent":
                seq = sig.supporting_data.get("cascade_sequence", [])
                if len(seq) >= 2:
                    return seq

        # Source 2: ContextAgent co-failure pairs
        for sig in signals:
            if sig.agent_name == "ContextAgent":
                pairs = sig.supporting_data.get("component_cofailure", {}).get(
                    "top_cofailure_pairs", []
                )
                if pairs:
                    # Parse "COMP_A + COMP_B" string
                    first_pair = pairs[0].get("pair", "")
                    comps = [c.strip() for c in first_pair.split("+")]
                    if len(comps) == 2:
                        return comps

        # Source 3: top components by error count order (weakest signal)
        for sig in signals:
            if sig.agent_name == "InfraAgent":
                comps = sig.supporting_data.get("component_error_counts", [])
                return [c["component"] for c in comps[:4]]

        return []

    def _build_evidence_logs(self, signals: List[RCASignal]) -> List[Dict]:
        """
        Pull structured log templates from LogInvestigationAgent as evidence.
        Returns the top_templates list enriched with agent attribution.
        """
        for sig in signals:
            if sig.agent_name == "LogInvestigationAgent":
                templates = sig.supporting_data.get("top_templates", [])
                return [
                    {
                        "template": t.get("template", ""),
                        "count":    t.get("count", 0),
                        "source":   "LogInvestigationAgent",
                    }
                    for t in templates[:10]
                ]
        return []

    def _build_evidence_metrics(self, signals: List[RCASignal]) -> Dict:
        """
        Extract key numeric metrics from agent supporting_data into a flat dict
        for easy consumption by downstream systems (correlation, dashboards).

        Pulls from InfraAgent and ContextAgent.
        """
        metrics: Dict = {}

        for sig in signals:
            if sig.agent_name == "InfraAgent":
                sd = sig.supporting_data
                burst = sd.get("burst_stats", {})
                metrics["peak_density_pct"]      = burst.get("peak_density_pct", 0)
                metrics["burst_shape"]           = burst.get("shape", "unknown")
                metrics["sustained_buckets"]     = burst.get("sustained_buckets", 0)
                metrics["error_velocity_per_100"] = sd.get("error_velocity_per_100", 0)
                metrics["total_error_entries"]   = sd.get("total_error_entries", 0)
                metrics["cascade_detected"]      = sd.get("cascade_detected", False)

            if sig.agent_name == "LogInvestigationAgent":
                sd = sig.supporting_data
                metrics["unique_error_templates"] = sd.get("unique_templates", 0)
                metrics["error_count_in_scope"]  = sd.get("error_count_in_scope", 0)
                metrics["error_progression"]     = sd.get("error_progression", "unknown")

            if sig.agent_name == "ContextAgent":
                sd = sig.supporting_data
                label_info = sd.get("label_correlation", {})
                if label_info:
                    metrics["gt_label_rate_pct"]  = label_info.get("label_rate_pct", 0)
                    metrics["gt_coverage_pct"]    = label_info.get("gt_coverage", 0)
                rack_info = sd.get("rack_topology", {})
                if rack_info:
                    metrics["racks_affected"]     = rack_info.get("total_racks_affected", 0)
                    metrics["failure_spread"]     = rack_info.get("spread", "unknown")
                # Top failure category
                failure_cats = sd.get("failure_categories", [])
                if failure_cats:
                    metrics["primary_failure_category"] = failure_cats[0].get("category", "")

        return metrics

    def _build_agent_contributions(
        self,
        signals: List[RCASignal],
        fused,          # FusedSignal
    ) -> Dict:
        """
        Build a per-agent contribution summary for report explainability.

        Structure:
            {
                "LogInvestigationAgent": {
                    "weight": 0.41,
                    "confidence": 0.97,
                    "key_finding": "Most frequent error pattern (56x): ...",
                    "recommendation_count": 3,
                },
                ...
            }
        """
        contribs: Dict = {}
        for sig in signals:
            key_finding = sig.findings[0] if sig.findings else sig.root_cause_hypothesis[:80]
            contribs[sig.agent_name] = {
                "weight":               fused.agent_weights.get(sig.agent_name, 0.0),
                "confidence":           sig.confidence,
                "key_finding":          key_finding,
                "recommendation_count": len(sig.recommendations),
                "findings_count":       len(sig.findings),
            }
        return contribs


# ── Convenience function ──────────────────────────────────────────────────────

def run_investigation(
    incidents:   List[Incident],
    parsed_logs: List[Dict],
    output_file: str = "outputs/investigation_reports.json",
    overwrite:   bool = True,
    agents:      Optional[List[BaseAgent]] = None,
) -> List[InvestigationReport]:
    """
    One-call convenience: run full multi-agent investigation and save results.
    """
    runner  = AgentRunner(agents=agents)
    reports = runner.run(incidents, parsed_logs)
    runner.save_reports(reports, output_file, overwrite=overwrite)
    return reports