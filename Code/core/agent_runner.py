"""
core/agent_runner.py
---------------------
Protocol Zero — Phase 3: Multi-Agent Investigation Orchestrator.

Responsibility:
    Receives List[Incident] + parsed_logs from Phase 2.
    Runs all registered agents on each incident.
    Merges per-agent RCASignals into a single InvestigationReport per incident.
    Saves investigation reports to JSON.

Architecture:
    - AgentRunner is the ONLY orchestration point. Agents do not call each other.
    - Each agent is run independently and produces one RCASignal per incident.
    - Signals are merged deterministically (highest confidence wins for hypothesis).
    - Designed for easy extension: add new agents by registering them below.

Pipeline position:
    Phase 2:  load_logs → parse → detect → List[Incident]
    Phase 3:  List[Incident] + parsed_logs → AgentRunner → List[InvestigationReport]
"""

import json
import os
import time
from typing import List, Dict, Optional

from core.incident       import Incident
from agents.base_agent   import BaseAgent
from agents.rca_signal   import RCASignal, InvestigationReport

# ── Default agent registry ────────────────────────────────────────────────────
# Import all built-in agents here. To add a new agent, import it and add to
# DEFAULT_AGENTS list.

from agents.log_investigation_agent import LogInvestigationAgent
from agents.infra_agent             import InfraAgent
from agents.context_agent           import ContextAgent

DEFAULT_AGENTS: List[BaseAgent] = [
    LogInvestigationAgent(context_window=50),
    InfraAgent(bucket_size=50),
    ContextAgent(),
]


# ── AgentRunner ───────────────────────────────────────────────────────────────

class AgentRunner:
    """
    Orchestrates multi-agent investigation for a list of incidents.

    Usage:
        runner  = AgentRunner()
        reports = runner.run(incidents, parsed_logs)
        runner.save_reports(reports, "outputs/investigation_reports.json")
    """

    def __init__(self, agents: Optional[List[BaseAgent]] = None):
        """
        Args:
            agents: List of agent instances to run. Defaults to DEFAULT_AGENTS.
                    Pass a custom list to override.
        """
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
        """
        Run all agents on every incident and return one InvestigationReport
        per incident.

        Args:
            incidents   : List of Incident objects from Phase 2 detection.
            parsed_logs : Full parsed log list (same as fed to detectors).

        Returns:
            List of InvestigationReport objects, one per incident.
        """
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
        """
        Save InvestigationReport list to JSON.

        Args:
            reports   : List of InvestigationReport objects.
            filepath  : Output file path.
            overwrite : If True (default), replace file. If False, append
                        with deduplication by report_id.
        """
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

    # ── Internal methods ──────────────────────────────────────────────────────

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
                # One agent failing should never block the others
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

    def _merge_signals(
        self,
        incident: Incident,
        signals:  List[RCASignal],
    ) -> InvestigationReport:
        if not signals:
            return InvestigationReport(
                incident_id=incident.incident_id,
                incident_summary=str(incident),
            )

        best_signal   = max(signals, key=lambda s: s.confidence)
        overall_conf  = best_signal.confidence

        # FIX: consensus hypothesis should come from the most confident
        # *incident-specific* agent (InfraAgent or LogInvestigationAgent),
        # not ContextAgent whose score reflects dataset-level coverage.
        DATASET_AGENTS = {"ContextAgent"}
        incident_signals = [s for s in signals if s.agent_name not in DATASET_AGENTS]
        consensus_signal = max(
            incident_signals,
            key=lambda s: s.confidence,
            default=best_signal,   # fallback if only ContextAgent ran
        )
        consensus_hyp = consensus_signal.root_cause_hypothesis

        top_components = self._extract_top_components(signals)
        top_patterns   = self._extract_top_patterns(signals)
        timeline       = self._extract_timeline(signals)
        action_items   = self._deduplicate_recommendations(signals)

        return InvestigationReport(
            incident_id=incident.incident_id,
            incident_summary=(
                f"{incident.anomaly_type} | {incident.severity} | "
                f"errors={incident.error_count} | source={incident.source}"
            ),
            signals=signals,
            consensus_hypothesis=consensus_hyp,
            overall_confidence=overall_conf,
            top_components=top_components,
            top_error_patterns=top_patterns,
            timeline=timeline,
            action_items=action_items,
        )

    def _extract_top_components(self, signals: List[RCASignal]) -> List[str]:
        """Pull component list from InfraAgent signal."""
        for sig in signals:
            if sig.agent_name == "InfraAgent":
                comps = sig.supporting_data.get("component_error_counts", [])
                return [c["component"] for c in comps[:5]]
        return []

    def _extract_top_patterns(self, signals: List[RCASignal]) -> List[str]:
        """Pull error templates from LogInvestigationAgent signal."""
        for sig in signals:
            if sig.agent_name == "LogInvestigationAgent":
                templates = sig.supporting_data.get("top_templates", [])
                return [t["template"][:80] for t in templates[:5]]
        return []

    def _extract_timeline(self, signals: List[RCASignal]) -> List[str]:
        """Pull timeline from LogInvestigationAgent signal."""
        for sig in signals:
            if sig.agent_name == "LogInvestigationAgent":
                return sig.supporting_data.get("timeline_events", [])
        return []

    def _deduplicate_recommendations(self, signals: List[RCASignal]) -> List[str]:
        """
        Collect all recommendations from all agents and deduplicate.
        Preserves order: higher-confidence agents go first.
        """
        sorted_signals = sorted(signals, key=lambda s: -s.confidence)
        seen: set = set()
        result = []
        for sig in sorted_signals:
            for rec in sig.recommendations:
                normalised = rec.strip().lower()
                if normalised not in seen:
                    seen.add(normalised)
                    result.append(rec)
        return result


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

    Args:
        incidents   : From Phase 2 detection.
        parsed_logs : Full parsed log corpus.
        output_file : Where to write reports JSON.
        overwrite   : Overwrite or append to existing file.
        agents      : Custom agent list (None = use DEFAULT_AGENTS).

    Returns:
        List of InvestigationReport objects.
    """
    runner  = AgentRunner(agents=agents)
    reports = runner.run(incidents, parsed_logs)
    runner.save_reports(reports, output_file, overwrite=overwrite)
    return reports