"""
agents/context_agent.py
------------------------
Context Extraction Agent — Phase 3, Protocol Zero.

Responsibility:
    Dataset-aware context extraction. Understands the specific semantics
    of the log source (BGL, generic, etc.) and extracts contextual signals
    that neither the Log Investigation Agent nor the Infra Agent can see:

    For BGL logs:
        - Ground-truth label correlation (what % of errors have anomaly labels?)
        - Node rack / midplane topology (R02-M1 style naming)
        - Component co-failure analysis (which components fail together?)
        - BGL-specific failure categories (ECC memory, CIOD, DMA, Torus, etc.)

    For generic logs:
        - Temporal gap analysis (are there suspicious log silences?)
        - Level transition analysis (INFO → ERROR sequences)
        - Message deduplication ratio (are there flood patterns?)

    The context agent enriches the investigation with dataset-specific
    knowledge that other agents intentionally ignore.
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set

from core.incident import Incident
from agents.base_agent import BaseAgent
from agents.rca_signal import RCASignal

ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}

# BGL-specific failure category keywords mapped to human-readable labels
BGL_FAILURE_CATEGORIES = {
    "memory":     ["memory", "ecc", "ddr", "dimm", "correctable", "uncorrectable"],
    "network":    ["torus", "link", "network", "routing", "packet"],
    "io":         ["ciod", "ionode", "io node", "ciostream", "socket"],
    "compute":    ["kernel", "tlb", "cache", "instruction", "interrupt"],
    "job":        ["job", "rank", "mpi", "exitstatus", "abort", "killed"],
    "storage":    ["disk", "filesystem", "mount", "write", "read"],
    "dma":        ["dma", "crossbar", "register"],
}

# BGL node topology regex: R<rack>-M<midplane>-N<node>-...
_BGL_NODE_RE = re.compile(r"R(\d+)-M(\d+)")


class ContextAgent(BaseAgent):
    """
    Dataset-aware context extraction agent.

    Extracts BGL-specific signals when component/node/label fields are
    present, and falls back to generic temporal/structural analysis
    for plain log formats.
    """

    name        = "ContextAgent"
    description = "Dataset-aware context extraction, BGL topology & label correlation"

    # ── Public interface ──────────────────────────────────────────────────────

    def investigate(self, incident: Incident, parsed_logs: List[Dict]) -> RCASignal:
        print(f"  [{self.name}] Investigating incident {incident.incident_id[:8]} ...")

        is_bgl      = self._detect_bgl_format(parsed_logs)
        scoped_logs  = self._scope_logs(incident, parsed_logs)
        error_entries = [e for e in scoped_logs if e["level"] in ERROR_LEVELS]

        if is_bgl:
            return self._investigate_bgl(incident, scoped_logs, error_entries)
        else:
            return self._investigate_generic(incident, scoped_logs, error_entries)

    # ── BGL-specific investigation ────────────────────────────────────────────

    def _investigate_bgl(
        self,
        incident:      Incident,
        parsed_logs:   List[Dict],
        error_entries: List[Dict],
    ) -> RCASignal:
        label_stats       = self._analyse_labels(parsed_logs)
        rack_stats        = self._analyse_rack_topology(error_entries)
        failure_cats      = self._categorise_bgl_failures(error_entries)
        cofailure         = self._detect_component_cofailure(parsed_logs)
        level_transitions = self._analyse_level_transitions(parsed_logs)

        top_cats   = sorted(failure_cats.items(), key=lambda x: -x[1])[:4]
        top_racks  = rack_stats["top_racks"][:3]

        hypothesis = self._build_bgl_hypothesis(
            top_cats, top_racks, rack_stats, cofailure, label_stats
        )
        findings   = self._build_bgl_findings(
            label_stats, rack_stats, top_cats, cofailure, level_transitions
        )
        recs       = self._build_bgl_recommendations(top_cats, rack_stats, cofailure)
        confidence = self._score_bgl_confidence(error_entries, label_stats, top_cats)

        supporting = {
            "dataset_format":      "BGL",
            "label_correlation":   label_stats,
            "rack_topology":       rack_stats,
            "failure_categories":  [{"category": c, "count": n} for c, n in top_cats],
            "component_cofailure": cofailure,
            "level_transitions":   level_transitions,
        }

        signal = RCASignal(
            agent_name=self.name,
            incident_id=incident.incident_id,
            confidence=confidence,
            root_cause_hypothesis=hypothesis,
            findings=findings,
            recommendations=recs,
            supporting_data=supporting,
        )
        print(f"    → {signal}")
        return signal

    def _analyse_labels(self, parsed_logs: List[Dict]) -> Dict:
        """Correlate BGL anomaly labels with detected error levels."""
        total        = len(parsed_logs)
        labelled     = sum(1 for e in parsed_logs if e.get("is_anomaly"))
        error_labelled = sum(
            1 for e in parsed_logs
            if e.get("is_anomaly") and e["level"] in ERROR_LEVELS
        )
        label_types = Counter(
            e.get("label") for e in parsed_logs
            if e.get("label") and e.get("label") != "-"
        )
        return {
            "total_lines":        total,
            "labelled_anomalies": labelled,
            "label_rate_pct":     round(labelled / max(total, 1) * 100, 2),
            "error_labelled":     error_labelled,
            "label_types":        dict(label_types.most_common(5)),
            "gt_coverage":        round(error_labelled / max(labelled, 1) * 100, 1),
        }

    def _analyse_rack_topology(self, error_entries: List[Dict]) -> Dict:
        """Extract which racks and midplanes are generating errors."""
        rack_counter = Counter()
        midplane_counter = Counter()
        for e in error_entries:
            node = e.get("node", "")
            m = _BGL_NODE_RE.match(node)
            if m:
                rack      = f"R{m.group(1)}"
                midplane  = f"R{m.group(1)}-M{m.group(2)}"
                rack_counter[rack] += 1
                midplane_counter[midplane] += 1

        total_racks = len(rack_counter)
        top_racks   = rack_counter.most_common(5)

        return {
            "total_racks_affected": total_racks,
            "top_racks":            [{"rack": r, "errors": c} for r, c in top_racks],
            "top_midplanes":        [{"midplane": m, "errors": c}
                                     for m, c in midplane_counter.most_common(3)],
            "spread":               "localised" if total_racks <= 2 else
                                    "moderate"  if total_racks <= 5 else "wide",
        }

    def _categorise_bgl_failures(self, error_entries: List[Dict]) -> Counter:
        """Map error messages to BGL failure category buckets."""
        cats: Counter = Counter()
        for e in error_entries:
            msg = e.get("message", "").lower()
            matched = False
            for cat, keywords in BGL_FAILURE_CATEGORIES.items():
                if any(kw in msg for kw in keywords):
                    cats[cat] += 1
                    matched = True
                    break
            if not matched:
                cats["other"] += 1
        return cats

    def _detect_component_cofailure(self, parsed_logs: List[Dict]) -> Dict:
        """
        Detect which pairs of components fail within close proximity (50 entries).
        Co-failure pairs suggest causal relationships.
        """
        window = 50
        pairs: Counter = Counter()
        for i, e in enumerate(parsed_logs):
            if e["level"] not in ERROR_LEVELS:
                continue
            comp_a = e.get("component")
            if not comp_a:
                continue
            nearby = parsed_logs[i+1: i+window]
            nearby_comps: Set[str] = set()
            for nb in nearby:
                if nb["level"] in ERROR_LEVELS:
                    comp_b = nb.get("component")
                    if comp_b and comp_b != comp_a and comp_b not in nearby_comps:
                        pairs[(comp_a, comp_b)] += 1
                        nearby_comps.add(comp_b)

        top_pairs = [
            {"pair": f"{a} + {b}", "co_occurrences": c}
            for (a, b), c in pairs.most_common(5)
        ]
        return {
            "top_cofailure_pairs": top_pairs,
            "total_unique_pairs":  len(pairs),
        }

    def _analyse_level_transitions(self, parsed_logs: List[Dict]) -> Dict:
        """Count INFO→ERROR, WARNING→ERROR, ERROR→CRITICAL transitions."""
        transitions: Counter = Counter()
        for i in range(1, len(parsed_logs)):
            prev_level = parsed_logs[i-1]["level"]
            curr_level = parsed_logs[i]["level"]
            if curr_level in ERROR_LEVELS and prev_level not in ERROR_LEVELS:
                transitions[f"{prev_level}→{curr_level}"] += 1
        return dict(transitions.most_common(8))

    def _build_bgl_hypothesis(self, top_cats, top_racks, rack_stats, cofailure, label_stats):
        parts = []
        if top_cats:
            cat, cnt = top_cats[0]
            parts.append(f"Primary failure category: {cat.upper()} ({cnt} errors)")
        if top_racks:
            rack   = top_racks[0]["rack"]
            spread = rack_stats.get("spread", "unknown").upper()
            parts.append(f"Most affected rack: {rack} | spread: {spread}")
        if cofailure["top_cofailure_pairs"]:
            pair = cofailure["top_cofailure_pairs"][0]["pair"]
            parts.append(f"Strongest co-failure: {pair}")
        coverage = label_stats.get("gt_coverage", 0)
        if coverage > 80:
            parts.append(f"High ground-truth label coverage ({coverage}%) confirms systematic failure")
        return " | ".join(parts) if parts else "Insufficient BGL context for hypothesis."

    def _build_bgl_findings(self, label_stats, rack_stats, top_cats, cofailure, transitions):
        findings = []
        findings.append(
            f"BGL ground-truth labels: {label_stats['labelled_anomalies']} anomalous lines "
            f"({label_stats['label_rate_pct']}% of dataset). "
            f"Label types: {label_stats['label_types']}"
        )
        spread = rack_stats.get("spread", "unknown")
        findings.append(
            f"Error spread: {spread.upper()} — "
            f"{rack_stats['total_racks_affected']} rack(s) affected."
        )
        if top_cats:
            cat_str = ", ".join(f"{c}({n})" for c, n in top_cats)
            findings.append(f"BGL failure categories: {cat_str}")
        if cofailure["top_cofailure_pairs"]:
            pair_str = ", ".join(
                f"{p['pair']}({p['co_occurrences']})"
                for p in cofailure["top_cofailure_pairs"][:3]
            )
            findings.append(f"Component co-failure pairs: {pair_str}")
        if transitions:
            trans_str = ", ".join(f"{k}({v})" for k, v in list(transitions.items())[:4])
            findings.append(f"Log level transition events: {trans_str}")
        return findings

    def _build_bgl_recommendations(self, top_cats, rack_stats, cofailure):
        recs = []
        if top_cats:
            cat = top_cats[0][0]
            if cat == "memory":
                recs.append("Run memory diagnostics (memtest) on affected nodes — ECC errors indicate hardware fault.")
            elif cat == "network":
                recs.append("Inspect torus network links — high torus error count suggests fabric instability.")
            elif cat == "io":
                recs.append("Check I/O node health — CIOD failures indicate I/O subsystem disruption.")
            elif cat == "compute":
                recs.append("Review kernel exception logs — TLB/cache errors may indicate compute node fault.")
            elif cat == "job":
                recs.append("Investigate job scheduler and MPI communicator health — application-level failures.")
            else:
                recs.append(f"Investigate '{cat}' subsystem — dominant failure category.")
        spread = rack_stats.get("spread", "")
        if spread == "localised":
            recs.append("Failure is LOCALISED — consider isolating the affected rack for hardware inspection.")
        elif spread == "wide":
            recs.append("Failure is WIDE-SPREAD — may indicate a systemic issue (firmware, cooling, power).")
        if cofailure["top_cofailure_pairs"]:
            pair = cofailure["top_cofailure_pairs"][0]["pair"]
            recs.append(f"Investigate causal link between co-failing components: {pair}")
        return recs

    def _score_bgl_confidence(self, error_entries, label_stats, top_cats):
        volume   = min(len(error_entries) / 50.0, 1.0)
        coverage = label_stats.get("gt_coverage", 0) / 100.0
        has_cats = 0.9 if top_cats else 0.3
        return round(volume * 0.3 + coverage * 0.4 + has_cats * 0.3, 2)

    # ── Generic investigation ─────────────────────────────────────────────────

    def _investigate_generic(
        self,
        incident:      Incident,
        parsed_logs:   List[Dict],
        error_entries: List[Dict],
    ) -> RCASignal:
        silence_gaps  = self._detect_log_silences(parsed_logs)
        transitions   = self._analyse_level_transitions(parsed_logs)
        dedup_ratio   = self._compute_dedup_ratio(error_entries)

        hypothesis = (
            f"Generic log context: {len(error_entries)} errors detected. "
            f"Dedup ratio: {dedup_ratio:.0%} unique messages. "
            + (f"Log silence gaps: {len(silence_gaps)}." if silence_gaps else "No silence gaps.")
        )
        findings = [
            f"Analysed {len(parsed_logs)} total log entries (generic format).",
            f"Error deduplication ratio: {dedup_ratio:.1%} unique messages (low = flood pattern).",
            f"Level transitions into error: {transitions}",
        ]
        if silence_gaps:
            findings.append(f"Detected {len(silence_gaps)} log silence gap(s) > 10 entries: {silence_gaps[:3]}")

        signal = RCASignal(
            agent_name=self.name,
            incident_id=incident.incident_id,
            confidence=0.5,
            root_cause_hypothesis=hypothesis,
            findings=findings,
            recommendations=[
                "Check for log rotation or collection gaps during silence periods.",
                "Review repeated error messages for flood/loop patterns.",
            ],
            supporting_data={
                "dataset_format":  "generic",
                "dedup_ratio":     dedup_ratio,
                "silence_gaps":    silence_gaps,
                "level_transitions": transitions,
            },
        )
        print(f"    → {signal}")
        return signal

    def _detect_log_silences(self, parsed_logs: List[Dict]) -> List[int]:
        """Detect indices where there is a gap of 10+ consecutive non-error entries."""
        gaps = []
        run = 0
        for i, e in enumerate(parsed_logs):
            if e["level"] not in ERROR_LEVELS:
                run += 1
            else:
                if run >= 10:
                    gaps.append(i - run)
                run = 0
        return gaps

    def _compute_dedup_ratio(self, error_entries: List[Dict]) -> float:
        if not error_entries:
            return 1.0
        unique = len({e.get("message", "") for e in error_entries})
        return unique / len(error_entries)

    def _scope_logs(self, incident: Incident, parsed_logs: List[Dict]) -> List[Dict]:
        """
        Narrow the log corpus to the window relevant to this incident.
        Mirrors InfraAgent._scope_logs so all agents analyse the same window.
        For ERROR_SPIKE incidents the full corpus is used (global analysis intended).
        """
        if incident.anomaly_type != "SLIDING_WINDOW_SPIKE" or not incident.sample_errors:
            return parsed_logs

        anchor_msg = incident.sample_errors[0]
        anchor_idx = None
        for i, entry in enumerate(parsed_logs):
            if anchor_msg in entry.get("message", "") or anchor_msg in entry.get("raw", ""):
                anchor_idx = i
                break

        if anchor_idx is None:
            return parsed_logs

        half  = 75
        start = max(0, anchor_idx - half)
        end   = min(len(parsed_logs), anchor_idx + half)
        return parsed_logs[start:end]

    def _detect_bgl_format(self, parsed_logs: List[Dict]) -> bool:
        """Detect BGL format by checking for BGL-specific fields on first entries."""
        sample = parsed_logs[:10]
        return any(e.get("node") or e.get("bgl_level") for e in sample)