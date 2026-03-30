"""
agents/context_agent.py
------------------------
Protocol Zero — Phase 3.5 upgrade.

Changes from Phase 3:
    Three new BGL-specific analysis methods:
        detect_node_failure_patterns()   — rack/midplane clusters with persistent failure
        detect_component_hotspots()      — components far above baseline error rate
        detect_cluster_failures()        — job/MPI group failures from log messages

    These are called from _investigate_bgl() and added to supporting_data under
    the keys "node_failure_patterns", "component_hotspots", "cluster_failures".

    The BGL hypothesis and findings builders are extended to incorporate
    these new signals when available.

    All Phase 3 logic is preserved unchanged.
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set

from core.incident import Incident
from agents.base_agent import BaseAgent
from agents.rca_signal import RCASignal

ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}

BGL_FAILURE_CATEGORIES = {
    "memory":     ["memory", "ecc", "ddr", "dimm", "correctable", "uncorrectable"],
    "network":    ["torus", "link", "network", "routing", "packet"],
    "io":         ["ciod", "ionode", "io node", "ciostream", "socket"],
    "compute":    ["kernel", "tlb", "cache", "instruction", "interrupt"],
    "job":        ["job", "rank", "mpi", "exitstatus", "abort", "killed"],
    "storage":    ["disk", "filesystem", "mount", "write", "read"],
    "dma":        ["dma", "crossbar", "register"],
}

_BGL_NODE_RE  = re.compile(r"R(\d+)-M(\d+)")
_JOB_ID_RE    = re.compile(r"(?i)(?:job|jobid)\s*[=:]?\s*(\d+)")
_RANK_RE      = re.compile(r"(?i)rank\s+(\d+)")
_MPI_RE       = re.compile(r"(?i)mpi")


class ContextAgent(BaseAgent):

    name        = "ContextAgent"
    description = "Dataset-aware context extraction, BGL topology & label correlation"

    # ── Public interface ──────────────────────────────────────────────────────

    def investigate(self, incident: Incident, parsed_logs: List[Dict]) -> RCASignal:
        print(f"  [{self.name}] Investigating incident {incident.incident_id[:8]} ...")

        is_bgl        = self._detect_bgl_format(parsed_logs)
        scoped_logs   = self._scope_logs(incident, parsed_logs)
        error_entries = [e for e in scoped_logs if e["level"] in ERROR_LEVELS]

        if is_bgl:
            return self._investigate_bgl(incident, scoped_logs, error_entries)
        else:
            return self._investigate_generic(incident, scoped_logs, error_entries)

    # ── BGL investigation (Phase 3 + Phase 3.5 additions) ────────────────────

    def _investigate_bgl(
        self,
        incident:      Incident,
        parsed_logs:   List[Dict],
        error_entries: List[Dict],
    ) -> RCASignal:
        # Phase 3 analyses
        label_stats       = self._analyse_labels(parsed_logs)
        rack_stats        = self._analyse_rack_topology(error_entries)
        failure_cats      = self._categorise_bgl_failures(error_entries)
        cofailure         = self._detect_component_cofailure(parsed_logs)
        level_transitions = self._analyse_level_transitions(parsed_logs)

        top_cats  = sorted(failure_cats.items(), key=lambda x: -x[1])[:4]
        top_racks = rack_stats["top_racks"][:3]

        # Phase 3.5 additions
        node_failure_patterns = self.detect_node_failure_patterns(error_entries)
        component_hotspots    = self.detect_component_hotspots(parsed_logs, error_entries)
        cluster_failures      = self.detect_cluster_failures(error_entries)

        hypothesis = self._build_bgl_hypothesis(
            top_cats, top_racks, rack_stats, cofailure, label_stats,
            node_failure_patterns, component_hotspots, cluster_failures,
        )
        findings = self._build_bgl_findings(
            label_stats, rack_stats, top_cats, cofailure, level_transitions,
            node_failure_patterns, component_hotspots, cluster_failures,
        )
        recs       = self._build_bgl_recommendations(top_cats, rack_stats, cofailure,
                                                     node_failure_patterns, cluster_failures)
        confidence = self._score_bgl_confidence(error_entries, label_stats, top_cats)

        supporting = {
            "dataset_format":        "BGL",
            "label_correlation":     label_stats,
            "rack_topology":         rack_stats,
            "failure_categories":    [{"category": c, "count": n} for c, n in top_cats],
            "component_cofailure":   cofailure,
            "level_transitions":     level_transitions,
            # Phase 3.5
            "node_failure_patterns": node_failure_patterns,
            "component_hotspots":    component_hotspots,
            "cluster_failures":      cluster_failures,
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

    # ── Phase 3.5: new dataset-aware methods ─────────────────────────────────

    def detect_node_failure_patterns(self, error_entries: List[Dict]) -> Dict:
        """
        Identify racks/midplanes with persistently high error rates.

        A "persistent" failure is defined as a rack or midplane that appears
        in errors in more than one distinct component (suggesting the problem
        is hardware/location-based rather than software-based).

        Returns:
            {
                "persistent_racks":   [{"rack", "error_count", "components"}],
                "hotspot_midplanes":  [{"midplane", "error_count"}],
                "pattern_type":       "single_rack" | "multi_rack" | "none"
            }
        """
        rack_components: Dict[str, Set[str]] = defaultdict(set)
        rack_counts:     Counter              = Counter()
        midplane_counts: Counter              = Counter()

        for e in error_entries:
            node = e.get("node", "")
            comp = e.get("component", "")
            m = _BGL_NODE_RE.match(node)
            if m:
                rack     = f"R{m.group(1)}"
                midplane = f"R{m.group(1)}-M{m.group(2)}"
                rack_counts[rack] += 1
                midplane_counts[midplane] += 1
                if comp:
                    rack_components[rack].add(comp)

        # Persistent = rack has errors in 2+ distinct components
        persistent = [
            {
                "rack":        rack,
                "error_count": rack_counts[rack],
                "components":  sorted(rack_components[rack]),
            }
            for rack, comps in rack_components.items()
            if len(comps) >= 2
        ]
        persistent.sort(key=lambda x: -x["error_count"])

        hotspot_midplanes = [
            {"midplane": mp, "error_count": cnt}
            for mp, cnt in midplane_counts.most_common(5)
        ]

        if len(persistent) == 0:
            pattern_type = "none"
        elif len(persistent) == 1:
            pattern_type = "single_rack"
        else:
            pattern_type = "multi_rack"

        return {
            "persistent_racks":  persistent[:5],
            "hotspot_midplanes": hotspot_midplanes,
            "pattern_type":      pattern_type,
        }

    def detect_component_hotspots(
        self,
        all_logs:      List[Dict],
        error_entries: List[Dict],
    ) -> List[Dict]:
        """
        Identify components whose error rate far exceeds their baseline
        log volume rate (i.e., they appear in errors disproportionately).

        A component is a "hotspot" if its error_rate / baseline_rate > 2.0.
        (Baseline = fraction of all log entries from that component.)

        Returns:
            [{"component", "error_count", "total_count",
              "error_rate_pct", "baseline_rate_pct", "hotspot_ratio"}]
            sorted by hotspot_ratio descending.
        """
        total_log_count: Counter = Counter()
        error_count:     Counter = Counter()

        for e in all_logs:
            comp = e.get("component")
            if comp:
                total_log_count[comp] += 1

        for e in error_entries:
            comp = e.get("component")
            if comp:
                error_count[comp] += 1

        total_logs = max(len(all_logs), 1)
        total_errors = max(len(error_entries), 1)
        hotspots = []

        for comp, err_cnt in error_count.items():
            total_cnt    = total_log_count.get(comp, 1)
            err_rate     = err_cnt / total_errors
            baseline     = total_cnt / total_logs
            ratio        = err_rate / max(baseline, 1e-9)

            if ratio > 2.0:
                hotspots.append({
                    "component":        comp,
                    "error_count":      err_cnt,
                    "total_count":      total_cnt,
                    "error_rate_pct":   round(err_rate * 100, 2),
                    "baseline_rate_pct": round(baseline * 100, 2),
                    "hotspot_ratio":    round(ratio, 2),
                })

        hotspots.sort(key=lambda x: -x["hotspot_ratio"])
        return hotspots[:6]

    def detect_cluster_failures(self, error_entries: List[Dict]) -> Dict:
        """
        Detect job / MPI communicator group failures from error messages.

        Extracts job IDs, MPI ranks, and failure patterns that suggest
        an entire compute job or job group was affected (as opposed to
        isolated node hardware failures).

        Returns:
            {
                "job_ids_affected":   [str],   # unique job IDs found in error msgs
                "mpi_failure":        bool,     # True if MPI errors detected
                "affected_rank_count": int,     # number of distinct MPI ranks
                "job_failure_messages": [str],  # representative messages
                "failure_scope":      "none" | "single_job" | "multi_job" | "mpi_communicator"
            }
        """
        job_ids:  Set[str] = set()
        ranks:    Set[str] = set()
        mpi_msgs: List[str] = []
        job_msgs: List[str] = []

        for e in error_entries:
            msg = e.get("message", "")

            m_job = _JOB_ID_RE.search(msg)
            if m_job:
                job_ids.add(m_job.group(1))
                if len(job_msgs) < 3:
                    job_msgs.append(msg[:100])

            m_rank = _RANK_RE.search(msg)
            if m_rank:
                ranks.add(m_rank.group(1))

            if _MPI_RE.search(msg):
                if len(mpi_msgs) < 3:
                    mpi_msgs.append(msg[:100])

        mpi_detected = bool(mpi_msgs)

        if not job_ids and not mpi_detected:
            scope = "none"
        elif mpi_detected and len(ranks) > 4:
            scope = "mpi_communicator"
        elif len(job_ids) > 1:
            scope = "multi_job"
        elif len(job_ids) == 1:
            scope = "single_job"
        else:
            scope = "none"

        return {
            "job_ids_affected":     sorted(job_ids),
            "mpi_failure":          mpi_detected,
            "affected_rank_count":  len(ranks),
            "job_failure_messages": job_msgs[:3] or mpi_msgs[:3],
            "failure_scope":        scope,
        }

    # ── Phase 3 BGL helpers (extended for 3.5 signals) ───────────────────────

    def _analyse_labels(self, parsed_logs: List[Dict]) -> Dict:
        total          = len(parsed_logs)
        labelled       = sum(1 for e in parsed_logs if e.get("is_anomaly"))
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
        rack_counter     = Counter()
        midplane_counter = Counter()
        for e in error_entries:
            node = e.get("node", "")
            m = _BGL_NODE_RE.match(node)
            if m:
                rack     = f"R{m.group(1)}"
                midplane = f"R{m.group(1)}-M{m.group(2)}"
                rack_counter[rack] += 1
                midplane_counter[midplane] += 1

        total_racks = len(rack_counter)
        return {
            "total_racks_affected": total_racks,
            "top_racks":   [{"rack": r, "errors": c} for r, c in rack_counter.most_common(5)],
            "top_midplanes": [{"midplane": m, "errors": c}
                              for m, c in midplane_counter.most_common(3)],
            "spread": "localised" if total_racks <= 2 else
                      "moderate"  if total_racks <= 5 else "wide",
        }

    def _categorise_bgl_failures(self, error_entries: List[Dict]) -> Counter:
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
        window = 50
        pairs:  Counter = Counter()
        for i, e in enumerate(parsed_logs):
            if e["level"] not in ERROR_LEVELS:
                continue
            comp_a = e.get("component")
            if not comp_a:
                continue
            nearby_comps: Set[str] = set()
            for nb in parsed_logs[i+1: i+window]:
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
        transitions: Counter = Counter()
        for i in range(1, len(parsed_logs)):
            prev = parsed_logs[i-1]["level"]
            curr = parsed_logs[i]["level"]
            if curr in ERROR_LEVELS and prev not in ERROR_LEVELS:
                transitions[f"{prev}→{curr}"] += 1
        return dict(transitions.most_common(8))

    def _build_bgl_hypothesis(
        self, top_cats, top_racks, rack_stats, cofailure, label_stats,
        node_failure_patterns=None, component_hotspots=None, cluster_failures=None,
    ) -> str:
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
            parts.append(f"High ground-truth label coverage ({coverage}%)")

        # Phase 3.5 additions
        if node_failure_patterns and node_failure_patterns.get("pattern_type") != "none":
            ptype = node_failure_patterns["pattern_type"].replace("_", " ").upper()
            parts.append(f"Node failure pattern: {ptype}")
        if component_hotspots:
            hs = component_hotspots[0]
            parts.append(
                f"Component hotspot: {hs['component']} "
                f"({hs['hotspot_ratio']:.1f}× above baseline)"
            )
        if cluster_failures and cluster_failures.get("failure_scope") not in ("none", None):
            scope = cluster_failures["failure_scope"].replace("_", " ").upper()
            parts.append(f"Cluster failure scope: {scope}")

        return " | ".join(parts) if parts else "Insufficient BGL context for hypothesis."

    def _build_bgl_findings(
        self, label_stats, rack_stats, top_cats, cofailure, transitions,
        node_failure_patterns=None, component_hotspots=None, cluster_failures=None,
    ) -> List[str]:
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

        # Phase 3.5 additions
        if node_failure_patterns:
            persistent = node_failure_patterns.get("persistent_racks", [])
            if persistent:
                rack_strs = ", ".join(
                    f"{r['rack']}(errs={r['error_count']}, comps={r['components']})"
                    for r in persistent[:3]
                )
                findings.append(f"Persistent rack failures (multi-component): {rack_strs}")
            hotspot_mps = node_failure_patterns.get("hotspot_midplanes", [])
            if hotspot_mps:
                mp_str = ", ".join(f"{h['midplane']}({h['error_count']})" for h in hotspot_mps[:3])
                findings.append(f"Hotspot midplanes: {mp_str}")

        if component_hotspots:
            hs_str = ", ".join(
                f"{h['component']}({h['hotspot_ratio']:.1f}×)"
                for h in component_hotspots[:4]
            )
            findings.append(f"Disproportionate component error rates (hotspots): {hs_str}")

        if cluster_failures:
            scope = cluster_failures.get("failure_scope", "none")
            if scope != "none":
                findings.append(
                    f"Cluster/job failure detected: scope={scope.upper()}, "
                    f"jobs={cluster_failures.get('job_ids_affected', [])}, "
                    f"MPI={cluster_failures.get('mpi_failure', False)}, "
                    f"ranks_affected={cluster_failures.get('affected_rank_count', 0)}"
                )

        return findings

    def _build_bgl_recommendations(
        self, top_cats, rack_stats, cofailure,
        node_failure_patterns=None, cluster_failures=None,
    ) -> List[str]:
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

        # Phase 3.5 additions
        if node_failure_patterns and node_failure_patterns.get("pattern_type") == "single_rack":
            persistent = node_failure_patterns.get("persistent_racks", [])
            if persistent:
                recs.append(
                    f"Rack {persistent[0]['rack']} shows multi-component failure — "
                    "escalate to hardware team for physical inspection."
                )
        elif node_failure_patterns and node_failure_patterns.get("pattern_type") == "multi_rack":
            recs.append(
                "Multiple racks show persistent failures — "
                "likely systemic: check shared cooling, power, or fabric infrastructure."
            )

        if cluster_failures:
            scope = cluster_failures.get("failure_scope", "none")
            if scope == "mpi_communicator":
                recs.append(
                    "MPI communicator failure detected — resubmit affected jobs after "
                    "resolving the underlying hardware fault; review MPI error logs."
                )
            elif scope in ("single_job", "multi_job"):
                jobs = cluster_failures.get("job_ids_affected", [])
                recs.append(
                    f"Job failure detected (job IDs: {jobs}) — "
                    "check job scheduler logs and node health before resubmitting."
                )

        return recs

    def _score_bgl_confidence(self, error_entries, label_stats, top_cats) -> float:
        volume   = min(len(error_entries) / 50.0, 1.0)
        coverage = label_stats.get("gt_coverage", 0) / 100.0
        has_cats = 0.9 if top_cats else 0.3
        return round(volume * 0.3 + coverage * 0.4 + has_cats * 0.3, 2)

    # ── Generic investigation (Phase 3 — unchanged) ───────────────────────────

    def _investigate_generic(
        self,
        incident:      Incident,
        parsed_logs:   List[Dict],
        error_entries: List[Dict],
    ) -> RCASignal:
        silence_gaps = self._detect_log_silences(parsed_logs)
        transitions  = self._analyse_level_transitions(parsed_logs)
        dedup_ratio  = self._compute_dedup_ratio(error_entries)

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
            findings.append(
                f"Detected {len(silence_gaps)} log silence gap(s) > 10 entries: {silence_gaps[:3]}"
            )

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
                "dataset_format":    "generic",
                "dedup_ratio":       dedup_ratio,
                "silence_gaps":      silence_gaps,
                "level_transitions": transitions,
            },
        )
        print(f"    → {signal}")
        return signal

    def _detect_log_silences(self, parsed_logs: List[Dict]) -> List[int]:
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
        sample = parsed_logs[:10]
        return any(e.get("node") or e.get("bgl_level") for e in sample)