"""
core/correlation.py
--------------------
Protocol Zero — Phase 4: Incident Correlation & Root Cause Consolidation.

Responsibility:
    Receives List[Incident] + List[InvestigationReport] from Phase 3.
    Groups related incidents into IncidentGroups via:
        1. Time-window clustering
        2. Component overlap expansion
        3. Node overlap expansion
    Consolidates root causes across each group.
    Produces correlated investigation reports.

Pipeline position:
    Phase 3:  List[Incident] + parsed_logs → AgentRunner → List[InvestigationReport]
    Phase 4:  List[InvestigationReport] → correlate_incidents → List[IncidentGroup]
              → generate_correlated_reports → List[Dict]
"""

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple, Optional

from core.incident import Incident
from agents.rca_signal import InvestigationReport

# Regex to extract the approximate log-event timestamp from Incident.details.
# SlidingWindowDetector writes: "... starting at index N (approx. <ISO_TS>)."
_DETAILS_LOG_TS_RE = re.compile(r"approx\.\s+(\d{4}-\d{2}-\d{2}T[\d:.]+)")

# Regex to extract the starting log index from Incident.details.
_DETAILS_LOG_IDX_RE = re.compile(r"starting at index (\d+)")

# ── Configuration ─────────────────────────────────────────────────────────────

TIME_CORRELATION_WINDOW    = 120      # seconds — max gap between incidents in same cluster
COMPONENT_OVERLAP_THRESHOLD = 1      # minimum shared components to merge clusters
NODE_OVERLAP_THRESHOLD      = 1      # minimum shared nodes to merge clusters
MAX_EXPANSION_WINDOW        = 86400  # seconds (24h) — max time gap allowed when merging
                                     # clusters by component/node overlap. Prevents globally
                                     # recurring components (e.g. KERNEL) from collapsing
                                     # temporally distant clusters into one mega-group.

# Node values that carry no topology information and should be excluded
# from overlap detection to prevent spurious cluster merges.
_NOISE_NODES = {"NULL", "UNKNOWN_LOCATION", ""}


# ── IncidentGroup dataclass ───────────────────────────────────────────────────

@dataclass
class IncidentGroup:
    """
    A correlated group of incidents sharing temporal proximity,
    component overlap, or node overlap.

    Fields
    ------
    group_id             : Deterministic MD5 hash of member incident IDs.
    incidents            : Member Incident objects.
    reports              : Corresponding InvestigationReport objects.
    start_time           : Earliest incident timestamp in group.
    end_time             : Latest incident timestamp in group.
    affected_components  : Union of components across all reports.
    affected_nodes       : Union of nodes across all reports.
    cascade_chain        : Ordered list of components in first-failure order.
    root_cause_hypothesis: Consolidated hypothesis (highest-weight signal).
    confidence           : Weighted confidence across all reports.
    """

    group_id:              str
    incidents:             List[Incident]      = field(default_factory=list)
    reports:               List[InvestigationReport] = field(default_factory=list)

    start_time:            Optional[datetime]  = None
    end_time:              Optional[datetime]  = None

    affected_components:   Set[str]            = field(default_factory=set)
    affected_nodes:        Set[str]            = field(default_factory=set)

    cascade_chain:         List[str]           = field(default_factory=list)
    root_cause_hypothesis: str                 = ""
    confidence:            float               = 0.0

    def to_dict(self) -> Dict:
        return {
            "group_id":              self.group_id,
            "incident_count":        len(self.incidents),
            "incidents":             [inc.incident_id for inc in self.incidents],
            "start_time":            self.start_time.isoformat() if self.start_time else None,
            "end_time":              self.end_time.isoformat() if self.end_time else None,
            "affected_components":   sorted(self.affected_components),
            "affected_nodes":        sorted(self.affected_nodes),
            "cascade_chain":         self.cascade_chain,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "confidence":            self.confidence,
        }

    def __str__(self) -> str:
        return (
            f"[IncidentGroup {self.group_id[:8]}] "
            f"incidents={len(self.incidents)} | "
            f"components={len(self.affected_components)} | "
            f"confidence={self.confidence:.0%} | "
            f"{self.root_cause_hypothesis[:70]}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamp string to datetime. Returns None on failure."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _log_event_ts(incident: Incident) -> Optional[datetime]:
    """
    Extract the real log-event timestamp for an incident.

    Priority:
        1. Parse 'approx. <ISO_TS>' from incident.details
           (written by SlidingWindowDetector when the window has a BGL timestamp).
        2. Fall back to incident.timestamp (object creation time — last resort).

    Background: incident.timestamp is set to datetime.now() at detection time,
    so all incidents created in a single pipeline run share nearly identical
    creation timestamps and are useless for temporal clustering. The log-event
    timestamp embedded in details reflects when the anomaly actually occurred
    in the source logs, which may span months.

    Note: BGL log timestamps are timezone-naive (no UTC offset). They are
    normalised to UTC-aware here so all datetimes are comparable.
    """
    m = _DETAILS_LOG_TS_RE.search(incident.details or "")
    if m:
        ts = _parse_ts(m.group(1))
        if ts:
            # Ensure UTC-aware for consistent comparisons
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
    # Fallback for ERROR_SPIKE (no window timestamp) or unparseable details
    return _parse_ts(incident.timestamp)


def _log_event_index(incident: Incident) -> int:
    """
    Extract the starting log-index from incident.details.
    Returns 0 if not found (e.g. ERROR_SPIKE incidents).
    """
    m = _DETAILS_LOG_IDX_RE.search(incident.details or "")
    return int(m.group(1)) if m else 0


def _group_id_from_incidents(incidents: List[Incident]) -> str:
    """Deterministic group ID from sorted member incident IDs."""
    key = "|".join(sorted(inc.incident_id for inc in incidents))
    return hashlib.md5(key.encode()).hexdigest()


def _components_from_report(report: InvestigationReport) -> Set[str]:
    """Extract component names from an InvestigationReport."""
    comps: Set[str] = set()
    for comp in report.top_components:
        if comp:
            comps.add(comp)
    # Also scan InfraAgent signal supporting_data for richer component list
    for sig in report.signals:
        if sig.agent_name == "InfraAgent":
            for entry in sig.supporting_data.get("component_error_counts", []):
                if entry.get("component"):
                    comps.add(entry["component"])
    return comps


def _nodes_from_report(report: InvestigationReport) -> Set[str]:
    """Extract node names from an InvestigationReport, excluding noise values."""
    nodes: Set[str] = set()
    for sig in report.signals:
        if sig.agent_name == "InfraAgent":
            for entry in sig.supporting_data.get("node_error_counts", []):
                node = entry.get("node", "")
                if node and node not in _NOISE_NODES:
                    nodes.add(node)
        if sig.agent_name == "ContextAgent":
            rack_info = sig.supporting_data.get("rack_topology", {})
            for rack_entry in rack_info.get("top_racks", []):
                rack = rack_entry.get("rack", "")
                if rack and rack not in _NOISE_NODES:
                    nodes.add(rack)
    return nodes


# ── Core Functions ────────────────────────────────────────────────────────────

def cluster_by_time_window(
    incidents: List[Incident],
    window_seconds: int = TIME_CORRELATION_WINDOW,
) -> List[List[Incident]]:
    """
    Group incidents into temporal clusters using LOG-EVENT timestamps.

    Uses the approximate log-event time extracted from incident.details
    (written by SlidingWindowDetector as 'approx. <ISO_TS>'). This correctly
    reflects when the anomaly occurred in the source logs, rather than when
    the Incident object was created (which is useless for clustering since all
    incidents in a single run share nearly identical creation timestamps).

    Incidents with no recoverable log-event timestamp are placed in their own
    single-incident cluster.
    """
    if not incidents:
        return []

    timed: List[Tuple[datetime, Incident]] = []
    untimed: List[Incident] = []

    for inc in incidents:
        ts = _log_event_ts(inc)
        if ts:
            timed.append((ts, inc))
        else:
            untimed.append(inc)

    timed.sort(key=lambda x: x[0])

    clusters: List[List[Incident]] = []
    current: List[Incident] = []
    last_ts: Optional[datetime] = None

    for ts, inc in timed:
        if last_ts is None:
            current.append(inc)
        else:
            gap = abs((ts - last_ts).total_seconds())
            if gap <= window_seconds:
                current.append(inc)
            else:
                clusters.append(current)
                current = [inc]
        last_ts = ts

    if current:
        clusters.append(current)

    for inc in untimed:
        clusters.append([inc])

    print(f"[Correlation] Time-window clustering → {len(clusters)} cluster(s)")
    return clusters


def _cluster_time_range(
    cluster: List[Incident],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Return (min_ts, max_ts) of log-event timestamps for a cluster."""
    timestamps = [t for t in (_log_event_ts(inc) for inc in cluster) if t is not None]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def _clusters_within_time_bound(
    cluster_a: List[Incident],
    cluster_b: List[Incident],
    max_gap_seconds: int,
) -> bool:
    """
    Return True if the two clusters are close enough in log-event time to be
    merged by overlap expansion.

    Two clusters are considered time-compatible if the gap between their
    nearest edges (end of earlier cluster to start of later cluster) is
    within max_gap_seconds. This prevents globally recurring components
    (e.g. KERNEL, which appears in every burst) from merging clusters that
    are hours or months apart.

    If either cluster has no parseable timestamps, returns True (safe merge).
    """
    start_a, end_a = _cluster_time_range(cluster_a)
    start_b, end_b = _cluster_time_range(cluster_b)

    if start_a is None or start_b is None:
        return True  # no timestamp info — allow merge

    # Determine which cluster comes first
    if end_a <= start_b:
        gap = (start_b - end_a).total_seconds()
    elif end_b <= start_a:
        gap = (start_a - end_b).total_seconds()
    else:
        gap = 0  # overlapping time ranges — definitely compatible

    return gap <= max_gap_seconds


def expand_cluster_by_component_overlap(
    clusters: List[List[Incident]],
    reports: Dict[str, InvestigationReport],
    threshold: int = COMPONENT_OVERLAP_THRESHOLD,
    max_expansion_window: int = MAX_EXPANSION_WINDOW,
) -> List[List[Incident]]:
    """
    Merge clusters that share at least `threshold` common components AND
    whose log-event time ranges are within `max_expansion_window` seconds.

    The time guard is critical: without it, a globally recurring component
    like KERNEL (present in every burst of this dataset) would merge all
    clusters into a single mega-group regardless of how far apart they are
    in actual log time.
    """
    # Build component sets per cluster
    def get_components(cluster: List[Incident]) -> Set[str]:
        comps: Set[str] = set()
        for inc in cluster:
            if inc.incident_id in reports:
                comps |= _components_from_report(reports[inc.incident_id])
        return comps

    merged = True
    result = [list(c) for c in clusters]

    while merged:
        merged = False
        new_result: List[List[Incident]] = []
        used = [False] * len(result)

        for i in range(len(result)):
            if used[i]:
                continue
            base = list(result[i])
            base_comps = get_components(base)

            for j in range(i + 1, len(result)):
                if used[j]:
                    continue
                other_comps = get_components(result[j])
                shared = base_comps & other_comps
                if (len(shared) >= threshold
                        and (base_comps or other_comps)
                        and _clusters_within_time_bound(base, result[j], max_expansion_window)):
                    base.extend(result[j])
                    base_comps |= other_comps
                    used[j] = True
                    merged = True

            new_result.append(base)
            used[i] = True

        result = new_result

    print(f"[Correlation] After component-overlap expansion → {len(result)} cluster(s)")
    return result


def expand_cluster_by_node_overlap(
    clusters: List[List[Incident]],
    reports: Dict[str, InvestigationReport],
    threshold: int = NODE_OVERLAP_THRESHOLD,
    max_expansion_window: int = MAX_EXPANSION_WINDOW,
) -> List[List[Incident]]:
    """
    Merge clusters that share at least `threshold` common (non-noise) nodes
    AND whose log-event time ranges are within `max_expansion_window` seconds.
    """
    def get_nodes(cluster: List[Incident]) -> Set[str]:
        nodes: Set[str] = set()
        for inc in cluster:
            if inc.incident_id in reports:
                nodes |= _nodes_from_report(reports[inc.incident_id])
        return nodes

    merged = True
    result = [list(c) for c in clusters]

    while merged:
        merged = False
        new_result: List[List[Incident]] = []
        used = [False] * len(result)

        for i in range(len(result)):
            if used[i]:
                continue
            base = list(result[i])
            base_nodes = get_nodes(base)

            for j in range(i + 1, len(result)):
                if used[j]:
                    continue
                other_nodes = get_nodes(result[j])
                shared = base_nodes & other_nodes
                if (len(shared) >= threshold
                        and (base_nodes or other_nodes)
                        and _clusters_within_time_bound(base, result[j], max_expansion_window)):
                    base.extend(result[j])
                    base_nodes |= other_nodes
                    used[j] = True
                    merged = True

            new_result.append(base)
            used[i] = True

        result = new_result

    print(f"[Correlation] After node-overlap expansion → {len(result)} cluster(s)")
    return result


def detect_cascade_chain(
    incidents: List[Incident],
    reports: Dict[str, InvestigationReport],
) -> List[str]:
    """
    Infer a causal cascade chain from the order in which components
    first appear in errors across temporally sorted incidents.

    Sorts by log-event index (extracted from incident.details) rather than
    by incident creation timestamp, which is meaningless for ordering since
    all incidents in a run are created within milliseconds of each other.

    Returns an ordered list like: ["KERNEL", "TORUS", "LINKCARD"]
    """
    # Sort by log-event index (position in log corpus); fall back to log-event ts
    sorted_incs = sorted(incidents, key=_log_event_index)

    seen: Set[str] = set()
    chain: List[str] = []

    for inc in sorted_incs:
        report = reports.get(inc.incident_id)
        if not report:
            continue
        # Prefer InfraAgent cascade_sequence (already ordered by first-error appearance)
        for sig in report.signals:
            if sig.agent_name == "InfraAgent":
                for comp in sig.supporting_data.get("cascade_sequence", []):
                    if comp not in seen:
                        chain.append(comp)
                        seen.add(comp)
        # Fallback: top_components order
        for comp in report.top_components:
            if comp not in seen:
                chain.append(comp)
                seen.add(comp)

    return chain


def consolidate_root_cause(
    reports: List[InvestigationReport],
) -> Tuple[str, float]:
    """
    Aggregate root-cause hypotheses across all reports, weighting by confidence.

    Strategy:
        - For each report, weight its consensus_hypothesis by overall_confidence.
        - Tally keyword frequency across all hypotheses (weighted).
        - Select the hypothesis from the highest-confidence report as the
          group-level statement; append shared dominant theme if different.

    Returns:
        (consolidated_hypothesis: str, group_confidence: float)
    """
    if not reports:
        return "No investigation reports available.", 0.0

    # Sort reports by confidence descending
    sorted_reports = sorted(reports, key=lambda r: r.overall_confidence, reverse=True)
    best = sorted_reports[0]

    # Weighted keyword vote across all hypotheses
    keyword_votes: Counter = Counter()
    total_weight = 0.0

    for report in sorted_reports:
        weight = max(report.overall_confidence, 0.01)
        total_weight += weight
        words = report.consensus_hypothesis.lower().split()
        for word in words:
            if len(word) > 4:  # skip short stopwords
                keyword_votes[word] += weight

    # Top shared theme (most voted keyword across reports)
    top_kw = keyword_votes.most_common(1)
    theme = top_kw[0][0] if top_kw else ""

    hypothesis = best.consensus_hypothesis
    if theme and theme not in hypothesis.lower():
        hypothesis += f" | Shared theme across group: '{theme}'"

    # Group confidence: weighted mean
    weighted_conf = sum(r.overall_confidence * max(r.overall_confidence, 0.01)
                        for r in sorted_reports)
    group_conf = round(weighted_conf / max(total_weight, 1e-9), 3)
    group_conf = min(group_conf, 1.0)

    return hypothesis, group_conf


def build_incident_group(
    incidents: List[Incident],
    reports: Dict[str, InvestigationReport],
) -> IncidentGroup:
    """
    Construct a single IncidentGroup from a list of correlated incidents.
    Time bounds (start_time, end_time) are derived from log-event timestamps,
    not object creation timestamps.
    """
    group_id = _group_id_from_incidents(incidents)
    group_reports = [reports[inc.incident_id]
                     for inc in incidents if inc.incident_id in reports]

    # Time bounds — use log-event timestamps, not creation timestamps
    timestamps = [
        _log_event_ts(inc) for inc in incidents
        if _log_event_ts(inc) is not None
    ]
    start_time = min(timestamps) if timestamps else None
    end_time   = max(timestamps) if timestamps else None

    # Component & node aggregation
    all_components: Set[str] = set()
    all_nodes: Set[str] = set()
    for report in group_reports:
        all_components |= _components_from_report(report)
        all_nodes      |= _nodes_from_report(report)

    # Cascade chain
    cascade = detect_cascade_chain(incidents, reports)

    # Root cause consolidation
    hypothesis, confidence = consolidate_root_cause(group_reports)

    group = IncidentGroup(
        group_id=group_id,
        incidents=incidents,
        reports=group_reports,
        start_time=start_time,
        end_time=end_time,
        affected_components=all_components,
        affected_nodes=all_nodes,
        cascade_chain=cascade,
        root_cause_hypothesis=hypothesis,
        confidence=confidence,
    )

    print(f"  [build_incident_group] {group}")
    return group


def correlate_incidents(
    incidents: List[Incident],
    reports: List[InvestigationReport],
    time_window: int = TIME_CORRELATION_WINDOW,
) -> List[IncidentGroup]:
    """
    Main correlation entry point.

    Steps:
        1. Sort incidents by timestamp.
        2. Time-window clustering.
        3. Expand by component overlap.
        4. Expand by node overlap.
        5. Build IncidentGroup per cluster.

    Args:
        incidents   : All Incident objects from Phase 2.
        reports     : All InvestigationReport objects from Phase 3.
        time_window : Max seconds between incidents in same cluster.

    Returns:
        List[IncidentGroup]
    """
    if not incidents:
        print("[Correlation] No incidents to correlate.")
        return []

    print(f"\n[Correlation] Starting correlation of {len(incidents)} incident(s) ...")

    # Index reports by incident_id for O(1) lookup
    report_index: Dict[str, InvestigationReport] = {
        r.incident_id: r for r in reports
    }

    # Step 1 — Sort by log-event timestamp (not creation timestamp)
    sorted_incs = sorted(
        incidents,
        key=lambda inc: _log_event_ts(inc) or datetime.min.replace(tzinfo=timezone.utc),
    )

    # Step 2 — Time clustering
    clusters = cluster_by_time_window(sorted_incs, window_seconds=time_window)

    # Step 3 — Component expansion
    clusters = expand_cluster_by_component_overlap(clusters, report_index)

    # Step 4 — Node expansion
    clusters = expand_cluster_by_node_overlap(clusters, report_index)

    # Step 5 — Deduplicate incidents across clusters (first assignment wins)
    seen_ids: Set[str] = set()
    deduped_clusters: List[List[Incident]] = []
    for cluster in clusters:
        unique = [inc for inc in cluster if inc.incident_id not in seen_ids]
        for inc in unique:
            seen_ids.add(inc.incident_id)
        if unique:
            deduped_clusters.append(unique)

    # Step 6 — Build groups
    groups: List[IncidentGroup] = []
    for cluster in deduped_clusters:
        group = build_incident_group(cluster, report_index)
        groups.append(group)

    print(
        f"[Correlation] Complete. "
        f"{len(incidents)} incident(s) → {len(groups)} group(s)."
    )
    return groups


def generate_correlated_reports(groups: List[IncidentGroup]) -> List[Dict]:
    """
    Serialise IncidentGroups into dicts suitable for JSON output.

    Output format per group:
    {
        group_id, incident_count, incidents,
        start_time, end_time,
        affected_components, affected_nodes,
        cascade_chain, root_cause_hypothesis, confidence
    }
    """
    return [group.to_dict() for group in groups]


def save_correlated_reports(
    groups: List[IncidentGroup],
    filepath: str = "outputs/correlated_reports.json",
    overwrite: bool = True,
) -> None:
    """Persist correlated reports to JSON."""
    if not groups:
        print("[Correlation] No groups to save.")
        return

    new_dicts = generate_correlated_reports(groups)

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
        existing_ids = {d["group_id"] for d in existing}
        merged = existing + [d for d in new_dicts if d["group_id"] not in existing_ids]

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(
        f"[Correlation] Saved {len(groups)} group(s) "
        f"(total in file: {len(merged)}) → '{filepath}'"
    )


# ── Example integration ───────────────────────────────────────────────────────

def run_correlation(
    incidents: List[Incident],
    reports: List[InvestigationReport],
    output_file: str = "outputs/correlated_reports.json",
    time_window: int = TIME_CORRELATION_WINDOW,
    overwrite: bool = True,
) -> List[IncidentGroup]:
    """
    One-call convenience for the Phase 4 correlation pipeline.

    Example usage in run_phase4.py:

        from core.agent_runner import run_investigation
        from core.correlation  import run_correlation

        incidents = load_incidents("outputs/bgl_incidents.json")
        reports   = run_investigation(incidents, parsed_logs)
        groups    = run_correlation(incidents, reports)
    """
    groups = correlate_incidents(incidents, reports, time_window=time_window)
    save_correlated_reports(groups, filepath=output_file, overwrite=overwrite)
    return groups