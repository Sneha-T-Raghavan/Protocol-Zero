"""
agents/log_investigation_agent.py
-----------------------------------
Log Investigation Agent — Phase 3, Protocol Zero.

Responsibility:
    Deep analysis of error messages within a detected incident's scope.
    Clusters similar errors, identifies dominant failure signatures,
    extracts temporal patterns, and proposes a root cause hypothesis
    grounded in the actual error text.

Techniques used (all stdlib, no ML deps):
    1. Keyword frequency analysis on error messages
    2. Template extraction — strips numbers/hex to find structural patterns
    3. Error progression timeline (first/last seen per pattern)
    4. Co-occurrence: which error types appear together most
    5. BGL-aware: uses component field when available
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

from core.incident import Incident
from agents.base_agent import BaseAgent
from agents.rca_signal import RCASignal
from utils.log_templates import extract_log_templates as _extract_templates_util

# Levels considered errors for investigation scope
ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}

# Regex to strip variable tokens (numbers, hex, IPs, paths) to get structural pattern
_VARIABLE_RE = re.compile(
    r"0x[0-9a-fA-F]+|"          # hex addresses
    r"\b\d{1,3}(?:\.\d{1,3}){3}\b|"  # IP addresses
    r"\b\d+\b|"                  # plain integers
    r"/[\w/\.\-]+"               # file paths
)


def _templatise(message: str) -> str:
    """Strip variable tokens from a message to get its structural template."""
    return _VARIABLE_RE.sub("<VAR>", message).strip()


def _top_n(counter: Counter, n: int = 5) -> List[Tuple[str, int]]:
    return counter.most_common(n)


class LogInvestigationAgent(BaseAgent):
    """
    Performs deep textual and structural analysis of error logs
    within the scope of a detected incident.

    For each incident it:
      - Isolates the relevant error window
      - Extracts error message templates (structural patterns)
      - Ranks patterns by frequency
      - Builds a timeline of when each pattern first/last appeared
      - Detects if errors are accelerating or decelerating
      - Extracts dominant keywords from error messages
      - Produces a root cause hypothesis from the top pattern + keywords
    """

    name        = "LogInvestigationAgent"
    description = "Deep error clustering, pattern extraction, and temporal analysis"

    def __init__(self, context_window: int = 50):
        """
        Args:
            context_window: Number of log lines to examine around the
                            incident's error spike for temporal context.
        """
        self.context_window = context_window

    # ── Public interface ──────────────────────────────────────────────────────

    def investigate(self, incident: Incident, parsed_logs: List[Dict]) -> RCASignal:
        print(f"  [{self.name}] Investigating incident {incident.incident_id[:8]} ...")

        error_entries = self._extract_error_scope(incident, parsed_logs)

        if not error_entries:
            return self._empty_signal(incident, "No error entries found in scope.")

        # Core analyses
        template_counts  = self._count_templates(error_entries)
        keyword_counts   = self._count_keywords(error_entries)
        timeline         = self._build_timeline(error_entries, template_counts)
        progression      = self._detect_progression(error_entries)
        component_counts = self._count_components(error_entries)

        # Synthesise
        top_templates = _top_n(template_counts, 5)
        top_keywords  = _top_n(keyword_counts, 8)
        top_comps     = _top_n(component_counts, 3)

        hypothesis    = self._build_hypothesis(top_templates, top_keywords, top_comps, progression)
        findings      = self._build_findings(
            error_entries, top_templates, top_keywords, top_comps, progression, timeline
        )
        recommendations = self._build_recommendations(top_templates, top_comps, progression)
        confidence    = self._score_confidence(error_entries, top_templates)

        # Build enriched template list with examples from the utility
        messages_in_scope = [e.get("message", "") for e in error_entries]
        enriched_templates = _extract_templates_util(messages_in_scope, max_templates=30)
        enriched_top = [
            t for t in enriched_templates
            if t["template"] in {tmpl for tmpl, _ in top_templates}
        ][:5]

        supporting = {
            "error_count_in_scope":  len(error_entries),
            "unique_templates":      len(template_counts),
            "top_templates":         [{"template": t, "count": c} for t, c in top_templates],
            "top_templates_enriched": enriched_top,   # includes examples[]
            "top_keywords":          [{"keyword": k, "count": c} for k, c in top_keywords],
            "top_components":        [{"component": c, "count": n} for c, n in top_comps],
            "error_progression":     progression,
            "timeline_events":       timeline,
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

    def _extract_error_scope(
        self, incident: Incident, parsed_logs: List[Dict]
    ) -> List[Dict]:
        """
        Extract the error entries most relevant to this incident.

        For SLIDING_WINDOW_SPIKE: scope raw logs to ±75 entries around the
        anchor (same window as InfraAgent and ContextAgent), then filter to
        errors within that window. This ensures all three agents analyse the
        same slice of the corpus.

        For ERROR_SPIKE / others: use all error entries across the full log.
        """
        if incident.anomaly_type != "SLIDING_WINDOW_SPIKE" or not incident.sample_errors:
            return [e for e in parsed_logs if e["level"] in ERROR_LEVELS]

        anchor_msg = incident.sample_errors[0]
        anchor_idx = None
        for i, entry in enumerate(parsed_logs):
            if anchor_msg in entry.get("message", "") or anchor_msg in entry.get("raw", ""):
                anchor_idx = i
                break

        if anchor_idx is None:
            return [e for e in parsed_logs if e["level"] in ERROR_LEVELS]

        half  = 75
        start = max(0, anchor_idx - half)
        end   = min(len(parsed_logs), anchor_idx + half)
        return [e for e in parsed_logs[start:end] if e["level"] in ERROR_LEVELS]

    def _count_templates(self, entries: List[Dict]) -> Counter:
        """
        Count structural error templates using the improved log_templates utility.

        The utility applies a richer multi-pass normalisation (hex, IP, timestamps,
        BGL node IDs, UUIDs, job/rank IDs, bare integers, paths) and clusters
        near-identical templates using token-set Jaccard similarity.

        Returns a Counter keyed by template string for backward compatibility
        with the rest of the agent (timeline, keyword extraction, etc.).
        """
        messages = [e.get("message", "") for e in entries]
        structured = _extract_templates_util(messages, max_templates=30)
        c = Counter()
        for t in structured:
            c[t["template"]] = t["count"]
        return c

    def _count_keywords(self, entries: List[Dict]) -> Counter:
        """Extract meaningful keywords from error messages, ignoring stopwords."""
        stopwords = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
            "or", "is", "are", "was", "be", "has", "have", "with", "from",
            "by", "not", "no", "it", "its", "this", "that", "error",
        }
        c = Counter()
        for e in entries:
            words = re.findall(r"[a-zA-Z][a-zA-Z_\-]{2,}", e.get("message", ""))
            for w in words:
                w_lower = w.lower()
                if w_lower not in stopwords and len(w_lower) >= 3:
                    c[w_lower] += 1
        return c

    def _build_timeline(
        self, entries: List[Dict], template_counts: Counter
    ) -> List[str]:
        """Build an ordered timeline of when each major error template first appeared."""
        seen: Dict[str, str] = {}
        events = []
        for e in entries:
            tmpl = _templatise(e.get("message", ""))
            if tmpl not in seen and template_counts[tmpl] >= 2:
                ts = e.get("timestamp") or "unknown time"
                seen[tmpl] = ts
                short = tmpl[:60] + ("…" if len(tmpl) > 60 else "")
                events.append(f"{ts} — FIRST OCCURRENCE: {short}")
        return events[:10]  # cap at 10 timeline events

    def _detect_progression(self, entries: List[Dict]) -> str:
        """
        Detect whether errors are accelerating, decelerating, or steady
        by comparing error density in the first vs second half.
        """
        half = len(entries) // 2
        if half == 0:
            return "insufficient_data"
        first_half  = len(entries[:half])
        second_half = len(entries[half:])
        ratio = second_half / max(first_half, 1)
        if ratio > 1.4:
            return "accelerating"
        if ratio < 0.7:
            return "decelerating"
        return "steady"

    def _count_components(self, entries: List[Dict]) -> Counter:
        """Count errors per component (BGL-specific field, gracefully absent)."""
        c = Counter()
        for e in entries:
            comp = e.get("component")
            if comp:
                c[comp] += 1
        return c

    def _build_hypothesis(
        self,
        top_templates: List[Tuple],
        top_keywords:  List[Tuple],
        top_comps:     List[Tuple],
        progression:   str,
    ) -> str:
        parts = []
        if top_templates:
            dominant = top_templates[0][0]
            count    = top_templates[0][1]
            short    = dominant[:70] + ("…" if len(dominant) > 70 else "")
            parts.append(f"Dominant failure pattern ({count}x): '{short}'")
        if top_keywords:
            kws = ", ".join(k for k, _ in top_keywords[:4])
            parts.append(f"Key error terms: {kws}")
        if top_comps:
            comp = top_comps[0][0]
            parts.append(f"Primary affected component: {comp}")
        if progression == "accelerating":
            parts.append("Error rate is accelerating — likely cascade in progress.")
        elif progression == "decelerating":
            parts.append("Error rate is decelerating — issue may be self-recovering.")

        return " | ".join(parts) if parts else "Insufficient data for hypothesis."

    def _build_findings(
        self,
        entries:       List[Dict],
        top_templates: List[Tuple],
        top_keywords:  List[Tuple],
        top_comps:     List[Tuple],
        progression:   str,
        timeline:      List[str],
    ) -> List[str]:
        findings = []
        findings.append(f"Analysed {len(entries)} error-level log entries in scope.")
        if top_templates:
            findings.append(
                f"Most frequent error pattern ({top_templates[0][1]}x): "
                f"'{top_templates[0][0][:80]}'"
            )
            if len(top_templates) > 1:
                findings.append(
                    f"{len(top_templates)} distinct error templates identified; "
                    f"top 3: {', '.join(t[:40] for t, _ in top_templates[:3])}"
                )
        if top_keywords:
            kws = ", ".join(f"'{k}' ({c})" for k, c in top_keywords[:5])
            findings.append(f"High-frequency error keywords: {kws}")
        if top_comps:
            comp_str = ", ".join(f"{c} ({n})" for c, n in top_comps)
            findings.append(f"Error distribution by component: {comp_str}")
        findings.append(f"Error progression trend: {progression.upper()}")
        if timeline:
            findings.append(f"Timeline has {len(timeline)} distinct error first-occurrence events.")
        return findings

    def _build_recommendations(
        self,
        top_templates: List[Tuple],
        top_comps:     List[Tuple],
        progression:   str,
    ) -> List[str]:
        recs = []
        if top_templates:
            pattern = top_templates[0][0]
            if "memory" in pattern.lower() or "ecc" in pattern.lower():
                recs.append("Check hardware memory diagnostics — ECC/memory errors suggest DIMM failure.")
            if "cache" in pattern.lower() or "ciod" in pattern.lower():
                recs.append("Investigate I/O node connectivity — CIOD errors indicate communication failure.")
            if "kernel" in pattern.lower() or "tlb" in pattern.lower():
                recs.append("Review kernel logs on affected nodes for hardware exceptions.")
            if "timeout" in pattern.lower():
                recs.append("Check network latency and service health for timeout root cause.")
            if not recs:
                recs.append(f"Investigate the dominant error pattern: '{pattern[:60]}'")
        if top_comps:
            recs.append(f"Focus investigation on component: {top_comps[0][0]}")
        if progression == "accelerating":
            recs.append("URGENT: Error rate is accelerating — consider immediate intervention.")
        elif progression == "decelerating":
            recs.append("Monitor for full recovery — errors decelerating but not yet resolved.")
        recs.append("Correlate error timestamps with deployment, config, or maintenance events.")
        return recs

    def _score_confidence(self, entries: List[Dict], top_templates: List[Tuple]) -> float:
        """
        Score confidence based on:
          - Number of error entries (more = more reliable)
          - Dominance of the top template (clear pattern = high confidence)
        top_templates is List[Tuple[str, int]] from _top_n().
        """
        if not entries:
            return 0.1
        volume_score = min(len(entries) / 50.0, 1.0)  # saturates at 50+ errors
        if top_templates:
            top_count = top_templates[0][1]  # count of most common template
            dominance = top_count / len(entries)
        else:
            dominance = 0.0
        # Weighted blend
        return round(0.5 * volume_score + 0.5 * dominance, 2)

    def _empty_signal(self, incident: Incident, reason: str) -> RCASignal:
        return RCASignal(
            agent_name=self.name,
            incident_id=incident.incident_id,
            confidence=0.0,
            root_cause_hypothesis=reason,
            findings=[reason],
            recommendations=["Verify log parsing — no error entries found in scope."],
        )