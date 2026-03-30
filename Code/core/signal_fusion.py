"""
core/signal_fusion.py
----------------------
Protocol Zero — Phase 3.5: Signal Weighting & Fusion Engine.

Responsibility:
    Takes raw RCASignals produced by multiple agents and applies structured
    weighting, cross-agent agreement scoring, and conflict resolution to
    produce a single fused, ranked output for each incident.

    This replaces the naïve "take the highest confidence signal" logic
    in agent_runner._merge_signals() with a principled fusion pipeline.

Pipeline position:
    Phase 3:  agents → List[RCASignal]
    Phase 3.5 (here): List[RCASignal] → FusedSignal (weighted_hypothesis,
                       final_confidence, ranked_hypotheses, conflict_notes)
    Phase 3:  FusedSignal → InvestigationReport fields

Design:
    - All functions are pure (no side effects, same input → same output)
    - No external dependencies — stdlib only
    - Agents are NOT modified; this module operates on their outputs only

Agent weight priors (baseline trust before adjustment):
    LogInvestigationAgent : 0.45  — directly analyses error text; most specific
    InfraAgent            : 0.35  — structural/infra signals; broad but reliable
    ContextAgent          : 0.20  — dataset-level context; useful but indirect
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agents.rca_signal import RCASignal

# ── Agent priors ──────────────────────────────────────────────────────────────
# Baseline trust weight per agent type before adjustments.
# Must sum to 1.0 for normalisation to be meaningful.
AGENT_WEIGHT_PRIORS: Dict[str, float] = {
    "LogInvestigationAgent": 0.45,
    "InfraAgent":            0.35,
    "ContextAgent":          0.20,
}

# Minimum confidence threshold below which a signal is treated as unreliable
# and its weight is halved before fusion.
LOW_CONFIDENCE_THRESHOLD = 0.30

# Jaccard similarity threshold above which two hypotheses are considered
# to be "in agreement" for cross-agent agreement boosting.
AGREEMENT_JACCARD_THRESHOLD = 0.20

# Agents whose signals inform evidence but must NOT win the consensus hypothesis.
# ContextAgent scores reflect dataset-level coverage (ground-truth labels),
# not the quality of the incident-specific RCA hypothesis.
DATASET_AGENTS: set = {"ContextAgent"}

# Words present in every hypothesis regardless of semantic content — excluded from
# Jaccard similarity so structural boilerplate does not cause false agreements.
_HYPOTHESIS_STOPWORDS: set = {
    "errors", "error", "component", "affected", "failure", "detected",
    "burst", "shape", "peak", "density", "hotspot", "velocity", "entries",
    "cascade", "most", "above", "baseline", "category", "coverage",
    "pattern", "dominant", "primary", "total", "count", "high", "node",
    "spread", "rack", "level", "rate", "analysis", "signal", "data",
}


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class FusedSignal:
    """
    The output of the signal fusion pipeline for one incident.

    Fields
    ------
    weighted_hypothesis  : The best single hypothesis after weighting + conflict resolution.
    final_confidence     : Fused confidence score (0.0–1.0).
    ranked_hypotheses    : All agent hypotheses ranked by fused weight, with scores.
    conflict_notes       : Human-readable notes where agents disagreed.
    agent_weights        : Final per-agent weights used in fusion (post-adjustment).
    agreement_pairs      : Pairs of agents that agreed on a hypothesis.
    """
    weighted_hypothesis:  str
    final_confidence:     float
    ranked_hypotheses:    List[Dict]   = field(default_factory=list)
    conflict_notes:       List[str]    = field(default_factory=list)
    agent_weights:        Dict[str, float] = field(default_factory=dict)
    agreement_pairs:      List[str]    = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """
    Extract semantically meaningful words from a hypothesis string.

    Filters out both short words (<4 chars) and hypothesis-structural stopwords
    that appear in every agent's output regardless of semantic content (e.g.
    "errors", "component", "burst"). Without this filter, Jaccard similarity
    between structurally different hypotheses is artificially inflated, causing
    false "agreement" detections between InfraAgent and ContextAgent.
    """
    return [
        w.lower()
        for w in re.findall(r"[a-zA-Z][a-zA-Z_\-]{3,}", text)
        if w.lower() not in _HYPOTHESIS_STOPWORDS
    ]


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between token sets of two hypothesis strings."""
    set_a = set(_tokenise(a))
    set_b = set(_tokenise(b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _earliest_component(signal: RCASignal) -> Optional[str]:
    """
    Extract the first component in a cascade sequence from a signal's
    supporting_data (InfraAgent) or the first top_component equivalent.
    Returns None if not available.
    """
    cascade = signal.supporting_data.get("cascade_sequence", [])
    if cascade:
        return cascade[0]
    comps = signal.supporting_data.get("component_error_counts", [])
    if comps:
        return comps[0].get("component")
    return None


# ── Core functions ─────────────────────────────────────────────────────────────

def normalize_signal_confidence(signals: List[RCASignal]) -> Dict[str, float]:
    """
    Return agent confidence scores as-is for use in weighting.

    Min-max normalisation was the original approach but is explicitly avoided
    here because it is unstable over narrow confidence ranges. When all three
    agents score in a tight band (e.g. 0.87–0.97), the lowest agent gets
    normalised to 0.0 and clipped to 0.10, which collapses its effective weight
    and can cause ContextAgent (prior=0.20) to outscore InfraAgent (prior=0.35)
    — a correctness inversion.

    Raw confidence is already on a meaningful 0–1 scale produced by each agent's
    own scoring function. The prior weights in AGENT_WEIGHT_PRIORS supply the
    inter-agent calibration; per-incident normalisation adds noise, not signal.

    Args:
        signals: Raw RCASignal list from all agents.

    Returns:
        Dict mapping signal_id → confidence (raw, unchanged).
    """
    return {s.signal_id: round(s.confidence, 4) for s in signals}


def weight_agent_signals(
    signals: List[RCASignal],
    normalised_confs: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute a final weight for each signal combining:
        1. Agent prior (AGENT_WEIGHT_PRIORS)
        2. Normalised self-reported confidence
        3. Penalty for low confidence (< LOW_CONFIDENCE_THRESHOLD)

    Args:
        signals          : RCASignal list from all agents.
        normalised_confs : Output of normalize_signal_confidence(). If None,
                           raw confidence values are used.

    Returns:
        Dict mapping signal_id → final weight (not yet normalised to sum=1).
    """
    norm = normalised_confs or {s.signal_id: s.confidence for s in signals}
    weights: Dict[str, float] = {}

    for sig in signals:
        prior = AGENT_WEIGHT_PRIORS.get(sig.agent_name, 0.25)
        conf  = norm.get(sig.signal_id, sig.confidence)

        # Penalise low-confidence signals
        if sig.confidence < LOW_CONFIDENCE_THRESHOLD:
            conf *= 0.5

        weight = prior * conf
        weights[sig.signal_id] = round(weight, 4)

    # Normalise so weights sum to 1.0
    total = sum(weights.values()) or 1.0
    return {sid: round(w / total, 4) for sid, w in weights.items()}


def resolve_conflicting_hypotheses(
    signals: List[RCASignal],
    weights: Dict[str, float],
) -> Tuple[str, List[str], List[str]]:
    """
    Detect conflicts between agent hypotheses and resolve them.

    Resolution priority:
        1. Prefer cross-agent agreement (boosted by AGREEMENT_JACCARD_THRESHOLD).
        2. Prefer the hypothesis with the highest weighted score.
        3. Among ties, prefer the earliest-failure component (cascade origin).

    Args:
        signals : RCASignal list.
        weights : Per-signal weights from weight_agent_signals().

    Returns:
        Tuple of:
            resolved_hypothesis : str   — the winning hypothesis
            conflict_notes      : List[str] — human-readable conflict descriptions
            agreement_pairs     : List[str] — agent pairs that agreed
    """
    if not signals:
        return "No signals available.", [], []

    if len(signals) == 1:
        return signals[0].root_cause_hypothesis, [], []

    conflict_notes: List[str] = []
    agreement_pairs: List[str] = []

    # ── Step 1: detect agreements ─────────────────────────────────────────────
    # Score each signal by how many other signals agree with it (Jaccard)
    agreement_bonus: Dict[str, float] = {s.signal_id: 0.0 for s in signals}

    for i, sig_a in enumerate(signals):
        for sig_b in signals[i+1:]:
            j = _jaccard(sig_a.root_cause_hypothesis, sig_b.root_cause_hypothesis)
            if j >= AGREEMENT_JACCARD_THRESHOLD:
                # Both get a bonus proportional to how much they agree
                agreement_bonus[sig_a.signal_id] += j * 0.15
                agreement_bonus[sig_b.signal_id] += j * 0.15
                agreement_pairs.append(
                    f"{sig_a.agent_name} ↔ {sig_b.agent_name} "
                    f"(similarity={j:.2f})"
                )

    # ── Step 2: detect conflicts ──────────────────────────────────────────────
    # ContextAgent is excluded from consensus: its confidence score reflects
    # dataset-level ground-truth label coverage, not incident-specific RCA quality.
    incident_sigs = [s for s in signals if s.agent_name not in DATASET_AGENTS]

    if len(incident_sigs) >= 2:
        hyps = [(s.agent_name, s.root_cause_hypothesis) for s in incident_sigs]
        for i, (name_a, hyp_a) in enumerate(hyps):
            for name_b, hyp_b in hyps[i+1:]:
                j = _jaccard(hyp_a, hyp_b)
                if j < AGREEMENT_JACCARD_THRESHOLD:
                    conflict_notes.append(
                        f"Conflict: {name_a} vs {name_b} "
                        f"(similarity={j:.2f}) — "
                        f"{name_a}: '{hyp_a[:50]}' | "
                        f"{name_b}: '{hyp_b[:50]}'"
                    )

    # ── Step 3: score + resolve ───────────────────────────────────────────────
    # Score all signals (including dataset agents) for ranking display,
    # but select the winning hypothesis only from incident-specific agents.
    scored: List[Tuple[float, RCASignal]] = []
    for sig in signals:
        score = weights.get(sig.signal_id, 0.0) + agreement_bonus.get(sig.signal_id, 0.0)
        scored.append((score, sig))

    scored.sort(key=lambda x: -x[0])

    # Tiebreak: prefer earliest cascade component
    if len(scored) >= 2 and abs(scored[0][0] - scored[1][0]) < 0.01:
        winner = scored[0][1]
        runner = scored[1][1]
        if _earliest_component(runner) and not _earliest_component(winner):
            scored[0], scored[1] = scored[1], scored[0]

    # Consensus must come from an incident-specific agent, not ContextAgent.
    # ContextAgent's confidence reflects dataset-level label coverage; allowing
    # it to win produces hypotheses about label statistics rather than the
    # actual incident root cause.
    incident_scored = [(sc, sig) for sc, sig in scored if sig.agent_name not in DATASET_AGENTS]
    consensus_source = incident_scored[0][1] if incident_scored else scored[0][1]
    resolved = consensus_source.root_cause_hypothesis
    return resolved, conflict_notes, agreement_pairs


def combine_signals(signals: List[RCASignal]) -> FusedSignal:
    """
    Full fusion pipeline: normalise → weight → resolve conflicts → rank.

    This is the main entry point for the signal fusion engine.

    Args:
        signals: All RCASignals from all agents for one incident.

    Returns:
        FusedSignal with weighted_hypothesis, final_confidence,
        ranked_hypotheses, and conflict metadata.
    """
    if not signals:
        return FusedSignal(
            weighted_hypothesis="No agent signals available.",
            final_confidence=0.0,
        )

    # Step 1 — Normalise
    norm_confs = normalize_signal_confidence(signals)

    # Step 2 — Weight
    weights = weight_agent_signals(signals, norm_confs)

    # Step 3 — Resolve conflicts
    hypothesis, conflict_notes, agreement_pairs = resolve_conflicting_hypotheses(
        signals, weights
    )

    # Step 4 — Final confidence: weighted average of raw confidences
    final_conf = sum(
        sig.confidence * weights.get(sig.signal_id, 0.0)
        for sig in signals
    )
    final_conf = round(min(final_conf, 1.0), 3)

    # Step 5 — Ranked hypotheses
    ranked = rank_root_cause_hypotheses(signals, weights)

    return FusedSignal(
        weighted_hypothesis=hypothesis,
        final_confidence=final_conf,
        ranked_hypotheses=ranked,
        conflict_notes=conflict_notes,
        agreement_pairs=agreement_pairs,
        agent_weights={
            sig.agent_name: weights.get(sig.signal_id, 0.0)
            for sig in signals
        },
    )


def rank_root_cause_hypotheses(
    signals: List[RCASignal],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Produce a ranked list of root cause hypotheses across all agents.

    Each entry includes the agent name, hypothesis text, raw confidence,
    and fused weight score.

    Example output:
        [
          {"rank": 1, "agent": "LogInvestigationAgent",
           "hypothesis": "data TLB error interrupt...", "confidence": 0.97, "score": 0.41},
          {"rank": 2, "agent": "InfraAgent",
           "hypothesis": "Burst shape: FLOOD...",       "confidence": 0.93, "score": 0.30},
          {"rank": 3, "agent": "ContextAgent",
           "hypothesis": "PRIMARY: compute...",         "confidence": 0.87, "score": 0.15},
        ]

    Args:
        signals : RCASignal list.
        weights : Pre-computed weights (computed internally if None).

    Returns:
        List of ranked hypothesis dicts, best first.
    """
    if not signals:
        return []

    if weights is None:
        norm = normalize_signal_confidence(signals)
        weights = weight_agent_signals(signals, norm)

    scored = [
        {
            "agent":      sig.agent_name,
            "hypothesis": sig.root_cause_hypothesis,
            "confidence": sig.confidence,
            "score":      round(weights.get(sig.signal_id, 0.0), 4),
        }
        for sig in signals
    ]
    scored.sort(key=lambda x: -x["score"])

    for i, entry in enumerate(scored, 1):
        entry["rank"] = i

    return scored
    