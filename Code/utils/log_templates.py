"""
utils/log_templates.py
-----------------------
Protocol Zero — Phase 3.5: Improved Log Template Extraction.

Responsibility:
    Extracts structural templates from raw log messages, clusters similar
    log lines together, and counts template frequencies.

    Improves on the basic _templatise() in log_investigation_agent.py by:
        1. Multi-pass variable normalisation (hex, IP, timestamps, UUIDs,
           node IDs, job IDs, numeric sequences).
        2. Wildcard collapsing — consecutive <VAR> tokens are merged into
           a single <*> to reduce template fragmentation.
        3. Cluster merging by edit-distance proxy (token-set Jaccard) so
           that near-identical templates produced by slight phrasing
           differences are grouped together.
        4. Frequency counting with representative example retention.

Typical usage inside an agent:
    from utils.log_templates import extract_log_templates

    templates = extract_log_templates(error_messages)
    # → [{"template": "data TLB error at <*>", "count": 42,
    #      "examples": ["data TLB error at 0x3f00"], "pattern_id": "a3f9..."}]

No external dependencies — stdlib only.
"""

import hashlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

# ── Normalisation patterns (applied in order) ─────────────────────────────────

_NORM_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Hex addresses / memory addresses
    (re.compile(r"0x[0-9a-fA-F]+"),                         "<HEX>"),
    # IPv4 addresses
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),            "<IP>"),
    # ISO timestamps / date-time fragments
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"), "<TS>"),
    # Time-only (HH:MM:SS)
    (re.compile(r"\b\d{2}:\d{2}:\d{2}\b"),                  "<TIME>"),
    # BGL node IDs like R02-M1-N3-C:J08-U11
    (re.compile(r"R\d+-M\d+[\w\-:.]*"),                     "<NODE>"),
    # UUID / hex-dash strings
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<UUID>"),
    # Job / rank / PID integers preceded by keyword (not 'node' — handled by NODE_RE)
    (re.compile(r"(?i)(?:rank|job|pid|tid|core|cpu|slot)\s+\d+"), lambda m: m.group().split()[0] + " <ID>"),
    # Bare integers (must come after more specific patterns)
    (re.compile(r"\b\d+\b"),                                 "<NUM>"),
    # Unix file paths
    (re.compile(r"/[\w/.\-]+"),                              "<PATH>"),
]

# Merge consecutive variable placeholders into a single <*>
_COLLAPSE_RE = re.compile(r"(<[A-Z*]+>\s*){2,}")

# Cluster merge threshold: templates with Jaccard token similarity above this
# are merged into the same cluster (the more frequent template wins label).
CLUSTER_MERGE_THRESHOLD = 0.60


# ── Public API ────────────────────────────────────────────────────────────────

def extract_log_templates(
    messages: List[str],
    max_templates: int = 20,
) -> List[Dict]:
    """
    Extract structural templates from a list of raw log messages.

    Steps:
        1. Normalise each message → template string.
        2. Collapse consecutive variable tokens.
        3. Count frequencies and retain examples.
        4. Cluster near-identical templates.
        5. Return sorted by frequency descending.

    Args:
        messages     : Raw log message strings (not full log lines).
        max_templates: Maximum number of templates to return.

    Returns:
        List of dicts:
            {
                "template":   str,    # structural pattern with <*> wildcards
                "count":      int,    # number of messages matching this template
                "examples":   list,   # up to 3 representative raw messages
                "pattern_id": str,    # MD5 hash of the template string
            }
        Sorted by count descending.
    """
    if not messages:
        return []

    # Step 1+2: normalise + collapse
    raw_to_template: Dict[str, str] = {}
    for msg in messages:
        raw_to_template[msg] = _normalise(msg)

    # Step 3: count + examples
    template_counts:   Dict[str, int]       = defaultdict(int)
    template_examples: Dict[str, List[str]] = defaultdict(list)

    for msg, tmpl in raw_to_template.items():
        template_counts[tmpl] += 1
        if len(template_examples[tmpl]) < 3:
            template_examples[tmpl].append(msg)

    # Step 4: cluster near-identical templates
    clusters = cluster_log_patterns(list(template_counts.keys()))

    # Build final output: one entry per cluster, using the most-frequent
    # template as the label, summing counts across cluster members.
    results: List[Dict] = []
    for cluster in clusters:
        total_count = sum(template_counts[t] for t in cluster)
        # Representative template: most frequent in cluster
        rep = max(cluster, key=lambda t: template_counts[t])
        examples: List[str] = []
        for t in cluster:
            for ex in template_examples[t]:
                if ex not in examples:
                    examples.append(ex)
                if len(examples) >= 3:
                    break

        results.append({
            "template":   rep,
            "count":      total_count,
            "examples":   examples[:3],
            "pattern_id": hashlib.md5(rep.encode()).hexdigest()[:12],
        })

    results.sort(key=lambda x: -x["count"])
    return results[:max_templates]


def cluster_log_patterns(templates: List[str]) -> List[List[str]]:
    """
    Group structurally similar templates into clusters using token-set
    Jaccard similarity.

    Two templates are merged if their token-set Jaccard similarity exceeds
    CLUSTER_MERGE_THRESHOLD. Uses single-link agglomerative clustering
    (greedy, one pass) — O(n²) but n is small (≤ unique templates per window).

    Args:
        templates: List of normalised template strings.

    Returns:
        List of clusters, each cluster is a list of template strings.
    """
    if not templates:
        return []

    clusters: List[List[str]] = [[t] for t in templates]

    merged = True
    while merged:
        merged = False
        new_clusters: List[List[str]] = []
        used = [False] * len(clusters)

        for i in range(len(clusters)):
            if used[i]:
                continue
            base = list(clusters[i])

            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                # Compare representative (first) template of each cluster
                if _template_jaccard(base[0], clusters[j][0]) >= CLUSTER_MERGE_THRESHOLD:
                    base.extend(clusters[j])
                    used[j] = True
                    merged = True

            new_clusters.append(base)
            used[i] = True

        clusters = new_clusters

    return clusters


def count_template_frequency(
    messages: List[str],
) -> List[Tuple[str, int]]:
    """
    Lightweight frequency count without clustering — returns (template, count)
    pairs sorted by count descending. Useful for quick analysis.

    Args:
        messages: Raw log message strings.

    Returns:
        List of (template_string, count) sorted descending.
    """
    counts: Dict[str, int] = defaultdict(int)
    for msg in messages:
        counts[_normalise(msg)] += 1
    return sorted(counts.items(), key=lambda x: -x[1])


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise(message: str) -> str:
    """
    Apply all normalisation patterns to a message and collapse variable runs.
    """
    result = message
    for pattern, replacement in _NORM_PATTERNS:
        if callable(replacement):
            result = pattern.sub(replacement, result)
        else:
            result = pattern.sub(replacement, result)

    # Collapse consecutive variable tokens into <*>
    result = _COLLAPSE_RE.sub("<*> ", result)
    # Tidy up whitespace
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _template_jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity between two template strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)