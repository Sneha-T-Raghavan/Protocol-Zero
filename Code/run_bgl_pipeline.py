"""
run_bgl_pipeline.py
--------------------
Protocol Zero — Phase 2: Full BGL Dataset Pipeline
====================================================

Runs the complete detection pipeline on a BGL log dataset (real or generated).
Produces:
  - outputs/bgl_incidents.json  — detected incidents
  - outputs/bgl_report.txt      — human-readable benchmark report

Usage:
    # Step 1: generate dataset (run once)
    python tools/generate_bgl_dataset.py --lines 50000

    # Step 2: run full pipeline
    python run_bgl_pipeline.py

    # Step 2 with options
    python run_bgl_pipeline.py --log-file data/BGL.log --lines 0  # 0 = all lines
    python run_bgl_pipeline.py --log-file data/BGL.log --lines 10000  # first 10k
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))

from utils.log_loader     import load_logs
from utils.bgl_parser     import parse_bgl_logs
from core.detection       import run_all_detectors, GlobalErrorSpikeDetector, SlidingWindowDetector
from core.incident        import Incident
from core.persistence     import save_incidents


# ── Detection config ─────────────────────────────────────────────────────────

DETECTION_CONFIG = {
    "global_threshold":       50,   # tuned for large dataset (50k+ lines)
    "window_size":           100,   # 100-entry sliding window
    "window_error_threshold": 20,   # ≥20 errors in 100 entries = burst
}


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_detection(parsed_logs: List[Dict], incidents: List[Incident]) -> Dict:
    """
    Compare detected incidents against BGL ground-truth labels.
    Returns a dict of evaluation metrics.
    """
    # Ground-truth: windows that contain at least one labelled anomaly
    total_anomalous_lines = sum(1 for e in parsed_logs if e.get("is_anomaly"))
    total_error_lines     = sum(1 for e in parsed_logs if e["level"] in {"ERROR", "CRITICAL"})

    # Error lines per component
    component_errors: Dict[str, int] = defaultdict(int)
    for e in parsed_logs:
        if e["level"] in {"ERROR", "CRITICAL"}:
            component_errors[e.get("component", "UNKNOWN")] += 1

    return {
        "total_log_lines":      len(parsed_logs),
        "total_anomalous_gt":   total_anomalous_lines,
        "total_error_lines":    total_error_lines,
        "incidents_detected":   len(incidents),
        "error_rate_pct":       round(total_error_lines / max(len(parsed_logs), 1) * 100, 2),
        "anomaly_rate_pct":     round(total_anomalous_lines / max(len(parsed_logs), 1) * 100, 2),
        "top_error_components": sorted(component_errors.items(), key=lambda x: -x[1])[:5],
    }


def print_report(metrics: Dict, incidents: List[Incident], elapsed: float) -> str:
    """Build and return a formatted text report."""
    lines = [
        "=" * 65,
        "  PROTOCOL ZERO — Phase 2 | BGL Dataset Benchmark Report",
        "=" * 65,
        "",
        "── Dataset Stats ──────────────────────────────────────────",
        f"  Total log lines     : {metrics['total_log_lines']:>10,}",
        f"  Labelled anomalies  : {metrics['total_anomalous_gt']:>10,}  ({metrics['anomaly_rate_pct']}%)",
        f"  Error-level entries : {metrics['total_error_lines']:>10,}  ({metrics['error_rate_pct']}%)",
        "",
        "── Detection Results ───────────────────────────────────────",
        f"  Incidents detected  : {metrics['incidents_detected']:>10,}",
        f"  Pipeline runtime    : {elapsed:>9.2f}s",
        f"  Lines/second        : {metrics['total_log_lines']/max(elapsed,0.001):>10,.0f}",
        "",
        "── Top Error Components ────────────────────────────────────",
    ]
    for comp, count in metrics["top_error_components"]:
        lines.append(f"  {comp:<20} {count:>6,} errors")
    lines.append("")

    if incidents:
        lines.append("── Incident Breakdown ──────────────────────────────────────")
        severity_counts: Dict[str, int] = defaultdict(int)
        type_counts: Dict[str, int] = defaultdict(int)
        for inc in incidents:
            severity_counts[inc.severity] += 1
            type_counts[inc.anomaly_type] += 1

        lines.append("  By anomaly type:")
        for atype, cnt in sorted(type_counts.items()):
            lines.append(f"    {atype:<30} {cnt:>4,}")
        lines.append("  By severity:")
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            cnt = severity_counts.get(sev, 0)
            bar = "█" * min(cnt, 40)
            lines.append(f"    {sev:<10} {cnt:>4,}  {bar}")
        lines.append("")

        lines.append("── Sample Incidents (first 5) ──────────────────────────────")
        for inc in incidents[:5]:
            lines.append(f"  [{inc.severity:8s}] {inc.anomaly_type}")
            lines.append(f"    ID       : {inc.incident_id}")
            lines.append(f"    Errors   : {inc.error_count}")
            lines.append(f"    Details  : {inc.details[:80]}…")
            if inc.sample_errors:
                lines.append(f"    Sample   : {inc.sample_errors[0][:70]}")
            lines.append("")

    lines += ["=" * 65, ""]
    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_bgl_pipeline(log_file: str, output_file: str, report_file: str, max_lines: int, append_mode: bool = False) -> None:

    banner = "=" * 65
    print(f"\n{banner}")
    print("  PROTOCOL ZERO — Phase 2: BGL Full Dataset Pipeline")
    print(f"{banner}\n")

    if not os.path.exists(log_file):
        print(f"❌  Log file not found: '{log_file}'")
        print("    Run first:  python tools/generate_bgl_dataset.py --lines 50000")
        sys.exit(1)

    t_start = time.time()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("▶ Step 1/4  Loading BGL logs …")
    raw_logs = load_logs(log_file, max_lines=max_lines if max_lines > 0 else None)
    print()

    # ── 2. Parse (BGL-specific) ───────────────────────────────────────────────
    print("▶ Step 2/4  Parsing BGL log format …")
    parsed_logs = parse_bgl_logs(raw_logs)
    print()

    # ── 3. Detect ─────────────────────────────────────────────────────────────
    print("▶ Step 3/4  Running anomaly detectors …")
    incidents = run_all_detectors(
        parsed_logs,
        source=log_file,
        **DETECTION_CONFIG,
    )
    print()

    # ── 4. Save ───────────────────────────────────────────────────────────────
    print("▶ Step 4/4  Persisting incidents …")
    save_incidents(incidents, output_file, overwrite=not append_mode)
    print()

    t_elapsed = time.time() - t_start

    # ── Evaluate & report ────────────────────────────────────────────────────
    metrics = evaluate_detection(parsed_logs, incidents)
    report  = print_report(metrics, incidents, t_elapsed)

    print(report)

    # Save text report
    os.makedirs(os.path.dirname(report_file) or ".", exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📄 Report saved → '{report_file}'")
    print(f"📦 Incidents   → '{output_file}'")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Protocol Zero BGL full dataset pipeline")
    p.add_argument("--log-file",    default="data/BGL.log",              help="BGL log file path")
    p.add_argument("--output",      default="outputs/bgl_incidents.json", help="Output incidents JSON")
    p.add_argument("--report",      default="outputs/bgl_report.txt",     help="Output report text file")
    p.add_argument("--lines",       type=int, default=0,                  help="Max lines to process (0 = all)")
    p.add_argument("--append",      action="store_true",                   help="Append to existing file instead of overwriting")
    args = p.parse_args()

    run_bgl_pipeline(
        log_file=args.log_file,
        output_file=args.output,
        report_file=args.report,
        max_lines=args.lines,
        append_mode=args.append,
    )