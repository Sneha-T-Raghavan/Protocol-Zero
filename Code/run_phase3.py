"""
run_phase3.py
-------------
Protocol Zero — Phase 3: Multi-Agent Investigation Pipeline
============================================================

Runs Phases 2 + 3 end-to-end on a BGL log dataset:
    load → parse → detect (Phase 2) → investigate (Phase 3) → report

Produces:
    outputs/bgl_incidents.json          — Phase 2 detected incidents
    outputs/investigation_reports.json  — Phase 3 investigation reports
    outputs/phase3_report.txt           — Human-readable Phase 3 summary

Usage:
    python run_phase3.py                              # uses data/BGL.log
    python run_phase3.py --log-file data/BGL.log
    python run_phase3.py --lines 2000                 # limit to 2k lines
    python run_phase3.py --incidents-only             # skip Phase 3, just detect
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))

from utils.log_loader       import load_logs
from utils.bgl_parser       import parse_bgl_logs
from core.detection         import run_all_detectors
from core.persistence       import save_incidents
from core.agent_runner      import run_investigation
from agents.rca_signal      import InvestigationReport

DETECTION_CONFIG = {
    "global_threshold":        50,
    "window_size":            100,
    "window_error_threshold":  20,
}


# ── Report printer ────────────────────────────────────────────────────────────

def format_phase3_report(
    reports: List[InvestigationReport],
    elapsed: float,
    n_logs:  int,
    n_incidents: int,
) -> str:
    lines = [
        "=" * 70,
        "  PROTOCOL ZERO — Phase 3 | Multi-Agent Investigation Report",
        "=" * 70,
        "",
        "── Pipeline Summary ────────────────────────────────────────────────",
        f"  Log lines processed : {n_logs:>10,}",
        f"  Incidents detected  : {n_incidents:>10,}",
        f"  Reports generated   : {len(reports):>10,}",
        f"  Total runtime       : {elapsed:>9.2f}s",
        "",
    ]

    for i, report in enumerate(reports, 1):
        lines += [
            f"── Report {i}/{len(reports)}: Incident {report.incident_id[:8]} ──────────────────",
            f"  Summary   : {report.incident_summary}",
            f"  Confidence: {report.overall_confidence:.0%}",
            "",
            f"  CONSENSUS HYPOTHESIS:",
            f"    {report.consensus_hypothesis}",
            "",
        ]

        if report.top_components:
            lines.append(f"  Top Components : {', '.join(report.top_components[:5])}")

        if report.top_error_patterns:
            lines.append("  Top Error Patterns:")
            for p in report.top_error_patterns[:3]:
                lines.append(f"    • {p[:75]}")

        if report.timeline:
            lines.append("  Timeline (first occurrences):")
            for t in report.timeline[:4]:
                lines.append(f"    → {t[:75]}")

        lines.append("")
        lines.append("  Action Items:")
        for j, item in enumerate(report.action_items[:6], 1):
            lines.append(f"    {j}. {item}")

        lines.append("")
        lines.append("  Agent Signals:")
        for sig in report.signals:
            lines.append(
                f"    [{sig.agent_name:<26}] "
                f"confidence={sig.confidence:.0%} | "
                f"{sig.root_cause_hypothesis[:60]}"
            )
        lines.append("")
        lines.append("─" * 70)
        lines.append("")

    lines += ["=" * 70, ""]
    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(
    log_file:      str,
    output_incidents: str,
    output_reports:   str,
    output_report_txt: str,
    max_lines:     int,
    skip_phase3:   bool,
) -> None:
    banner = "=" * 70
    print(f"\n{banner}")
    print("  PROTOCOL ZERO — Phases 2 + 3 Full Pipeline")
    print(f"{banner}\n")

    if not os.path.exists(log_file):
        print(f"❌  Log file not found: '{log_file}'")
        print("    Run: python tools/generate_bgl_dataset.py --lines 50000")
        sys.exit(1)

    t_start = time.time()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print("▶ Phase 2 — Incident Detection\n")

    print("  Step 1/4  Loading logs ...")
    raw_logs = load_logs(log_file, max_lines=max_lines if max_lines > 0 else None)
    print()

    print("  Step 2/4  Parsing BGL format ...")
    parsed_logs = parse_bgl_logs(raw_logs)
    print()

    print("  Step 3/4  Running anomaly detectors ...")
    incidents = run_all_detectors(parsed_logs, source=log_file, **DETECTION_CONFIG)
    print()

    print("  Step 4/4  Saving incidents ...")
    save_incidents(incidents, output_incidents, overwrite=True)
    print()

    if skip_phase3:
        print("  --incidents-only flag set. Skipping Phase 3.")
        return

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    print(f"▶ Phase 3 — Multi-Agent Investigation ({len(incidents)} incidents)\n")
    reports = run_investigation(
        incidents=incidents,
        parsed_logs=parsed_logs,
        output_file=output_reports,
        overwrite=True,
    )

    t_elapsed = time.time() - t_start

    # ── Format and save text report ───────────────────────────────────────────
    report_txt = format_phase3_report(reports, t_elapsed, len(raw_logs), len(incidents))
    print(report_txt)

    os.makedirs(os.path.dirname(output_report_txt) or ".", exist_ok=True)
    with open(output_report_txt, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"📄 Phase 3 text report → '{output_report_txt}'")
    print(f"📦 Investigation JSON  → '{output_reports}'")
    print(f"📦 Incidents JSON      → '{output_incidents}'")
    print(f"\n⏱  Total runtime: {t_elapsed:.2f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Protocol Zero Phase 2+3 Pipeline")
    p.add_argument("--log-file",        default="data/BGL.log")
    p.add_argument("--output-incidents",default="outputs/bgl_incidents.json")
    p.add_argument("--output-reports",  default="outputs/investigation_reports.json")
    p.add_argument("--output-report-txt",default="outputs/phase3_report.txt")
    p.add_argument("--lines",           type=int, default=0, help="0 = all lines")
    p.add_argument("--incidents-only",  action="store_true", help="Skip Phase 3")
    args = p.parse_args()

    run_full_pipeline(
        log_file=args.log_file,
        output_incidents=args.output_incidents,
        output_reports=args.output_reports,
        output_report_txt=args.output_report_txt,
        max_lines=args.lines,
        skip_phase3=args.incidents_only,
    )