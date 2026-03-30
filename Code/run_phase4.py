"""
run_phase4.py
--------------
Protocol Zero — Phase 4 entry point.

Full pipeline: Log ingestion → Detection → Investigation → Correlation

Usage:
    python run_phase4.py --log-file data/BGL.log
    python run_phase4.py --log-file data/BGL.log --lines 5000
    python run_phase4.py --log-file data/BGL.log --time-window 300
    python run_phase4.py --log-file data/BGL.log --investigation-only  # skip correlation
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.log_loader     import load_logs
from utils.bgl_parser     import parse_bgl_line
from utils.parser         import parse_line as parse_log_line
from core.detection       import run_all_detectors
from core.persistence     import save_incidents
from core.agent_runner    import run_investigation
from core.correlation     import run_correlation, TIME_CORRELATION_WINDOW

DETECTION_CONFIG = {
    "global_threshold":       50,
    "window_size":           100,
    "window_error_threshold": 20,
}


def main():
    parser = argparse.ArgumentParser(description="Protocol Zero — Phase 4 Pipeline")
    parser.add_argument("--log-file",   default="data/BGL.log", help="Path to log file")
    parser.add_argument("--lines",      type=int, default=None,  help="Limit to first N lines")
    parser.add_argument("--time-window",type=int, default=TIME_CORRELATION_WINDOW,
                        help=f"Correlation time window in seconds (default: {TIME_CORRELATION_WINDOW})")
    parser.add_argument("--investigation-only", action="store_true",
                        help="Run Phase 2+3 only, skip Phase 4 correlation")
    args = parser.parse_args()

    # ── Phase 2: Ingest + Detect ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Protocol Zero — Phase 4 Pipeline")
    print(f"  Log file : {args.log_file}")
    print(f"  Lines    : {args.lines or 'ALL'}")
    print(f"{'='*60}\n")

    raw_lines = load_logs(args.log_file, max_lines=args.lines)
    print(f"[Phase 2] Loaded {len(raw_lines):,} log lines.")

    # Auto-detect BGL format
    is_bgl = False
    if raw_lines:
        first = raw_lines[0].split()
        is_bgl = (len(first) >= 9 and (first[0] == "-" or first[0].startswith("BOOT")))

    parse_fn = parse_bgl_line if is_bgl else parse_log_line
    parsed   = [parse_fn(line) for line in raw_lines]
    parsed   = [p for p in parsed if p]
    print(f"[Phase 2] Parsed {len(parsed):,} entries ({'BGL' if is_bgl else 'generic'} format).")

    incidents = run_all_detectors(parsed, source=args.log_file, **DETECTION_CONFIG)
    save_incidents(incidents, "outputs/bgl_incidents.json", overwrite=True)
    print(f"[Phase 2] {len(incidents)} incident(s) detected.\n")

    if not incidents:
        print("[Phase 4] No incidents — nothing to investigate or correlate.")
        return

    # ── Phase 3: Multi-Agent Investigation ───────────────────────────────────
    reports = run_investigation(
        incidents, parsed,
        output_file="outputs/investigation_reports.json",
        overwrite=True,
    )
    print(f"[Phase 3] {len(reports)} investigation report(s) generated.\n")

    if args.investigation_only:
        print("[Phase 4] Skipped (--investigation-only flag set).")
        return

    # ── Phase 4: Incident Correlation ─────────────────────────────────────────
    groups = run_correlation(
        incidents, reports,
        output_file="outputs/correlated_reports.json",
        time_window=args.time_window,
        overwrite=True,
    )

    # Human-readable summary
    summary_path = "outputs/phase4_report.txt"
    _write_summary(groups, summary_path)
    print(f"\n[Phase 4] Summary written → '{summary_path}'")
    print(f"[Phase 4] Done. {len(incidents)} incidents → {len(groups)} correlated group(s).")


def _write_summary(groups, path):
    lines = [
        "=" * 70,
        "  PROTOCOL ZERO — PHASE 4 CORRELATED REPORT",
        "=" * 70,
        f"  Total groups : {len(groups)}",
        f"  Total incidents: {sum(len(g.incidents) for g in groups)}",
        "",
    ]
    for i, group in enumerate(groups, 1):
        hyp = group.root_cause_hypothesis
        # Wrap hypothesis across lines cleanly at pipe separators
        hyp_parts = [p.strip() for p in hyp.split("|")]
        hyp_lines = ["  Root cause       : " + hyp_parts[0]]
        for part in hyp_parts[1:]:
            hyp_lines.append("                     | " + part)

        lines += [
            f"── Group {i}: {group.group_id[:12]} ──",
            f"  Incidents        : {len(group.incidents)}",
            f"  Time span        : {group.start_time} → {group.end_time}",
            f"  Components       : {', '.join(sorted(group.affected_components)) or 'N/A'}",
            f"  Nodes            : {', '.join(sorted(group.affected_nodes)[:5]) or 'N/A'}",
            f"  Cascade chain    : {' → '.join(group.cascade_chain) or 'N/A'}",
            f"  Confidence       : {group.confidence:.0%}",
        ] + hyp_lines + [""]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()