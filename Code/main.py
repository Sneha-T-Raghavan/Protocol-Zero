"""
main.py
-------
Protocol Zero — Phase 2: Incident Detection Module
===================================================

Pipeline:
    load_logs → parse_logs → detect_anomalies → create_incidents → save_incidents

Run:
    python main.py
    python main.py --log-file data/logs.txt --output outputs/incidents.json
"""

import argparse
import sys
import os
import json

# Ensure project root is on the path regardless of CWD
sys.path.insert(0, os.path.dirname(__file__))

from utils.log_loader import load_logs
from utils.parser import parse_logs
from core.detection import run_all_detectors
from core.persistence import save_incidents, load_incidents


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    log_file: str,
    output_file: str,
    global_threshold: int,
    window_size: int,
    window_error_threshold: int,
) -> None:
    """Execute the full incident detection pipeline end-to-end."""

    banner = "=" * 60
    print(f"\n{banner}")
    print("  PROTOCOL ZERO — Phase 2: Incident Detection")
    print(f"{banner}\n")

    # ── Step 1: Load logs ──────────────────────────────────────────────────
    print("▶ Step 1/4  Loading logs …")
    raw_logs = load_logs(log_file)
    print()

    # ── Step 2: Parse logs ─────────────────────────────────────────────────
    print("▶ Step 2/4  Parsing log entries …")
    parsed_logs = parse_logs(raw_logs)
    print()

    # ── Step 3: Detect anomalies ───────────────────────────────────────────
    print("▶ Step 3/4  Running anomaly detectors …")
    incidents = run_all_detectors(
        parsed_logs,
        source=log_file,
        global_threshold=global_threshold,
        window_size=window_size,
        window_error_threshold=window_error_threshold,
    )
    print()

    # ── Step 4: Persist incidents ──────────────────────────────────────────
    print("▶ Step 4/4  Persisting incidents …")
    save_incidents(incidents, output_file)
    print()

    # ── Summary ────────────────────────────────────────────────────────────
    print(banner)
    print(f"  PIPELINE COMPLETE")
    print(f"  Logs processed  : {len(raw_logs)}")
    print(f"  Incidents found : {len(incidents)}")
    print(f"  Output file     : {output_file}")
    print(banner)

    if incidents:
        print("\n📋 Incident Summary:")
        for inc in incidents:
            print(f"  • [{inc.severity:8s}] {inc.anomaly_type:25s} | errors={inc.error_count:3d} | id={inc.incident_id[:8]}")
        print()

    # Pretty-print the JSON output for visual confirmation
    all_saved = load_incidents(output_file)
    print("📄 Sample incident JSON (first entry):")
    if all_saved:
        print(json.dumps(all_saved[0], indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Protocol Zero — Phase 2 Incident Detection"
    )
    p.add_argument(
        "--log-file",
        default="data/logs.txt",
        help="Path to input log file (default: data/logs.txt)",
    )
    p.add_argument(
        "--output",
        default="outputs/incidents.json",
        help="Path to output JSON file (default: outputs/incidents.json)",
    )
    p.add_argument(
        "--global-threshold",
        type=int,
        default=5,
        help="GlobalErrorSpikeDetector: min total errors to trigger (default: 5)",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="SlidingWindowDetector: entries per window (default: 10)",
    )
    p.add_argument(
        "--window-errors",
        type=int,
        default=3,
        help="SlidingWindowDetector: errors within window to trigger (default: 3)",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(
        log_file=args.log_file,
        output_file=args.output,
        global_threshold=args.global_threshold,
        window_size=args.window_size,
        window_error_threshold=args.window_errors,
    )