"""
core/persistence.py
--------------------
Handles saving Incident objects to JSON.

Behaviour:
  - Creates the output file if it does not exist.
  - Appends to an existing file, but deduplicates by incident_id.
  - overwrite=True replaces the file entirely (clean slate).
  - Writes pretty-printed JSON for human readability.
"""

import json
import os
from typing import List

from core.incident import Incident


def save_incidents(
    incidents: List[Incident],
    filepath: str,
    overwrite: bool = False,
) -> None:
    """
    Persist a list of Incident objects to a JSON file.

    Args:
        incidents : List of Incident objects to save.
        filepath  : Destination JSON file path.
        overwrite : If True, replace file contents entirely (no appending).
                    If False (default), append but deduplicate by incident_id
                    so re-running never creates duplicate entries.
    """
    if not incidents:
        print("[Persistence] No incidents to save.")
        return

    new_dicts = [inc.to_dict() for inc in incidents]

    if overwrite or not os.path.exists(filepath):
        merged = new_dicts
        skipped = 0
    else:
        existing: List[dict] = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, ValueError):
            existing = []

        existing_ids = {e["incident_id"] for e in existing}
        truly_new = [d for d in new_dicts if d["incident_id"] not in existing_ids]
        skipped = len(new_dicts) - len(truly_new)
        merged = existing + truly_new

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    action = "Overwrote" if overwrite else "Saved"
    skip_note = f" (skipped {skipped} duplicates)" if skipped else ""
    print(
        f"[Persistence] {action} {len(incidents)} incident(s){skip_note} "
        f"| total in file: {len(merged)} -> '{filepath}'"
    )


def load_incidents(filepath: str) -> List[dict]:
    """
    Load previously saved incidents from a JSON file.
    Returns an empty list if the file does not exist or is malformed.
    """
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        return []