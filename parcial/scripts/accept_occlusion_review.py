"""Apply the user's explicit exclusions, conflict resolution and note waiver atomically."""

from __future__ import annotations

import argparse
import csv
import io
import os
from datetime import UTC, datetime
from pathlib import Path

from occlusion_accepted_review import accepted_inputs
from wildlife_occlusion_sample import (
    FIELDS,
    HUMAN_FIELDS,
    digest,
    read_json,
    read_protocol,
    validate_reviews,
    verify_parent,
    write_json,
    write_reviews,
)

USER_INSTRUCTION = (
    "no tomar en cuenta los 18 casos sin nivel y el caso 121 tomar en cuenta lo que dice Miguel; "
    "me lo entregaron sin nota ya no puedo hacer nada"
)


def accept(parent: Path, output: Path, consolidation: Path) -> dict:
    """Keep the five originals and the old CSV; change only the authorized sample record."""
    protocol = read_protocol(output)
    verify_parent(parent, protocol["parent_sha256"])
    receipt = read_json(consolidation / "consolidation_receipt.json")
    if (output / "review_acceptance.json").exists():
        return accepted_inputs(output)[3]
    draft_path = consolidation / "occlusion_review_consolidated_DRAFT.csv"
    if digest(draft_path) != receipt["draft_sha256"]:
        raise ValueError("Provisional consolidation changed.")
    for entry in receipt["source_files"]:
        if digest(output / entry["file"]) != entry["sha256"]:
            raise ValueError("Returned human CSV changed.")
    with draft_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    validate_reviews(rows, protocol)
    numbers = {row["annotation_id"]: index + 1 for index, row in enumerate(protocol["selected"])}
    case_121 = next(row["annotation_id"] for row in protocol["selected"] if numbers[row["annotation_id"]] == 121)
    miguel_name = next(
        entry["file"] for entry in receipt["source_files"] if entry["file"].startswith("occlusion_review_Miguel_")
    )
    with (output / miguel_name).open(encoding="utf-8-sig", newline="") as stream:
        miguel = next(row for row in csv.DictReader(stream) if row["annotation_id"] == case_121)
    if miguel["reviewer"] != "Miguel" or miguel["level"] != "medium":
        raise ValueError("The selected case-121 record is not the reviewed Miguel record.")
    selected_row = next(row for row in rows if row["annotation_id"] == case_121)
    for field in HUMAN_FIELDS:
        selected_row[field] = miguel[field]
    excluded = [row["annotation_id"] for row in rows if not row["level"]]
    if (
        sorted(numbers[identifier] for identifier in excluded) != receipt["unreviewed_case_numbers"]
        or len(excluded) != 18
    ):
        raise ValueError("Excluded cases differ from the user's 18 ungraded cases.")
    validate_reviews(rows, protocol)
    canonical = output / "occlusion_review_sample_600.csv"
    before = canonical.read_bytes()
    initial = consolidation / "received_originals"  # Genuine source snapshots remain independently preserved.
    for entry in receipt["source_files"]:
        if digest(initial / entry["file"]) != entry["sha256"]:
            raise ValueError("Archived human source changed.")
    archive = output / "review_archive"
    archive.mkdir(exist_ok=True)
    backup = archive / "before_acceptance_2026-10-08.csv"
    if backup.exists() and backup.read_bytes() != before:
        raise ValueError("Refusing to replace a different review backup.")
    backup.write_bytes(before)
    acceptance = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "user_instruction": USER_INSTRUCTION,
        "protocol_sha256": digest(output / "protocol.json"),
        "original_cases": 600,
        "accepted_cases": 582,
        "excluded_cases": 18,
        "excluded_annotation_ids": excluded,
        "excluded_case_numbers": receipt["unreviewed_case_numbers"],
        "exclusion_reason": "No human occlusion level; user explicitly excluded these cases.",
        "notes_required": False,
        "human_notes_fabricated": False,
        "resolved_case_121": {
            "annotation_id": case_121,
            "source_file": miguel_name,
            **{field: miguel[field] for field in HUMAN_FIELDS},
        },
        "source_files_sha256": {entry["file"]: entry["sha256"] for entry in receipt["source_files"]},
        "previous_canonical_sha256": digest(backup),
        "previous_canonical_backup": str(backup),
        "teacher_approval_claimed": False,
        "original_sampling_receipt_changed": False,
        "population_proportion_ci_with_complete_600_assumptions_allowed": False,
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows([row for row in rows if row["annotation_id"] not in excluded])
    accepted_csv = output / "occlusion_review_accepted_582.csv"
    accepted_csv.write_text(stream.getvalue(), encoding="utf-8-sig", newline="")
    # Publication follows a final check so an intervening live edit is never overwritten.
    if canonical.read_bytes() != before:
        raise ValueError("Live review changed during acceptance.")
    write_reviews(output, rows)
    acceptance["reviews_sha256"] = digest(canonical)
    acceptance["accepted_csv_sha256"] = digest(accepted_csv)
    temporary = output / "review_acceptance.json.tmp"
    write_json(temporary, acceptance)
    os.replace(temporary, output / "review_acceptance.json")
    status = accepted_inputs(output)[3]
    write_json(output / "accepted_review_status.json", status)
    verify_parent(parent, protocol["parent_sha256"])
    return status


def main() -> None:
    """Apply the recorded user decision using the verified provisional consolidation."""
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consolidation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(accept(args.parent, args.output, args.consolidation), indent=2))


if __name__ == "__main__":
    main()
