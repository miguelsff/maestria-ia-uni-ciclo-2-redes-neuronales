"""Validate the explicitly accepted 582-case analysis without inventing human notes."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from wildlife_occlusion_sample import LEVELS, digest, read_json, read_protocol, read_reviews, validate_reviews


def accepted_inputs(output: Path) -> tuple[dict, list[dict], dict, dict]:
    """Bind grades and exclusions to the unchanged 600-case sampling receipt."""
    protocol = read_protocol(output)
    acceptance = read_json(output / "review_acceptance.json")
    if acceptance["protocol_sha256"] != digest(output / "protocol.json"):
        raise ValueError("Accepted review uses a different sample protocol.")
    if acceptance["reviews_sha256"] != digest(output / "occlusion_review_sample_600.csv"):
        raise ValueError("Accepted human review changed after the recorded decision.")
    if acceptance["accepted_csv_sha256"] != digest(output / "occlusion_review_accepted_582.csv"):
        raise ValueError("Exported accepted cohort changed after the recorded decision.")
    if acceptance["notes_required"] is not False or acceptance["human_notes_fabricated"] is not False:
        raise ValueError("Missing notes must remain empty under the explicit user waiver.")
    rows = read_reviews(output)
    validate_reviews(rows, protocol)
    excluded = acceptance["excluded_annotation_ids"]
    if len(excluded) != len(set(excluded)) or len(excluded) != acceptance["excluded_cases"]:
        raise ValueError("Duplicate or inconsistent exclusions.")
    if set(excluded) != {row["annotation_id"] for row in rows if not row["level"]}:
        raise ValueError("Excluded IDs must exactly match the ungraded cases.")
    selected = [row for row in rows if row["annotation_id"] not in excluded]
    if len(selected) != acceptance["accepted_cases"] or len(selected) + len(excluded) != 600:
        raise ValueError("Accepted and excluded cases do not cover the fixed sample.")
    if any(row["level"] not in LEVELS or not row["reviewer"].strip() for row in selected):
        raise ValueError("An included case lacks a genuine human level or reviewer.")
    by_id = {row["annotation_id"]: row for row in rows}
    decision = acceptance["resolved_case_121"]
    if decision["annotation_id"] != protocol["selected"][120]["annotation_id"]:
        raise ValueError("Conflict decision refers to a different numbered case.")
    if any(
        by_id[decision["annotation_id"]][field] != decision[field] for field in ("level", "visual_evidence", "reviewer")
    ):
        raise ValueError("Case 121 differs from the user-selected human record.")
    counts = Counter(row["level"] for row in selected)
    status = {
        "status": "accepted_with_documented_exclusions_and_note_waiver",
        "original_sample": 600,
        "accepted": len(selected),
        "excluded": len(excluded),
        "counts": dict(counts),
        "accepted_cases_without_notes": sum(not row["visual_evidence"].strip() for row in selected),
        "minimum_support": protocol["minimum_detection_support_per_level"],
        "missing_support": {
            level: max(0, protocol["minimum_detection_support_per_level"] - counts[level]) for level in LEVELS[:3]
        },
    }
    return protocol, selected, acceptance, status


def describe_accepted_reviews(output: Path) -> dict:
    """Report observed ratios, retaining weights but making no complete-sample CI claim."""
    _, selected, acceptance, status = accepted_inputs(output)
    all_rows = read_reviews(output)
    accepted_weight = math.fsum(float(row["sample_weight"]) for row in selected)
    full_weight = math.fsum(float(row["sample_weight"]) for row in all_rows)
    missing_weight = math.fsum(float(row["sample_weight"]) for row in all_rows if not row["level"])
    levels = {}
    for level in LEVELS:
        rows = [row for row in selected if row["level"] == level]
        weight = math.fsum(float(row["sample_weight"]) for row in rows)
        levels[level] = {
            "cases": len(rows),
            "observed_case_proportion": len(rows) / len(selected),
            "original_weight_ratio_among_accepted_cases": weight / accepted_weight,
        }
    return {
        "method": "Descriptive proportions among accepted cases; unchanged original design weights.",
        "status": status,
        "levels": levels,
        "original_design_weight_sum": full_weight,
        "accepted_original_design_weight_sum": accepted_weight,
        "excluded_original_design_weight_sum": missing_weight,
        "excluded_design_weight_fraction": missing_weight / full_weight,
        "confidence_intervals_for_population_proportions": None,
        "original_95pct_plus_minus_5pp_planning_guarantee_claimed": False,
        "limitation": "The 18 excluded cases are human nonresponse, not a new random sample. "
        "Observed ratios are conditional on the 582 accepted cases. Original complete-sample "
        "proportion intervals are not reused and population representativeness is not asserted.",
        "user_decision": acceptance["user_instruction"],
    }
