"""Cluster bootstrap of saved detections with a fixed model and protocol."""

from collections import defaultdict

import numpy as np

from neural_exam.wildlife.metrics import coco_evaluate


def bootstrap_metrics(
    records: list[dict], predictions: list[dict], names: list[str], replicates: int, seed: int, **kwargs
) -> dict:
    """Resample complete sequences; report undefined replicates without replacing them."""
    if replicates < 2:
        raise ValueError("Bootstrap requires at least two replicates.")
    identifiers = {row["image_id"] for row in records}
    if len(identifiers) != len(records):
        raise ValueError("Bootstrap input contains duplicate frame IDs.")
    if any(row["image_id"] not in identifiers for row in predictions):
        raise ValueError("Bootstrap prediction refers to a frame outside this split.")
    if any(not row.get("sequence_id") for row in records):
        raise ValueError("Bootstrap requires a real sequence identifier for each frame.")
    groups, by_image = defaultdict(list), defaultdict(list)
    for row in records:
        groups[row["sequence_id"]].append(row)
    for row in predictions:
        by_image[row["image_id"]].append(row)
    sequences = sorted(groups)
    if len(sequences) < 2:
        raise ValueError("Bootstrap requires multiple independent sequences.")
    generator = np.random.default_rng(seed)
    values = {key: [] for key in ("map50", "map50_95")}
    for replicate in range(replicates):
        sampled, detected = [], []
        for index in generator.integers(0, len(sequences), len(sequences)):
            for row in groups[sequences[index]]:
                identifier = f"bootstrap-{len(sampled)}-{row['image_id']}"
                sampled.append({**row, "image_id": identifier})
                detected.extend({**prediction, "image_id": identifier} for prediction in by_image[row["image_id"]])
        metrics = coco_evaluate(sampled, detected, names, **kwargs)
        for key in values:
            if metrics[key] is not None:
                values[key].append(metrics[key])
        if (replicate + 1) % 10 == 0 or replicate + 1 == replicates:
            print(
                f"Sequence bootstrap: {replicate + 1}/{replicates}; frames: {len(records)}; "
                f"area: {kwargs.get('area_range', 'all')}; occlusion: {kwargs.get('occlusion_group', 'all')}",
                flush=True,
            )
    return {
        "method": "percentile sequence-cluster bootstrap; fixed checkpoint; not between-seed uncertainty",
        "replicates": replicates,
        "seed": seed,
        "clusters": len(sequences),
        "confidence_level": 0.95,
        "intervals": {
            key: {
                "lower": float(np.quantile(value, 0.025)) if value else None,
                "upper": float(np.quantile(value, 0.975)) if value else None,
                "defined_replicates": len(value),
            }
            for key, value in values.items()
        },
    }


def assess_uncertainty(
    manifest: dict, predictions: dict, robustness: dict, replicates: int, seed: int, reviews: list[dict]
) -> dict:
    """Use declared strata and exclude unfinished human occlusion estimates."""
    result = {
        "global": {},
        "size": {},
        "occlusion": {},
        "limitations": "Conditional on sampled cameras and one training seed; sequence resampling does not measure new-region uncertainty.",
    }
    sizes = {
        "small_lt_1pct": (0, float(np.nextafter(10000.0, 0))),
        "medium_1_to_5pct": (10000, 50000),
        "large_gt_5pct": (float(np.nextafter(50000.0, np.inf)), 1e10),
    }
    for split in ("cis_test", "trans_test"):
        args = (manifest["records"][split], predictions[split], manifest["names"], replicates, seed)
        result["global"][split] = bootstrap_metrics(*args)
        result["size"][split] = {name: bootstrap_metrics(*args, area_range=bounds) for name, bounds in sizes.items()}
    if robustness["occlusion_status"] == "reviewed":
        levels = {row["annotation_id"]: row["level"] for row in reviews}
        for level in ("low", "medium", "high"):
            result["occlusion"][level] = bootstrap_metrics(
                manifest["records"]["trans_test"],
                predictions["trans_test"],
                manifest["names"],
                replicates,
                seed,
                occlusions=levels,
                occlusion_group=level,
            )
    result["occlusion_status"] = robustness["occlusion_status"]
    return result
