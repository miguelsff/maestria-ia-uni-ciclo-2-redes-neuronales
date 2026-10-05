"""Explicit ecological activity counts, not estimates of unique animal populations."""

from collections import defaultdict
from datetime import datetime


def activity_report(frames: list[dict], detections: list[dict], names: list[str], confidence=0.25) -> dict:
    """Aggregate detections and empty-photo rates by camera and capture day."""
    ids = [frame["image_id"] for frame in frames]
    identifiers = set(ids)
    if len(ids) != len(identifiers):
        raise ValueError("Duplicate frames would inflate ecological activity counts.")
    by_image = defaultdict(list)
    for detection in detections:
        if detection["image_id"] not in identifiers:
            raise ValueError("Detection refers to a frame without camera metadata.")
        if not 0 <= detection["class_id"] < len(names) or not 0 <= detection["confidence"] <= 1:
            raise ValueError("Invalid detection class or confidence.")
        if detection["confidence"] >= confidence:
            by_image[detection["image_id"]].append(detection)
    groups = defaultdict(
        lambda: {
            "photos": 0,
            "positive_photos": 0,
            "species": set(),
            "counts": defaultdict(int),
            "species_photos": defaultdict(int),
        }
    )
    for frame in frames:
        try:
            date = datetime.fromisoformat((frame.get("timestamp") or "").replace("Z", "+00:00")).date().isoformat()
        except (ValueError, TypeError):
            date = "unknown"
        group = groups[(frame["camera_id"], date)]
        group["photos"] += 1
        predictions = by_image[frame["image_id"]]
        group["positive_photos"] += bool(predictions)
        present = set()
        for prediction in predictions:
            species = names[prediction["class_id"]]
            group["species"].add(species)
            group["counts"][species] += 1
            present.add(species)
        for species in present:
            group["species_photos"][species] += 1
    cameras, species_rows = [], []
    for (camera, date), group in sorted(groups.items()):
        cameras.append(
            {
                "camera_id": camera,
                "date": date,
                "photos": group["photos"],
                "photos_with_fauna": group["positive_photos"],
                "empty_photo_rate": 1 - group["positive_photos"] / group["photos"],
                "distinct_species": len(group["species"]),
            }
        )
        for species, count in sorted(group["counts"].items()):
            species_rows.append(
                {
                    "camera_id": camera,
                    "date": date,
                    "species": species,
                    "detections": count,
                    "photos_with_species": group["species_photos"][species],
                }
            )
    return {
        "camera_days": cameras,
        "species_days": species_rows,
        "confidence": confidence,
        "interpretation": "counts are detections and photographs, never unique individual animals",
    }
