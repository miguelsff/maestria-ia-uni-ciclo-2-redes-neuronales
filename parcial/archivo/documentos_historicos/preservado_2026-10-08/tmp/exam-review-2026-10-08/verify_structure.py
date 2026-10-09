"""Check dataset membership, reviewed inputs, preserved assets and publication."""

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DESTINATION = WORKSPACE / "audits/revision-2026-10-08"
DATA = Path("E:/Kaggle")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run():
    root = DATA / "exam-output-wildlife-v6/wildlife"
    manifest = read_json(root / "manifest.json")
    original_root = DATA / "CCT20/eccv_18_annotation_files"
    filenames = {"train": "train_annotations.json", "val": "cis_val_annotations.json",
                 "cis_test": "cis_test_annotations.json", "trans_val": "trans_val_annotations.json",
                 "trans_test": "trans_test_annotations.json"}
    records = manifest["records"]
    image_sets = {s: {r["image_id"] for r in rows} for s,rows in records.items()}
    sequence_sets = {s: {r["sequence_id"] for r in rows} for s,rows in records.items()}
    camera_sets = {s: {r["camera_id"] for r in rows} for s,rows in records.items()}
    assert sum(map(len, image_sets.values())) == len(set().union(*image_sets.values()))
    assert not any(sequence_sets["train"] & sequence_sets[s] for s in ("val", "cis_test", "trans_val", "trans_test"))
    assert not any(camera_sets["train"] & camera_sets[s] for s in ("trans_val", "trans_test"))
    assert camera_sets["cis_test"] <= camera_sets["train"]
    original_counts = {}
    box_checks, added_backgrounds = 0, []
    corrections = read_json(WORKSPACE / "configs/wildlife_label_corrections_v6.json")
    correction_ids = {str(row["image_id"]) for row in corrections["patches"]}
    for split, filename in filenames.items():
        path = original_root / filename
        assert digest(path) == manifest["annotation_sha256"][split]
        data = read_json(path)
        categories = {r["id"]: r["name"] for r in data["categories"]}
        assert {name for name in categories.values() if name not in {"empty", "car"}} == set(manifest["names"])
        originals = {r["id"]: r for r in data["images"]}
        annotations = {str(r["id"]): r for r in data["annotations"] if "bbox" in r}
        original_counts[split] = len(originals)
        for record in records[split]:
            if split == "train" and record["image_id"] not in originals:
                assert not record["boxes"]
                added_backgrounds.append(record)
                continue
            raw = originals[record["image_id"]]
            assert str(raw["location"]) == record["camera_id"]
            assert raw["seq_id"] == record["sequence_id"] and raw["date_captured"] == record["timestamp"]
            assert (raw["width"], raw["height"]) == (record["original_width"], record["original_height"])
            for box in record["boxes"]:
                if record["image_id"] in correction_ids:
                    continue
                annotation = annotations[box["annotation_id"]]
                assert categories[annotation["category_id"]] == manifest["names"][box["class_id"]]
                x,y,w,h = annotation["bbox"]
                expected = [(x+w/2)/raw["width"], (y+h/2)/raw["height"], w/raw["width"], h/raw["height"]]
                assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(expected,box["box"],strict=True))
                box_checks += 1
    assert len(added_backgrounds) == 1000
    metadata_root = DATA / "exam-output-wildlife-v2/inputs/backgrounds"
    cohort = read_json(metadata_root / "cohort.json")
    metadata_hashes = {name: digest(metadata_root/name) == value
                       for name,value in manifest["background_cohort"]["metadata_sha256"].items()}
    assert all(metadata_hashes.values())
    assert digest(metadata_root / "cohort.json") == manifest["background_cohort"]["manifest_sha256"]
    full_metadata = read_json(metadata_root / "full_metadata.json")
    full_splits = read_json(metadata_root / "CaltechCameraTrapsSplits_v0.json")["splits"]
    categories_by_image = {}
    for annotation in full_metadata["annotations"]:
        categories_by_image.setdefault(annotation["image_id"], set()).add(annotation["category_id"])
    official_backgrounds = {r["id"]: r for r in full_metadata["images"]}
    benchmark_cameras = set().union(*camera_sets.values()) - {r["camera_id"] for r in added_backgrounds}
    for background in added_backgrounds:
        assert categories_by_image[background["image_id"]] == {30}
        official = official_backgrounds[background["image_id"]]
        assert str(official["location"]) in full_splits["train"]
        assert str(official["location"]) not in benchmark_cameras
        assert official["seq_id"] == background["sequence_id"]
    assert len({r["sequence_id"] for r in added_backgrounds}) == 1000
    reviews = read_csv(root / "label_review.csv")
    assert len(reviews) == 90 and all(r["status"] == "reviewed" for r in reviews)
    sample_root = DATA / "exam-output-wildlife-v6-occlusion-sample-600"
    acceptance = read_json(sample_root/"review_acceptance.json")
    sample = read_csv(sample_root/"occlusion_review_sample_600.csv")
    accepted = read_csv(sample_root/"occlusion_review_accepted_582.csv")
    assert len(sample) == 600 and len(accepted) == 582
    assert digest(sample_root/"protocol.json") == acceptance["protocol_sha256"]
    assert digest(sample_root/"occlusion_review_sample_600.csv") == acceptance["reviews_sha256"]
    assert digest(sample_root/"occlusion_review_accepted_582.csv") == acceptance["accepted_csv_sha256"]
    assert all(digest(sample_root/name) == value for name,value in acceptance["source_files_sha256"].items())
    assert len({r["annotation_id"] for r in accepted}) == 582
    assert not ({r["annotation_id"] for r in accepted} & set(acceptance["excluded_annotation_ids"]))
    levels = Counter(row["level"] for row in accepted)
    assert levels == {"low": 340, "medium": 69, "high": 48, "ambiguous": 125}
    by_id = {r["annotation_id"]:r for r in accepted}
    assert by_id["23098"]["level"] == "medium" and by_id["23098"]["reviewer"] == "Miguel"
    assert not acceptance["notes_required"]
    historical_hashes = read_json(DATA / "exam-output-mri/submission-2026-10-04/audit/original_scientific_hashes.json")
    preservation = {name: digest(DATA / "exam-output-mri/stratified-v2" / name) == value
                    for name,value in historical_hashes.items()}
    assert all(preservation.values())
    result = dict(status="passed", audited_at_utc=datetime.now(timezone.utc).isoformat(),
        cct_original_split_images=original_counts, official_boxes_verified=box_checks,
        original_category_count=16, animal_categories=len(manifest["names"]),
        prepared_images=sum(map(len,records.values())), all_images_disjoint=True,
        training_sequences_disjoint_from_holdouts=True, new_cameras_disjoint=True,
        background_images=len(added_backgrounds), background_metadata_hashes=metadata_hashes,
        added_backgrounds_official_empty=True, added_backgrounds_official_training_cameras=True,
        human_label_reviews=len(reviews), accepted_occlusion=levels,
        ambiguous_cases=125, excluded_cases=18, accepted_cases_without_notes=sum(not r["visual_evidence"] for r in accepted),
        mri_historical_files_unchanged=len(preservation))
    (DESTINATION/"structure.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
