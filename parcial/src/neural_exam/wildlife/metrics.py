"""COCO box evaluation with object-aware size and manually reviewed occlusion strata."""

from __future__ import annotations

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_evaluate(
    records: list[dict],
    predictions: list[dict],
    names: list[str],
    area_range: tuple[float, float] | None = None,
    occlusions: dict | None = None,
    occlusion_group: str | None = None,
) -> dict:
    """Evaluate fixed categories at IoU 0.50:0.95 using 101 recall points.

    Coordinates use a virtual 1000x1000 image, preserving each image's IoU while
    allowing size strata based on relative box area. Other occlusion strata are
    represented as ignored crowd regions, so correct detections of them are not FPs.
    """
    ids = {record["image_id"]: index + 1 for index, record in enumerate(records)}
    if len(ids) != len(records) or not names or len(set(names)) != len(names):
        raise ValueError("Evaluation requires unique images and category names.")
    for prediction in predictions:
        coordinates = prediction["bbox"]
        if prediction["image_id"] not in ids or not 0 <= prediction["class_id"] < len(names):
            raise ValueError("Prediction refers to an unknown image or category.")
        if (
            len(coordinates) != 4
            or not np.isfinite([*coordinates, prediction["confidence"]]).all()
            or min(coordinates[2:]) < 0
            or not 0 <= prediction["confidence"] <= 1
        ):
            raise ValueError("Invalid prediction coordinates or confidence.")
        # Clipped zero-area predictions remain false-positive candidates in COCO.
    images, annotations = [], []
    annotation_id = 0
    eligible = 0
    for record in records:
        images.append({"id": ids[record["image_id"]], "width": 1000, "height": 1000})
        for box in record["boxes"]:
            if not 0 <= box["class_id"] < len(names):
                raise ValueError("Ground truth refers to an unknown category.")
            annotation_id += 1
            cx, cy, width, height = box["box"]
            crowd = int(occlusion_group is not None and (occlusions or {}).get(box["annotation_id"]) != occlusion_group)
            area = width * height * 1e6
            eligible += int(not crowd and (area_range is None or area_range[0] <= area <= area_range[1]))
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": ids[record["image_id"]],
                    "category_id": box["class_id"] + 1,
                    "bbox": [(cx - width / 2) * 1000, (cy - height / 2) * 1000, width * 1000, height * 1000],
                    "area": area,
                    "iscrowd": crowd,
                }
            )
    if not eligible:
        return {
            "map50": None,
            "map50_95": None,
            "eligible_objects": 0,
            "max_detections": 300,
            "per_species": [{"species": name, "support": 0, "map50": None, "map50_95": None} for name in names],
        }
    detections = []
    for prediction in predictions:
        x, y, width, height = prediction["bbox"]
        detections.append(
            {
                "image_id": ids[prediction["image_id"]],
                "category_id": prediction["class_id"] + 1,
                "bbox": [x * 1000, y * 1000, width * 1000, height * 1000],
                "score": prediction["confidence"],
            }
        )
    # pycocotools.loadRes requires at least one detection; empty predictions are valid zero AP.
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {
            "info": {},
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i + 1, "name": name} for i, name in enumerate(names)],
        }
        truth.createIndex()
        if detections:
            detected = truth.loadRes(detections)
        else:
            detected = COCO()
            detected.dataset = {"images": images, "annotations": [], "categories": truth.dataset["categories"]}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.imgIds = list(ids.values())
        evaluator.params.catIds = list(range(1, len(names) + 1))
        evaluator.params.maxDets = [1, 10, 300]
        if area_range:
            evaluator.params.areaRng = [list(area_range)]
            evaluator.params.areaRngLbl = ["selected"]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]

    def mean_defined(values):
        defined = values[values >= 0]
        return float(defined.mean()) if defined.size else None

    return {
        "map50": mean_defined(precision[0]),
        "map50_95": mean_defined(precision),
        "eligible_objects": eligible,
        "max_detections": 300,
        "per_species": [
            {
                "species": name,
                "support": sum(
                    annotation["category_id"] == index + 1
                    and not annotation["iscrowd"]
                    and (area_range is None or area_range[0] <= annotation["area"] <= area_range[1])
                    for annotation in annotations
                ),
                "map50": mean_defined(precision[0, :, index]),
                "map50_95": mean_defined(precision[:, :, index]),
            }
            for index, name in enumerate(names)
        ],
    }
