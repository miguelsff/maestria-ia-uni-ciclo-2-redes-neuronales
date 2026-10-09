"""Check the installed YOLO verifier accepts every prepared image and target."""

from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

import numpy as np


def verify_manifest_images(manifest: dict) -> dict:
    """Reject silent sample loss, duplicate target removal or missing background labels."""
    from ultralytics.data.utils import verify_image_label

    summaries, problems = {}, []
    for split, records in manifest["records"].items():
        arguments = []
        for row in records:
            image = Path(row["image"])
            label = image.parents[2] / "labels" / split / f"{image.stem}.txt"
            arguments.append((str(image), str(label), "", False, len(manifest["names"]), 0, 0, False))
        accepted = backgrounds = boxes = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, (row, result) in enumerate(
                zip(records, pool.map(verify_image_label, arguments), strict=True), 1
            ):
                image, labels, _, _, _, missing, found, empty, corrupt, message = result
                expected = np.asarray([[box["class_id"], *box["box"]] for box in row["boxes"]], dtype=np.float32)
                valid = image is not None and not corrupt and not missing and found == 1 and not message
                valid = valid and len(labels) == len(expected) and bool(empty) == (len(expected) == 0)
                if valid and len(expected):
                    valid = np.allclose(labels, expected, rtol=0, atol=5e-7)
                if valid:
                    accepted += 1
                    backgrounds += int(empty)
                    boxes += len(labels)
                else:
                    problems.append(
                        {"split": split, "image_id": row["image_id"], "message": message or "Targets differ"}
                    )
                if index % 5000 == 0 or index == len(records):
                    print(
                        f"YOLO loader verification {split}: {index}/{len(records)}; problems: {len(problems)}",
                        flush=True,
                    )
        summaries[split] = {
            "expected_images": len(records),
            "accepted_images": accepted,
            "backgrounds": backgrounds,
            "boxes": boxes,
        }
    return {
        "status": "passed" if not problems else "failed",
        "manifest_identity": manifest["identity"],
        "ultralytics_version": metadata.version("ultralytics"),
        "verifier": "ultralytics.data.utils.verify_image_label",
        "splits": summaries,
        "problem_count": len(problems),
        "problems": problems,
    }
