"""Generate and verify the sequential PeruFoodNet data partition."""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIELDS = ("relative_path", "class_name", "class_index", "image_number", "split", "subset_10", "subset_40")
COUNT_FIELDS = ("class_name", "class_index", "train", "test", "val", "subset_10", "subset_40", "total")


def scan_dataset(dataset: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read and validate class folders and their numbered image files."""
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    entries = list(dataset.iterdir())
    if any(not entry.is_dir() for entry in entries):
        raise ValueError("Dataset root must contain only class directories")
    classes = sorted((entry.name for entry in entries), key=lambda name: (name.casefold(), name))
    if len(classes) != 40:
        raise ValueError(f"Expected 40 classes, found {len(classes)}")
    rows = []
    for index, class_name in enumerate(classes):
        numbered = {}
        for image in (dataset / class_name).iterdir():
            match = re.match(r"^(\d+)(?=\D|$)", image.name)
            if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not match:
                raise ValueError(f"Unexpected image entry: {image}")
            number = int(match.group(1))
            if number in numbered:
                raise ValueError(f"Duplicate image number {number} in {class_name}")
            numbered[number] = image
        if set(numbered) != set(range(1, 101)):
            raise ValueError(f"Expected unique image numbers 1–100 in {class_name}")
        for number, image in sorted(numbered.items()):
            rows.append({
                "relative_path": image.relative_to(dataset).as_posix(),
                "class_name": class_name,
                "class_index": str(index),
                "image_number": str(number),
                "split": "train" if number <= 70 else "test" if number <= 85 else "val",
                "subset_10": str(int(number <= 10)),
                "subset_40": str(int(number <= 40)),
            })
    return classes, rows


def make_counts(classes: list[str], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Summarize partitions and training subsets for each class."""
    counts = []
    for index, name in enumerate(classes):
        class_rows = [row for row in rows if row["class_name"] == name]
        splits = Counter(row["split"] for row in class_rows)
        counts.append({
            "class_name": name, "class_index": str(index),
            "train": str(splits["train"]), "test": str(splits["test"]), "val": str(splits["val"]),
            "subset_10": str(sum(int(row["subset_10"]) for row in class_rows)),
            "subset_40": str(sum(int(row["subset_40"]) for row in class_rows)),
            "total": str(len(class_rows)),
        })
    return counts


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    """Write a UTF-8 CSV with a stable column and row order."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_csv(path: Path, fields: tuple[str, ...], expected: list[dict[str, str]]) -> None:
    """Compare a CSV schema and every row with the dataset-derived records."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(fields):
            raise ValueError(f"Incorrect columns in {path}")
        actual = list(reader)
    if actual != expected:
        raise ValueError(f"Contents differ from the dataset-derived records: {path}")


def main() -> int:
    """Generate artifacts or verify existing artifacts without modifying them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the dataset root")
    parser.add_argument("--verify", action="store_true", help="Check existing artifacts without writing")
    args = parser.parse_args()
    data_dir = PROJECT_DIR / "data"
    try:
        classes, rows = scan_dataset(args.dataset)
        counts = make_counts(classes, rows)
        if args.verify:
            with (data_dir / "classes.json").open(encoding="utf-8") as file:
                if json.load(file) != classes:
                    raise ValueError("Class list differs from the dataset")
            verify_csv(data_dir / "manifest.csv", FIELDS, rows)
            verify_csv(data_dir / "counts.csv", COUNT_FIELDS, counts)
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            write_csv(data_dir / "manifest.csv", FIELDS, rows)
            write_csv(data_dir / "counts.csv", COUNT_FIELDS, counts)
            (data_dir / "classes.json").write_text(
                json.dumps(classes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"{'Verified' if args.verify else 'Generated'}: 40 classes, 4000 images; "
          "train=2800, test=600, val=600; subset_10=400, subset_40=1600")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
