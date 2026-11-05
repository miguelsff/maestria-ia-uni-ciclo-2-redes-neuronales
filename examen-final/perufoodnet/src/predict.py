"""Predict one local image from a local checkpoint, without MLflow or the dataset."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from datasets import image_transform
from models import build_model


def predict(image_path: Path, checkpoint_path: Path, classes_path: Path, ingredients_path: Path | None = None) -> dict:
    """Return the proposed class and its uncalibrated softmax score."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = build_model(config["family"], None, config["regime"], checkpoint.get("architecture"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with Image.open(image_path) as image:
        tensor = image_transform(config["family"])(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        scores = model(tensor).softmax(1)[0]
    score, index = scores.max(0)
    classes = json.loads(classes_path.read_text(encoding="utf-8-sig"))
    class_name = classes[index.item()]
    ingredients_path = ingredients_path or classes_path.with_name("ingredients.json")
    catalog = json.loads(ingredients_path.read_text(encoding="utf-8-sig"))
    entry = catalog["dishes"][class_name]
    codes = sorted(set(entry["efsa"] or []) | set(entry["usda"] or []))
    return {
        "class_name": class_name,
        "score": score.item(),
        "uncertain": score.item() < 0.5,
        "recipe_reference": {
            "paper_name": entry["paper_name"],
            "efsa_codes": entry["efsa"],
            "usda_codes": entry["usda"],
            "union": [{"code": code, "name": catalog["ingredients"][code]["es"]} for code in codes],
            "source": catalog["source"],
            "limitation": "Ingredientes de una receta documentada; la imagen no detecta ingredientes. Una lista vacía no garantiza ausencia.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--ingredients", type=Path)
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.checkpoint, args.classes, args.ingredients), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
