"""Stop before training if Ultralytics would silently reject prepared inputs."""

from neural_exam.common.io import read_json, write_json
from neural_exam.config import load_settings
from neural_exam.wildlife.loader import verify_manifest_images


def main() -> None:
    settings = load_settings()
    manifest = read_json(settings.output_root / "wildlife/manifest.json")
    receipt = verify_manifest_images(manifest)
    write_json(settings.output_root / "receipts/yolo-loader-verification.json", receipt)
    if receipt["status"] != "passed":
        raise ValueError("YOLO rejected prepared inputs; inspect receipts/yolo-loader-verification.json.")
    print({"status": "passed", "splits": receipt["splits"]})


if __name__ == "__main__":
    main()
