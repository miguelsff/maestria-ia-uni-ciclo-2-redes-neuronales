"""Download only the official MRI backbone weights and NeurIPS author kit."""

from neural_exam.common.download import STYLE_URL, download_file, safe_extract
from neural_exam.common.io import file_sha256, write_json
from neural_exam.config import load_settings


def main() -> None:
    """Save provenance and hashes without accessing or modifying the dataset."""
    settings = load_settings("configs/mri.toml")
    weights_url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
    weights = download_file(weights_url, settings.weights_root / "resnet50-11ad3fa6.pth")
    checksum = file_sha256(weights)
    if not checksum.startswith("11ad3fa6"):
        raise ValueError("The downloaded weights do not match the official hash prefix.")
    assets = settings.output_root / "report-assets"
    archive = download_file(STYLE_URL, assets / "Formatting_Instructions_For_NeurIPS_2026.zip")
    safe_extract(archive, assets)
    write_json(
        settings.output_root / "receipts/mri_assets.json",
        {
            "weights": {
                "source": weights_url,
                "sha256": checksum,
                "bytes": weights.stat().st_size,
                "torchvision_enum": "ResNet50_Weights.IMAGENET1K_V2",
            },
            "style": {"source": STYLE_URL, "sha256": file_sha256(archive), "bytes": archive.stat().st_size},
        },
    )
    print("Official MRI assets downloaded and hashed.", flush=True)


if __name__ == "__main__":
    main()
