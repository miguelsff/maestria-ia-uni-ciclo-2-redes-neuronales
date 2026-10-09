"""Configuration shared by native Python, Docker and notebooks."""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Resolved data locations and experiment parameters."""

    kaggle_root: Path
    metadata_root: Path
    mri_root: Path
    flood_root: Path
    wildlife_root: Path
    output_root: Path
    training: dict[str, Any]
    mri: dict[str, Any]
    flood: dict[str, Any]
    wildlife: dict[str, Any]
    wildlife_images_root: Path | None = None
    wildlife_annotations_root: Path | None = None

    @property
    def weights_root(self) -> Path:
        """Share explicitly downloaded assets across scientific and smoke outputs."""
        return Path(os.environ.get("PRETRAINED_WEIGHTS_DIR", str(self.output_root / "cache/weights")))

    def quick(self) -> Settings:
        """Return explicitly non-scientific, small real-data smoke parameters."""
        result = copy.deepcopy(self)
        result.training.update(epochs=1, patience=1, workers=0)
        result.mri.update(max_studies=74, stacks=2, batch_size=2)
        result.flood.update(weak_max=4, batch_size=1)
        result.wildlife.update(batch_size=2)
        return result


def load_settings(path: str | Path | None = None, device: str | None = None) -> Settings:
    """Read TOML and optional environment overrides without opening data files."""
    config_path = Path(path or os.environ.get("NEURAL_EXAM_CONFIG") or PROJECT_ROOT / "configs/default.toml")
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    root = Path(os.environ.get("KAGGLE_ROOT", config["paths"]["kaggle_root"]))
    training = config["training"]
    if device:
        training["device"] = device
    if training["device"] not in {"cpu", "cuda", "auto"}:
        raise ValueError("Device must be cpu, cuda or auto.")
    for key in ("epochs", "patience"):
        if training[key] < 1:
            raise ValueError(f"{key} must be positive.")
    if training["workers"] < 0:
        raise ValueError("workers must be nonnegative.")
    if not 0 < config["mri"]["validation_fraction"] < 1:
        raise ValueError("MRI validation_fraction must be between zero and one.")
    return Settings(
        kaggle_root=root,
        metadata_root=Path(os.environ.get("RSNA_METADATA_DIR", str(PROJECT_ROOT / "data"))),
        mri_root=Path(os.environ.get("RSNA_DATA_DIR", str(root))),
        flood_root=Path(os.environ.get("SEN1_DATA_DIR", str(root / "Sen1Floods11"))),
        wildlife_root=Path(os.environ.get("CCT_DATA_DIR", str(root / "CCT20"))),
        output_root=Path(os.environ.get("OUTPUT_DIR", str(root / "exam-output"))),
        training=training,
        mri=config["mri"],
        flood=config["flood"],
        wildlife=config["wildlife"],
        wildlife_images_root=Path(
            os.environ.get("CCT_IMAGES_DIR")
            or config["paths"].get("cct_images_dir")
            or os.environ.get("CCT_DATA_DIR", str(root / "CCT20"))
        ),
        wildlife_annotations_root=Path(
            os.environ.get("CCT_ANNOTATIONS_DIR")
            or config["paths"].get("cct_annotations_dir")
            or os.environ.get("CCT_DATA_DIR", str(root / "CCT20"))
        ),
    )
