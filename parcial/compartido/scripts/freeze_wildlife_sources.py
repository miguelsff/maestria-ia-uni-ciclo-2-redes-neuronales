"""Preserve the code and configuration that can affect wildlife training."""

import os
import shutil
from importlib import metadata
from pathlib import Path

from neural_exam.common.io import file_sha256, read_json, write_json
from neural_exam.config import PROJECT_ROOT, load_settings


def main() -> None:
    settings = load_settings()
    package = PROJECT_ROOT / "src/neural_exam"
    paths = [*(package / "wildlife").glob("*.py"), *(package / "common").glob("*.py")]
    paths += [package / filename for filename in ("config.py", "cli.py", "__main__.py", "__init__.py")]
    paths += [PROJECT_ROOT / filename for filename in ("requirements.txt", "Dockerfile", "configs/wildlife.toml")]
    if settings.wildlife.get("annotation_corrections"):
        paths += [Path(settings.wildlife["annotation_corrections"]), Path(os.environ["NEURAL_EXAM_CONFIG"])]
        paths += [
            PROJECT_ROOT / filename
            for filename in (
                "scripts/run_wildlife.ps1",
                "scripts/freeze_wildlife_sources.py",
                "scripts/audit_wildlife_preflight.py",
                "scripts/migrate_wildlife_reviews.py",
                "scripts/check_wildlife_gate.py",
                "scripts/bootstrap_wildlife_v3.py",
                "compose.wildlife.yaml",
                "compose.wildlife.v2.yaml",
                "compose.wildlife.v3.yaml",
            )
        ]
    paths = sorted(set(paths))
    config_name = Path(os.environ.get("NEURAL_EXAM_CONFIG", "")).name
    if config_name in {"wildlife_v4.toml", "wildlife_v5.toml", "wildlife_v6.toml"}:
        paths.append(PROJECT_ROOT / "compose.wildlife.v4.yaml")
    if config_name in {"wildlife_v5.toml", "wildlife_v6.toml"}:
        paths.append(PROJECT_ROOT / "compose.wildlife.v5.yaml")
    if config_name == "wildlife_v6.toml":
        paths.append(PROJECT_ROOT / "compose.wildlife.v6.yaml")
        paths.append(PROJECT_ROOT / "scripts/verify_wildlife_loader.py")
    hashes = {path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path) for path in sorted(paths)}
    receipt_path = settings.output_root / "receipts/source-freeze.json"
    destination = settings.output_root / "receipts/source-freeze"
    receipt = {
        "scope": "Training, data conversion, prediction and configuration; editorial report files are independent.",
        "sha256": hashes,
        "versions": {name: metadata.version(name) for name in ("torch", "torchvision", "ultralytics", "pillow")},
    }
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError("Frozen experimental sources changed. Preserve this execution and use a new output.")
    for path in paths:
        target = destination / path.relative_to(PROJECT_ROOT)
        if target.exists():
            if file_sha256(target) != hashes[path.relative_to(PROJECT_ROOT).as_posix()]:
                raise ValueError("Preserved source snapshot changed.")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    write_json(receipt_path, receipt)
    print({"status": "sources frozen", "files": len(hashes)})


if __name__ == "__main__":
    main()
