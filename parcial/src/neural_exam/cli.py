"""Explicit project commands; help/configuration never download or train anything."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import replace
from pathlib import Path

from neural_exam.common.io import read_json, write_json
from neural_exam.config import load_settings

TASKS = ("mri", "flood", "wildlife")


def doctor(settings, gpu_check=False, amp_check=False) -> dict:
    """Inspect dependency versions and data locations; GPU operations require an explicit flag."""
    from importlib import metadata

    versions = {}
    for dependency in ("torch", "torchvision", "ultralytics", "pydicom", "rasterio", "scikit-learn"):
        try:
            versions[dependency] = metadata.version(dependency)
        except metadata.PackageNotFoundError:
            versions[dependency] = "not installed"
    result = {
        "dependencies": versions,
        "locations": {
            key: str(getattr(settings, key))
            for key in ("metadata_root", "mri_root", "flood_root", "wildlife_root", "output_root")
        },
        "data_directories_exist": {
            key: getattr(settings, key).is_dir() for key in ("metadata_root", "mri_root", "flood_root", "wildlife_root")
        },
        "gpu_check": "not requested",
        "wildlife_roots": {
            "images": str(settings.wildlife_images_root),
            "annotations": str(settings.wildlife_annotations_root),
        },
        "pretrained_weights_available": {
            name: (settings.weights_root / name).is_file()
            for name in ("resnet50-11ad3fa6.pth", settings.wildlife["model"], "yolo26n.pt")
        },
    }
    if gpu_check:
        import torch
        from torchvision.ops import nms

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside this runtime.")
        matrix = torch.randn(32, 32, device="cuda", requires_grad=True)
        (matrix @ matrix).square().mean().backward()
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], device="cuda", dtype=torch.float32)
        kept = nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5).cpu().tolist()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(matrix.grad).all()) or kept != [0]:
            raise RuntimeError("CUDA gradients or torchvision NMS failed their expected checks.")
        result["gpu_check"] = {
            "name": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "compiled_architectures": torch.cuda.get_arch_list(),
            "gradient_finite": bool(torch.isfinite(matrix.grad).all()),
            "nms_kept": kept,
        }
    if amp_check:
        from neural_exam.wildlife.runtime import check_amp

        result["amp_check"] = check_amp(settings)
    receipt_name = "doctor_gpu.json" if gpu_check or amp_check else "doctor.json"
    write_json(settings.output_root / "receipts" / receipt_name, result)
    return result


def parser() -> argparse.ArgumentParser:
    """Define stable command interfaces used by README instructions and notebooks."""
    root = argparse.ArgumentParser(description="Local neural-network exam workflows")
    root.add_argument("--config", type=Path, help="TOML experiment configuration")
    root.add_argument("--device", choices=["cpu", "cuda", "auto"])
    commands = root.add_subparsers(dest="command", required=True)
    diagnostics = commands.add_parser("doctor", help="Inspect dependencies and data paths")
    diagnostics.add_argument("--gpu-check", action="store_true")
    diagnostics.add_argument("--amp-check", action="store_true")
    commands.add_parser("verify-wildlife-resume", help="Check YOLO epoch/optimizer recovery on isolated synthetic data")
    for command in ("prepare", "run", "smoke"):
        subcommand = commands.add_parser(command)
        subcommand.add_argument("--task", choices=[*TASKS, "all"], default="all")
        if command != "prepare":
            subcommand.add_argument("--resume", action="store_true")
    downloads = commands.add_parser("download", help="Explicit downloads, never started by run")
    downloads.add_argument(
        "--dataset", choices=["flood", "cct", "weights", "style", "cct-annotations", "wildlife-weights"], required=True
    )
    reports = commands.add_parser("report", help="Generate tables from actual metrics without compiling PDFs")
    reports.add_argument("--task", choices=[*TASKS, "all"], default="all")
    review = commands.add_parser("review-occlusion", help="Prepare the manual visual-review sheet")
    review.add_argument(
        "--extend", type=int, default=0, help="Add candidate objects without replacing existing reviews"
    )
    commands.add_parser("review-labels", help="Prepare or verify the human label-review gallery")
    return root


def main() -> None:
    """Dispatch only the operation explicitly requested by the caller."""
    arguments = parser().parse_args()
    settings = load_settings(arguments.config, arguments.device)
    for variable, relative in (("YOLO_CONFIG_DIR", "cache/ultralytics"), ("TORCH_HOME", "cache/torch")):
        directory = Path(os.environ.setdefault(variable, str(settings.output_root / relative)))
        directory.mkdir(parents=True, exist_ok=True)
    if arguments.command == "doctor":
        result = doctor(settings, arguments.gpu_check, arguments.amp_check)
    elif arguments.command == "verify-wildlife-resume":
        from neural_exam.common.locking import execution_lock
        from neural_exam.wildlife.runtime import verify_resume

        with execution_lock(settings.output_root):
            result = verify_resume(settings)
    elif arguments.command == "download":
        from neural_exam.common.download import download_dataset

        result = download_dataset(settings, arguments.dataset)
    elif arguments.command == "report":
        from neural_exam.reporting import prepare_report_assets

        result = prepare_report_assets(settings, arguments.task)
    else:
        from neural_exam.common.locking import execution_lock

        quick = arguments.command == "smoke"
        os.environ.setdefault("PRETRAINED_WEIGHTS_DIR", str(settings.weights_root))
        if quick:
            settings = replace(settings.quick(), output_root=settings.output_root / "smoke")
        os.environ.setdefault("TORCH_HOME", str(settings.output_root / "cache/torch"))
        os.environ.setdefault("YOLO_CONFIG_DIR", str(settings.output_root / "cache/ultralytics"))
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        if arguments.command in {"review-occlusion", "review-labels"}:
            from neural_exam.wildlife.data import prepare
            from neural_exam.wildlife.review import label_review_status, prepare_label_review, prepare_occlusion_review

            with execution_lock(settings.output_root):
                manifest_path = settings.output_root / "wildlife/manifest.json"
                manifest = read_json(manifest_path) if manifest_path.exists() else prepare(settings)
                if arguments.command == "review-labels":
                    path = prepare_label_review(manifest, settings.output_root / "wildlife", settings.training["seed"])
                    result = {"review_csv": str(path), **label_review_status(path)}
                else:
                    result = {
                        "review_csv": str(
                            prepare_occlusion_review(
                                manifest,
                                settings.output_root / "wildlife",
                                settings.training["seed"],
                                extend=arguments.extend,
                            )
                        )
                    }
        else:
            tasks = TASKS if arguments.task == "all" else (arguments.task,)
            if arguments.command != "prepare":
                required = []
                if "mri" in tasks and not quick and "resnet50" in settings.mri["architectures"]:
                    required.append("resnet50-11ad3fa6.pth")
                if "wildlife" in tasks:
                    required.append(settings.wildlife["model"])
                    if settings.training["amp"] and settings.training["device"] != "cpu":
                        required.append("yolo26n.pt")
                missing = [name for name in required if not (settings.weights_root / name).is_file()]
                if missing:
                    raise FileNotFoundError(f"Explicit weight download required before training: {', '.join(missing)}")
            with execution_lock(settings.output_root):
                manifests = {
                    task: (
                        importlib.import_module("neural_exam.wildlife.data").prepare(settings, quick=quick)
                        if task == "wildlife"
                        else importlib.import_module(f"neural_exam.{task}.data").prepare(settings)
                    )
                    for task in tasks
                }
                if arguments.command == "prepare":
                    if "wildlife" in manifests:
                        from neural_exam.wildlife.review import prepare_label_review

                        prepare_label_review(
                            manifests["wildlife"], settings.output_root / "wildlife", settings.training["seed"]
                        )
                    result = {
                        task: {"identity": manifest["identity"], "status": "prepared"}
                        for task, manifest in manifests.items()
                    }
                else:
                    # All requested inputs are validated before the first training run starts.
                    results = [
                        importlib.import_module(f"neural_exam.{task}.pipeline").run(
                            settings, resume=arguments.resume, quick=quick, manifest=manifests[task]
                        )
                        for task in tasks
                    ]
                    result = {"scientific_run": not quick, "tasks": results}
                    write_json(settings.output_root / "execution_summary.json", result)
                    if not quick:
                        from neural_exam.reporting import prepare_report_assets

                        prepare_report_assets(settings, arguments.task)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
