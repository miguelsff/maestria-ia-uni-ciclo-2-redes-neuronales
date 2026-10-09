"""Exercise both MRI architectures at the full planned batch and resolution."""

import torch

from neural_exam.common.io import read_csv, read_json, write_json
from neural_exam.common.training import seed_everything
from neural_exam.config import load_settings
from neural_exam.mri.data import decode_stacks, find_dicom_root
from neural_exam.mri.labels import extract_labels
from neural_exam.mri.loss import classification_loss
from neural_exam.mri.model import StudyClassifier


def main() -> None:
    """Check real-data decoding, AMP forward/backward and bounded VRAM without scientific training."""
    settings = load_settings("configs/mri.toml")
    selected = read_json(settings.output_root / "audit/selected_series.json")
    record = next(row for row in selected if row["split"] == "train")
    metadata = next(
        row for row in read_csv(settings.metadata_root / "train.csv") if row["StudyInstanceUID"] == record["study_id"]
    )
    weak = extract_labels(metadata["Report"])
    image, details = decode_stacks(
        find_dicom_root(settings.mri_root) / record["study_id"] / record["series_id"], 8, 224
    )
    assert str(image.dtype) == "float32"
    target = torch.tensor(
        [float("nan") if value is None else value for value in weak["values"]], device="cuda", dtype=torch.float32
    )
    weights = torch.tensor(weak["confidence"], device="cuda")
    checks = []
    for architecture in ("scratch", "resnet50"):
        seed_everything(42)
        torch.cuda.reset_peak_memory_stats()
        model = StudyClassifier(architecture, weights_path=settings.weights_root / "resnet50-11ad3fa6.pth").cuda()
        batch = torch.from_numpy(image).cuda().unsqueeze(0).repeat(4, 1, 1, 1, 1)
        if architecture == "resnet50":
            mean = batch.new_tensor((0.485, 0.456, 0.406))[None, None, :, None, None]
            std = batch.new_tensor((0.229, 0.224, 0.225))[None, None, :, None, None]
            batch = (batch - mean) / std
        for loss_name in ("bce", "focal"):
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(batch)
                loss = classification_loss(logits, target.repeat(4, 1), weights.repeat(4, 1), kind=loss_name)
            loss.backward()
            torch.cuda.synchronize()
            assert logits.shape == (4, 12) and torch.isfinite(loss)
            assert all(
                torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None
            )
            checks.append(
                {
                    "architecture": architecture,
                    "loss": loss_name,
                    "loss_finite": True,
                    "gradients_finite": True,
                    "shape": list(logits.shape),
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                }
            )
        del model, batch, logits, loss
        torch.cuda.empty_cache()
    write_json(
        settings.output_root / "audit/mri_runtime.json",
        {
            "scientific_run": False,
            "source_split": "train",
            "image_dtype": "float32",
            "decoded": details,
            "checks": checks,
        },
    )
    print("Full-shape MRI runtime check passed for both architectures and losses.")


if __name__ == "__main__":
    main()
