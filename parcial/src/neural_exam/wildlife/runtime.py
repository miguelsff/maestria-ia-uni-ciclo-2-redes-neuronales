"""Strict offline diagnostics for the pinned YOLO runtime."""

import logging

from neural_exam.common.io import write_json


def check_amp(settings) -> dict:
    """Reject skipped AMP checks and compare the exercise detector in FP32/AMP."""
    import torch
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings
    from ultralytics.utils import ASSETS, LOGGER
    from ultralytics.utils.checks import check_amp as upstream_check

    for name in (settings.wildlife["model"], "yolo26n.pt"):
        if not (settings.weights_root / name).is_file():
            raise FileNotFoundError(f"Missing local weights: {name}.")
    if not (ASSETS / "bus.jpg").is_file():
        raise FileNotFoundError("The pinned package lacks its built-in AMP-check image.")
    yolo_settings.update({"sync": False, "weights_dir": str(settings.weights_root)})
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture()
    LOGGER.addHandler(handler)
    model = YOLO(str(settings.weights_root / settings.wildlife["model"])).model.to("cuda").float().eval()
    try:
        passed = upstream_check(model)
    finally:
        LOGGER.removeHandler(handler)
    if not passed or any("AMP" in message and ("skipped" in message or "failed" in message) for message in messages):
        raise RuntimeError("AMP check was skipped or failed; training is blocked.")
    tensor = torch.rand(2, 3, 256, 256, device="cuda")
    with torch.no_grad():
        fp32 = model(tensor)[0]
        with torch.amp.autocast("cuda"):
            mixed = model(tensor)[0].float()
    finite = bool(torch.isfinite(mixed).all())
    close = fp32.shape == mixed.shape and bool(torch.allclose(fp32, mixed, atol=0.5, rtol=0.01))
    if not finite or not close:
        raise RuntimeError("YOLOv8s FP32/AMP comparison failed.")
    result = {
        "status": "passed",
        "auxiliary": "yolo26n.pt",
        "detector": settings.wildlife["model"],
        "finite": finite,
        "fp32_amp_close": close,
        "messages": messages,
    }
    write_json(settings.output_root / "receipts/amp.json", result)
    del model, tensor, fp32, mixed
    torch.cuda.empty_cache()
    return result


def verify_resume(settings) -> dict:
    """Interrupt YOLO after a saved epoch and check actual optimizer/epoch restoration."""
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings

    yolo_settings.update({"sync": False, "weights_dir": str(settings.weights_root)})
    output = settings.output_root / "resume-check"
    if (output / "passed.json").is_file():
        from neural_exam.common.io import read_json

        return read_json(output / "passed.json")
    dataset = output / "dataset"
    generator = np.random.default_rng(42)
    for split, count in (("train", 8), ("val", 4)):
        for index in range(count):
            image = dataset / "images" / split / f"fixture_{index}.jpg"
            label = dataset / "labels" / split / f"fixture_{index}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(image)
            label.write_text("0 0.5 0.5 0.4 0.4\n" if index % 2 else "", encoding="utf-8")
    config = dataset / "data.yaml"
    config.write_text(
        yaml.safe_dump({"path": str(dataset), "train": "images/train", "val": "images/val", "names": ["fixture"]}),
        encoding="utf-8",
    )
    model = YOLO(str(settings.weights_root / settings.wildlife["model"]))

    def interrupt(trainer):
        if trainer.epoch == 0:
            raise RuntimeError("controlled_resume_interruption")

    model.add_callback("on_model_save", interrupt)
    try:
        model.train(
            data=str(config),
            epochs=3,
            patience=10,
            imgsz=64,
            batch=2,
            workers=0,
            device=0,
            amp=False,
            seed=42,
            deterministic=True,
            optimizer="AdamW",
            lr0=0.0003,
            project=str(output),
            name="train",
            exist_ok=True,
            plots=False,
            mosaic=0.0,
        )
    except RuntimeError as error:
        if str(error) != "controlled_resume_interruption":
            raise
    last = output / "train/weights/last.pt"
    state = torch.load(last, map_location="cpu", weights_only=False)
    if state["epoch"] != 0 or not state.get("optimizer"):
        raise RuntimeError("First checkpoint lacks the expected optimizer or epoch.")
    observations = {}

    def observe(trainer):
        observations.update(start_epoch=trainer.start_epoch, optimizer_states=len(trainer.optimizer.state))

    resumed = YOLO(str(last))
    resumed.add_callback("on_train_start", observe)
    resumed.train(resume=True)
    rows = pd.read_csv(output / "train/results.csv")
    if observations.get("start_epoch") != 1 or not observations.get("optimizer_states") or len(rows) != 3:
        raise RuntimeError(f"YOLO resume did not restore its state: {observations}, {len(rows)} rows.")
    result = {
        "status": "passed",
        "completed_epochs": len(rows),
        **observations,
        "scope": "YOLO optimizer and epoch restoration; no bitwise equivalence claim",
    }
    write_json(output / "passed.json", result)
    return result
