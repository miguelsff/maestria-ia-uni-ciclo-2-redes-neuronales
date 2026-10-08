"""Preserve early-stopping state alongside interrupted YOLO checkpoints."""

from pathlib import Path

from neural_exam.common.io import file_sha256, read_json, write_json


def attach_resume_callbacks(model, run_dir: Path, resume: bool) -> None:
    """Restore patience only from a sidecar matching the actual last checkpoint."""
    path = run_dir / "resume_state.json"
    saved = None
    if resume:
        if not path.is_file():
            raise ValueError("Interrupted run lacks its patience state; unsafe resume is blocked.")
        saved = read_json(path)
        if saved["checkpoint_sha256"] != file_sha256(run_dir / "weights/last.pt"):
            raise ValueError("Resume sidecar does not match last.pt.")

    def restore(trainer):
        if saved is not None:
            if trainer.start_epoch != saved["epoch"] + 1:
                raise ValueError("Resume epoch differs from the patience sidecar.")
            if trainer.args.patience != saved["stopper"]["patience"]:
                raise ValueError("Resume patience differs from the original protocol.")
            for name, value in saved["stopper"].items():
                setattr(trainer.stopper, name, value)

    def persist(trainer):
        write_json(
            path,
            {
                "epoch": trainer.epoch,
                "checkpoint_sha256": file_sha256(trainer.last),
                "stopper": {
                    key: getattr(trainer.stopper, key)
                    for key in ("best_fitness", "best_epoch", "patience", "possible_stop")
                },
            },
        )

    model.add_callback("on_train_start", restore)
    model.add_callback("on_model_save", persist)
