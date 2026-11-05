"""MLflow setup and explicit experiment evidence."""
import json
import importlib.metadata
import os
import sys
import platform
import subprocess
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature


def setup(config: dict) -> None:
    """Connect to the designated tracking server and experiment."""
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment"])


def log_context(config: dict, train_rows, val_rows, project: Path) -> None:
    """Log effective settings and dataset inputs."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    params = dict(config, commit=commit, split_policy="sequential", train_count=len(train_rows),
                  val_count=len(val_rows), torch_version=torch.__version__, mlflow_version=mlflow.__version__,
                  python_version=platform.python_version(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                  transforms="resize 224x224; train horizontal flip; ResNet ImageNet normalization; YOLO 0/1")
    for package in ("torchvision", "ultralytics", "pandas", "scikit-learn", "pillow"):
        params[f"version_{package}"] = importlib.metadata.version(package)
    params["cuda_version"] = str(torch.version.cuda)
    mlflow.log_params(params)
    for context, rows in (("training", train_rows), ("validation", val_rows)):
        dataset = mlflow.data.from_pandas(rows, source=str(project / "data/manifest.csv"),
                                         targets="class_index", name=f"perufoodnet_{context}")
        mlflow.log_input(dataset, context=context)
    mlflow.log_artifact(str(project / "data/manifest.csv"), "data")
    mlflow.log_artifact(str(project / "data/classes.json"), "data")


def log_native_model(model, sample: torch.Tensor, output_dir: Path, model_name: str, phase: str) -> str:
    """Log a native model and check its recovered logits against the local model."""
    model = model.cpu().eval()
    sample = sample.cpu()
    with torch.no_grad():
        expected = model(sample).numpy()
    info = mlflow.pytorch.log_model(model, name=model_name, input_example=sample.numpy(),
                                   tags={"phase": phase, "protocol": "R03.1",
                                         "selection_eligible": str(phase == "matrix").lower()},
                                   signature=infer_signature(sample.numpy(), expected),
                                   code_paths=[str(Path(__file__).with_name("models.py"))],
                                   serialization_format="pickle",
                                   pip_requirements=["torch==2.11.0", "torchvision==0.26.0", "ultralytics==8.4.202", "cloudpickle==3.1.2"])
    sample_path = output_dir / "sample.npy"
    result_path = output_dir / "recovered.npy"
    np.save(sample_path, sample.numpy())
    code = """
import mlflow
import mlflow.pytorch
import numpy as np
import sys
import torch
mlflow.set_tracking_uri(sys.argv[1])
model = mlflow.pytorch.load_model(sys.argv[2], map_location="cpu").eval()
inputs = torch.from_numpy(np.load(sys.argv[3]))
outputs = model(inputs).detach().numpy()
np.save(sys.argv[4], outputs)
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-c", code, mlflow.get_tracking_uri(), info.model_uri,
                    str(sample_path.resolve()), str(result_path.resolve())],
                   cwd=output_dir, env=env, check=True)
    actual = np.load(result_path)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    sample_path.unlink()
    result_path.unlink()
    (output_dir / "model_uri.json").write_text(json.dumps({"uri": info.model_uri, "max_abs_diff": float(np.max(np.abs(actual - expected))), "independent_process": True}, indent=2), encoding="utf-8")
    return info.model_uri
