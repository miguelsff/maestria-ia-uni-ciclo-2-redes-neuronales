"""Prepare a private runtime and dispatch closure checks or an original scientific runner."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from closure_runtime import materialize

DEFAULT_PARENT = Path("E:/Kaggle/exam-output-wildlife-v6")
DEFAULT_OCCLUSION = Path("E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600")


def example_values(path: Path) -> dict[str, str]:
    """Read public examples only; never persist the calling process's secrets."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if key.replace("_", "").isalnum() and key.isupper():
                values[key] = value.strip()
    return values


def prepare_run(exam: Path, parent: Path, occlusion: Path, task: str, image: str,
                scientific_output: Path | None = None) -> tuple[dict, dict[str, str]]:
    """Build a command plan without invoking Docker, training or inference."""
    if task != "closure" and (scientific_output is None or scientific_output.exists()):
        raise ValueError("Scientific runs require an explicit fresh output path that does not exist.")
    private = exam.resolve() / ".runtime"
    private.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix=f"{task}-", dir=private))
    workspace = job / "workspace"
    copied = materialize(exam, workspace, parent / "receipts/source-freeze.json")
    derived = job / "derived"
    (derived / "cache/ipython").mkdir(parents=True)
    environment = os.environ.copy()
    if task == "closure":
        values = {"RUNTIME_HOST_DIR": str(workspace), "CLOSURE_OUTPUT_HOST_DIR": str(derived),
                  "CLOSURE_BUILD_CONTEXT": str(workspace), "CLOSURE_IMAGE": image,
                  "WILDLIFE_ORIGINAL_HOST_DIR": str(parent.resolve()),
                  "OCCLUSION_ORIGINAL_HOST_DIR": str(occlusion.resolve())}
        environment.update(values)
        compose = ["docker", "compose", "-f", str(workspace / "compose.closure.yaml")]
        commands = [compose + ["config", "--quiet"], compose + ["run", "--rm", "verify"]]
    else:
        public = ".env.wildlife-v6.example" if task == "wildlife" else ".env.example"
        values = example_values(workspace / public)
        for key in ("KAGGLE_ROOT", "RSNA_HOST_DIR", "SEN1_HOST_DIR", "CCT_IMAGES_HOST_DIR",
                    "CCT_ANNOTATIONS_HOST_DIR", "PARENT_OUTPUT_HOST_DIR"):
            if key in os.environ:
                values[key] = os.environ[key]
        output = str(scientific_output.resolve())
        values.update(OUTPUT_HOST_DIR=output, OUTPUT_DIR=output, MRI_OUTPUT_DIR=output, FLOOD_OUTPUT_DIR=output,
                      MRI_PYTHON=sys.executable, CCT_BACKGROUND_HOST_DIR=str(scientific_output.resolve() / "inputs/backgrounds"),
                      JUPYTER_TOKEN=secrets.token_urlsafe(32))
        environment.update(values)
        envfile = workspace / (".env.wildlife-v6" if task == "wildlife" else ".env")
        envfile.write_text("\n".join(f"{key}={value}" for key, value in sorted(values.items())) + "\n", encoding="utf-8")
        shell = shutil.which("pwsh") or shutil.which("powershell.exe") or "pwsh"
        command = [shell, "-NoProfile", "-File", str(workspace / f"scripts/run_{task}.ps1")]
        if task == "wildlife":
            command += ["-Variant", "v6", "-EnvironmentFile", str(envfile)]
        commands = [command]
    plan = {"task": task, "workspace": str(workspace), "derived": str(derived), "commands": commands,
            "frozen_sources_verified": len(copied["frozen_sources"]), "scientific_runner_executed": False,
            "fresh_training_reproduction_verified": False}
    return plan, environment


def execute_plan(plan: dict, environment: dict[str, str], tests: bool = False) -> dict:
    """Execute a previously prepared plan and preserve real exit statuses."""
    commands = list(plan["commands"])
    if tests:
        if plan["task"] != "closure":
            raise ValueError("The --tests option belongs to closure checks only.")
        commands.append(commands[-1][:-1] + ["tests"])
    results = []
    for command in commands:
        process = subprocess.run(command, cwd=plan["workspace"], env=environment, check=False)
        results.append({"command": command, "exit_code": process.returncode})
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    return {**plan, "commands_executed": results, "scientific_runner_executed": plan["task"] != "closure"}


def main() -> None:
    """Default to closure verification; scientific tasks require explicit execution."""
    own_root = Path(__file__).resolve().parents[1]
    default_exam = own_root.parent if own_root.name == "compartido" else own_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("closure", "mri", "flood", "wildlife"), nargs="?", default="closure")
    parser.add_argument("--exam-root", type=Path, default=default_exam)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--occlusion", type=Path, default=DEFAULT_OCCLUSION)
    parser.add_argument("--image", default="neural-exam-wildlife:0.2")
    parser.add_argument("--scientific-output", type=Path)
    parser.add_argument("--execute-science", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--tests", action="store_true")
    args = parser.parse_args()
    if args.task != "closure" and args.tests:
        parser.error("--tests is available only for closure")
    plan, environment = prepare_run(args.exam_root, args.parent, args.occlusion, args.task,
                                    args.image, args.scientific_output)
    execute = not args.plan_only and (args.task == "closure" or args.execute_science)
    result = execute_plan(plan, environment, args.tests) if execute else {**plan, "status": "plan_only"}
    path = Path(plan["derived"]) / "command_receipt.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(path), "executed": execute, "task": args.task}))


if __name__ == "__main__":
    main()
