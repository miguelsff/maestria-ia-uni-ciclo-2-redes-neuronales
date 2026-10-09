"""Materialize legacy scientific paths without changing signed source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

QUESTION_TASKS = {"pregunta_1_mri": "mri", "pregunta_2_inundaciones": "flood", "pregunta_3_fauna": "wildlife"}
SOURCE_TREES = ("src", "configs", "scripts", "tests", "reviews")
ROOT_FILES = ("Dockerfile", "pyproject.toml", "requirements.txt")


def digest(path: Path) -> str:
    """Hash the physical bytes, including original line endings."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    """Read UTF-8 evidence without modifying its serialization."""
    return json.loads(path.read_text(encoding="utf-8"))


def source_path(exam: Path, relative: str) -> Path:
    """Resolve a scientific path in the shared or legacy layout."""
    shared = exam / "compartido" / relative
    return shared if shared.is_file() else exam / relative


def verify_frozen(exam: Path, receipt: Path) -> list[dict]:
    """Reject physical-byte drift before any runtime copy is created."""
    results = []
    for name, expected in read_json(receipt)["sha256"].items():
        path = source_path(exam, name)
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Frozen source changed: {name}: {actual} != {expected}")
        results.append({"path": name, "sha256": actual})
    return results


def _files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(directory).parts)
        and path.suffix not in {".pyc", ".pyo"}
    )


def _register(mapping: dict[str, Path], source: Path, relative: str) -> None:
    if relative in mapping and digest(mapping[relative]) != digest(source):
        raise ValueError(f"Conflicting legacy/shared inputs: {relative}")
    mapping[relative] = source


def workspace_inputs(exam: Path) -> dict[str, Path]:
    """Map shared sources and per-question material to unchanged /workspace paths."""
    mapping: dict[str, Path] = {}
    for root in (exam, exam / "compartido"):
        for tree in SOURCE_TREES:
            for source in _files(root / tree):
                _register(mapping, source, f"{tree}/{source.relative_to(root / tree).as_posix()}")
        for source in root.glob("compose*.yaml"):
            _register(mapping, source, source.name)
    for name in ROOT_FILES:
        mapping[name] = source_path(exam, name)
    for root in (exam, exam / "compartido"):
        for source in root.glob(".env*.example"):
            _register(mapping, source, source.name)
    for source in _files(exam / "notebooks"):
        _register(mapping, source, f"notebooks/{source.relative_to(exam / 'notebooks').as_posix()}")
    # A migrated per-question notebook supersedes the corresponding legacy copy.
    for question in QUESTION_TASKS:
        for source in _files(exam / question / "notebooks"):
            mapping[f"notebooks/{source.relative_to(exam / question / 'notebooks').as_posix()}"] = source
    for source in _files(exam / "reports"):
        mapping[f"reports/{source.relative_to(exam / 'reports').as_posix()}"] = source
    for question, task in QUESTION_TASKS.items():
        for source in _files(exam / question / "fuentes_informe"):
            mapping[f"reports/{task}/{source.relative_to(exam / question / 'fuentes_informe').as_posix()}"] = source
    templates = exam / "compartido/plantillas"
    # Migrated groups lose their organizational prefix in the scientific runtime.
    for group, target in (("informes", "reports"), ("reviews", "reviews")):
        for source in _files(templates / group):
            mapping[f"{target}/{source.relative_to(templates / group).as_posix()}"] = source
    # Retain compatibility with earlier flat common-report resources.
    for source in templates.glob("*"):
        if source.is_file() and source.name != "notebook_template.ipynb":
            mapping[f"reports/{source.name}"] = source
    for root in (exam / "compartido", exam):
        template = root / "tmp/occlusion-accepted-delivery/notebook_template.ipynb"
        if template.is_file():
            mapping["tmp/occlusion-accepted-delivery/notebook_template.ipynb"] = template
            break
    template = exam / "compartido/plantillas/notebook_template.ipynb"
    if template.is_file():
        mapping["tmp/occlusion-accepted-delivery/notebook_template.ipynb"] = template
    for question in QUESTION_TASKS:
        for source in _files(exam / question / "entregable"):
            mapping[f"deliveries/{question}/{source.relative_to(exam / question / 'entregable').as_posix()}"] = source
    return mapping


def materialize(exam: Path, destination: Path, receipt: Path) -> dict:
    """Build a new private runtime workspace; never overwrite an existing directory."""
    exam = exam.resolve()
    destination = destination.resolve()
    if not destination.is_relative_to(exam / ".runtime"):
        raise ValueError("Runtime destination must be inside the calling worktree's parcial/.runtime directory.")
    if destination.exists():
        raise ValueError("Runtime destination already exists; choose a fresh directory.")
    frozen = verify_frozen(exam, receipt)
    mapping = workspace_inputs(exam)
    for relative, source in mapping.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(source) != digest(target):
            raise ValueError(f"Runtime copy changed bytes: {relative}")
    verify_frozen(destination, receipt)
    result = {"status": "verified", "exam_root": str(exam), "runtime_root": str(destination),
              "frozen_sources": frozen, "files_sha256": {name: digest(path) for name, path in mapping.items()},
              "trained_or_ran_inference": False}
    (destination / "runtime_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Expose a byte-preserving adapter for the agreed exam directory migration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.exam_root, args.destination, args.freeze_receipt)
    print(json.dumps({"status": result["status"], "runtime_root": result["runtime_root"],
                      "frozen_sources": len(result["frozen_sources"]), "files": len(result["files_sha256"])}))


if __name__ == "__main__":
    main()
