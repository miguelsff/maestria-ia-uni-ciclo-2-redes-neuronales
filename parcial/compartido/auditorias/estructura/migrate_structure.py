"""Validate the migration; apply only a regenerated, integrated D2 plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    """Hash bytes using a bounded stream."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root: Path, name: str) -> Path:
    """Reject traversal, absolute names, links and destinations outside worktree."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not name.startswith("parcial/"):
        raise ValueError(f"Unsafe relative path: {name}")
    target = root / relative
    for component in [target, *target.parents]:
        if component == root:
            break
        if component.is_symlink() or component.is_junction():
            raise ValueError(f"Linked path: {component}")
    resolved = target.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escaped worktree: {target}")
    return resolved


def validate(plan: dict, root: Path) -> dict:
    """Check every source and reconstruction before allowing any mutation."""
    if Path(plan["worktree"]).resolve() != root.resolve():
        raise ValueError("Plan belongs to another worktree; regenerate it here.")
    seen_sources, seen_targets = set(), set()
    counts = {"keep": 0, "move": 0, "retire_copy": 0}
    for row in plan["operations"]:
        source = contained(root, row["source"])
        if source in seen_sources:
            raise ValueError(f"Duplicate source: {source}")
        seen_sources.add(source)
        if not source.is_file() or digest(source) != row["sha256"]:
            raise ValueError(f"Source changed or missing: {source}")
        action = row["action"]
        if action not in counts:
            raise ValueError(f"Unsupported action: {action}")
        counts[action] += 1
        if action == "retire_copy":
            recipe = row.get("reconstruction", {})
            mirror = Path(recipe.get("source", ""))
            if row["frozen"] or recipe.get("method") != "copy_bytes" or not mirror.is_file():
                raise ValueError(f"Missing reconstruction: {source}")
            if digest(mirror) != row["sha256"] or recipe.get("sha256") != row["sha256"]:
                raise ValueError(f"Reconstruction changed: {mirror}")
            continue
        target = contained(root, row["destination"])
        if target in seen_targets:
            raise ValueError(f"Destination collision: {target}")
        seen_targets.add(target)
        if action == "move" and target.exists():
            raise ValueError(f"Destination already exists: {target}")
        if action == "keep" and target != source:
            raise ValueError("Keep must preserve its original location.")
    return {"status": "passed", "files": len(seen_sources), "operations": counts, "mutations": 0}


def apply(plan: dict, root: Path) -> dict:
    """Back up verified files, then perform byte-preserving native operations."""
    if plan.get("stage") != "D2" or plan.get("apply_authorized") is not True:
        raise ValueError("D1 cannot be applied. Wait for integrated A+B+C SHAs and regenerate D2.")
    commits = plan.get("integration_commits", [])
    if len(commits) < 3:
        raise ValueError("Missing A+B+C integration commits.")
    for commit in commits:
        subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "HEAD"], check=True)
    validate(plan, root)
    freeze = json.loads((root / "parcial/compartido/auditorias/estructura/inventario.json").read_text(encoding="utf-8"))["frozen_sources"]
    if len(freeze) != 44:
        raise ValueError("Frozen source inventory must contain exactly 44 entries.")
    for entry in freeze:
        if digest(contained(root, entry["path"])) != entry["expected_sha256"]:
            raise ValueError(f"Frozen physical bytes need restoration before D2: {entry['path']}")
    recovery = root / ".migration-recovery-d2"
    if recovery.exists():
        raise ValueError("Recovery directory exists; inspect its receipt before retrying.")
    recovery.mkdir()
    receipt = {"status": "backing_up", "operations": plan["operations"], "completed": []}
    receipt_path = recovery / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for row in plan["operations"]:
        if row["action"] == "keep":
            continue
        backup = recovery / row["source"]
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained(root, row["source"]), backup)
        if digest(backup) != row["sha256"]:
            raise ValueError(f"Backup mismatch: {backup}")
    receipt["status"] = "applying"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for row in plan["operations"]:
        source = contained(root, row["source"])
        if row["action"] == "keep":
            continue
        if digest(source) != row["sha256"]:
            raise ValueError(f"Source changed after backup: {source}")
        if row["action"] == "move":
            target = contained(root, row["destination"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"Destination appeared after preflight: {target}")
            source.rename(target)
            if digest(target) != row["sha256"]:
                raise ValueError(f"Post-move hash mismatch: {target}")
        else:
            if digest(Path(row["reconstruction"]["source"])) != row["sha256"]:
                raise ValueError(f"Reconstruction changed after preflight: {source}")
            source.unlink()
        receipt["completed"].append(row["source"])
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    receipt["status"] = "completed"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return {"status": "completed", "mutations": len(receipt["completed"]), "recovery": str(recovery)}


def main() -> None:
    """Use this script only from its own registered managed Git worktree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("migration_plan.json"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    if not (root / ".git").is_file():
        raise ValueError("Expected a managed worktree with a .git indirection file.")
    if not args.plan.resolve().is_relative_to(root):
        raise ValueError("Plan must reside inside this worktree.")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    print(json.dumps(apply(plan, root) if args.apply else validate(plan, root), ensure_ascii=False))


if __name__ == "__main__":
    main()
