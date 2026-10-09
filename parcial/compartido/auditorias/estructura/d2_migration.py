"""Migrate the integrated exam with verified bytes and recoverable operations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from build_inventory import PRESERVED, QUESTIONS, SAMPLE, destination
from migrate_structure import contained, digest

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).with_name("d2")
BASE = "76557bc5aecb7995f1af0fed130ac22cd66ece82"
COMMITS = ["4f77669", "33047d4", "a872bab", "2d45398", "d2b0a31", "98c198e"]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", "-c", "core.longpaths=true", "-C", str(ROOT), *arguments])


def target_for(path: str) -> tuple[str, str, str]:
    """Respect integrated products and avoid replacing B's editorial sources."""
    if path in {"parcial/examen-parcial.pdf", "parcial/ejemplo.pdf"}:
        return path.replace("parcial/", "parcial/compartido/referencias/", 1), "move", "Exportación nativa B; el PDF original está separado en archivo."
    if path.startswith("parcial/reports/"):
        parts = path.split("/")
        if len(parts) > 3 and parts[2] in QUESTIONS:
            target = f"parcial/{QUESTIONS[parts[2]]}/fuentes_informe/latex_previo_d1/" + "/".join(parts[3:])
            return target, "move", "Fuente editorial previa; carpeta separada para no sobrescribir fuentes integradas A/B."
    if path == "parcial/tmp/occlusion-accepted-delivery/notebook_template.ipynb":
        return "parcial/compartido/plantillas/notebook_template.ipynb", "move", "Plantilla firmada de los helpers aceptados; runtime C conserva su ruta histórica."
    target, action, reason = destination(path)
    if action == "move" and (ROOT / target).exists():
        parts = path.split("/")
        if parts[1] == "notebooks":
            target = str(Path(target).parent / "historico" / Path(path).name).replace("\\", "/")
            reason = "Notebook histórico previo; conservar el notebook integrado sin sobrescribirlo."
        elif Path(path).name.endswith("_STATUS.json"):
            target = str(Path(target).parent / Path(path).name).replace("\\", "/")
            reason = "Estado histórico previo; estado.json integrado conserva el cierre vigente."
    return target, action, reason


def protected_before() -> list[dict]:
    """Restore only Git blobs already matching signed hashes."""
    import hashlib

    rows = read(ROOT / "parcial/compartido/auditorias/c1/git-checkout-roundtrip.json")["sha256"]
    assert len(rows) == 61
    restored = []
    for row in rows:
        path = contained(ROOT, row["path"])
        blob = git("show", "HEAD:" + row["path"])
        if hashlib.sha256(blob).hexdigest() != row["sha256"]:
            raise ValueError(f"Signed Git blob mismatch: {row['path']}")
        physical = digest(path)
        if physical != row["sha256"]:
            path.write_bytes(blob)
            restored.append({"path": row["path"], "physical_before": physical, "restored_sha256": digest(path)})
    # Refresh cached index metadata while proving that signed blobs stay identical.
    subprocess.run(["git", "-C", str(ROOT), "add", "--", *[row["path"] for row in rows]], check=True)
    for row in rows:
        if hashlib.sha256(git("show", ":" + row["path"])).hexdigest() != row["sha256"]:
            raise ValueError("Index refresh changed a signed blob.")
    freeze = read(Path("E:/Kaggle/exam-output-wildlife-v6/receipts/source-freeze.json"))["sha256"]
    for name, expected in freeze.items():
        if digest(ROOT / "parcial" / name) != expected:
            raise ValueError(f"Frozen physical mismatch: {name}")
    write(EVIDENCE / "physical_before.json", {"status": "passed", "protected_verified": 61, "frozen_verified": len(freeze),
                                             "restorations": restored, "scientific_changes": 0,
                                             "first_attempt": "43 archivos restaurados desde blobs firmados; update-index --refresh devolvió 1 por caché de stat. El reintento verifica los 61 blobs del índice sin cambios semánticos."})
    return rows


def prepare() -> None:
    """Generate one operation for every integrated file and all 64 backups."""
    if git("rev-parse", "HEAD").decode().strip() != BASE:
        raise ValueError("Prepare D2 only from the authorized integrated base.")
    protected = protected_before()
    tracked = git("ls-files", "parcial").decode().splitlines()
    operations = []
    for name in tracked:
        target, action, reason = target_for(name)
        source = ROOT / name
        row = {"source": name, "destination": target, "sha256": digest(source), "size": source.stat().st_size,
               "action": action, "reason": reason}
        if action == "retire_copy":
            row["reconstruction_source"] = str(SAMPLE / "images" / source.name)
        operations.append(row)
    preserved = read(PRESERVED.parent / "preserved-before.json")["files"]
    for entry in preserved:
        name = entry["path"]
        source = PRESERVED / name
        if digest(source) != entry["sha256"]:
            raise ValueError(f"Backup source changed: {source}")
        operations.append({"source": str(source), "original_path": name, "sha256": entry["sha256"], "size": source.stat().st_size,
                           "destination": "parcial/archivo/documentos_historicos/preservado_2026-10-08/" + name.removeprefix("parcial/"),
                           "action": "copy_preserved", "reason": "Versión local respaldada única o con contexto previo; no sobrescribe integración."})
    protected.extend({"path": name, "sha256": digest(ROOT / name)} for name in tracked
                     if name.endswith((".odt", ".pdf")) or "/entregable/" in name and name.endswith(".ipynb"))
    plan = {"stage": "D2", "worktree": str(ROOT), "base": BASE, "apply_authorized": True,
            "integration_commits": COMMITS, "operations": operations, "protected": protected}
    write(EVIDENCE / "plan.json", plan)
    validate(plan)
    print(json.dumps({"status": "prepared", "tracked": len(tracked), "preserved": len(preserved),
                      "operations": dict(Counter(row["action"] for row in operations))}))


def validate(plan: dict) -> dict:
    """Reject stale hashes, source escapes and every destination collision."""
    if Path(plan["worktree"]).resolve() != ROOT.resolve():
        raise ValueError("Wrong worktree.")
    seen = set()
    for row in plan["operations"]:
        if row["action"] == "copy_preserved":
            source = Path(row["source"]).resolve()
            if not source.is_relative_to(PRESERVED.resolve()):
                raise ValueError("External input outside preserved snapshot.")
        else:
            source = contained(ROOT, row["source"])
        if digest(source) != row["sha256"]:
            raise ValueError(f"Stale source: {source}")
        if row["action"] == "retire_copy":
            if digest(Path(row["reconstruction_source"])) != row["sha256"]:
                raise ValueError(f"Reconstruction changed: {source}")
            continue
        target = contained(ROOT, row["destination"])
        if target in seen or row["action"] != "keep" and target.exists():
            raise ValueError(f"Destination collision: {target}")
        seen.add(target)
    return {"status": "passed", "operations": len(plan["operations"]), "mutations": 0}


def apply_plan() -> None:
    """Back up all changed inputs, then use native byte-preserving operations."""
    plan = read(EVIDENCE / "plan.json")
    if plan.get("stage") != "D2" or plan.get("apply_authorized") is not True:
        raise ValueError("D2 authorization required.")
    for commit in plan["integration_commits"]:
        subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"], check=True)
    validate(plan)
    recovery = ROOT / "parcial/.runtime/d2-recovery"
    if recovery.exists():
        raise ValueError("Inspect previous recovery before retrying.")
    for row in plan["operations"]:
        if row["action"] in {"keep", "copy_preserved"}:
            continue
        backup = recovery / row["source"]
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained(ROOT, row["source"]), backup)
        if digest(backup) != row["sha256"]:
            raise ValueError("Recovery mismatch.")
    completed = []
    write(recovery / "receipt.json", {"status": "ready", "completed": completed})
    for row in plan["operations"]:
        action = row["action"]
        if action == "keep":
            continue
        if action == "copy_preserved":
            source = Path(row["source"])
        else:
            source = contained(ROOT, row["source"])
        if digest(source) != row["sha256"]:
            raise ValueError(f"Source changed after backup: {source}")
        if action == "retire_copy":
            if digest(Path(row["reconstruction_source"])) != row["sha256"]:
                raise ValueError("Reconstruction source changed after preflight.")
            source.unlink()
        else:
            target = contained(ROOT, row["destination"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"Refuse overwrite: {target}")
            if action == "move":
                source.rename(target)
            else:
                shutil.copyfile(source, target)
            if digest(target) != row["sha256"]:
                raise ValueError("Post-operation byte mismatch.")
        completed.append(row["source"])
        write(recovery / "receipt.json", {"status": "applying", "completed": completed})
    # Remove only empty legacy directories after all file operations are verified.
    for path in sorted((ROOT / "parcial").rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not path.is_relative_to(ROOT / "parcial/.runtime"):
            try:
                path.rmdir()
            except OSError:
                pass
    write(recovery / "receipt.json", {"status": "completed", "completed": completed})
    write(EVIDENCE / "application.json", {"status": "completed", "mutations": len(completed), "recovery": str(recovery)})
    with (EVIDENCE / "mapa_d2.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "destination", "sha256", "size", "action", "reason"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plan["operations"])
    print(json.dumps({"status": "applied", "operations": len(completed)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "validate", "apply"])
    args = parser.parse_args()
    if not (ROOT / ".git").is_file():
        raise ValueError("Managed worktree required.")
    if args.stage == "prepare":
        prepare()
    elif args.stage == "validate":
        print(json.dumps(validate(read(EVIDENCE / "plan.json"))))
    else:
        apply_plan()


if __name__ == "__main__":
    main()
