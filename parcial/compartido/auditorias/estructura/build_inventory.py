"""Create a byte-level inventory without moving or deleting exam files."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
PRESERVED = Path("E:/Kaggle/exam-closure-2026-10-08/preserved-before")
ORIGINAL = Path("D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales")
FROZEN = Path("E:/Kaggle/exam-output-wildlife-v6/receipts/source-freeze")
SAMPLE = Path("E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600")
QUESTIONS = {"mri": "pregunta_1_mri", "flood": "pregunta_2_inundaciones", "wildlife": "pregunta_3_fauna"}


def digest(path: Path) -> str:
    """Hash physical bytes, without Git newline normalization."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args: str, root: Path = ROOT) -> str:
    """Read Git metadata without altering any checkout."""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


def write_json(name: str, value: object) -> None:
    """Write derived evidence exclusively beside this script."""
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def destination(path: str) -> tuple[str, str, str]:
    """Classify by role, retaining unique historical and scientific material."""
    relative = path.removeprefix("parcial/")
    parts = relative.split("/")
    group = parts[0]
    if group in {"compartido", "archivo"} or group.startswith("pregunta_"):
        return path, "keep", "Estructura aprobada; conservar los archivos nuevos de A+B+C."
    if group in {"src", "scripts", "configs", "tests"}:
        return "parcial/compartido/" + relative, "move", "Motor, configuración o herramientas comunes; conservar contenido."
    if group == "audits":
        return "parcial/compartido/auditorias/" + "/".join(parts[1:]), "move", "Auditoría común con fecha y contexto originales."
    if group == "reviews":
        return "parcial/compartido/plantillas/reviews/" + "/".join(parts[1:]), "move", "Plantillas de revisión, no decisiones humanas nuevas."
    if group == "notebooks":
        task = next((name for name in QUESTIONS if name in parts[-1]), None)
        if task:
            return f"parcial/{QUESTIONS[task]}/notebooks/{parts[-1]}", "move", "Notebook fuente de la pregunta; entrega canónica proviene de A+B+C."
    if group == "reports":
        if len(parts) > 2 and parts[1] in QUESTIONS:
            return f"parcial/{QUESTIONS[parts[1]]}/fuentes_informe/" + "/".join(parts[2:]), "move", "Fuentes editoriales y decisiones de finalización de la pregunta."
        return "parcial/compartido/plantillas/informes/" + "/".join(parts[1:]), "move", "Preambulo y bibliografía comunes."
    if group == "delivery":
        return "parcial/archivo/entregas_anteriores/" + "/".join(parts[1:]), "move", "Entrega histórica única; no confundir PDF previo con PDF final exportado desde ODT."
    if relative in {"examen-parcial.pdf", "ejemplo.pdf"}:
        return "parcial/archivo/referencias_originales/" + relative, "move", "PDF original inalterado; referencia compartida será exportada del ODT nativo de B."
    if relative in {"examen-parcial.odt", "ejemplo.odt"}:
        return "parcial/compartido/referencias/" + relative, "move", "Referencia del examen; conservar origen y bytes."
    if group == "tmp":
        if "/images/" in relative and relative.endswith(".jpg"):
            return "", "retire_copy", "JPEG extraído regenerable solo tras comprobar la copia original y SHA-256."
        if parts[-1] in {"check_package.py", "check_browser_export.py", "analyze.py"}:
            return "parcial/compartido/scripts/qa_historico/" + "/".join(parts[1:]), "move", "Código auxiliar único; adaptar rutas por separado antes de reutilizarlo."
        return "parcial/archivo/documentos_historicos/tmp/" + "/".join(parts[1:]), "move", "Conservar instantáneas, reseñas, notebook anotado, logs y vistas únicas; no borrar por estar en tmp."
    if "/" not in relative:
        if relative == "README.md":
            return path, "keep", "Índice general del examen; conserva enlaces hacia preguntas y motor compartido."
        if relative.startswith((".env", "compose")) or relative in {"Dockerfile", ".dockerignore", "requirements.txt", "pyproject.toml"}:
            return "parcial/compartido/" + relative, "move", "Entrada del motor común: contexto compartido conserva src/configs y rutas congeladas."
        task = next((name for name in QUESTIONS if name.upper() in relative.upper()), None)
        if task:
            if relative == f"{task.upper()}_STATUS.json":
                return f"parcial/{QUESTIONS[task]}/evidencias/estado.json", "move", "Estado de la pregunta; D2 debe integrar la versión final del equipo."
            folder = "evidencias" if relative.endswith(".json") else "docs"
            return f"parcial/{QUESTIONS[task]}/{folder}/{relative}", "move", "Documentación o evidencia de la pregunta; preservar estados anteriores con su contexto."
        return "parcial/archivo/documentos_historicos/" + relative, "move", "Revisión o preparación histórica común; conservar evidencia."
    raise ValueError(f"Unclassified path: {path}")


def main() -> None:
    """Record every tracked exam file and every preserved local version."""
    tracked = [path for path in git("ls-files", "parcial").splitlines()
               if not path.startswith("parcial/compartido/auditorias/estructura/")]
    freeze = json.loads((FROZEN.parent / "source-freeze.json").read_text(encoding="utf-8"))["sha256"]
    frozen_rows = []
    for name, expected in freeze.items():
        local = ROOT / "parcial" / name
        canonical = FROZEN / name
        physical = digest(local)
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:parcial/{name}"])
        frozen_rows.append({"path": "parcial/" + name, "expected_sha256": expected,
                            "original": str(canonical), "original_sha256": digest(canonical),
                            "worktree_sha256": physical, "git_blob_sha256": hashlib.sha256(blob).hexdigest(),
                            "physical_match": physical == expected,
                            "lf_normalization_matches": hashlib.sha256(local.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected})
    rows = []
    for path in tracked:
        target, action, reason = destination(path)
        source = ROOT / path
        row = {"source": path, "destination": target, "sha256": digest(source), "size": source.stat().st_size,
               "action": action, "reason": reason, "frozen": path.removeprefix("parcial/") in freeze}
        if action == "retire_copy":
            mirror = SAMPLE / "images" / source.name
            row["reconstruction"] = {"method": "copy_bytes", "source": str(mirror),
                                     "sha256": digest(mirror) if mirror.is_file() else None,
                                     "destination": path, "requires_training_or_inference": False}
            if row["reconstruction"]["sha256"] != row["sha256"]:
                raise ValueError(f"Cannot reconstruct image: {path}")
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row["source"])
    duplicates = [{"sha256": sha, "paths": paths, "decision": "Conservar salvo JPEG regenerable con verificación explícita."}
                  for sha, paths in groups.items() if len(paths) > 1]
    preserved_manifest = json.loads((PRESERVED.parent / "preserved-before.json").read_text(encoding="utf-8"))
    preserved_rows = []
    tracked_by_path = {row["source"]: row for row in rows}
    for entry in preserved_manifest["files"]:
        path = entry["path"]
        source = PRESERVED / path
        actual = digest(source)
        if actual != entry["sha256"]:
            raise ValueError(f"Preservation manifest mismatch: {path}")
        same = path in tracked_by_path and tracked_by_path[path]["sha256"] == actual
        target, _, reason = destination(path)
        preserved_rows.append({"source": str(source), "original_path": path, "sha256": actual,
                               "size": source.stat().st_size, "matches_worktree_bytes": same,
                               "proposed_destination": target if path not in tracked_by_path else
                               "parcial/archivo/documentos_historicos/preservado_2026-10-08/" + path.removeprefix("parcial/"),
                               "action": "reference_only_d1", "reason": reason,
                               "d2_policy": "Integrar versión única en destino activo; archivar versión anterior si A+B+C la sustituye. No sobrescribir."})
    ignored = git("ls-files", "--others", "--ignored", "--exclude-standard", "parcial", root=ORIGINAL).splitlines()
    secrets = [path for path in ignored if Path(path).name.startswith(".env") and not path.endswith(".example")]
    base_count = len(git("ls-tree", "-r", "--name-only", "0dc2fd2", "parcial").splitlines())
    summary = {"stage": "D1", "worktree": str(ROOT), "branch": git("branch", "--show-current"),
               "base_main": "0dc2fd20f424941919c38ff213894541e60ab200", "audit_commit_original": "8bb5a39",
               "audit_commit_local": "7bd7bece23638cb990ab7397b08796b11bf8a4cc",
               "inventory_source_commit": git("rev-parse", "HEAD"), "tracked_before_audit": base_count,
               "tracked_inventoried": len(rows), "operations": dict(Counter(row["action"] for row in rows)),
               "jpeg_unique": sum(row["action"] == "retire_copy" for row in rows), "sample_cases": 600,
               "exact_duplicate_groups": len(duplicates), "preserved_versions": len(preserved_rows),
               "frozen_files": len(freeze), "frozen_originals_match": all(row["original_sha256"] == row["expected_sha256"] for row in frozen_rows),
               "frozen_worktree_mismatches": sum(not row["physical_match"] for row in frozen_rows),
               "requested_configuration": {"model": "Según creación: gpt-6.1-sol.",
                                           "effort": "Según creación: high.",
                                           "speed": "Estándar heredada (default); herramienta sin ajuste individual."},
               "effective_configuration": {"model": "gpt-6.1-sol", "effort": "high", "speed": "default"},
               "effective_configuration_verified": True,
               "configuration_evidence": "Chat principal verificó turn_context y velocidad global; comunicado en seguimiento de D1, no inferido por este script.",
               "ignored_secret_paths": secrets,
               "scope_exclusions": ["parcial/compartido/auditorias/estructura/: auditoría D1 conservada in situ; evita hashes autorreferenciales al regenerar."],
               "secrets_content_read_or_published": False, "files_moved_or_deleted": 0}
    write_json("inventario.json", {"summary": summary, "files": rows, "exact_duplicates": duplicates,
                                   "semantic_exceptions": [{"paths": ["parcial/configs/default.toml", "parcial/configs/flood.toml"],
                                                            "decision": "Conservar ambas: fallback sin argumento y configuración explícita de Flood."}],
                                   "preserved_local_versions": preserved_rows, "frozen_sources": frozen_rows})
    write_json("migration_plan.json", {"stage": "D1", "worktree": str(ROOT), "operations": rows,
                                       "integration_commits": [], "apply_authorized": False})
    with (OUTPUT / "mapa.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "destination", "sha256", "size", "action", "reason", "frozen"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json("evidencia_d1.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
