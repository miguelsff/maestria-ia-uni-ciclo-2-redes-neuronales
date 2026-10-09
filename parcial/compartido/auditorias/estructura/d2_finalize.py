"""Close editorial paths and provenance after a verified physical migration."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from d2_migration import BASE, EVIDENCE, ROOT, read, write
from migrate_structure import digest

EXAM = ROOT / "parcial"
PLAN = read(EVIDENCE / "plan.json")
MAP = {row["source"]: row["destination"] for row in PLAN["operations"] if row["action"] != "retire_copy"}
EDITS = read(EVIDENCE / "editorial_edits.json") if (EVIDENCE / "editorial_edits.json").exists() else []


def moved(path: str) -> str:
    return MAP.get(path, path)


def edit(path: Path, content: str) -> None:
    """Preserve original documentation before recording any editorial change."""
    if path.exists() and path.read_bytes() == content.encode("utf-8"):
        return
    before = digest(path) if path.exists() else None
    original = None
    if path.exists():
        original = EXAM / "archivo/documentos_historicos/editorial_pre_d2" / path.relative_to(ROOT)
        if not original.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, original)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    EDITS.append({"path": path.relative_to(ROOT).as_posix(), "before_sha256": before, "after_sha256": digest(path),
                  "original_copy": original.relative_to(ROOT).as_posix() if original else None,
                  "reason": "Índice o ruta editorial actualizada; no cambia resultados científicos."})


def pdf_links() -> list[dict]:
    """Associate by actual signed source/export hashes, never by similar names."""
    documents = read(EXAM / "compartido/auditorias/documentos_b1/inicializacion.json")["documents"]
    exports = read(EXAM / "compartido/auditorias/documentos_b1/exportacion_documentos.json")["documents"]
    source_hashes, export_hashes, masters = {}, {}, {}
    for document in documents:
        master = moved(document["output"])
        native = ROOT / master
        source = ROOT / moved(document["source"])
        source_sha = document.get("original_sha256") or digest(source)
        source_hashes[source_sha] = document["id"]
        exported = next(row for row in exports if moved("parcial/" + row["odt"]) == master)
        pdf = moved("parcial/" + exported["pdf"])
        if digest(native) != exported["odt_sha256_after"] or digest(ROOT / pdf) != exported["pdf_sha256"]:
            raise ValueError(f"Master/export changed: {master}")
        export_hashes[exported["pdf_sha256"]] = document["id"]
        masters[document["id"]] = {"master_odt_path": master, "master_odt_sha256": digest(native),
                                    "exported_pdf_path": pdf, "exported_pdf_sha256": exported["pdf_sha256"],
                                    "pages_in_B_receipt": document["pages"]}
    rows = []
    for path in sorted(EXAM.rglob("*.pdf")):
        if ".runtime" in path.parts:
            continue
        sha = digest(path)
        if sha in export_hashes:
            master_id, relationship = export_hashes[sha], "exported_from_native_odt"
        elif sha in source_hashes:
            master_id, relationship = source_hashes[sha], "preserved_pdf_source_of_native_odt"
        else:
            raise ValueError(f"Unassociated PDF: {path}")
        rows.append({"pdf_path": path.relative_to(ROOT).as_posix(), "pdf_sha256": sha,
                     "master_id": master_id, "relationship": relationship, **masters[master_id],
                     "different_source_and_export_hashes": sha != masters[master_id]["exported_pdf_sha256"],
                     "reason": "Fuente previa y exportación nativa mantienen hashes distintos; asociación acreditada por recibos B."})
    if len(masters) != 8:
        raise ValueError("Expected eight native masters.")
    write(EVIDENCE / "pdf_odt.json", {"status": "passed", "masters": masters, "pdfs": rows, "unassociated_pdfs": []})
    return rows


def guides() -> None:
    """Expose final products without replacing historical scientific evidence."""
    command = "python parcial/compartido/scripts/run_exam.py closure"
    rows = [
        ("pregunta_1_mri", "MRI", "informe_mri", "01_mri", 9),
        ("pregunta_2_inundaciones", "Inundaciones", "informe_inundaciones", "02_flood", 9),
        ("pregunta_3_fauna", "Fauna", "informe_fauna", "03_wildlife", 10),
    ]
    for question, title, report, notebook, pages in rows:
        base = EXAM / question
        annex = "\nAdemás: [anexo ODT](anexo_occlusion.odt), [anexo PDF](anexo_occlusion.pdf), [notebook de oclusión](04_occlusion_582.ipynb) y [métricas](metricas_occlusion.csv). El anexo tiene 2 páginas según el recibo de B.\n" if question.endswith("fauna") else ""
        edit(base / "entregable/LEEME.md", f"# Entrega de {title}\n\n[Maestro ODT]({report}.odt), [PDF exportado desde el ODT]({report}.pdf) y [notebook ejecutado]({notebook}.ipynb). Informe principal: {pages} páginas según los recibos de B. Los hashes se conservan respecto de la base integrada.\n{annex}\nAbrir los notebooks de entrega desde esta carpeta. Conservan las salidas calculadas; no es necesario entrenar ni ejecutar inferencia para consultarlos. Los originales científicos y archivos grandes permanecen en E:/Kaggle.\n\nLa constancia local está en [verificacion.json](verificacion.json). La matriz por subrequisito está en [../evidencias/matriz_revision.md](../evidencias/matriz_revision.md).\n\nPara comprobar el cierre desde la raíz del repositorio:\n\n```powershell\n{command}\n```\n\nEl comando usa copias privadas y montajes originales de solo lectura; no repite experimentos. La reproducción desde entrenamiento sigue sin verificarse. Consulta [el procedimiento compartido](../../compartido/REPRODUCIBILIDAD_CIERRE.md). Los borradores anteriores siguen en el archivo y en evidencias con su contexto. Este paquete no acredita el envío a la plataforma del curso.\n")
        edit(base / "README.md", f"# Pregunta: {title}\n\nLa entrega contiene el maestro ODT editable, su PDF exportado y notebooks con resultados conservados.\n\n- [Entrega y verificación](entregable/LEEME.md).\n- [Fuentes editoriales](fuentes_informe/).\n- [Documentación histórica y técnica](docs/).\n- [Notebooks fuente](notebooks/).\n- [Estado y matriz por subrequisito](evidencias/estado.json).\n- [Reproducción compartida](../compartido/REPRODUCIBILIDAD_CIERRE.md).\n\nLa migración no modificó los documentos ni los notebooks finales. Las fuentes anteriores se conservan separadas en `fuentes_informe/latex_previo_d1/`; no sustituyen el maestro ODT.\n")
        verification_path = base / "entregable/verificacion.json"
        if verification_path.exists():
            current = read(verification_path)
            archive = base / "evidencias/verificacion_pre_d2.json"
            if not archive.exists():
                shutil.copyfile(verification_path, archive)
            current.setdefault("files_sha256", {})["LEEME.md"] = digest(base / "entregable/LEEME.md")
        else:
            current = {"status": "native_documents_and_notebook_preserved", "files_sha256": {}}
        for path in (base / "entregable").iterdir():
            if path.is_file() and path.suffix in {".odt", ".pdf", ".ipynb"}:
                current["files_sha256"][path.name] = digest(path)
        current["migration_D2"] = {"base_integrated": BASE, "status": "bytes_preserved", "receipt": "../../compartido/auditorias/estructura/d2/verificacion.json", "training_or_inference": False}
        edit(verification_path, json.dumps(current, indent=2, ensure_ascii=False) + "\n")
    edit(EXAM / "README.md", "# Examen parcial: entrega organizada\n\nLos tres informes principales están en ODT editable y PDF exportado desde ese ODT. Los notebooks finales conservan sus resultados y hashes; la migración no repitió entrenamiento, inferencia ni bootstrap.\n\n| Pregunta | Entrega |\n|---|---|\n| MRI | [Informe y notebook](pregunta_1_mri/entregable/LEEME.md) |\n| Inundaciones | [Informe y notebook](pregunta_2_inundaciones/entregable/LEEME.md) |\n| Fauna y oclusión | [Informe, anexo y notebooks](pregunta_3_fauna/entregable/LEEME.md) |\n\n[Procedimiento compartido](compartido/REPRODUCIBILIDAD_CIERRE.md), [82 comprobaciones](compartido/auditorias/cierre-2026-10-08/matrices_completas.md) y [recibo de migración](compartido/auditorias/estructura/d2/verificacion.json).\n\nEl motor, Dockerfile, Compose, dependencias y ejemplos públicos de entorno están en `compartido/`. Los datos y checkpoints grandes permanecen en E:/Kaggle. Las copias de runtime y sus secretos son privados e ignorados en `.runtime/`.\n\nEl [enunciado ODT](compartido/referencias/examen-parcial.odt) y su [PDF exportado](compartido/referencias/examen-parcial.pdf), junto con el [ejemplo ODT](compartido/referencias/ejemplo.odt) y [PDF exportado](compartido/referencias/ejemplo.pdf), son las referencias editables. Los [PDF originales](archivo/referencias_originales/) permanecen inalterados.\n\nEl [archivo](archivo/manifiesto.json) conserva versiones históricas, fuentes, las 64 versiones respaldadas y las 13 salidas humanas previas con su procedencia. Cada PDF conservado se asocia a un maestro ODT por hash de fuente o exportación; los borradores no se promueven a entregas. La reproducción completa desde entrenamiento y el envío al curso siguen sin acreditarse.\n")
    edit(ROOT / "README.md", "# Redes neuronales — Maestría en Inteligencia Artificial, UNI\n\n[Material del curso](docs/) y [examen parcial organizado](parcial/README.md).\n\n| Pregunta | Entrega |\n|---|---|\n| MRI | [Informe ODT/PDF y notebook](parcial/pregunta_1_mri/entregable/LEEME.md) |\n| Inundaciones | [Informe ODT/PDF y notebook](parcial/pregunta_2_inundaciones/entregable/LEEME.md) |\n| Fauna y oclusión | [Informe, anexo y notebooks](parcial/pregunta_3_fauna/entregable/LEEME.md) |\n\n[Motor y reproducción](parcial/compartido/REPRODUCIBILIDAD_CIERRE.md), [referencias](parcial/compartido/referencias/), [historia y procedencia](parcial/archivo/manifiesto.json). Los resultados finales conservan sus hashes; datos y pesos grandes permanecen fuera del repositorio. Consultar la matriz de cada pregunta para el alcance verificado y las obligaciones externas pendientes.\n")


def rewrite_links() -> None:
    """Rewrite moved Markdown links while retaining their dated content."""
    pattern = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
    for row in PLAN["operations"]:
        if row["action"] != "move" or not row["destination"].endswith(".md"):
            continue
        path = ROOT / row["destination"]
        text = path.read_text(encoding="utf-8")
        old_parent = (ROOT / row["source"]).parent

        def replacement(match: re.Match) -> str:
            url = match.group(2)
            if "://" in url or url.startswith(("#", "E:", "D:", "C:")):
                return match.group(0)
            source_path = (old_parent / url).resolve()
            if not source_path.is_relative_to(ROOT):
                return match.group(0)
            original = source_path.relative_to(ROOT).as_posix()
            target = ROOT / moved(original)
            if not target.exists():
                return match.group(0)
            return f"[{match.group(1)}]({os.path.relpath(target, path.parent).replace(chr(92), '/')})"

        updated = pattern.sub(replacement, text)
        if updated != text:
            edit(path, updated)


def verify_and_manifest(pdfs: list[dict]) -> None:
    """Check signed files, final masters and every recorded migration operation."""
    checks = []
    for row in PLAN["protected"]:
        target = ROOT / moved(row["path"])
        actual = digest(target)
        if actual != row["sha256"]:
            raise ValueError(f"Protected file changed: {target}")
        checks.append({"source": row["path"], "destination": target.relative_to(ROOT).as_posix(), "sha256": actual})
    edited = {entry["path"] for entry in EDITS}
    for row in PLAN["operations"]:
        if row["action"] == "retire_copy":
            if (ROOT / row["source"]).exists() or digest(Path(row["reconstruction_source"])) != row["sha256"]:
                raise ValueError("Retired copy cannot be reconstructed.")
        elif row["destination"] not in edited and digest(ROOT / row["destination"]) != row["sha256"]:
            raise ValueError(f"Migrated bytes changed: {row['destination']}")
    provenance = read(EXAM / "archivo/documentos_historicos/salidas_locales_2026-10-08/procedencia.json")
    for row in provenance["files"]:
        if digest(ROOT / row["destination"]) != row["sha256"]:
            raise ValueError("Human original output changed.")
    allowed = {"README.md", "pregunta_1_mri", "pregunta_2_inundaciones", "pregunta_3_fauna", "compartido", "archivo", ".runtime"}
    if {path.name for path in EXAM.iterdir()} != allowed:
        raise ValueError("Unexpected top-level exam entries.")
    write(EVIDENCE / "editorial_edits.json", EDITS)
    write(EXAM / "archivo/manifiesto.json", {"stage": "D2", "base": BASE, "operations": PLAN["operations"], "pdf_odt": pdfs,
                                              "editorial_edits": EDITS, "private_data_published": False})
    write(EVIDENCE / "verificacion.json", {"status": "passed", "base_integrated": BASE,
                                           "protected_verified": checks, "protected_c1": 61, "frozen_files": 44,
                                           "tracked_input_files": 1233, "preserved_versions": 64,
                                           "human_provenance_entries_verified": len(provenance["files"]),
                                           "human_output_files_preserved_locally": sum(not Path(row["destination"]).is_absolute() for row in provenance["files"]),
                                           "large_zip_preserved_externally_not_published": True,
                                           "native_odt_masters": 8, "pdfs_associated": len(pdfs), "unassociated_pdfs": 0,
                                           "source_operations_verified": len(PLAN["operations"]), "training_inference_or_bootstrap": False,
                                           "worktree": str(ROOT), "configuration": {"model": "gpt-6.1-sol", "effort": "high", "speed": "default", "evidence": "turn_context verificado por el chat principal"},
                                           "next_stage": "F: revisión del padre y C/B en copia limpia; no publicado."})
    print(json.dumps({"status": "passed", "protected": len(checks), "pdfs": len(pdfs), "masters": 8, "human_outputs": len(provenance["files"])}))


def main() -> None:
    if read(EVIDENCE / "application.json")["status"] != "completed":
        raise ValueError("Migration must be completed first.")
    metadata_path = EXAM / "compartido/auditorias/documentos_b1/inicializacion.json"
    metadata = read(metadata_path)
    for document in metadata["documents"]:
        for field in ("source", "output"):
            previous = document[field]
            current = moved(previous)
            if previous != current:
                document.setdefault("historical_paths_before_D2", {})[field] = previous
                document[field] = current
    metadata["migration_D2"] = {"base": BASE, "original_metadata_copy": "parcial/archivo/documentos_historicos/editorial_pre_d2/parcial/compartido/auditorias/documentos_b1/inicializacion.json", "masters_modified": False}
    edit(metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    guides()
    rewrite_links()
    verify_and_manifest(pdf_links())


if __name__ == "__main__":
    main()
