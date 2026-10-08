"""Bundle the fixed review snapshot and original photos for file:// use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from wildlife_occlusion_sample import FIELDS, HUMAN_FIELDS, digest, read_protocol, review_status, validate_reviews

PROJECT = Path(__file__).resolve().parents[1]
ROOT = "revision_occlusion_600"
TEMPLATE = PROJECT / "reviews/occlusion_offline.html"
CORE = PROJECT / "reviews/occlusion_offline.js"


def json_bytes(value: dict) -> bytes:
    """Serialize traceability metadata in UTF-8."""
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def embedded_json(value: dict) -> str:
    """Keep arbitrary human notes inert inside an HTML script data element."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def package(source: Path, output: Path) -> dict:
    """Read one atomic CSV snapshot; never write to the live review directory."""
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Package output must be outside the live review directory.")
    if output.suffix.lower() != ".zip":
        raise ValueError("Output must be a ZIP file.")
    protocol = read_protocol(source)
    protocol_sha256 = digest(source / "protocol.json")
    snapshot = (source / "occlusion_review_sample_600.csv").read_bytes()
    snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
    reader = csv.DictReader(io.StringIO(snapshot.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != list(FIELDS):
        raise ValueError("Source CSV columns changed.")
    rows = list(reader)
    if any(set(row) != set(FIELDS) for row in rows):
        raise ValueError("Source CSV contains malformed rows.")
    validate_reviews(rows, protocol)
    if any(not isinstance(row.get(field), str) for row in rows for field in HUMAN_FIELDS):
        raise ValueError("Source CSV contains missing human review cells.")
    geometry = {row["annotation_id"]: row["normalized_box"] for row in protocol["selected"]}
    payload = {
        "protocol_sha256": protocol_sha256,
        "snapshot_sha256": snapshot_sha256,
        "rows": [{**row, "normalized_box": geometry[row["annotation_id"]]} for row in rows],
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    core = CORE.read_text(encoding="utf-8")
    if template.count("__SAMPLE_DATA__") != 1 or template.count("__REVIEW_CORE__") != 1:
        raise ValueError("Offline template placeholders are incomplete.")
    if re.search(r"fetch\s*\(|XMLHttpRequest|https?://|/api/|__REVIEW_TOKEN__", template + core):
        raise ValueError("Offline HTML must not require a server or external resource.")
    html = template.replace("__REVIEW_CORE__", core).replace("__SAMPLE_DATA__", embedded_json(payload))
    images = {}
    for row in protocol["selected"]:
        name = row["gallery"]
        if not re.fullmatch(r"images/[a-zA-Z0-9_-]+\.jpg", name):
            raise ValueError("Unsafe or unsupported image path.")
        if name in images and images[name] != row["source_sha256"]:
            raise ValueError("Conflicting original image hashes.")
        images[name] = row["source_sha256"]
    status = review_status(rows, protocol)
    readme = f"""REVISIÓN DE OCLUSIÓN — 600 CASOS, SIN CONEXIÓN

1. Extraiga TODO el ZIP en una carpeta. No abra la página dentro del ZIP.
2. Abra index.html con Chrome, Edge o Firefox. No necesita instalar programas,
   iniciar un servidor ni conectarse a Internet.
3. Ingrese su nombre. Revise el animal del recuadro amarillo y complete el nivel
   de oclusión y una evidencia breve. Haga clic en la foto para ampliarla.
4. Use Descargar CSV al terminar CADA sesión. Conserve el archivo descargado
   y envíelo al responsable. Puede enviar una revisión parcial.
5. Para continuar en otro equipo o navegador, abra index.html y use Cargar CSV.

Si el responsable distribuyó el trabajo, revise solo sus números de caso.
El CSV siempre incluye los 600 casos; deje pendientes los que no le asignaron.
Cada persona trabaja en su propia copia. Los CSV pueden cargarse sucesivamente
para reunir revisiones. Si hay diferencias en un mismo caso, la página no
sobrescribe los datos: conserve ambos CSV y envíelos al responsable.

CRITERIOS
Baja: más del 75 % del animal visible.
Media: entre el 25 % y el 75 % visible, incluidos ambos límites.
Alta: menos del 25 % visible.
Ambigua: no se puede estimar o el borde de la imagen corta al animal.
El tamaño pequeño, la oscuridad y el desenfoque no implican oclusión por sí solos.

GUARDADO
El navegador intenta respaldar el avance localmente. Ese respaldo depende del
navegador y de la carpeta; puede perderse en modo privado o al borrar sus datos.
Descargar CSV es el respaldo portátil. La página no modifica automáticamente
el CSV incluido ni envía información. Al cerrar con cambios sin descargar,
el navegador puede mostrar un recordatorio.

CONTENIDO
600 recuadros, {len(images)} fotos originales, sin modificar su calidad.
La copia inicial conserva el avance existente al empaquetar:
{status['complete']} casos completos, {status['pending']} pendientes.
Conserve index.html junto a la carpeta images. Mantenga las 600 filas y las
13 columnas del CSV descargado; así se conservan las identidades de la muestra.
protocol.json y manifest_package.json documentan la muestra y los archivos.
"""
    files = {
        "index.html": html.encode("utf-8"),
        "LEEME.txt": readme.encode("utf-8-sig"),
        "occlusion_review_sample_600.csv": snapshot,
        "protocol.json": (source / "protocol.json").read_bytes(),
        "protocol.sha256": (source / "protocol.sha256").read_bytes(),
        "sampling_analysis.json": (source / "sampling_analysis.json").read_bytes(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    hashes, image_bytes = {}, 0
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in files.items():
                hashes[name] = hashlib.sha256(data).hexdigest()
                archive.writestr(f"{ROOT}/{name}", data)
            for name, expected in sorted(images.items()):
                data = (source / name).read_bytes()
                actual = hashlib.sha256(data).hexdigest()
                if actual != expected:
                    raise ValueError(f"Selected original image changed: {name}")
                image_bytes += len(data)
                hashes[name] = actual
                archive.writestr(f"{ROOT}/{name}", data, compress_type=zipfile.ZIP_STORED)
            manifest = {
                "schema_version": 1,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "protocol_sha256": protocol_sha256,
                "snapshot_sha256": snapshot_sha256,
                "cases": len(rows),
                "images": len(images),
                "original_image_bytes": image_bytes,
                "complete_reviews_at_snapshot": status["complete"],
                "assigned_levels_at_snapshot": sum(bool(row["level"]) for row in rows),
                "requires_server": False,
                "requires_network": False,
                "source_review_written": False,
                "human_reviews_created_by_packager": False,
                "sha256": hashes,
            }
            archive.writestr(f"{ROOT}/manifest_package.json", json_bytes(manifest))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity verification failed.")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output": str(output),
        "zip_bytes": output.stat().st_size,
        "zip_sha256": digest(output),
        "cases": len(rows),
        "images": len(images),
        "protocol_sha256": protocol_sha256,
        "snapshot_sha256": snapshot_sha256,
        "source_review_changed_during_build": digest(source / "occlusion_review_sample_600.csv") != snapshot_sha256,
    }


def main() -> None:
    """Create the distributable ZIP in a separate authorized output folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
