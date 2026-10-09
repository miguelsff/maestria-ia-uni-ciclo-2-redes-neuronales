"""Build a portable, executed notebook and PDF annex from completed accepted-cohort results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

from evaluate_occlusion_accepted import inputs
from wildlife_occlusion_sample import digest, read_json, write_json

LABELS = {"low": "Baja", "medium": "Media", "high": "Alta", "ambiguous": "Ambigua"}
PROJECT = Path(__file__).resolve().parents[1]


def completed_result(parent: Path, output: Path) -> tuple[dict, Path]:
    """Reject incomplete results or a result from changed human/scientific inputs."""
    signature = inputs(parent, output)[4]
    latest = read_json(output / "accepted_evaluation_latest.json")
    path = output / "accepted_evaluation" / signature[:16] / "results.json"
    result = read_json(path)
    if latest["signature"] != signature or result["signature"] != signature:
        raise ValueError("Completed evaluation does not match the current accepted review.")
    if result["trained_or_ran_inference"] or result["review_status"]["accepted"] != 582:
        raise ValueError("Unexpected analysis scope.")
    for level in LABELS.keys() - {"ambiguous"}:
        group = result["sample_detection"][level]
        if group["signature"] != signature or group["uncertainty"]["replicates"] != 200:
            raise ValueError("A level lacks its completed matching sequence intervals.")
    return result, path


def cell(kind: str, source: str, identifier: str) -> dict:
    """Create a stable notebook cell using the scaffold's schema."""
    value = {"cell_type": kind, "id": identifier, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def prepare(parent: Path, output: Path, delivery: Path, template: Path) -> None:
    """Copy completed evidence into an independently portable supplement."""
    result, results_path = completed_result(parent, output)
    data = delivery / "data"
    data.mkdir(parents=True, exist_ok=True)
    source_files = {
        "results.json": results_path,
        "review_acceptance.json": output / "review_acceptance.json",
        "protocol.json": output / "protocol.json",
        "occlusion_review_sample_600.csv": output / "occlusion_review_sample_600.csv",
        "occlusion_review_accepted_582.csv": output / "occlusion_review_accepted_582.csv",
    }
    for name, source in source_files.items():
        shutil.copyfile(source, data / name)
    write_json(
        data / "integrity.json",
        {
            "sha256": {name: digest(data / name) for name in source_files},
            "signature": result["signature"],
            "parent_files_and_frozen_sources_verified": True,
            "source_results_sha256": digest(results_path),
        },
    )
    notebook = read_json(template)
    notebook["cells"] = [
        cell(
            "markdown",
            """# Fauna V6: oclusión en 582 casos aceptados

Este complemento verifica la revisión humana consolidada y presenta métricas ya calculadas sobre las predicciones congeladas de V6. No entrena ni ejecuta inferencia.

**Criterios:** conservar exactamente los 582 casos aceptados, excluir los 18 sin nivel, usar la revisión de Miguel en el caso 121, no completar notas ausentes y comprobar los intervalos de 200 réplicas por nivel. Ejecutar desde esta carpeta, junto con `data/`.
""",
            "accepted-objective",
        ),
        cell(
            "code",
            """from pathlib import Path
import csv
import hashlib
import json
from collections import Counter

data = Path.cwd() / "data"
integrity = json.loads((data / "integrity.json").read_text(encoding="utf-8"))
for name, expected in integrity["sha256"].items():
    assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
results = json.loads((data / "results.json").read_text(encoding="utf-8"))
acceptance = json.loads((data / "review_acceptance.json").read_text(encoding="utf-8"))
protocol = json.loads((data / "protocol.json").read_text(encoding="utf-8"))
with (data / "occlusion_review_sample_600.csv").open(encoding="utf-8-sig", newline="") as stream:
    full = list(csv.DictReader(stream))
with (data / "occlusion_review_accepted_582.csv").open(encoding="utf-8-sig", newline="") as stream:
    accepted = list(csv.DictReader(stream))
assert results["signature"] == integrity["signature"]
assert not results["trained_or_ran_inference"]
print("Integridad de archivos verificada; análisis del modelo fijo V6.")
""",
            "verify-evidence",
        ),
        cell(
            "markdown",
            """## Decisiones humanas y soporte

Los 18 casos sin nivel se excluyen por instrucción del usuario. Las notas son opcionales bajo esa misma instrucción; no se rellenan. Los ambiguos permanecen en el resumen descriptivo y no se asignan a baja, media o alta.
""",
            "review-decisions",
        ),
        cell(
            "code",
            """excluded = set(acceptance["excluded_annotation_ids"])
assert len(full) == 600 and len(accepted) == 582 and len(excluded) == 18
assert {row["annotation_id"] for row in full if not row["level"]} == excluded
assert accepted == [row for row in full if row["annotation_id"] not in excluded]
assert acceptance["notes_required"] is False and acceptance["human_notes_fabricated"] is False
case_121 = full[120]
assert case_121["annotation_id"] == protocol["selected"][120]["annotation_id"] == "23098"
assert (case_121["level"], case_121["reviewer"], case_121["visual_evidence"]) == ("medium", "Miguel", "")
counts = Counter(row["level"] for row in accepted)
assert counts == {"low": 340, "medium": 69, "high": 48, "ambiguous": 125}
missing_notes = sum(not row["visual_evidence"].strip() for row in accepted)
assert missing_notes == results["review_status"]["accepted_cases_without_notes"] == 398
labels = {"low": "Baja", "medium": "Media", "high": "Alta", "ambiguous": "Ambigua"}
print(f"Aceptados: {len(accepted)}; excluidos: {len(excluded)}; sin nota: {missing_notes}")
for level, label in labels.items():
    print(f"{label}: {counts[level]} ({counts[level] / len(accepted):.1%})")
print("Caso 121: media, Miguel; nota vacía preservada.")
""",
            "verify-human-cohort",
        ),
        cell(
            "markdown",
            """## Métricas e intervalos condicionales

Se conservan todos los fotogramas de `trans_test`, incluidos fondos sin animales, para que sus falsos positivos cuenten. Los objetos sin revisión, excluidos, ambiguos o de otro nivel se ignoran mediante las reglas COCO congeladas. AP no usa los pesos de muestreo. Los intervalos del 95 % remuestrean secuencias completas, con 200 réplicas y semilla 42; mantienen fijo el modelo.
""",
            "metric-method",
        ),
        cell(
            "code",
            """for level in ("low", "medium", "high"):
    group = results["sample_detection"][level]
    metrics, uncertainty = group["metrics"], group["uncertainty"]
    assert group["signature"] == results["signature"]
    assert metrics["eligible_objects"] == counts[level] >= 30
    assert uncertainty["replicates"] == 200 and uncertainty["seed"] == 42
    assert uncertainty["confidence_level"] == 0.95
    print(f"{labels[level]}: {metrics['eligible_objects']} objetos; {uncertainty['clusters']} secuencias")
    for key in ("map50", "map50_95"):
        bounds = uncertainty["intervals"][key]
        assert metrics[key] is not None and bounds["lower"] is not None and bounds["upper"] is not None
        assert bounds["lower"] <= bounds["upper"] and 0 < bounds["defined_replicates"] <= 200
        print(f"  {key}: {metrics[key]:.4f}; IC95% [{bounds['lower']:.4f}, {bounds['upper']:.4f}]; réplicas definidas {bounds['defined_replicates']}/200")
""",
            "actual-metrics",
        ),
        cell(
            "markdown",
            """## Alcance del resumen descriptivo

Las proporciones observadas se refieren a los 582 casos aceptados. Se conservan los pesos originales para describir su distribución, pero las 18 exclusiones son falta de respuesta humana y no constituyen una nueva muestra aleatoria. No se reutiliza la garantía de planificación de 600 casos ni se afirma representatividad poblacional después de las exclusiones.
""",
            "descriptive-scope",
        ),
        cell(
            "code",
            """descriptive = results["descriptive_occlusion"]
assert descriptive["confidence_intervals_for_population_proportions"] is None
assert descriptive["original_95pct_plus_minus_5pp_planning_guarantee_claimed"] is False
for level, label in labels.items():
    record = descriptive["levels"][level]
    print(f"{label}: observada {record['observed_case_proportion']:.2%}; razón de pesos entre aceptados {record['original_weight_ratio_among_accepted_cases']:.2%}")
print(f"Fracción del peso original excluido: {descriptive['excluded_design_weight_fraction']:.2%}")
print("No se estimaron intervalos poblacionales de proporciones.")
""",
            "descriptive-results",
        ),
        cell(
            "markdown",
            """## Limitaciones y entrega

Los intervalos de detección son condicionales al modelo y a las secuencias observadas. No miden desacuerdos entre revisores, variación entre entrenamientos ni rendimiento en cámaras futuras. Las 398 notas ausentes y el desacuerdo original del caso 121 quedan documentados; la decisión del usuario permite continuar sin inventar evidencia. El PDF es un anexo de trabajo para revisión editorial, no una aprobación final del informe ni del docente.
""",
            "limitations",
        ),
    ]
    write_json(delivery / "04_occlusion_582.ipynb", notebook)
    write_summary(delivery / "occlusion_metrics_582.csv", result)


def write_summary(path: Path, result: dict) -> None:
    """Export exact numeric values rather than rounded PDF presentation values."""
    columns = [
        "level",
        "accepted_objects",
        "map50",
        "map50_lower",
        "map50_upper",
        "map50_95",
        "map50_95_lower",
        "map50_95_upper",
        "replicates",
        "seed",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for level in ("low", "medium", "high"):
            group = result["sample_detection"][level]
            uncertainty = group["uncertainty"]
            row = {
                "level": level,
                "accepted_objects": group["metrics"]["eligible_objects"],
                "replicates": uncertainty["replicates"],
                "seed": uncertainty["seed"],
            }
            for metric in ("map50", "map50_95"):
                row[metric] = group["metrics"][metric]
                row[f"{metric}_lower"] = uncertainty["intervals"][metric]["lower"]
                row[f"{metric}_upper"] = uncertainty["intervals"][metric]["upper"]
            writer.writerow(row)


def execute_notebook(delivery: Path) -> None:
    """Execute every cell in a network-isolated container, stopping on any error."""
    command = (
        "from pathlib import Path; import nbformat; from nbclient import NotebookClient; "
        "p=Path('/delivery/04_occlusion_582.ipynb'); n=nbformat.read(p,as_version=4); "
        "NotebookClient(n,timeout=120,kernel_name='python3',allow_errors=False,resources={'metadata':{'path':'/delivery'}}).execute(); "
        "nbformat.write(n,p); print('Notebook executed without errors')"
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "1g",
            "--entrypoint",
            "python",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--mount",
            f"type=bind,source={delivery.resolve()},target=/delivery",
            "neural-exam-wildlife:0.2",
            "-c",
            command,
        ],
        check=True,
    )


def build_pdf(delivery: Path, result: dict) -> None:
    """Create a compact two-page annex without altering the frozen ten-page draft."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy, teal = colors.HexColor("#17324D"), colors.HexColor("#147D83")
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle("AnnexTitle", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=navy, spaceAfter=12)
    )
    styles.add(
        ParagraphStyle(
            "AnnexHeading",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=teal,
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle("AnnexBody", fontName="Helvetica", fontSize=10.5, leading=15, textColor=navy, spaceAfter=9)
    )
    styles.add(ParagraphStyle("AnnexSmall", parent=styles["AnnexBody"], fontSize=9, leading=12))
    styles.add(ParagraphStyle("AnnexTable", parent=styles["AnnexSmall"], alignment=TA_CENTER, spaceAfter=0))
    story = []

    def para(text: str, style: str = "AnnexBody") -> Paragraph:
        return Paragraph(text, styles[style])

    def table(rows: list, widths: list) -> Table:
        item = Table(
            [[para(str(value), "AnnexTable") for value in row] for row in rows], colWidths=widths, repeatRows=1
        )
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2F0F1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, teal),
                ]
            )
        )
        return item

    status = result["review_status"]
    story.extend(
        [
            para("Fauna V6<br/>Revisión de oclusión", "AnnexTitle"),
            para("Anexo de trabajo | 582 casos aceptados | Revisión editorial pendiente", "AnnexSmall"),
            para("Decisiones incorporadas", "AnnexHeading"),
            para(
                f"De los 600 casos originales se aceptaron <b>{status['accepted']}</b> y se excluyeron <b>{status['excluded']}</b> por no tener nivel. En el caso <b>121</b> se conserva la calificación <b>media de Miguel</b>, con la nota vacía tal como fue entregada."
            ),
            para(
                f"Las notas no se exigen por instrucción expresa del usuario: <b>{status['accepted_cases_without_notes']}</b> casos aceptados no tienen nota y 184 sí la tienen. No se agregó evidencia escrita ni se reinterpretaron las revisiones."
            ),
        ]
    )
    rows = [["Nivel", "Casos", "% de aceptados"]]
    for level, label in LABELS.items():
        rows.append([label, status["counts"][level], f"{status['counts'][level] / status['accepted']:.1%}"])
    story.append(table(rows, [170, 130, 190]))
    story.extend(
        [
            para("Reglas y exclusiones", "AnnexHeading"),
            para(
                "Baja: más del 75 % del cuerpo visible; media: entre 25 % y 75 %; alta: menos del 25 %. La ambigüedad indica que la estimación no es defendible. Los 125 ambiguos cuentan en este resumen, pero no se asignan a los tres niveles evaluados."
            ),
            para(
                "Casos excluidos: 253, 302, 385-393, 409-414 y 439. Se preservan sus filas e identificadores en el CSV de 600 casos y se omiten del CSV de 582 aceptados.",
                "AnnexSmall",
            ),
            para(
                "Los CSV originales de los cinco revisores y las dos versiones del caso 121 se conservaron. La selección de Miguel es una decisión del usuario y no representa aprobación del docente.",
                "AnnexSmall",
            ),
            PageBreak(),
            para("Resultados del detector", "AnnexTitle"),
            para("AP sobre la cohorte aceptada, con el modelo y las predicciones de V6 congelados.", "AnnexSmall"),
        ]
    )
    rows = [["Oclusión / soporte", "AP50<br/>(IC 95 %)", "AP50-95<br/>(IC 95 %)"]]
    for level in ("low", "medium", "high"):
        group = result["sample_detection"][level]
        values = [f"{LABELS[level]} / {group['metrics']['eligible_objects']}"]
        for metric in ("map50", "map50_95"):
            bounds = group["uncertainty"]["intervals"][metric]
            values.append(f"<b>{group['metrics'][metric]:.3f}</b><br/>[{bounds['lower']:.3f}, {bounds['upper']:.3f}]")
        rows.append(values)
    story.append(table(rows, [150, 170, 170]))
    story.extend(
        [
            Spacer(1, 8),
            para("Método e interpretación", "AnnexHeading"),
            para(
                "Se conservaron los 18 647 fotogramas de trans_test, incluidos los fondos: sus falsos positivos siguen contando. Los objetos sin revisión, excluidos, ambiguos o de otro nivel se ignoran con las reglas COCO congeladas. AP se calcula sin pesos de muestreo."
            ),
            para(
                "Los intervalos son percentiles del bootstrap por secuencias completas, con 200 réplicas por nivel, semilla 42 y confianza del 95 %. El modelo permanece fijo. No miden incertidumbre de las anotaciones, variación entre entrenamientos ni rendimiento en cámaras futuras."
            ),
        ]
    )
    defined = []
    for level in ("low", "medium", "high"):
        group = result["sample_detection"][level]
        counts = [group["uncertainty"]["intervals"][key]["defined_replicates"] for key in ("map50", "map50_95")]
        defined.append(f"{LABELS[level].lower()}: {counts[0]}/200 y {counts[1]}/200")
    story.extend(
        [
            para("Réplicas definidas para AP50 y AP50-95, respectivamente: " + "; ".join(defined) + ".", "AnnexSmall"),
            para("Alcance y pendiente editorial", "AnnexHeading"),
            para(
                "Las 18 exclusiones son falta de respuesta humana. Las proporciones se describen entre los 582 aceptados; no se reutilizan intervalos poblacionales del diseño completo ni se afirma la garantía original de 95 % y ±5 puntos porcentuales."
            ),
            para(
                "Este anexo y el notebook ejecutado complementan el borrador de V6. Queda pendiente integrar y revisar la redacción del informe final dentro del límite de páginas. No se repitió entrenamiento ni inferencia."
            ),
            para(
                "Trazabilidad: firma "
                + escape(result["signature"][:16])
                + "; decisiones en review_acceptance.json; métricas completas en data/results.json.",
                "AnnexSmall",
            ),
        ]
    )

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(teal)
        canvas.line(52, 43, A4[0] - 52, 43)
        canvas.setFillColor(navy)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(52, 29, "Fauna V6 | Anexo de oclusión | Borrador editorial")
        canvas.drawRightString(A4[0] - 52, 29, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(delivery / "occlusion_582.pdf"),
        pagesize=A4,
        rightMargin=52,
        leftMargin=52,
        topMargin=48,
        bottomMargin=61,
        title="Fauna V6 - Oclusión en 582 casos aceptados",
        author="Grupo de investigación",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def finalize(parent: Path, output: Path, delivery: Path) -> None:
    """Verify genuine notebook execution and publish a provenance-bound PDF."""
    result, results_path = completed_result(parent, output)
    integrity = read_json(delivery / "data/integrity.json")
    if integrity["source_results_sha256"] != digest(results_path):
        raise ValueError("Source metrics changed after preparing the notebook.")
    for name, expected in integrity["sha256"].items():
        if digest(delivery / "data" / name) != expected:
            raise ValueError("Portable evidence changed during notebook execution.")
    notebook = read_json(delivery / "04_occlusion_582.ipynb")
    code = [value for value in notebook["cells"] if value["cell_type"] == "code"]
    if len(code) != 4 or any(value["execution_count"] != index for index, value in enumerate(code, 1)):
        raise ValueError("Notebook has not executed completely in order.")
    if any(value["output_type"] == "error" for block in code for value in block["outputs"]):
        raise ValueError("Notebook execution contains errors.")
    build_pdf(delivery, result)
    (delivery / "LEEME.txt").write_text(
        "Complemento de oclusión V6\n\n"
        "occlusion_582.pdf: anexo de trabajo para revisión editorial.\n"
        "04_occlusion_582.ipynb: notebook ejecutado; abrir desde esta carpeta conservando data/.\n"
        "occlusion_metrics_582.csv: métricas con precisión completa.\n"
        "data/: decisiones humanas, revisiones y resultados verificables.\n\n"
        "582 casos aceptados, 18 sin nivel excluidos, caso 121 según Miguel.\n"
        "Las 398 notas ausentes se conservaron vacías. No se repitió entrenamiento ni inferencia.\n"
        "El anexo no constituye aprobación final del informe ni del docente.\n",
        encoding="utf-8",
    )
    completed_result(parent, output)
    write_json(
        delivery / "delivery_manifest.json",
        {
            "status": "generated_from_completed_results_pending_visual_and_editorial_review",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "signature": result["signature"],
            "results_sha256": digest(results_path),
            "builder_sha256": digest(Path(__file__)),
            "notebook_executed_code_cells": len(code),
            "notebook_errors": 0,
            "editorial_reviewed": False,
            "files_sha256": {
                str(path.relative_to(delivery)): digest(path)
                for path in sorted(delivery.rglob("*"))
                if path.is_file() and path.name != "delivery_manifest.json"
            },
        },
    )


def main() -> None:
    """Produce only the separate, already-authorized post-review supplement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "finalize", "all", "wait"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument(
        "--template", type=Path, default=PROJECT / "tmp/occlusion-accepted-delivery/notebook_template.ipynb"
    )
    args = parser.parse_args()
    if args.action == "wait":
        signature = inputs(args.parent, args.output)[4]
        receipt = args.output / "accepted_delivery_job.json"
        write_json(
            receipt,
            {
                "status": "waiting_for_completed_evaluation",
                "pid": os.getpid(),
                "signature": signature,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "delivery": str(args.delivery),
            },
        )
        try:
            deadline = time.monotonic() + 7200
            while not (args.output / "accepted_evaluation_latest.json").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "Accepted evaluation did not complete within two hours; no duplicate evaluation started."
                    )
                state = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Running}}", "neural-exam-occlusion-582-20261008"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if state.returncode or state.stdout.strip() != "true":
                    raise RuntimeError(
                        "The accepted evaluation container stopped without a completed result; inspect its saved log."
                    )
                time.sleep(30)
            if completed_result(args.parent, args.output)[0]["signature"] != signature:
                raise ValueError("Accepted review changed while the delivery waited.")
            prepare(args.parent, args.output, args.delivery, args.template)
            execute_notebook(args.delivery)
            finalize(args.parent, args.output, args.delivery)
            write_json(
                receipt,
                {
                    "status": "generated_pending_visual_and_editorial_review",
                    "signature": signature,
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "delivery": str(args.delivery),
                },
            )
        except Exception as error:
            write_json(
                receipt,
                {
                    "status": "failed",
                    "signature": signature,
                    "failed_at_utc": datetime.now(UTC).isoformat(),
                    "error": str(error),
                    "delivery": str(args.delivery),
                },
            )
            raise
        print(json.dumps({"action": args.action, "delivery": str(args.delivery)}))
        return
    if args.action in ("prepare", "all"):
        prepare(args.parent, args.output, args.delivery, args.template)
    if args.action in ("execute", "all"):
        execute_notebook(args.delivery)
    if args.action in ("finalize", "all"):
        finalize(args.parent, args.output, args.delivery)
    print(json.dumps({"action": args.action, "delivery": str(args.delivery)}))


if __name__ == "__main__":
    main()
