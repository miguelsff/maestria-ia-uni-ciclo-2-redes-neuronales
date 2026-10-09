"""Comprueba resultados portables y, opcionalmente, originales de solo lectura."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "entregable/data"
SIGNATURE = "6b2983c33abccbc18e293cc3a10ca30a710669c2ba8055a76bb462b3348a13a9"


def read_json(path: Path) -> dict:
    """Lee un documento de evidencia."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Obtiene un hash físico sin normalizar saltos de línea."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict]:
    """Lee CSV preservando campos humanos vacíos."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    """Verifica hashes, conteos, tablas, ejecución, fuentes y PDFs provisionales."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--originals", action="store_true")
    args = parser.parse_args()
    integrity = read_json(ROOT / "entregable/integridad_datos.json")
    for name, expected in integrity["sha256"].items():
        assert digest(DATA / name) == expected, name
    results = read_json(DATA / "results.json")
    acceptance = read_json(DATA / "review_acceptance.json")
    assert results["signature"] == integrity["signature"] == SIGNATURE
    assert results["trained_or_ran_inference"] is False
    accepted = csv_rows(DATA / "occlusion_review_accepted_582.csv")
    full = csv_rows(DATA / "occlusion_review_sample_600.csv")
    counts = Counter(row["level"] for row in accepted)
    assert len(full) == 600 and len(accepted) == 582
    assert counts == {"low": 340, "medium": 69, "high": 48, "ambiguous": 125}
    excluded = set(acceptance["excluded_annotation_ids"])
    assert len(excluded) == 18
    assert accepted == [row for row in full if row["annotation_id"] not in excluded]
    assert {row["annotation_id"] for row in full if not row["level"]} == excluded
    assert (full[120]["level"], full[120]["reviewer"], full[120]["visual_evidence"]) == ("medium", "Miguel", "")
    assert sum(not row["visual_evidence"].strip() for row in accepted) == 398
    assert acceptance["notes_required"] is False
    assert acceptance["human_notes_fabricated"] is False
    assert results["descriptive_occlusion"]["confidence_intervals_for_population_proportions"] is None

    numerical = csv_rows(ROOT / "entregable/metricas_occlusion.csv")
    source = (ROOT / "fuentes_informe/occlusion_tables.tex").read_text(encoding="utf-8")
    species_source = (ROOT / "fuentes_informe/occlusion_species_tables.tex").read_text(encoding="utf-8")
    for row in numerical:
        group = results["sample_detection"][row["level"]]
        metrics, uncertainty = group["metrics"], group["uncertainty"]
        assert group["signature"] == SIGNATURE
        assert int(row["accepted_objects"]) == metrics["eligible_objects"] == counts[row["level"]]
        assert uncertainty["replicates"] == int(row["replicates"]) == 200
        assert uncertainty["seed"] == int(row["seed"]) == 42
        assert uncertainty["clusters"] == 7438
        assert uncertainty["confidence_level"] == 0.95
        for metric in ["map50", "map50_95"]:
            bounds = uncertainty["intervals"][metric]
            assert float(row[metric]) == metrics[metric]
            assert float(row[metric + "_lower"]) == bounds["lower"]
            assert float(row[metric + "_upper"]) == bounds["upper"]
            assert bounds["defined_replicates"] == 200
            assert f"{metrics[metric]:.4f} [{bounds['lower']:.4f}, {bounds['upper']:.4f}]" in source
        assert sum(s["support"] for s in metrics["per_species"]) == metrics["eligible_objects"]

    def formatted(value: float | None) -> str:
        return "ND" if value is None else f"{value:.4f}"

    for index, species in enumerate(results["sample_detection"]["low"]["metrics"]["per_species"]):
        score_row = [species["species"]]
        ap50_row = [species["species"]]
        for level in ["low", "medium", "high"]:
            item = results["sample_detection"][level]["metrics"]["per_species"][index]
            score_row.extend([str(item["support"]), formatted(item["map50_95"])])
            ap50_row.append(formatted(item["map50"]))
        assert " & ".join(score_row) in species_source
        assert " & ".join(ap50_row) in species_source

    notebooks = {}
    for name, expected_cells in [("03_wildlife.ipynb", 16), ("04_occlusion_582.ipynb", 4)]:
        notebook = read_json(ROOT / "entregable" / name)
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert len(cells) == expected_cells
        assert [cell["execution_count"] for cell in cells] == list(range(1, expected_cells + 1))
        assert not any(out["output_type"] == "error" for cell in cells for out in cell["outputs"])
        notebooks[name] = {"executed_cells": len(cells), "errors": 0}

    pages = {}
    for name, expected_pages in [("informe_fauna_provisional.pdf", 10), ("anexo_occlusion_provisional.pdf", 2)]:
        reader = PdfReader(ROOT / "evidencias" / name)
        assert len(reader.pages) == expected_pages
        text = "\n".join(page.extract_text() for page in reader.pages)
        assert all(value in text for value in ["0.0875", "0.0271", "0.0244", "582", "398"])
        assert "permanece pendiente de anotación" not in text
        pages[name] = len(reader.pages)
    main = (ROOT / "fuentes_informe/main.tex").read_text(encoding="utf-8")
    assert len(re.findall(r"\\cite\{", main)) >= 5
    assert all(term in main for term in ["0.4358", "0.2625", "1\\,853", "occlusion_tables"])

    originals_checked = set()
    if args.originals:
        provenance = read_json(ROOT / "evidencias/procedencia.json")
        for original, entry in provenance["copied_files"].items():
            assert digest(Path(original)) == entry["sha256"], original
            assert digest(ROOT / entry["destination"]) == entry["sha256"], entry["destination"]
            originals_checked.add(str(Path(original)))
        for original, expected in provenance["preserved_inputs"].items():
            assert digest(Path(original)) == expected, original
            originals_checked.add(str(Path(original)))
        protocol = read_json(DATA / "protocol.json")
        parent_root = Path("E:/Kaggle/exam-output-wildlife-v6")
        for name, expected in protocol["parent_sha256"].items():
            original = parent_root / name
            assert digest(original) == expected, name
            originals_checked.add(str(original))
        sample_root = Path("E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600")
        for name, expected in acceptance["source_files_sha256"].items():
            original = sample_root / name
            assert digest(original) == expected, name
            originals_checked.add(str(original))
        original_root = Path("D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial")
        frozen = read_json(ROOT / "evidencias/fuentes_congeladas.json")["files"]
        assert len(frozen) == 44
        for entry in frozen:
            assert digest(original_root / entry["path"]) == entry["expected"]
            assert digest(ROOT.parent / entry["path"]) == entry["worktree"]
            originals_checked.add(str(original_root / entry["path"]))

    receipt = {
        "status": "passed", "data_files_verified": len(integrity["sha256"]),
        "counts": dict(counts), "excluded": 18, "missing_notes": 398,
        "completed_levels": 3, "replicates_per_level": 200,
        "notebooks": notebooks, "provisional_pdf_pages": pages,
        "original_files_verified": len(originals_checked),
        "scope": "Consistencia e integridad de resultados guardados; no recalcula AP ni bootstrap.",
    }
    (ROOT / "evidencias/validation.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
