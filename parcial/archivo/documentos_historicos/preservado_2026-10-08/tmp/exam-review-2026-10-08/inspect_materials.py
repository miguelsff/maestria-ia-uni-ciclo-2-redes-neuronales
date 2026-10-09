"""Inspect saved documents, notebooks, source provenance and remote publication."""

import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

WORKSPACE = Path(__file__).resolve().parents[2]
DESTINATION = WORKSPACE / "audits/revision-2026-10-08"
DATA = Path("E:/Kaggle")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=WORKSPACE, text=True, encoding="utf-8").strip()


def inspect():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    pdfs = {
        "exam": WORKSPACE / "examen-parcial.pdf",
        "mri": DATA / "exam-output-mri/submission-2026-10-04/reports/mri.pdf",
        "flood": DATA / "exam-delivery-flood/review-2026-10-04/reports/flood_draft.pdf",
        "wildlife": DATA / "exam-output-wildlife-v6/reports/wildlife_draft.pdf",
    }
    result = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "pdfs": {}, "notebooks": {}}
    preservation = {}
    for task, path in pdfs.items():
        reader = PdfReader(path)
        text = "\n\n".join(f"=== PAGE {index+1} ===\n{page.extract_text()}" for index, page in enumerate(reader.pages))
        (DESTINATION / f"{task}-extracted.txt").write_text(text, encoding="utf-8")
        result["pdfs"][task] = {"path": str(path), "pages": len(reader.pages), "sha256": digest(path)}
        preservation[str(path)] = digest(path)
    notebooks = {
        "mri": DATA / "exam-output-mri/submission-2026-10-04/notebooks/01_mri.ipynb",
        "flood": DATA / "exam-output-flood/notebooks/02_flood.ipynb",
        "wildlife": DATA / "exam-output-wildlife-v6/notebooks/03_wildlife.ipynb",
    }
    for task, path in notebooks.items():
        notebook = read_json(path)
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        errors = [output for cell in code for output in cell.get("outputs", []) if output["output_type"] == "error"]
        strings = []
        for cell in notebook["cells"]:
            strings.append("".join(cell.get("source", [])))
            for output in cell.get("outputs", []):
                strings.append("".join(output.get("text", [])))
                for kind in ("text/plain", "text/html"):
                    strings.append("".join(output.get("data", {}).get(kind, [])))
        (DESTINATION / f"{task}-notebook-extracted.txt").write_text("\n\n".join(strings), encoding="utf-8")
        result["notebooks"][task] = dict(path=str(path), sha256=digest(path), code_cells=len(code),
            executed=sum(cell.get("execution_count") is not None for cell in code), error_count=len(errors))
        preservation[str(path)] = digest(path)
    freeze = DATA / "exam-output-wildlife-v6/receipts/source-freeze.json"
    result["wildlife_frozen_source_matches"] = {
        name: digest(WORKSPACE/name) == expected for name,expected in read_json(freeze)["sha256"].items()
    }
    result["remote"] = {
        "heads": command("git", "ls-remote", "--heads", "origin").splitlines(),
        "repository": json.loads(command("gh", "api", "repos/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales")),
        "branches": json.loads(command("gh", "api", "repos/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/branches?per_page=100")),
        "local_head": command("git", "rev-parse", "HEAD"),
    }
    # Keep the remote receipt small and do not store account metadata.
    repo = result["remote"]["repository"]
    result["remote"]["repository"] = {key: repo[key] for key in ("full_name", "private", "default_branch", "html_url")}
    result["remote"]["branches"] = [{"name": row["name"], "sha": row["commit"]["sha"]}
                                      for row in result["remote"]["branches"]]
    tree = json.loads(command("gh", "api", "repos/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/git/trees/main?recursive=1"))
    paths = {row["path"] for row in tree["tree"]}
    result["remote"]["tree_sha"] = tree["sha"]
    result["remote"]["accepted_workflow_published"] = {name: "parcial/scripts/"+name in paths for name in (
        "accept_occlusion_review.py", "evaluate_occlusion_accepted.py", "build_occlusion_accepted_delivery.py",
        "occlusion_accepted_review.py")}
    result["remote"]["main_source_matches"] = {
        row["path"]: command("git", "hash-object", "--path", row["path"], str(WORKSPACE.parent/row["path"])) == row["sha"]
        for row in tree["tree"] if row["type"] == "blob" and row["path"].startswith("parcial/")
        and (WORKSPACE.parent/row["path"]).is_file()
        and any(row["path"].startswith("parcial/"+prefix) for prefix in ("src/", "scripts/", "configs/", "reports/", "notebooks/"))
    }
    for root, items in (
        (DATA / "exam-output-mri/stratified-v2/mri", ["manifest.json", "resnet50_bce_weighted/best.pt"]),
        (DATA / "exam-output-flood/flood", ["manifest.json", "gold_bce/best.pt"]),
        (DATA / "exam-output-wildlife-v6/wildlife", ["manifest.json", "yolov8s/weights/best.pt", "label_review.csv", "occlusion_review.csv"]),
        (DATA / "exam-output-wildlife-v6-occlusion-sample-600", ["protocol.json", "review_acceptance.json", "occlusion_review_sample_600.csv", "occlusion_review_accepted_582.csv"]),
    ):
        for name in items:
            preservation[str(root/name)] = digest(root/name)
    receipt = DATA / "exam-output-wildlife-v6-occlusion-sample-600/review_acceptance.json"
    result["occlusion_acceptance"] = read_json(receipt)
    (DESTINATION/"materials.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (DESTINATION/"preservation-before.json").write_text(json.dumps(preservation, indent=2), encoding="utf-8")
    print(json.dumps({"pdfs": result["pdfs"], "notebooks": result["notebooks"],
          "remote": result["remote"]["branches"],
          "main_source_mismatches": [k for k,v in result["remote"]["main_source_matches"].items() if not v],
          "accepted_workflow_published": result["remote"]["accepted_workflow_published"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    inspect()
