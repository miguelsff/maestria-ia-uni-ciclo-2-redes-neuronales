"""Compile the multi-file reports explicitly; never called during preparation."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["all", "mri", "flood", "wildlife"], default="all")
    parser.add_argument("--draft", action="store_true", help="Allow pending editorial work and page-count deviations")
    args = parser.parse_args()
    output = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
    source = Path(__file__).resolve().parents[1] / "reports"
    styles = list((output / "report-assets").rglob("neurips_2026.sty"))
    if len(styles) != 1:
        raise SystemExit("Download the official NeurIPS author kit with download --dataset style first.")
    tasks = ("mri", "flood", "wildlife") if args.task == "all" else (args.task,)
    for task in tasks:
        assets = output / "reports" / task
        status_path = assets / "status.json"
        if not status_path.is_file():
            raise SystemExit(f"Generate actual {task} report assets before compilation.")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not args.draft:
            approval = source / task / "finalized.json"
            if not approval.is_file() or not json.loads(approval.read_text(encoding="utf-8")).get("reviewed"):
                raise SystemExit(
                    f"Finish and review {task}/discussion.tex, then record reviewed=true in finalized.json; use --draft for preview."
                )
            if status.get("missing_figures") or status.get("occlusion_status") == "manual_review_pending":
                raise SystemExit(f"Unresolved {task} figures or occlusion review; regenerate assets after fixing them.")
            if task == "wildlife" and (
                status.get("label_review_status") != "reviewed" or status.get("cam_cases", 0) < 12
            ):
                raise SystemExit("Finish the wildlife label review and twelve CAM cases before final compilation.")
            if "completar nombres" in (source / "preamble.tex").read_text(encoding="utf-8").casefold():
                raise SystemExit("Enter the five real authors before final compilation.")
        build = output / "reports/build" / task
        build.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / task, build, dirs_exist_ok=True)
        for filename in ("preamble.tex", "references.bib"):
            shutil.copyfile(source / filename, build / filename)
        shutil.copyfile(styles[0], build / "neurips_2026.sty")
        for filename in ("tables.tex", "figures.tex"):
            shutil.copyfile(assets / filename, build / filename)
        for filename in (
            "inventory.tex",
            "resources.tex",
            "abstract_results.tex",
            "results_analysis.tex",
            "conclusions.tex",
            "qualitative_figures.tex",
        ):
            if (assets / filename).is_file():
                shutil.copyfile(assets / filename, build / filename)
        if (assets / "figures").is_dir():
            shutil.copytree(assets / "figures", build / "figures", dirs_exist_ok=True)
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=build, check=True
        )
        information = subprocess.check_output(["pdfinfo", str(build / "main.pdf")], text=True)
        pages = int(next(line.split(":", 1)[1] for line in information.splitlines() if line.startswith("Pages:")))
        if not args.draft and not 6 <= pages <= 10:
            raise SystemExit(f"{task}: {pages} pages; the exam requires 6–10. Adjust the report and compile again.")
        destination = output / "reports" / f"{task}{'_draft' if args.draft else ''}.pdf"
        shutil.copyfile(build / "main.pdf", destination)
        print(f"{destination}: {pages} pages; visual review {'pending' if args.draft else 'required'}")


if __name__ == "__main__":
    main()
