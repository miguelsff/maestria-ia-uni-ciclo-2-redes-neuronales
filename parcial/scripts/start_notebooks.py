"""Create editable notebook copies outside the read-only source mount."""

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    output = Path(os.environ.get("OUTPUT_DIR", "/outputs")) / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    task = os.environ.get("NOTEBOOK_TASK")
    pattern = f"*_{task}.ipynb" if task else "*.ipynb"
    for source in (Path(__file__).resolve().parents[1] / "notebooks").glob(pattern):
        destination = output / source.name
        if not destination.exists():
            shutil.copyfile(source, destination)
    os.execvp(
        "jupyter",
        [
            "jupyter",
            "lab",
            "--ip=0.0.0.0",
            "--port=8888",
            "--no-browser",
            "--allow-root",
            f"--notebook-dir={output.parent if task else output}",
            *sys.argv[1:],
        ],
    )


if __name__ == "__main__":
    main()
