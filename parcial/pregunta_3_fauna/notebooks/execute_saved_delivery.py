"""Ejecuta el notebook de consulta en un kernel limpio y registra el resultado."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import nbformat
from nbclient import NotebookClient


root = Path(__file__).resolve().parents[1]
delivery = root / "entregable"
path = delivery / "03_wildlife.ipynb"
notebook = nbformat.read(root / "notebooks/03_wildlife.ipynb", as_version=4)
NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(delivery)}}).execute()
nbformat.validate(notebook)
code = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert len(code) == 16
assert [cell.execution_count for cell in code] == list(range(1, 17))
assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
nbformat.write(notebook, path)
receipt = {
    "notebook": "entregable/03_wildlife.ipynb",
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "executed_code_cells": len(code),
    "errors": 0,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "execution": "Kernel limpio; contenedor sin red ni GPU; solo consulta de resultados guardados.",
    "training_or_inference": False,
}
(root / "evidencias/notebook_execution.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt, ensure_ascii=False))
