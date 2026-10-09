"""The PDF compiler must not turn missing scientific evidence into a deliverable."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", [[], ["--draft"]])
def test_compiler_rejects_missing_scientific_assets_even_for_a_draft(tmp_path, mode):
    style = tmp_path / "report-assets/neurips_2026.sty"
    style.parent.mkdir()
    style.write_text("% Unused synthetic style fixture: compilation must never start.\n")
    compiler = Path(__file__).resolve().parents[1] / "scripts/compile_reports.py"
    result = subprocess.run(
        [sys.executable, str(compiler), "--task", "wildlife", *mode],
        env={**os.environ, "OUTPUT_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Generate actual wildlife report assets before compilation" in result.stderr
    assert not (tmp_path / "reports/build").exists()
