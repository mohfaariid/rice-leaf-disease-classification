"""Execute code cells from visual_experiment_report.ipynb.

This lightweight runner is useful when Jupyter/nbconvert is unavailable.
It executes the notebook code cells in order so generated figures are saved.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "visual_reports" / "visual_experiment_report.ipynb"

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}

    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue

        print(f"Running code cell {index}...")
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)

    print("Visual report code cells executed successfully.")


if __name__ == "__main__":
    main()
