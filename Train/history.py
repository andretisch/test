from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def append_history_row(
    run_dir: Path,
    row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows.append(row)
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "history.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = run_dir / "history.csv"
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows
