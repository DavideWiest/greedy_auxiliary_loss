from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
        fieldnames = list(dict.fromkeys([*(existing_rows[0].keys() if existing_rows else []), *row.keys()]))
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(existing_row)
            writer.writerow(row)
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
