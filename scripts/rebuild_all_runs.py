from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    run_root = Path("results/runs")
    rows: list[dict] = []
    fieldnames: list[str] = []

    for result_path in sorted(run_root.glob("*/result.json")):
        with result_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        rows.append(row)
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    output_path = Path("results/all_runs.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
