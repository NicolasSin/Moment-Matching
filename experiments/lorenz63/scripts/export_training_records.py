"""Export archived monitoring values; this does not run training or new audits."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = (
    "stage", "update", "global_update", "elapsed_seconds", "train_loss",
    "validation_loss", "gaussian_order", "training_optimization_loss", "training_gaussian_order",
)
ENDPOINT_FIELDS = (
    "stage", "update", "global_update", "train_loss", "validation_loss", "gaussian_order",
    "training_optimization_loss", "training_gaussian_order", "train_quadrature_nodes",
    "validation_quadrature_nodes", "last_full_training_audit", "last_full_validation_audit",
    "numerical_error_interpretation",
)


def export_records(records_path, output):
    items = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items or items[0].get("record_type") != "historical_monitor_records":
        raise ValueError("Expected bundled historical monitor records with a source header")
    records = items[1:]
    if not records or (records[0]["stage"], records[0]["global_update"]) != ("adam", 0):
        raise ValueError("Historical monitoring must begin at Adam update zero")
    endpoint = {field: records[-1][field] for field in ENDPOINT_FIELDS}
    if (endpoint["stage"], endpoint["update"], endpoint["global_update"]) != ("gh", 4000, 5000):
        raise ValueError("Expected the recorded endpoint after 1,000 Adam and 4,000 L-BFGS updates")
    previous = -1
    for row in records:
        if any(field not in row for field in CSV_FIELDS) or row["global_update"] < previous:
            raise ValueError("Missing fields or unordered historical monitor records")
        previous = row["global_update"]
    encoded_endpoint = json.dumps(endpoint, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "loss_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    # Preserve the existing archived JSON/CSV line endings on every platform.
    with (output / "training_endpoint.json").open("w", encoding="utf-8", newline="\r\n") as stream:
        stream.write(encoded_endpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/training_records.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training_records")
    args = parser.parse_args()
    output = args.output.resolve()
    protected = ("code", "configs", "data", "docs", "models", "results", "scripts")
    if output == ROOT or any(output.is_relative_to(ROOT / name) for name in protected):
        parser.error("Use a new output directory outside packaged inputs, code, and results")
    export_records(args.records.resolve(), output)
    print(f"Exported historical loss_history.csv and training_endpoint.json to {output}")


if __name__ == "__main__":
    main()
