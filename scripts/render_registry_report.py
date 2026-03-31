from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_registry(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def score(entry: dict[str, object]) -> float:
    value = entry.get("monitor_value")
    if isinstance(value, (int, float)):
        return float(value)
    mode = entry.get("monitor_mode")
    return float("inf") if mode == "min" else float("-inf")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render markdown report from run registry")
    parser.add_argument("registry", type=str, help="Path to run_registry.jsonl")
    parser.add_argument("--output", type=str, default="artifacts/benchmark_report.md")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    entries = load_registry(registry_path)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in entries:
        monitor = str(row.get("monitor", ""))
        mode = str(row.get("monitor_mode", ""))
        grouped.setdefault((monitor, mode), []).append(row)

    lines = ["# Benchmark Report", ""]
    lines.append(f"Source: {registry_path}")
    lines.append("")

    for (monitor, mode), rows in sorted(grouped.items()):
        ranked = sorted(rows, key=score, reverse=(mode == "max"))[: max(1, args.top_k)]
        lines.append(f"## Monitor: {monitor} ({mode})")
        lines.append("")
        lines.append("| Rank | Run | Value | Epochs | Early Stop |")
        lines.append("|---:|---|---:|---:|---:|")
        for idx, row in enumerate(ranked, start=1):
            value = row.get("monitor_value")
            run_name = row.get("run_name")
            epochs = row.get("epochs_completed")
            stopped = row.get("stopped_early")
            lines.append(f"| {idx} | {run_name} | {value} | {epochs} | {stopped} |")
        lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
