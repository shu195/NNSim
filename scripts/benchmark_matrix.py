from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark matrix over optimizer/scheduler combinations"
    )
    parser.add_argument("--artifact-root", type=str, default="artifacts/benchmark_matrix")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="xor")
    parser.add_argument("--layers", type=str, default="2,8,1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    optimizers = ["sgd", "momentum", "adam"]
    schedulers = ["constant", "exp", "step"]

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    for optimizer, scheduler in itertools.product(optimizers, schedulers):
        run_name = f"{args.dataset}-{optimizer}-{scheduler}"
        artifact_dir = artifact_root / run_name
        command = [
            sys.executable,
            "-m",
            "nnsim",
            "--dataset",
            args.dataset,
            "--samples",
            str(args.samples),
            "--layers",
            args.layers,
            "--optimizer",
            optimizer,
            "--scheduler",
            scheduler,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--val-split",
            "0.5",
            "--monitor",
            "val_loss",
            "--monitor-mode",
            "min",
            "--save-registry",
            "--save-manifest",
            "--run-name",
            run_name,
            "--artifact-dir",
            str(artifact_dir),
            "--seed",
            str(args.seed),
            "--quiet",
        ]
        print(f"Running: optimizer={optimizer} scheduler={scheduler}")
        run_command(command)

    print(f"Benchmark matrix complete. Artifacts in: {artifact_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
