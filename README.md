# nnsim

[![CI](https://github.com/shu195/NNSim/actions/workflows/ci.yml/badge.svg)](https://github.com/shu195/NNSim/actions/workflows/ci.yml)
[![Release](https://github.com/shu195/NNSim/actions/workflows/release.yml/badge.svg)](https://github.com/shu195/NNSim/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade neural network simulator written in pure Python with a clean architecture, test suite, and packaging support.

CI is included via GitHub Actions in `.github/workflows/ci.yml`.

## Highlights

- Modular design (`src/nnsim`) with clear separation of datasets, layers, model, and trainer
- Config-driven network construction
- Production trainer stack:
	- Mini-batch training
	- Validation split with per-epoch metrics
	- Early stopping controls
	- Optimizer subsystem (`sgd`, `momentum`, `adam`)
	- Regularization controls (weight decay, gradient clipping)
	- Learning-rate scheduling (`constant`, `exp`, `step`)
	- Callback plugin pack:
		- Console logging
		- Best-validation checkpoint saving
		- Last-epoch checkpoint saving
		- Experiment manifest generation
		- Run registry and leaderboard generation
		- Final validation confusion-matrix reporting
- Multiple layer types:
	- Dense (trainable)
	- Reservoir (fixed sparse random weights)
	- Lattice (multi-input directional connectivity)
- Multiple datasets:
	- Logic: `xor`, `and`, `or`
	- Reconstruction manifolds: `circles`, `spiral`
	- Lightweight sentiment text dataset: `text`
- Reproducibility via deterministic seeding
- Artifact output for training loss history
- Model checkpoint save/load for repeatable experiments and resume workflows
- Professional project setup: `pyproject.toml`, CLI entry point, lint/type config, tests

## Repository Structure

```text
.
|-- src/nnsim/
|   |-- activations.py
|   |-- cli.py
|   |-- config.py
|   |-- datasets.py
|   |-- layers.py
|   |-- losses.py
|   |-- network.py
|   |-- trainer.py
|   `-- types.py
|-- tests/
|-- nn_simulator.py
|-- pyproject.toml
`-- README.md
```

## Installation

```powershell
python -m pip install -e .
```

For development tools:

```powershell
python -m pip install -e .[dev]
```

## Quickstart

Train and persist full experiment artifacts in one command:

```powershell
nnsim --dataset xor --layers 2,8,1 --optimizer adam --epochs 120 --val-split 0.5 --save-best-model --save-last-model --save-manifest --save-registry --run-name quickstart-xor --artifact-dir artifacts/quickstart
```

## Usage

### Package CLI

```powershell
nnsim --dataset xor --layers 2,6,1 --activation sigmoid --lr 0.5 --epochs 2000
```

### Backward-Compatible Script Entrypoint

```powershell
python nn_simulator.py --dataset xor --layers 2,6,1 --activation sigmoid --lr 0.5 --epochs 2000
```

### Common Scenarios

```powershell
# XOR
nnsim --dataset xor --layers 2,6,1 --activation sigmoid --lr 0.5 --epochs 2000

# XOR with validation, early stopping and step scheduler
nnsim --dataset xor --layers 2,8,1 --activation sigmoid --lr 0.5 --epochs 600 --batch-size 2 --val-split 0.5 --early-stop-patience 15 --early-stop-min-delta 0.0001 --scheduler step --scheduler-step-size 50 --scheduler-gamma 0.9

# XOR with Adam, weight decay and gradient clipping
nnsim --dataset xor --layers 2,8,1 --activation sigmoid --optimizer adam --lr 0.1 --weight-decay 0.0001 --grad-clip 1.0 --epochs 300 --batch-size 2

# Save best/last checkpoints and run manifest
nnsim --dataset xor --layers 2,8,1 --val-split 0.5 --save-best-model --save-last-model --save-manifest --run-name xor-baseline --artifact-dir artifacts/xor_baseline

# Monitor validation accuracy and restore best checkpoint on early stop
nnsim --dataset xor --layers 2,8,1 --val-split 0.5 --monitor val_accuracy --monitor-mode max --early-stop-patience 10 --restore-best-on-early-stop --save-best-model --artifact-dir artifacts/xor_monitored

# Composite monitor: optimize val_accuracy, tie-break with lower val_loss
nnsim --dataset xor --layers 2,8,1 --val-split 0.5 --monitor val_accuracy --monitor-mode max --secondary-monitor val_loss --secondary-monitor-mode min --save-best-model --artifact-dir artifacts/xor_composite

# Track runs in registry and maintain monitor-aware leaderboard
nnsim --dataset xor --layers 2,8,1 --val-split 0.5 --monitor val_loss --monitor-mode min --save-registry --leaderboard-top-k 25 --run-name xor-run-a --artifact-dir artifacts/xor_registry

# Reconstruction (input size == output size == 2)
nnsim --dataset circles --layers 2,16,2 --activation sigmoid --lr 0.5 --epochs 800 --samples 256

# Reservoir hidden layer
nnsim --dataset circles --layers 2,24,2 --reservoir 0 --reservoir-sparsity 0.2 --activation sigmoid --epochs 600

# Lattice hidden layer
nnsim --dataset circles --layers 2,12,12,2 --lattice 1 --activation sigmoid --epochs 600
```

## Artifacts

By default, training history is written to:

- `artifacts/training_history.json`

Disable saving history:

```powershell
nnsim --dataset xor --layers 2,6,1 --no-save-history
```

### Model Checkpoints

Save a trained model:

```powershell
nnsim --dataset xor --layers 2,6,1 --epochs 1200 --save-model artifacts/model_xor.json
```

Resume from a saved model:

```powershell
nnsim --dataset xor --load-model artifacts/model_xor.json --epochs 300 --save-model artifacts/model_xor_v2.json
```

## Development Workflow

Run tests:

```powershell
pytest
```

Lint:

```powershell
ruff check .
```

Type check:

```powershell
mypy src tests
```

## Benchmarking

Run a matrix benchmark over optimizer and scheduler combinations:

```powershell
python scripts/benchmark_matrix.py --dataset xor --epochs 120 --artifact-root artifacts/benchmark_matrix
```

Render a markdown report from the generated registry:

```powershell
python scripts/render_registry_report.py artifacts/benchmark_matrix/xor-adam-constant/run_registry.jsonl --output artifacts/benchmark_report.md
```

## Release Process

1. Update version in `pyproject.toml`.
2. Update `CHANGELOG.md` with release notes.
3. Create and push a tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

Tag pushes trigger `.github/workflows/release.yml` to build and attach wheel/sdist artifacts to a GitHub release.

## Community

- Contribution guide: `CONTRIBUTING.md`
- Bug report template: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Feature request template: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Pull request template: `.github/pull_request_template.md`

## Advanced Trainer Flags

- `--optimizer`: `sgd`, `momentum`, `adam`
- `--momentum`: momentum factor when `--optimizer momentum`
- `--adam-beta1`, `--adam-beta2`, `--adam-epsilon`: Adam hyperparameters
- `--weight-decay`: L2-style decay added to trainable weight gradients
- `--grad-clip`: clip gradient magnitude per parameter update (set negative to disable)
- `--batch-size`: mini-batch size (default: `16`)
- `--val-split`: validation ratio in `[0.0, 1.0)` (default: `0.0`)
- `--monitor`: primary metric to track, `train_loss`, `val_loss`, or `val_accuracy`
- `--monitor-mode`: metric direction, `auto`, `min`, or `max`
- `--secondary-monitor`: optional tie-break metric with the same choices as `--monitor`
- `--secondary-monitor-mode`: mode for secondary monitor (`auto`, `min`, `max`)
- `--secondary-monitor-min-delta`: minimum improvement for tie-break updates
- `--early-stop-patience`: stop when validation stalls for N epochs (default: `0`, disabled)
- `--early-stop-min-delta`: minimum monitored-metric improvement to reset patience
- `--restore-best-on-early-stop`: reload best checkpoint after early stop (requires `--save-best-model`)
- `--scheduler`: `constant`, `exp`, or `step`
- `--scheduler-gamma`: decay factor for `exp` and `step`
- `--scheduler-step-size`: epochs per decay step for `step`
- `--min-lr`: lower bound for scheduled learning rate
- `--quiet`: disable per-epoch console logging
- `--save-best-model`: write best validation model checkpoint to `artifact-dir/model_best.json`
- `--save-last-model`: write final model checkpoint to `artifact-dir/model_last.json`
- `--save-manifest`: write run manifest to `artifact-dir/experiment_manifest.json`
- `--save-registry`: append run into `artifact-dir/run_registry.jsonl` and update `artifact-dir/leaderboard.json`
- `--leaderboard-top-k`: keep top-k entries in generated leaderboard
- `--run-name`: custom run identifier stored in manifest

## Design Notes

- Output layer uses sigmoid by default for stable binary outputs.
- Reservoir layers are intentionally non-trainable to preserve fixed sparse dynamics.
- Lattice layers consume outputs from prior states and aggregate directional gradients during backpropagation.

## Roadmap

- Mini-batch and optimizer abstraction (SGD, Momentum, Adam)
- Cross-entropy and multi-class softmax outputs
- Model serialization/deserialization
- Web dashboard for live metrics
- Benchmark suite for performance profiling

## License

MIT. See `LICENSE`.
