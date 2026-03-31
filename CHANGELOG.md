# Changelog

All notable changes to this project are documented in this file.

The format is inspired by Keep a Changelog and this project follows Semantic Versioning.

## [0.1.0] - 2026-03-31

### Added
- Modular neural network simulator package with trainable dense, reservoir, and lattice layers.
- Production-grade trainer features: mini-batching, validation split, early stopping, LR schedulers, optimizer stack.
- Optimizers: SGD, Momentum, Adam with weight decay and gradient clipping.
- Callback plugin system for console logging, checkpoints, manifests, run registry, and leaderboard generation.
- Monitor policies with primary/secondary metric support and restore-best-on-early-stop behavior.
- CLI for end-to-end training, checkpointing, monitoring, and registry management.
- Test suite with linting, strict typing, and CI workflow.

### Changed
- Project evolved from a single script into a package-based architecture under src/nnsim.

### Notes
- This is the first stable tagged baseline for public GitHub release.
