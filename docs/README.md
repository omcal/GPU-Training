# Documentation

For executable examples, start with the [CUDA guide](../cuda/README.md) or
[Metal lab guide](../labs/README.md). The [main README](../README.md) covers both setups.

## Learning routes

| Guide | Purpose |
|---|---|
| [CUDA program: 8 October–1 November 2026](08-cuda-inference-study-plan.md) | PMPP + AI Systems Performance Engineering; chapter pairing, exercises, and completion criteria |
| [Independent study workspace](../study/README.md) | First session, 12 exercise briefs, build commands, and note templates |
| [Progress tracker](../study/progress.md) | Evidence of learning and outstanding work |
| [Metal learning route](05-learning-route.md) | Session-based reading and lab sequence for Apple Silicon |
| [PMPP topic map](04-pmpp-roadmap.md) | Chapter topics and their lab counterparts |

## Concepts and measurements

| Guide | Purpose |
|---|---|
| [GPU mental model](00-mental-model.md) | Execution hierarchy, memory, occupancy, roofline |
| [CUDA → Metal translation](01-cuda-to-metal.md) | Syntax, launch geometry, memory, synchronization |
| [Measuring GPU work](03-how-to-measure.md) | Warm-up, synchronization, working sets, and timing interpretation |
| [M4 measurements](02-your-hardware.md) | Tested Apple Silicon environment and observed behavior |
| [Ubuntu CUDA inventory](06-ubuntu-cuda-inventory.md) | Historical setup and migration record |
| [CUDA validation: 27 September 2026](07-cuda-validation.md) | Correctness, sanitizer results, and CUDA timings from that session |
| [Validation: 8 October 2026](10-validation-2026-10-08.md) | Latest recorded hardware checks and CI validation |

## Repository maintenance

[CI and release guide](09-ci-and-releases.md) describes portable checks, CUDA compilation,
and source releases. [Contributing](../CONTRIBUTING.md) covers the experiment and review
workflow. Hardware measurements and validation records are dated snapshots; distinguish
those results from your current environment and from planned study work.
