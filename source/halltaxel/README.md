# halltaxel

`halltaxel` simulates tactile sensors on robot hands. A sensing surface,
such as a fingertip pad, is covered by small sensing points called
taxels, and the simulation reports the contact force each taxel feels at
every step. The sensor model is based on Hall-effect sensing elements and
runs on [Newton](https://github.com/newton-physics/newton)/Warp. It offers
two modeling approaches behind one interface, so you can switch between
them with a single argument:

- **Contact patch:** the surface is split into one patch per taxel, and
  each taxel's force is the contact force on its patch.
- **Spring skin:** the surface is covered by a spring-damper net that
  deforms on contact, and each taxel's force comes from the net's local
  deformation.

The library is generic: it works on any Newton body and mesh, and you set up a sensor
the same way whether you simulate one scene in a viewer or thousands of
worlds in parallel for training. Hand-specific integration code (OrcaHand
geometry, RL training environments) lives in a separate repo,
[`newton-tactile-hand`](https://github.com/srl-ethz/newton-tactile-hand).

## Installation

This project uses [uv](https://docs.astral.sh/uv/). Clone the repo, then
install the package and its development tools:

```sh
uv sync --group dev
uv run pytest          # optional: check that everything works
```

## Running the examples

Each example opens an interactive Newton viewer. Add `--viewer viser` to
view it in a browser, or `--viewer null --test` to run it headless as a
quick check.

**Cube pressed onto a fingertip:** a cube drops onto an OrcaHand thumb
pad. Each taxel lights up with the force it feels, and an arrow shows the
force's direction. You can drag the cube around to press different spots.

```sh
uv run python examples/contact_patch_press/press_test.py
uv run python examples/contact_patch_press/press_test.py --show-taxels
```

**Full OrcaHand with sensors on every fingertip:** the complete hand,
turned palm-up, with a tactile sensor on each of the five fingertips. A
ball drops into the palm, and you can drag it onto the fingertips.

```sh
uv run python examples/orcahand_real_hand/real_hand.py
uv run python examples/orcahand_real_hand/real_hand.py --show-taxels
```

**Inspect how a fingertip is split into taxels:** shows how the sensing
surface of one finger is divided into taxel patches.

```sh
uv run python examples/orcahand_fingertips/inspect_discretization.py
uv run python examples/orcahand_fingertips/inspect_discretization.py --finger pinky
```

All the example data is included in the repo, so no external checkout is
needed. The example folders have their own READMEs with more details.

## Repository structure

```
halltaxel/
├── pyproject.toml                # package metadata and dependencies
├── src/halltaxel/
│   ├── __init__.py               # public API
│   ├── spec.py                   # TactileSurfaceSpec: describes one sensing surface
│   │                             #   (from_files() or from_urdf())
│   ├── segmentation.py           # surface-mesh tools: project_to_surface(), select_region(),
│   │                             #   segment_surface() (taxel patches, used by contact_patch)
│   ├── layout.py                 # TaxelLayout: taxel positions + sensor frame
│   ├── backends/
│   │   ├── base.py               # TactileBackend interface + attach_tactile() entry point
│   │   ├── contact_patch.py      # contact-patch backend (implemented): one collision
│   │   │                         #   shape per taxel, forces from Newton's SensorContact
│   │   └── spring_skin.py        # spring-model backend (not yet implemented)
│   ├── kernels/                  # shared Warp kernels
│   ├── runtime.py                # TactileRuntime: finalizes backends and updates readings
│   ├── reading.py                # TactileReading: per-taxel forces in the sensor frame,
│   │                             #   as NumPy or torch arrays
│   ├── _collision.py             # tactile shapes inherit their body's collision filters
│   ├── viz.py                    # heatmap colors, taxel markers, force arrows
│   └── examples.py               # shared scene helpers for the examples
├── examples/
│   ├── contact_patch_press/      # cube pressed onto one thumb pad
│   ├── orcahand_real_hand/       # full OrcaHand from its URDF, sensors on all fingertips
│   └── orcahand_fingertips/      # per-finger meshes, taxel positions, and an inspector
├── tests/                        # pytest suite, one module per source module
├── docs/                         # documentation site sources (not built yet)
└── .github/workflows/ci.yml
```

The library follows three design rules:

1. **No hand-specific dependencies.** `halltaxel` never depends on
   `orca_core` or any other hand package, only on generic Newton bodies
   and meshes.
2. **torch and RL libraries are optional.** They are imported only when
   you actually use them, for example `TactileReading.torch()`.
3. **No batching flags.** The number of parallel worlds is read from the
   `newton.Model`, so a single world and 4096 worlds use the same code.

## Future work

- **Spring-model backend (`spring_skin`):** simulate the sensor skin as a
  mass-spring-damper network using Newton's particle and spring system.
  It will be a second backend behind the same interface as
  `contact_patch`, so the two methods can be compared directly by
  switching `backend=`. So far it exists only as a placeholder module.
- **Documentation site:** build out the mkdocs site in `docs/`.
