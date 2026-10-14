# FrankaOrca robot asset

The Franka FR3 arm with the right ORCA hand, as `FRANKA_ORCA_CFG`
(`source/isaaclab_assets/isaaclab_assets/robots/franka_orca.py`) spawns it. Everything here is tracked in plain git
(not LFS, see `.gitattributes`), so a clone runs the FrankaOrca tasks without building anything.

## Layout

- `sources/`: the inputs of `scripts/tools/franka_orca/build_asset.py`, vendored so the build needs nothing outside
  this repository.
  - `franka_emika_panda/`: the FR3 URDF (`urdf/fr3.urdf`) and the meshes it references, from
    [newton-assets](https://github.com/newton-physics/newton-assets) `franka_emika_panda` at the revision Newton
    1.6.1 downloads (`newton.utils.download_asset`). Apache 2.0, see its `LICENSE`.
  - `orcahand_description/`: the right ORCA hand URDF and its meshes, with the five `<fingertip>_tactile_region`
    fixture links halltaxel adds (see its `README.md`), from halltaxel's
    `examples/orcahand_real_hand/data/orcahand_description`.
  - `tactile_regions/`: the sensor footprints those fixture links reference, from halltaxel's
    `examples/orcahand_real_hand/data/tactile_regions`.
  - `fingertip_taxels/<finger>/`: halltaxel's taxel data per fingertip (`mesh.stl`, `taxels.npy`,
    `sensor_mount.npy`, `<finger>_sensor.stl`), from halltaxel's `examples/orcahand_fingertips/data`.
- `<method>/`: one build per fingertip collision method (`FINGERTIP_METHODS`): `primitives`, `no_taxels`,
  `convex_taxels`, `mesh_taxels` and `mesh_fingertips`. Each holds the composed `franka_orca.urdf` with its baked
  fingertip `meshes/`, the converted `usd/`, the top layer `franka_orca.usda` the robot config loads and, with
  taxels, the tactile sidecars `tactile/<finger>.npz`.

The halltaxel data comes from halltaxel at commit `d849d20` (2026-10-10), the same revision as the vendored package
in `source/halltaxel`.

## Rebuilding

Rebuild after changing the build script or the sources, and commit the result. The build converts the URDF with
Isaac Sim's URDF importer, so it needs the `importers` extra (`uv sync --extra importers`); the environments do
not.

```bash
uv run python scripts/tools/franka_orca/build_asset.py                          # every method
uv run python scripts/tools/franka_orca/build_asset.py --fingertips mesh_taxels # one method
uv run python scripts/tools/franka_orca/check_asset.py --fingertips mesh_taxels
```

The CoACD decomposition of `mesh_taxels` runs single-threaded per finger, so it is reproducible.
