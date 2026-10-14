# Vendored halltaxel

This is [halltaxel](https://github.com/srl-ethz/halltaxel) (MIT, see `LICENSE`) at commit `d849d20` (2026-10-10):
its `src/halltaxel` package, `pyproject.toml`, `README.md` and `LICENSE`, unchanged. It is a uv workspace source of
Isaac Lab (`pyproject.toml`), so `uv sync` installs it.

Only `scripts/tools/franka_orca/build_asset.py` uses it, to cut the ORCA fingertips into taxel colliders; the
environments read the baked result through `isaaclab_newton.sensors.TactileSensor` and do not import it. Its example
data that the build uses is vendored in `assets/franka_orca/sources/`.

To update, copy the package from a newer halltaxel commit, record the commit here and in
`assets/franka_orca/README.md`, and rebuild the FrankaOrca asset.
