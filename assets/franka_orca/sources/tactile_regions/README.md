# Tactile-region sensor meshes

One STL per finger (`<finger>_sensor.stl`), referenced by that finger's
`_tactile_region` fixture link in
`../orcahand_description/v2/models/urdf/orcahand_right.urdf`
(`TactileSurfaceSpec.from_urdf()`'s input - see `../../real_hand.py`'s
docstring). Each is the hand-selected sensor footprint on the
corresponding real fingertip mesh in `../orcahand_description/`
(`T-DP.stl`, `I-FingerTipAssembly.stl`, `M-FingerTipAssembly.stl` x2,
`P-FingerTipAssembly.stl`), authored in that fingertip's own frame and
units (mm), so the fixture joints stay at zero offset.

`select_region()` marks the fingertip faces under each footprint as
sensed; the rest of the fingertip mesh is rendered and collides as plain,
unsensed skin (`replace_body_mesh()`), keeping the fingertip's own color.

## Replacing a footprint

Replace the corresponding file here (or point the fixture's
`<mesh filename>` at a new one) - no other change is needed; `from_urdf()`
reads the fixture's geometry and joint origin at runtime, not anything
cached here. If a new footprint isn't authored in the fingertip's frame,
set the fixture joint's `<origin>` accordingly.
