# Vendored orcahand_description (partial)

`v2/models/urdf/orcahand_right.urdf` and `v2/models/assets/right/*.stl` are
copied from [`orcahand_description`](https://github.com/orcahand/orcahand_description)
(v2, right hand) - only the URDF and the 16 mesh files it actually
references, not the full `assets/` tree (which also holds unused CAD
export variants and a second, much richer collision-mesh set, ~59 MB
total upstream). The directory structure is kept exactly as upstream
(`<pkg>/v2/models/...`) so `package://orcahand_description/...` URIs in
the URDF resolve unmodified - newton's URDF importer finds the package
root by walking up from the URDF file looking for a directory literally
named `orcahand_description`, wherever it happens to live on disk.

## What was added on top of upstream

Five `<link>`/`<joint>` pairs, one per real fingertip link, named
`<fingertip-link>_tactile_region` - not part of upstream
`orcahand_description`. See `../../real_hand.py`'s docstring and
`../../README.md` for what they're for (`TactileSurfaceSpec.from_urdf()`'s
input). Each is a zero-offset fixed joint whose own `<visual>` mesh points
at the hand-selected sensor footprint
`../../tactile_regions/<finger>_sensor.stl` rather than an
upstream asset - see that directory's README.

## Updating

If `orcahand_description` changes upstream (CAD revision, new URDF), re-copy
`orcahand_right.urdf` and its referenced assets, then re-apply the 5
fixture link/joint additions - there's no automated sync script for this
yet, since this is currently a small, static one-time copy.
