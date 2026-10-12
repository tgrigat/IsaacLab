# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the Franka (FR3) + ORCA-hand robot asset with halltaxel tactile fingertips.

The URDFs are the source of truth; everything written here is a gitignored build output under
``assets/franka_orca_mesh_fingertips/`` (or ``$FRANKA_ORCA_ASSET_DIR``). Two steps:

1. :func:`compose_urdf`: FR3 URDF + ORCA right-hand URDF -> ``franka_orca.urdf``. The hand links collide
   through their visual meshes, which the simulator imports as convex hulls, and each sensed fingertip
   is cut by halltaxel into one triangle-mesh collision per taxel (``<finger>_taxel_NNN``), a pad proxy and the
   rest of halltaxel's fingertip surface (``<finger>_unsensed``), all kept as triangle meshes. A sidecar
   ``tactile/<finger>.npz`` records the taxel order and the sensor frame.
2. :func:`urdf_to_usd`: ``franka_orca.urdf`` -> USD with Isaac Lab's URDF converter, plus a top layer
   ``franka_orca.usda`` holding the collision filter pairs (bodies, and the fingertip colliders, which meet only
   pad to pad and unsensed to unsensed) and keeping the fingertip colliders as triangle meshes. This is the file the
   robot config loads.

The fingertips collide as halltaxel simulates them, exactly as the sensing surface is, but a pair of the large pad or
unsensed meshes is slow to collide: a step of ``Isaac-Lift-FrankaOrca`` with 16 environments takes over 200 ms,
against 9 ms with convex hulls.

The inputs come from installed packages: the ORCA hand URDF and the taxel data from the halltaxel checkout (see
:data:`HALLTAXEL_DIR`), the FR3 arm from newton-assets. halltaxel is only needed here, at build time; nothing in the
environment imports it. It is not an Isaac Lab dependency, so install it from its checkout, again after a ``uv sync``,
which removes it.

Usage:

.. code-block:: bash

    uv pip install -e ../halltaxel                                    # once, and after every uv sync
    uv run python scripts/tools/franka_orca/build_asset.py            # all five fingers
    uv run python scripts/tools/franka_orca/build_asset.py --fingers thumb
    # without taxels, for comparison
    uv run python scripts/tools/franka_orca/build_asset.py --fingers \
        --out-dir assets/franka_orca_mesh_fingertips/no_taxels
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import newton.utils
import numpy as np
import trimesh
import warp as wp

try:
    import halltaxel
    from halltaxel import TactileSurfaceSpec, segment_surface
except ModuleNotFoundError as error:
    raise SystemExit("halltaxel is not installed; run `uv pip install -e <halltaxel checkout>` first") from error

from isaaclab_assets.robots.franka_orca import FRANKA_ORCA_ASSET_DIR, ORCA_FINGER_CHAINS

##
# Inputs
##

HALLTAXEL_DIR = Path(os.environ.get("HALLTAXEL_DIR", Path(halltaxel.__file__).resolve().parents[2]))
"""halltaxel checkout, whose examples hold the ORCA hand and the taxel data; the editable install's checkout unless
``$HALLTAXEL_DIR`` is set."""

HAND_URDF = HALLTAXEL_DIR / "examples/orcahand_real_hand/data/orcahand_description/v2/models/urdf/orcahand_right.urdf"
"""Right ORCA hand, including the ``<fingertip>_tactile_region`` fixture links halltaxel reads."""

TAXEL_DATA_DIR = HALLTAXEL_DIR / "examples/orcahand_fingertips/data"
"""Per finger: ``mesh.stl``, ``taxels.npy``, ``sensor_mount.npy`` and ``<finger>_sensor.stl``."""

FR3_PACKAGE = newton.utils.download_asset("franka_emika_panda")
"""Root of the ``franka_emika_panda`` package, at the newton-assets revision the installed Newton pins."""

FR3_URDF = FR3_PACKAGE / "urdf" / "fr3.urdf"
"""Franka FR3 arm URDF; ends at the ``fr3_link8`` flange, no gripper."""

##
# Robot layout
##

FINGERTIP_LINKS = {
    "thumb": "T_DP_b7429e50",
    "index": "I_FingerTipAssembly_ec49c16c",
    "middle": "M_FingerTipAssembly_34afb748",
    "ring": "M_FingerTipAssembly_424a8e75",
    "pinky": "P_FingerTipAssembly_cd219176",
}
"""Fingertip link of each sensed finger.

Link names here and below are the sanitized ones :func:`compose_urdf` writes (hyphens -> underscores)."""

HAND_MOUNT_XYZ = "0 0 0.0575"
HAND_MOUNT_RPY = "1.5707963 0 3.1415927"
"""Fixed joint ``fr3_link8`` -> hand root (``ForeArmStructure``), validated on the Newton path."""

HAND_INERTIA_SCALE = 1e-4
"""The hand URDF's ``<inertia>`` values are in kg cm^2, not kg m^2 (radii of gyration of 1-4 m on 2-5 cm links).
MuJoCo never read them, as the hand's MJCF computes inertia from the geoms."""

WRIST_LINKS = ["ForeArmStructure_Model_e18f2368", "TopTower_Model_4a80d30e", "R_Carpals_8d1f1041"]
"""Hand bodies at the flange, which permanently overlap the last arm links."""

ARM_EXCLUDES = [
    ("fr3_link0", "fr3_link2"),
    ("fr3_link1", "fr3_link3"),
    ("fr3_link2", "fr3_link4"),
    ("fr3_link3", "fr3_link5"),
    ("fr3_link5", "fr3_link7"),
]
"""Arm links whose hulls overlap at rest (Panda topology; re-check for the FR3 links)."""


def collision_excludes(fingers: list[str]) -> list[tuple[str, str]]:
    """Body pairs that never collide.

    Every pair of hand bodies is excluded except two sensed fingertips, which meet only through their pad proxies
    and unsensed meshes (see :func:`fingertip_collider_excludes`). So the rest of the fingers pass through each other,
    which is not physical, but fingertips pressed side by side would otherwise need over 1000 MJWarp constraint rows per
    environment, as every taxel touching the neighbouring finger is its own contact.

    Args:
        fingers: Sensed fingers, keys of :data:`FINGERTIP_LINKS`.
    """
    sensed_tips = {FINGERTIP_LINKS[finger] for finger in fingers}
    hand = WRIST_LINKS + [link for chain in ORCA_FINGER_CHAINS for link in chain]
    pairs = list(ARM_EXCLUDES)
    pairs += itertools.product(WRIST_LINKS, ["fr3_link5", "fr3_link6", "fr3_link7"])
    pairs += [(a, b) for a, b in itertools.combinations(hand, 2) if not (a in sensed_tips and b in sensed_tips)]
    return pairs


def fingertip_collider_excludes(colliders: dict[str, dict[str, list]]) -> list[tuple]:
    """Collider pairs of two sensed fingertips that never collide, so that only their pad proxies and unsensed meshes
    meet.

    The taxels skip each other through their shared collision group, and the pad proxies, in their own group, skip
    the unsensed meshes (see ``TactileSensorCfg``). What is left is filtered here: each pad proxy and each unsensed
    mesh against the other fingertip's taxels. A fingertip's taxels thus touch only the object, the table and the arm,
    and two fingertips touch pad to pad and unsensed to unsensed.

    Args:
        colliders: Per finger, its collider paths by kind: ``"taxels"``, ``"pad"`` and ``"unsensed"``.

    Returns:
        The collider path pairs.
    """
    pairs = []
    for a, b in itertools.permutations(colliders, 2):
        for kind in ("pad", "unsensed"):
            pairs += itertools.product(colliders[a][kind], colliders[b]["taxels"])
    return pairs


##
# Tactile bake
##

# Same defaults as halltaxel's ContactPatchBackend, so the patches match its runtime ones.
RELAX_ITERS = 0
SMOOTH_ITERS = 1
MAX_FRAME_ERROR = 0.05e-3  # m
"""halltaxel data further than this from the URDF visual mesh means the frames disagree."""
TAXEL_CONTACT_GAP = 2e-3  # m
"""Newton contact gap of the taxel and pad-proxy colliders, instead of Isaac Lab's 10 mm default. With hundreds of
taxels per hand, a 10 mm gap reports thousands of speculative contacts between neighbouring fingertips."""


def load_spec(finger: str) -> TactileSurfaceSpec:
    """Load a finger's halltaxel surface spec from :data:`TAXEL_DATA_DIR`."""
    data = TAXEL_DATA_DIR / finger
    region = data / f"{finger}_sensor.stl"
    return TactileSurfaceSpec.from_files(
        data / "mesh.stl",
        data / "taxels.npy",
        label_prefix=finger,
        sensor_region_path=region if region.is_file() else None,
        sensor_xform=wp.transform(*np.load(data / "sensor_mount.npy")),
    )


def bake_finger(finger: str, visual: trimesh.Trimesh) -> tuple[dict[str, trimesh.Trimesh], dict]:
    """Cut one fingertip into per-taxel mesh colliders, a pad proxy and an unsensed mesh.

    As in halltaxel's ``ContactPatchBackend``, each taxel collider is its patch of the sensing surface, an open
    triangle mesh, so the taxels are the fingertip's pad as it is. The rest of halltaxel's fingertip surface is the
    unsensed mesh ``<finger>_unsensed``; together they cover halltaxel's ``mesh.stl`` once, which stops about 5 mm
    short of the bottom of the fingertip link. The pad proxy ``<finger>_pad`` is the whole sensing surface, in the
    same place as the taxels. It only exists for other fingers' taxels to press against (see
    ``TactileSensorCfg.pad_collision_group``), so two pads touching give one contact pair per taxel rather than one
    per pair of taxels.

    Args:
        finger: Finger name, a key of :data:`FINGERTIP_LINKS`.
        visual: The fingertip's visual mesh in its link frame [m], to check the halltaxel data's frame.

    Returns:
        The colliders by name (``<finger>_taxel_NNN`` in taxel order, ``<finger>_pad``, ``<finger>_unsensed``),
        and the sidecar data: ``patch_names``, ``taxel_positions`` (N, 3) and ``sensor_xform`` (7,) in the link frame.
    """
    spec = load_spec(finger)
    _, dist, _ = trimesh.proximity.closest_point(visual, spec.mesh.vertices[::10])
    if np.median(dist) > MAX_FRAME_ERROR:
        raise RuntimeError(f"{finger}: halltaxel mesh.stl is {np.median(dist) * 1e3:.3f} mm off the URDF mesh")

    sensing = spec.sensing_mesh
    segmentation = segment_surface(
        sensing, seeds=spec.taxel_positions, relax_iters=RELAX_ITERS, smooth_iters=SMOOTH_ITERS
    )
    colliders = {}
    for patch_id in range(segmentation.num_regions):
        faces = np.flatnonzero(segmentation.labels == patch_id)
        if len(faces) == 0:
            raise ValueError(f"{finger}: patch {patch_id} has no faces")
        colliders[f"{finger}_taxel_{patch_id:03d}"] = sensing.submesh([faces], append=True)
    sidecar = {
        "patch_names": np.array(list(colliders)),
        "taxel_positions": segmentation.seeds.copy(),
        "sensor_xform": np.array(spec.sensor_xform, dtype=np.float64),
    }
    colliders[f"{finger}_pad"] = sensing
    if spec.unsensed_mesh is None:
        raise ValueError(f"{finger}: the sensing region covers the whole fingertip, nothing is left unsensed")
    colliders[f"{finger}_unsensed"] = spec.unsensed_mesh
    print(f"{finger}: {segmentation.num_regions} taxel patches")
    return colliders, sidecar


##
# Build steps
##


def compose_urdf(fingers: list[str], out_dir: Path) -> Path:
    """Mount the hand on the FR3 flange, add the hand colliders and bake the taxels, as one URDF.

    Args:
        fingers: Fingers to bake, keys of :data:`FINGERTIP_LINKS`.
        out_dir: Output directory; meshes go to ``meshes/``, sidecars to ``tactile/``.

    Returns:
        Path of ``franka_orca.urdf``.
    """
    mesh_dir = out_dir / "meshes"
    sidecar_dir = out_dir / "tactile"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    robot = ET.parse(FR3_URDF).getroot()
    robot.set("name", "franka_orca")
    resolve_mesh_paths(robot, {"franka_emika_panda": FR3_PACKAGE}, FR3_URDF.parent)

    hand = ET.parse(HAND_URDF).getroot()
    resolve_mesh_paths(hand, {"orcahand_description": HAND_URDF.parents[3]}, HAND_URDF.parent)
    # the tactile regions are read from the halltaxel data, not needed as bodies in the robot
    for element in list(hand):
        if element.get("name", "").endswith(("_tactile_region", "_tactile_region_joint")):
            hand.remove(element)

    for inertia in hand.iter("inertia"):
        for attr in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
            inertia.set(attr, repr(float(inertia.get(attr)) * HAND_INERTIA_SCALE))

    # USD identifiers have no hyphens; renamed here, the converter would escape them as ``tn__...``
    for element in hand.iter():
        for attr in ("name", "link"):
            if element.tag in ("link", "joint", "parent", "child") and element.get(attr):
                element.set(attr, sanitize(element.get(attr)))

    sensed = {FINGERTIP_LINKS[finger]: finger for finger in fingers}
    visuals = {sensed[link.get("name")]: visual_mesh(link) for link in hand.iter("link") if link.get("name") in sensed}
    for link in hand.iter("link"):
        name = link.get("name")
        if name in sensed:
            colliders, sidecar = bake_finger(sensed[name], visuals[sensed[name]])
            np.savez(sidecar_dir / f"{sensed[name]}.npz", body=name, **sidecar)
            for collider_name, mesh in colliders.items():
                mesh.export(mesh_dir / f"{collider_name}.stl")
                add_collision(link, collider_name, mesh_dir / f"{collider_name}.stl", scale=1.0)
        else:
            add_visual_collision(link, f"{name}_hull")

    robot.extend(hand.findall("material") + hand.findall("link") + hand.findall("joint"))
    mount = ET.SubElement(robot, "joint", name="orcahand_mount", type="fixed")
    ET.SubElement(mount, "parent", link="fr3_link8")
    ET.SubElement(mount, "child", link="ForeArmStructure_Model_e18f2368")
    ET.SubElement(mount, "origin", xyz=HAND_MOUNT_XYZ, rpy=HAND_MOUNT_RPY)

    urdf = out_dir / "franka_orca.urdf"
    ET.indent(robot)
    ET.ElementTree(robot).write(urdf, xml_declaration=True, encoding="utf-8")
    print(f"wrote {urdf}")
    return urdf


def urdf_to_usd(urdf: Path, fingers: list[str], out_dir: Path) -> Path:
    """Convert the combined URDF to USD and add a top layer with the collision filter pairs, the taxel gap and the
    fingertip mesh colliders.

    Args:
        urdf: The combined URDF from :func:`compose_urdf`.
        fingers: Sensed fingers, as baked by :func:`compose_urdf`.
        out_dir: Output directory.

    Returns:
        Path of ``franka_orca.usda``, the layer the robot config spawns.
    """
    from pxr import Sdf, Usd, UsdPhysics

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(urdf),
            usd_dir=str(out_dir / "usd"),
            force_usd_conversion=True,
            fix_base=True,
            merge_fixed_joints=False,
            self_collision=True,
            collision_from_visuals=False,
            joint_drive=None,  # gains come from the actuators in FRANKA_ORCA_CFG
        )
    )

    top = out_dir / "franka_orca.usda"
    layer = Sdf.Layer.CreateNew(str(top))
    layer.subLayerPaths.append(os.path.relpath(converter.usd_path, out_dir))
    # stage metadata is not inherited from sublayers; without it USD falls back to Y-up
    converted = Sdf.Layer.FindOrOpen(converter.usd_path)
    for key in ("upAxis", "metersPerUnit", "kilogramsPerUnit"):
        layer.pseudoRoot.SetInfo(key, converted.pseudoRoot.GetInfo(key))
    stage = Usd.Stage.Open(layer)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/franka_orca"))
    bodies = {prim.GetName(): prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    body_pairs = collision_excludes(fingers)
    colliders = {
        finger: fingertip_colliders(stage.GetPrimAtPath(bodies[FINGERTIP_LINKS[finger]])) for finger in fingers
    }
    collider_pairs = fingertip_collider_excludes(colliders)
    keep_triangle_meshes(stage, [path for kinds in colliders.values() for paths in kinds.values() for path in paths])
    for a, b in body_pairs:
        filter_collision(stage.GetPrimAtPath(bodies[a]), bodies[b])
    for a, b in collider_pairs:
        filter_collision(stage.GetPrimAtPath(a), b)
    num_taxels = set_sensor_contact_gap(stage)
    layer.Save()
    print(
        f"wrote {top} ({len(bodies)} bodies, {len(body_pairs)} filtered body pairs, {len(collider_pairs)} filtered"
        f" fingertip collider pairs, {num_taxels} taxels)"
    )
    return top


def fingertip_colliders(body) -> dict[str, list]:
    """Collider paths of a baked fingertip by kind, ``"taxels"``, ``"pad"`` and ``"unsensed"``, made editable.

    The converter instances each collision (a prim named after it, holding one mesh), and an instance cannot be
    edited, so these prims are made non-instanced in this layer. Fingertips with the same meshes share the
    instance, whose mesh keeps the first fingertip's name, so the mesh is looked up rather than named.
    """
    colliders = {"taxels": [], "pad": [], "unsensed": []}
    for prim in body.GetChildren():
        name = prim.GetName()
        kind = "taxels" if "_taxel_" in name else next((k for k in ("pad", "unsensed") if name.endswith(f"_{k}")), None)
        if kind is not None:
            prim.SetInstanceable(False)
            colliders[kind] += [mesh.GetPath() for mesh in prim.GetChildren()]
    return colliders


def filter_collision(prim, target) -> None:
    """Author ``target`` as a filtered pair of ``prim``; both may be rigid bodies or colliders."""
    from pxr import UsdPhysics

    UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(target)


def keep_triangle_meshes(stage, paths: list) -> None:
    """Author ``physics:approximation = "none"`` on the given mesh colliders.

    The converter marks every mesh collision ``convexHull``, which the simulator would replace by a hull of at most
    64 vertices. Newton's collision pipeline collides these as the triangle meshes they are instead, at a cost: a pair
    of large meshes, such as two pads, is far slower to collide than a pair of hulls.
    """
    from pxr import UsdPhysics

    for path in paths:
        UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(path)).GetApproximationAttr().Set(UsdPhysics.Tokens.none)


def set_sensor_contact_gap(stage) -> int:
    """Author :data:`TAXEL_CONTACT_GAP` as ``newton:contactGap`` on the taxel and pad-proxy colliders.

    The prims holding these colliders must be non-instanced already (see :func:`fingertip_colliders`).

    Returns:
        The number of taxels.
    """
    from pxr import Sdf, UsdPhysics

    # the prims named after the collisions, not the meshes they hold
    sensor_colliders = [
        prim
        for prim in stage.Traverse()
        if ("_taxel_" in prim.GetName() or prim.GetName().endswith("_pad"))
        and prim.GetParent().HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    for prim in sensor_colliders:
        for collider in prim.GetChildren():
            collider.CreateAttribute("newton:contactGap", Sdf.ValueTypeNames.Float).Set(TAXEL_CONTACT_GAP)
    return sum("_taxel_" in prim.GetName() for prim in sensor_colliders)


##
# URDF helpers
##


def sanitize(name: str) -> str:
    """Prim name the URDF converter gives a link (USD identifiers have no hyphens)."""
    return name.replace("-", "_")


def resolve_mesh_paths(robot: ET.Element, packages: dict[str, Path], urdf_dir: Path) -> None:
    """Rewrite ``package://`` and relative mesh filenames as absolute paths."""
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename")
        if filename.startswith("package://"):
            package, _, rest = filename.removeprefix("package://").partition("/")
            path = packages[package] / rest
        else:
            path = urdf_dir / filename
        mesh.set("filename", str(path.resolve()))


def visual_mesh(link: ET.Element) -> trimesh.Trimesh:
    """Return a link's single visual mesh in the link frame [m]."""
    (visual,) = link.findall("visual")
    mesh_element = visual.find("geometry/mesh")
    mesh = trimesh.load(mesh_element.get("filename"), force="mesh", process=False)
    mesh.apply_scale([float(s) for s in mesh_element.get("scale", "1 1 1").split()])
    origin = visual.find("origin")
    if origin is not None:
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        transform = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
        transform[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        mesh.apply_transform(transform)
    return mesh


def add_collision(link: ET.Element, name: str, mesh_path: Path, scale: float) -> None:
    """Append a named mesh ``<collision>`` in the link frame."""
    collision = ET.SubElement(link, "collision", name=name)
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "mesh", filename=str(mesh_path), scale=f"{scale} {scale} {scale}")


def add_visual_collision(link: ET.Element, name: str) -> None:
    """Append a named ``<collision>`` that is the link's visual mesh.

    The converter marks it ``convexHull``, so the simulator collides with the mesh's convex hull, which it computes
    at import (Newton caps it at 64 vertices). A hull fills concave parts, such as the palm's hollow.
    """
    (visual,) = link.findall("visual")
    collision = ET.SubElement(link, "collision", name=name)
    collision.extend(copy.deepcopy(child) for child in visual if child.tag in ("origin", "geometry"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fingers", nargs="*", default=["all"], help="Fingers to bake, 'all' (default), or none.")
    parser.add_argument("--out-dir", type=Path, default=FRANKA_ORCA_ASSET_DIR)
    args = parser.parse_args()
    fingers = list(FINGERTIP_LINKS) if args.fingers == ["all"] else args.fingers

    # the URDF names the generated meshes by path, which must not depend on the working directory
    out_dir = args.out_dir.resolve()
    urdf = compose_urdf(fingers, out_dir)
    urdf_to_usd(urdf, fingers, out_dir)


if __name__ == "__main__":
    main()
