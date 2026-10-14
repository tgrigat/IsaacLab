# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the Franka (FR3) + ORCA-hand robot asset, with one of several fingertip collision methods.

Everything is built from the sources vendored under ``assets/franka_orca/sources/`` (see its README) into
``assets/franka_orca/<method>/``, both tracked in git, so the environments need no build. Two steps:

1. :func:`compose_urdf`: FR3 URDF + ORCA right-hand URDF -> ``franka_orca.urdf``. The hand links collide
   through their visual meshes, which the simulator imports as convex hulls, and each sensed fingertip is cut by
   halltaxel into one collision per taxel (``<finger>_taxel_NNN``), a pad proxy (``<finger>_pad``) and the unsensed
   rest (``<finger>_unsensed*``), as the fingertip method says. A sidecar ``tactile/<finger>.npz`` records the taxel
   order and the sensor frame.
2. :func:`urdf_to_usd`: ``franka_orca.urdf`` -> USD with Isaac Lab's URDF converter, plus a top layer
   ``franka_orca.usda`` holding the collision filter pairs (bodies, and the fingertip colliders, which meet only
   pad to pad and unsensed to unsensed) and the colliders kept as triangle meshes. This is the file the robot config
   loads.

The fingertip methods (:data:`FINGERTIP_METHODS`), from cheapest to most exact; the step times are of
``Isaac-Lift-FrankaOrca-*`` with 16 environments:

* ``primitives``: no taxels, and every link collides as one capsule or box fitted to its meshes, for coarse training.
* ``no_taxels``: no taxels, each fingertip collides as one convex hull of its visual mesh.
* ``convex_taxels``: each taxel is a convex solid, its patch of the sensing surface thickened by
  :data:`PATCH_THICKNESS`; the pad proxy is the same for the whole pad, and the rest of the fingertip is one convex
  hull. About 9 ms per step, but the sensing surface is bumpy where the solver hulls a taxel to 64 vertices.
* ``mesh_taxels``: each taxel is its patch of the sensing surface as a triangle mesh, as in halltaxel's
  ``ContactPatchBackend``; the rest of the fingertip is a CoACD decomposition into up to :data:`COACD_MAX_HULLS`
  hulls. About 10 ms per step.
* ``mesh_fingertips``: as ``mesh_taxels``, but the rest of the fingertip is its visual mesh without the pad, kept as a
  triangle mesh, and so is the pad proxy. The fingertips collide exactly as they look, but a pair of these large
  meshes is slow to collide: about 760 ms per step.

Usage:

.. code-block:: bash

    uv run python scripts/tools/franka_orca/build_asset.py                         # every method
    uv run python scripts/tools/franka_orca/build_asset.py --fingertips mesh_taxels
    uv run python scripts/tools/franka_orca/build_asset.py --fingertips mesh_taxels --fingers thumb \
        --out-dir /tmp/franka_orca_thumb
"""

from __future__ import annotations

import argparse
import copy
import itertools
import multiprocessing
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import warp as wp
from halltaxel import TactileSurfaceSpec, segment_surface, select_region

from isaaclab_assets.robots.franka_orca import FINGERTIP_METHODS, FRANKA_ORCA_ASSET_DIR, ORCA_FINGER_CHAINS

##
# Inputs
##

SOURCES_DIR = FRANKA_ORCA_ASSET_DIR / "sources"
"""Vendored inputs of the build, see ``assets/franka_orca/README.md``."""

HAND_URDF = SOURCES_DIR / "orcahand_description/v2/models/urdf/orcahand_right.urdf"
"""Right ORCA hand, including the ``<fingertip>_tactile_region`` fixture links halltaxel reads."""

TAXEL_DATA_DIR = SOURCES_DIR / "fingertip_taxels"
"""Per finger: ``mesh.stl``, ``taxels.npy``, ``sensor_mount.npy`` and ``<finger>_sensor.stl``."""

FR3_PACKAGE = SOURCES_DIR / "franka_emika_panda"
"""Root of the ``franka_emika_panda`` package of newton-assets."""

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

BOX_LINKS = ["fr3_link0", "fr3_link4", "fr3_link6", *WRIST_LINKS]
"""Links that :func:`replace_with_primitives` collides as a box: the arm base, the L-shaped elbow and wrist links, whose
capsules would be up to 10 cm thick, and the blocks at the flange and the palm. So are the fingers' abduction links
(``_AP_``), short knuckle blocks whose capsules would reach into the neighbouring fingers'. Every other link, an arm
segment or a finger phalanx, is long and round enough for a capsule."""


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
PATCH_THICKNESS = 1.5e-3  # m
"""Inward depth of the ``convex_taxels`` patch solids: well above the solver's penetration under fingertip loads, and
inside the gap down to the unsensed hull."""
UNSENSED_INSET = 1.5e-3  # m
"""Depth the sensed pad is pushed inward by before the ``mesh_taxels`` fingertip is decomposed, so the hulls stay
below it."""
COACD_THRESHOLD = 0.05
"""CoACD concavity threshold; lower gives more, tighter hulls."""
COACD_MAX_HULLS = 16
"""Most ``mesh_taxels`` unsensed hulls per fingertip."""
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


def pad_mask(visual: trimesh.Trimesh, spec: TactileSurfaceSpec) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Return the fingertip's visual mesh with merged vertices and the mask of its faces on the sensed pad."""
    mesh = visual.copy()
    mesh.merge_vertices()  # keeps the faces in order, so the pad's face mask still applies
    return mesh, select_region(mesh, spec.sensor_region if spec.sensor_region is not None else spec.mesh)


def convex_patch(patch: trimesh.Trimesh, inside: np.ndarray) -> trimesh.Trimesh:
    """Return the convex hull of a patch surface and its copy offset :data:`PATCH_THICKNESS` towards ``inside``."""
    outward = np.sign(patch.face_normals.mean(axis=0) @ (patch.vertices.mean(axis=0) - inside)) or 1.0
    inner = patch.vertices - outward * patch.vertex_normals * PATCH_THICKNESS
    return trimesh.convex.convex_hull(np.vstack([patch.vertices, inner]))


def decompose_fingers(visuals: dict[str, trimesh.Trimesh]) -> dict[str, list[trimesh.Trimesh]]:
    """Run :func:`decompose_unsensed` for each finger, in parallel.

    CoACD gives a different decomposition on every run when it runs multithreaded, even with a fixed seed. Each
    finger is therefore decomposed in its own single-threaded process, so the build is reproducible and still takes
    about as long as the slowest finger.

    Args:
        visuals: Per finger to decompose (a key of :data:`FINGERTIP_LINKS`), its fingertip's visual mesh in its
            link frame [m].

    Returns:
        The hulls of each finger.
    """
    from concurrent.futures import ProcessPoolExecutor

    # the workers inherit the environment when they start, before anything in them loads OpenMP, which CoACD runs on
    threads = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(max_workers=len(visuals), mp_context=multiprocessing.get_context("spawn")) as pool:
            return dict(zip(visuals, pool.map(decompose_unsensed, visuals, visuals.values())))
    finally:
        if threads is None:
            del os.environ["OMP_NUM_THREADS"]
        else:
            os.environ["OMP_NUM_THREADS"] = threads


def decompose_unsensed(finger: str, visual: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Decompose a fingertip into convex hulls that follow its unsensed surface.

    CoACD runs on the fingertip's whole visual mesh, which reaches further than halltaxel's ``mesh.stl``, down to the
    joint, with the sensed pad pushed :data:`UNSENSED_INSET` inward, so the hulls stay below the taxels. They are not
    exactly below: CoACD remeshes its input and hulls are wider than the parts they wrap, so near the pad's rim some
    reach up to about 1 mm above the pad, where they take contacts the taxels would otherwise sense. The simulator
    reduces each hull to at most 64 vertices when it imports it.

    Args:
        finger: Finger name, a key of :data:`FINGERTIP_LINKS`.
        visual: The fingertip's visual mesh in its link frame [m].

    Returns:
        The hulls, in the link frame [m].
    """
    import coacd

    mesh, pad = pad_mask(visual, load_spec(finger))
    # orient the vertex normals outward, away from the fingertip's middle
    pad_outward = mesh.face_normals[pad].mean(axis=0) @ (mesh.triangles_center[pad].mean(axis=0) - mesh.centroid)
    outward = 1.0 if pad_outward > 0 else -1.0
    pad_vertices = np.unique(mesh.faces[pad])
    vertices = mesh.vertices.copy()
    vertices[pad_vertices] -= outward * mesh.vertex_normals[pad_vertices] * UNSENSED_INSET
    parts = coacd.run_coacd(
        coacd.Mesh(vertices, mesh.faces),
        threshold=COACD_THRESHOLD,
        max_convex_hull=COACD_MAX_HULLS,
        preprocess_mode="on",  # the visual mesh is a CAD assembly, neither watertight nor a single shell
        seed=0,
    )
    hulls = [trimesh.Trimesh(part_vertices, part_faces).convex_hull for part_vertices, part_faces in parts]
    print(f"{finger}: {len(hulls)} unsensed hulls")
    return hulls


def bake_finger(
    finger: str, visual: trimesh.Trimesh, method: str, unsensed_hulls: list[trimesh.Trimesh] | None = None
) -> tuple[dict[str, trimesh.Trimesh], dict]:
    """Cut one fingertip into per-taxel colliders, a pad proxy and unsensed colliders, as ``method`` says.

    The taxels are the patches of halltaxel's sensing surface, as in its ``ContactPatchBackend``: open triangle meshes
    (``mesh_taxels``, ``mesh_fingertips``) or solids thickened inward (``convex_taxels``). The pad proxy
    ``<finger>_pad`` is the whole sensing surface, in the same place as the taxels. It only exists for other fingers'
    taxels to press against (see ``TactileSensorCfg.pad_collision_group``), so two pads touching give one contact pair
    per taxel rather than one per pair of taxels. The unsensed colliders are the rest of the fingertip's visual mesh,
    which reaches further than halltaxel's ``mesh.stl``, the skin shell: down to the joint, with the nail and, as the
    visual mesh is a CAD assembly, the inner parts. They are one convex hull ``<finger>_unsensed`` (``convex_taxels``),
    the CoACD hulls ``<finger>_unsensed_NN`` (``mesh_taxels``) or the mesh itself ``<finger>_unsensed``
    (``mesh_fingertips``).

    Args:
        finger: Finger name, a key of :data:`FINGERTIP_LINKS`.
        visual: The fingertip's visual mesh in its link frame [m].
        method: Fingertip method with taxels, see :data:`FINGERTIP_METHODS`.
        unsensed_hulls: The fingertip's hulls from :func:`decompose_unsensed`, for ``mesh_taxels``.

    Returns:
        The colliders by name (``<finger>_taxel_NNN`` in taxel order, ``<finger>_pad``, ``<finger>_unsensed*``),
        and the sidecar data: ``patch_names``, ``taxel_positions`` (N, 3) and ``sensor_xform`` (7,) in the link frame.
    """
    spec = load_spec(finger)
    _, dist, _ = trimesh.proximity.closest_point(visual, spec.mesh.vertices[::10])
    if np.median(dist) > MAX_FRAME_ERROR:
        raise RuntimeError(f"{finger}: halltaxel mesh.stl is {np.median(dist) * 1e3:.3f} mm off the URDF mesh")

    mesh, pad = pad_mask(visual, spec)
    unsensed = mesh.submesh([np.flatnonzero(~pad)], append=True)
    unsensed_hull = unsensed.convex_hull
    sensing = spec.sensing_mesh
    segmentation = segment_surface(
        sensing, seeds=spec.taxel_positions, relax_iters=RELAX_ITERS, smooth_iters=SMOOTH_ITERS
    )
    colliders = {}
    for patch_id in range(segmentation.num_regions):
        faces = np.flatnonzero(segmentation.labels == patch_id)
        if len(faces) == 0:
            raise ValueError(f"{finger}: patch {patch_id} has no faces")
        patch = sensing.submesh([faces], append=True)
        name = f"{finger}_taxel_{patch_id:03d}"
        colliders[name] = convex_patch(patch, unsensed_hull.centroid) if method == "convex_taxels" else patch
    sidecar = {
        "patch_names": np.array(list(colliders)),
        "taxel_positions": segmentation.seeds.copy(),
        "sensor_xform": np.array(spec.sensor_xform, dtype=np.float64),
    }
    if method == "convex_taxels":
        colliders[f"{finger}_pad"] = convex_patch(sensing, unsensed_hull.centroid)
        colliders[f"{finger}_unsensed"] = unsensed_hull
    elif method == "mesh_taxels":
        colliders[f"{finger}_pad"] = sensing
        for i, hull in enumerate(unsensed_hulls):
            colliders[f"{finger}_unsensed_{i:02d}"] = hull
    else:
        colliders[f"{finger}_pad"] = sensing
        colliders[f"{finger}_unsensed"] = unsensed
    print(f"{finger}: {segmentation.num_regions} taxel patches")
    return colliders, sidecar


##
# Build steps
##


def compose_urdf(fingers: list[str], method: str, out_dir: Path) -> Path:
    """Mount the hand on the FR3 flange, add the hand colliders and bake the taxels, as one URDF.

    Mesh paths in the URDF are relative to it, so the build can move with the repository.

    Args:
        fingers: Fingers to bake, keys of :data:`FINGERTIP_LINKS`.
        method: Fingertip method of the sensed fingers, see :data:`FINGERTIP_METHODS`.
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
    resolve_mesh_paths(robot, {"franka_emika_panda": FR3_PACKAGE}, FR3_URDF.parent, out_dir)

    hand = ET.parse(HAND_URDF).getroot()
    resolve_mesh_paths(hand, {"orcahand_description": HAND_URDF.parents[3]}, HAND_URDF.parent, out_dir)
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
    visuals = {
        sensed[link.get("name")]: visual_mesh(link, out_dir) for link in hand.iter("link") if link.get("name") in sensed
    }
    unsensed_hulls = decompose_fingers(visuals) if visuals and method == "mesh_taxels" else {}
    for link in hand.iter("link"):
        name = link.get("name")
        if name in sensed:
            finger = sensed[name]
            colliders, sidecar = bake_finger(finger, visuals[finger], method, unsensed_hulls.get(finger))
            np.savez(sidecar_dir / f"{finger}.npz", body=name, **sidecar)
            for collider_name, mesh in colliders.items():
                mesh.export(mesh_dir / f"{collider_name}.stl")
                add_collision(link, collider_name, Path("meshes") / f"{collider_name}.stl", scale=1.0)
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


def urdf_to_usd(urdf: Path, fingers: list[str], method: str, out_dir: Path) -> Path:
    """Convert the combined URDF to USD and add a top layer with the collision filter pairs, the taxel gap and the
    fingertip mesh colliders.

    Args:
        urdf: The combined URDF from :func:`compose_urdf`.
        fingers: Sensed fingers, as baked by :func:`compose_urdf`.
        method: Fingertip method, see :data:`FINGERTIP_METHODS`.
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
    # which colliders the simulator takes as they are, instead of as their convex hulls
    mesh_kinds = {"mesh_taxels": ["taxels"], "mesh_fingertips": ["taxels", "pad", "unsensed"]}.get(method, [])
    keep_triangle_meshes(stage, [path for kinds in colliders.values() for kind in mesh_kinds for path in kinds[kind]])
    for a, b in body_pairs:
        filter_collision(stage.GetPrimAtPath(bodies[a]), bodies[b])
    for a, b in collider_pairs:
        filter_collision(stage.GetPrimAtPath(a), b)
    num_taxels = set_sensor_contact_gap(stage)
    if method == "primitives":
        replace_with_primitives(stage, bodies)
    layer.Save()
    strip_build_paths(out_dir)
    print(
        f"wrote {top} ({len(bodies)} bodies, {len(body_pairs)} filtered body pairs, {len(collider_pairs)} filtered"
        f" fingertip collider pairs, {num_taxels} taxels)"
    )
    return top


def strip_build_paths(out_dir: Path) -> None:
    """Clear the converter's layer docs that name the build's absolute paths, so the build is the same on every
    machine."""
    from pxr import Sdf

    for path in sorted((out_dir / "usd").rglob("*.usda")):
        layer = Sdf.Layer.FindOrOpen(str(path))
        if str(out_dir) in layer.documentation:
            layer.documentation = ""
            layer.Save()


def replace_with_primitives(stage, bodies: dict[str, Path]) -> None:
    """Collide every link as one primitive fitted to its collision meshes, instead of the meshes.

    The links in :data:`BOX_LINKS` and the fingers' abduction links get the oriented bounding box of their meshes'
    vertices, every other link a capsule along the longest axis of that box, with its ends at the box's ends. A finger
    capsule is as wide as the box's narrower side: the phalanges are about 18 by 21 mm across, so a capsule as wide as
    the wider side would be wider than the finger, and curled fingers lying side by side would overlap. An arm capsule
    is as wide as the wider side, so the arm keeps its clearance from the table. The mesh colliders are deactivated in
    this layer. MJWarp collides primitive pairs in closed form, with fewer contacts than pairs of convex hulls.

    Args:
        stage: Stage of the top layer, with the converter's mesh colliders.
        bodies: Rigid body prim paths by link name.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    xform_cache = UsdGeom.XformCache()
    for name, body_path in bodies.items():
        body = stage.GetPrimAtPath(body_path)
        to_body = xform_cache.GetLocalToWorldTransform(body).GetInverse()
        points = []
        # the converter puts each collision in a child prim of the body, holding the (instanced) collision mesh;
        # other children hold the child bodies, some through a massless link such as fr3_link8
        for child in body.GetChildren():
            subtree = list(Usd.PrimRange(child, Usd.TraverseInstanceProxies()))
            if any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in subtree):
                continue
            meshes = [prim for prim in subtree if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh)]
            if not meshes:
                continue
            for mesh in meshes:
                to_body_mesh = xform_cache.GetLocalToWorldTransform(mesh) * to_body
                points += [to_body_mesh.Transform(Gf.Vec3d(p)) for p in UsdGeom.Mesh(mesh).GetPointsAttr().Get()]
            child.SetActive(False)
        if not points:
            continue
        points = np.asarray(points)
        # oriented_bounds maps the points into a frame centred on the box, along its axes
        to_box, extents = trimesh.bounds.oriented_bounds(points)
        box_to_body = np.linalg.inv(to_box)
        if name in BOX_LINKS or "_AP_" in name:
            shape = UsdGeom.Cube.Define(stage, body_path.AppendChild(f"{name}_box"))
            shape.CreateSizeAttr(1.0)
            scale = Gf.Vec3f(*extents)
        else:
            # rotate the box frame so its longest axis is the capsule's z axis
            axis = int(np.argmax(extents))
            order = [(axis + 1) % 3, (axis + 2) % 3, axis]
            permute = np.eye(4)
            permute[:3, :3] = np.eye(3)[:, order]
            box_to_body = box_to_body @ permute
            width = max if name.startswith("fr3_") else min
            radius = float(width(extents[order[0]], extents[order[1]])) / 2.0
            shape = UsdGeom.Capsule.Define(stage, body_path.AppendChild(f"{name}_capsule"))
            shape.CreateAxisAttr(UsdGeom.Tokens.z)
            shape.CreateRadiusAttr(radius)
            shape.CreateHeightAttr(max(float(extents[axis]) - 2.0 * radius, 0.0))
            scale = Gf.Vec3f(1.0)
        rotation = Gf.Matrix3d(*box_to_body[:3, :3].T.flatten().tolist()).ExtractRotation().GetQuat()
        shape.AddTranslateOp().Set(Gf.Vec3d(*box_to_body[:3, 3]))
        shape.AddOrientOp().Set(Gf.Quatf(rotation))
        shape.AddScaleOp().Set(scale)
        shape.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())


def fingertip_colliders(body) -> dict[str, list]:
    """Collider paths of a baked fingertip by kind, ``"taxels"``, ``"pad"`` and ``"unsensed"``, made editable.

    The converter instances each collision (a prim named after it, holding one mesh), and an instance cannot be
    edited, so these prims are made non-instanced in this layer. Fingertips with the same meshes share the
    instance, whose mesh keeps the first fingertip's name, so the mesh is looked up rather than named.
    """
    colliders = {"taxels": [], "pad": [], "unsensed": []}
    for prim in body.GetChildren():
        name = prim.GetName()
        if "_taxel_" in name:
            kind = "taxels"
        elif name.endswith("_pad"):
            kind = "pad"
        elif re.search(r"_unsensed(_\d+)?$", name):  # one mesh or hull, or CoACD hulls
            kind = "unsensed"
        else:
            kind = None
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


def resolve_mesh_paths(robot: ET.Element, packages: dict[str, Path], urdf_dir: Path, out_dir: Path) -> None:
    """Rewrite ``package://`` and relative mesh filenames as paths relative to ``out_dir``, where the URDF goes."""
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename")
        if filename.startswith("package://"):
            package, _, rest = filename.removeprefix("package://").partition("/")
            path = packages[package] / rest
        else:
            path = urdf_dir / filename
        mesh.set("filename", os.path.relpath(path.resolve(), out_dir))


def visual_mesh(link: ET.Element, base_dir: Path) -> trimesh.Trimesh:
    """Return a link's single visual mesh in the link frame [m]; its filename is relative to ``base_dir``."""
    (visual,) = link.findall("visual")
    mesh_element = visual.find("geometry/mesh")
    mesh = trimesh.load(base_dir / mesh_element.get("filename"), force="mesh", process=False)
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


def build(method: str, fingers: list[str], out_dir: Path) -> None:
    """Build the asset with one fingertip method into a fresh ``out_dir``."""
    if method in ("no_taxels", "primitives"):
        fingers = []
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    urdf = compose_urdf(fingers, method, out_dir)
    urdf_to_usd(urdf, fingers, method, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fingertips", nargs="+", choices=FINGERTIP_METHODS, default=list(FINGERTIP_METHODS), help="Methods to build."
    )
    parser.add_argument("--fingers", nargs="+", choices=FINGERTIP_LINKS, default=list(FINGERTIP_LINKS))
    parser.add_argument("--out-dir", type=Path, help="Output directory, with a single method.")
    args = parser.parse_args()
    if args.out_dir and len(args.fingertips) > 1:
        parser.error("--out-dir takes a single --fingertips method")

    for method in args.fingertips:
        out_dir = (args.out_dir or FRANKA_ORCA_ASSET_DIR / method).resolve()
        print(f"== {method} -> {out_dir}")
        build(method, args.fingers, out_dir)


if __name__ == "__main__":
    main()
