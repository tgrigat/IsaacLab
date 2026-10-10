# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton tactile sensor.

A pad carries a 2x2 grid of taxel colliders (``<name>_taxel_000`` .. ``_003``) and a pad proxy ``<name>_pad`` covering
them, like a fingertip baked by halltaxel.

* A box dropped onto one taxel checks the taxel order, the sensor frame and the sign convention (reaction force); the
  sidecar lists the taxels in reverse order and rotates the sensor frame.
* A second pad resting face-down on the first checks pad-to-pad contact: each side's taxels press on the other side's
  pad proxy, and with the finger-contact correction each side reads the full weight.
"""

# pyright: reportPrivateUsage=none

import sys
from pathlib import Path

from isaaclab.test.utils import test_devices

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sensors import TactileSensor, TactileSensorCfg
from isaaclab_newton.sensors.tactile_sensor.kernels import taxel_reaction_forces_kernel
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from physics.physics_test_utils import make_sim_cfg, perform_sim_step

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils import configclass

SIM_DT = 1.0 / 120.0
GRAVITY = 9.81
TAXEL_SIZE = 0.05  # m, edge of the square taxel colliders
TAXEL_PITCH = 0.06  # m, between taxel centres
LAYER_THICKNESS = 0.01  # m, of the taxels and the pad proxy
HALF = math.sqrt(0.5)


def taxel_names(name: str) -> list[str]:
    """Taxels of a pad, in the order they are authored."""
    return [f"{name}_taxel_{i:03d}" for i in range(4)]


def taxel_xy(index: int) -> tuple[float, float]:
    """Centre of the ``index``-th taxel [m], in a 2x2 grid around the pad origin."""
    return (index % 2 - 0.5) * TAXEL_PITCH, (index // 2 - 0.5) * TAXEL_PITCH


def write_pad(
    directory: Path, name: str, mass: float, kinematic: bool, sidecar_order: list[str], sensor_quat: tuple
) -> tuple[Path, Path]:
    """Write a pad body with a 2x2 grid of taxel colliders and a pad proxy covering them, and its sidecar.

    Args:
        directory: Output directory.
        name: Prefix of the taxel and pad-proxy names.
        mass: Body mass [kg].
        kinematic: Whether the body is kinematic.
        sidecar_order: Taxel order in the sidecar.
        sensor_quat: Sensor frame rotation ``(qx, qy, qz, qw)`` in the body frame.

    Returns:
        Paths of the pad USD and the sidecar.
    """
    usd_path = directory / f"{name}.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    pad = UsdGeom.Xform.Define(stage, "/Pad").GetPrim()
    stage.SetDefaultPrim(pad)
    UsdPhysics.RigidBodyAPI.Apply(pad).CreateKinematicEnabledAttr(kinematic)
    UsdPhysics.MassAPI.Apply(pad).CreateMassAttr(mass)

    def add_collider(prim_name: str, xy: tuple[float, float], size_xy: float):
        UsdGeom.Xform.Define(stage, f"/Pad/{prim_name}").AddTranslateOp().Set(Gf.Vec3d(*xy, 0.0))
        shape = UsdGeom.Cube.Define(stage, f"/Pad/{prim_name}/shape")
        shape.CreateSizeAttr(1.0)
        shape.AddScaleOp().Set(Gf.Vec3f(size_xy, size_xy, LAYER_THICKNESS))
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())

    for index, taxel in enumerate(taxel_names(name)):
        add_collider(taxel, taxel_xy(index), TAXEL_SIZE)
    add_collider(f"{name}_pad", (0.0, 0.0), TAXEL_PITCH + TAXEL_SIZE)
    stage.GetRootLayer().Save()

    sidecar_path = directory / f"{name}.npz"
    names = taxel_names(name)
    np.savez(
        sidecar_path,
        body="Pad",
        patch_names=np.array(sidecar_order),
        taxel_positions=np.array([(*taxel_xy(names.index(taxel)), 0.0) for taxel in sidecar_order]),
        sensor_xform=np.array([0.0, 0.0, 0.0, *sensor_quat]),  # [x, y, z, qx, qy, qz, qw]
    )
    return usd_path, sidecar_path


def pad_cfg(usd_path: Path, prim_name: str, pos: tuple, rot: tuple = (0.0, 0.0, 0.0, 1.0)) -> RigidObjectCfg:
    """A pad written by :func:`write_pad`, subject to gravity unless kinematic."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=[PhysxRigidBodyCfg(disable_gravity=False, linear_damping=0.5, angular_damping=0.5)],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def settled(sim, scene, sensors: list[TactileSensor], steps: int = 120, window: int = 20) -> list[torch.Tensor]:
    """Step the scene and return each sensor's ``forces_s`` averaged over the last ``window`` steps."""
    samples = [[] for _ in sensors]
    for step in range(steps):
        perform_sim_step(sim, scene, SIM_DT)
        if step >= steps - window:
            for sample, sensor in zip(samples, sensors):
                sample.append(sensor.data.forces_s.torch.clone())
    return [torch.stack(sample).mean(dim=0) for sample in samples]


@configclass
class TactileTestSceneCfg(InteractiveSceneCfg):
    """Up to two pads, a box, and a tactile sensor per pad."""

    pad_a: RigidObjectCfg | None = None
    pad_b: RigidObjectCfg | None = None
    box: RigidObjectCfg | None = None
    tactile_a: TactileSensorCfg | None = None
    tactile_b: TactileSensorCfg | None = None


@pytest.mark.parametrize("device", test_devices())
def test_box_resting_on_one_taxel(device: str, tmp_path: Path):
    """A box resting on one taxel loads only that taxel, with its weight along the sensor's +y.

    The sensor frame is rotated 90 deg about x, so the pad's +z (up) is the sensor's +y. The box is in the default
    collision group, so it ignores the pad proxy.
    """
    num_envs = 2
    mass = 0.5
    sidecar_order = taxel_names("a")[::-1]
    loaded = 2  # the authored index of the loaded taxel
    usd_path, sidecar_path = write_pad(tmp_path, "a", 1.0, True, sidecar_order, (HALF, 0.0, 0.0, HALF))

    sim_cfg = make_sim_cfg(device=device, gravity=(0.0, 0.0, -GRAVITY))
    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = TactileTestSceneCfg(num_envs=num_envs, env_spacing=1.0, lazy_sensor_update=False)
        scene_cfg.pad_a = pad_cfg(usd_path, "PadA", pos=(0.0, 0.0, 0.2))
        scene_cfg.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=[PhysxRigidBodyCfg(disable_gravity=False, linear_damping=0.5, angular_damping=0.5)],
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                mass_props=sim_utils.MassCfg(mass=mass),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(*taxel_xy(loaded), 0.24)),
        )
        scene_cfg.tactile_a = TactileSensorCfg(
            prim_path="{ENV_REGEX_NS}/PadA", sidecar_path=str(sidecar_path), pad_collision_group=2
        )

        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()
        sensor: TactileSensor = scene["tactile_a"]
        assert sensor.taxel_names == sidecar_order
        model = NewtonManager.get_model()
        groups = dict(zip(model.shape_label, model.shape_collision_group.numpy()))
        assert {int(g) for label, g in groups.items() if "_taxel_" in label} == {-1}
        assert {int(g) for label, g in groups.items() if "_pad/" in label} == {2}

        (forces,) = settled(sim, scene, [sensor])  # (num_envs, num_taxels, 3)
        row = sidecar_order.index(taxel_names("a")[loaded])
        expected = torch.tensor([0.0, mass * GRAVITY, 0.0], device=forces.device)
        others = [i for i in range(len(sidecar_order)) if i != row]
        for env in range(num_envs):
            torch.testing.assert_close(forces[env, row], expected, atol=0.05 * mass * GRAVITY, rtol=0.0)
            assert forces[env, others].abs().max() < 1e-3, f"env {env}: unloaded taxels report {forces[env, others]}"


@pytest.mark.parametrize("device", test_devices())
def test_pad_resting_on_pad(device: str, tmp_path: Path):
    """A pad resting face-down on another: each side's taxels together read the full weight, normal to the pad.

    The taxels of the two pads skip each other; pad B's taxels press on pad A's proxy and pad A's taxels on pad B's.
    Each side carries about half of the weight, which ``finger_contact_scale=2`` restores.
    """
    num_envs = 2
    mass = 0.5
    usd_a, sidecar_a = write_pad(tmp_path, "a", 1.0, True, taxel_names("a"), (0.0, 0.0, 0.0, 1.0))
    usd_b, sidecar_b = write_pad(tmp_path, "b", mass, False, taxel_names("b"), (0.0, 0.0, 0.0, 1.0))
    pads = ["{ENV_REGEX_NS}/Pad.*/.*_pad/.*"]

    sim_cfg = make_sim_cfg(device=device, gravity=(0.0, 0.0, -GRAVITY))
    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = TactileTestSceneCfg(num_envs=num_envs, env_spacing=1.0, lazy_sensor_update=False)
        scene_cfg.pad_a = pad_cfg(usd_a, "PadA", pos=(0.0, 0.0, 0.2))
        # flipped 180 deg about x, so its layer faces down onto pad A's
        scene_cfg.pad_b = pad_cfg(usd_b, "PadB", pos=(0.0, 0.0, 0.2 + LAYER_THICKNESS + 0.002), rot=(1.0, 0, 0, 0))
        scene_cfg.tactile_a = TactileSensorCfg(
            prim_path="{ENV_REGEX_NS}/PadA",
            sidecar_path=str(sidecar_a),
            pad_collision_group=2,
            filter_shape_prim_expr=pads,
        )
        scene_cfg.tactile_b = TactileSensorCfg(
            prim_path="{ENV_REGEX_NS}/PadB",
            sidecar_path=str(sidecar_b),
            pad_collision_group=3,
            filter_shape_prim_expr=pads,
        )

        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()
        forces_a, forces_b = settled(sim, scene, [scene["tactile_a"], scene["tactile_b"]])

        # both sensor frames are their pad's frame, and both pads push on the other along their own +z
        expected = torch.tensor([0.0, 0.0, mass * GRAVITY], device=forces_a.device)
        for env in range(num_envs):
            for side, forces in (("a", forces_a), ("b", forces_b)):
                torch.testing.assert_close(
                    forces[env].sum(dim=0),
                    expected,
                    atol=0.1 * mass * GRAVITY,
                    rtol=0.0,
                    msg=lambda m, side=side, env=env: f"pad {side}, env {env}: {m}",
                )


def test_diverged_environment_reads_no_force():
    """A non-finite shape pose (a diverged environment before its reset is simulated) yields zero, not NaN."""
    nan = float("nan")
    identity = wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))
    diverged = wp.transformf(wp.vec3f(nan), wp.quatf(nan, nan, nan, nan))
    forces_s = wp.zeros((2, 1), dtype=wp.vec3f, device="cpu")
    wp.launch(
        taxel_reaction_forces_kernel,
        dim=(2, 1),
        inputs=[
            None,
            1,
            0,
            2.0,
            wp.array([0], dtype=wp.int32, device="cpu"),
            wp.array([wp.quatf(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device="cpu"),
            wp.array([[wp.vec3f(0.0, 0.0, 1.0)], [wp.vec3f(0.0)]], dtype=wp.vec3f, device="cpu"),
            None,
            wp.array([identity, diverged], dtype=wp.transformf, device="cpu"),
        ],
        outputs=[forces_s],
        device="cpu",
    )
    np.testing.assert_allclose(forces_s.numpy(), [[[0.0, 0.0, -1.0]], [[0.0, 0.0, 0.0]]])
