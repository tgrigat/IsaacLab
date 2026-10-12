# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fingertip inverse-kinematics variant of the Franka + ORCA-hand lift environment.

The arm and the hand wrist place the palm (one pose IK term), and each finger places its fingertip (one position IK
term per finger, over that finger's joints only). The terms take absolute targets in the robot root frame, which
suits scripted tests such as ``scripts/environments/state_machine/lift_cube_franka_orca_sm.py``.
"""

from isaaclab_newton.sim.schemas import MujocoCollisionCfg, MujocoRigidBodyCfg

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.franka_orca import ORCA_FINGER_CHAINS

from ... import mdp
from .actions import VisualizedDifferentialInverseKinematicsActionCfg
from .franka_orca_env_cfg import PALM_BODY, FrankaOrcaLiftEnvCfg, FrankaOrcaObjectCfg, remove_taxels

HAND_WRIST_JOINT = "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e"
"""Wrist joint of the ORCA hand, between the arm flange and the palm."""

PALM_IK_JOINTS = ["fr3_joint.*", HAND_WRIST_JOINT]
"""Joints placing the palm: the seven arm joints and the hand wrist."""

FINGER_IK_NAMES = ["index", "middle", "ring", "pinky", "thumb"]
"""Action term names of the fingers, in the order of :data:`ORCA_FINGER_CHAINS`."""

FINGERTIP_PAD_OFFSETS = [
    (-0.0117, 0.0, 0.027),  # index
    (-0.0117, 0.0, 0.027),  # middle
    (-0.0117, 0.0, 0.027),  # ring
    (-0.0127, 0.0, 0.025),  # pinky
    (-0.0076, 0.0, 0.020),  # thumb
]
"""Fingertip IK points in the fingertip body frames [m], in the order of :data:`ORCA_FINGER_CHAINS`: on the skin of
the pads, along the pad normal (the body's -x axis) from about the pad centers, so a target on an object surface puts
the skin there. The fingertip body origins sit on the last finger joint, which would leave that joint without any
effect on the IK point."""


CUBE_CONTACT = MujocoCollisionCfg(priority=1, solref=(0.01, 1.0), solimp=(0.95, 0.99, 0.001, 0.5, 2.0))
"""MuJoCo contact parameters of the cube, which govern all of its contacts: a time constant of 0.01 s (the default
0.02 s; MuJoCo's floor is about twice the 1/240 s solver step) and a harder impedance beyond 1 mm of overlap."""


def _finger_joints(chain: list[str]) -> list[str]:
    """Joints of a finger chain; every joint is named after its child body, ``<child>_to_<parent>``."""
    return [f"{body}_to_.*" for body in chain]


def _palm_ik(relative: bool) -> VisualizedDifferentialInverseKinematicsActionCfg:
    """Pose IK of the palm over the arm and the hand wrist."""
    return VisualizedDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        debug_vis=True,
        marker_name="palm",
        joint_names=PALM_IK_JOINTS,
        body_name=PALM_BODY,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=relative, ik_method="dls"),
        scale=0.05 if relative else 1.0,
    )


def _fingertip_ik(finger: int, relative: bool) -> VisualizedDifferentialInverseKinematicsActionCfg:
    """Position IK of a fingertip pad over its finger's joints."""
    chain = ORCA_FINGER_CHAINS[finger]
    return VisualizedDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        debug_vis=True,
        marker_name=FINGER_IK_NAMES[finger],
        joint_names=_finger_joints(chain),
        body_name=chain[-1],
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=FINGERTIP_PAD_OFFSETS[finger]),
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=relative, ik_method="dls"),
        scale=0.01 if relative else 1.0,
    )


@configclass
class FrankaOrcaFingertipIKActionCfg:
    """Absolute palm pose and fingertip pad position targets in the robot root frame.

    The action is ``[palm (x, y, z, qx, qy, qz, qw), index, middle, ring, pinky, thumb (x, y, z) each]``, 22 values.
    """

    palm = _palm_ik(relative=False)
    index = _fingertip_ik(0, relative=False)
    middle = _fingertip_ik(1, relative=False)
    ring = _fingertip_ik(2, relative=False)
    pinky = _fingertip_ik(3, relative=False)
    thumb = _fingertip_ik(4, relative=False)


@configclass
class FrankaOrcaFingertipIKLiftEnvCfg(FrankaOrcaLiftEnvCfg):
    """Franka + ORCA-hand cube lift driven by fingertip IK, for scripted tests of the IK.

    Strips the training machinery that would get in the way of a scripted grasp: no curriculum (gravity and the
    out-of-bounds limits sit at their final values), no actuator or mass randomization, and a plain reset with the
    robot at its home pose and a single cube on the table. Gravity is cancelled on the robot, and the fingertips are
    plain hulls without taxels or tactile sensors.
    """

    actions: FrankaOrcaFingertipIKActionCfg = FrankaOrcaFingertipIKActionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        cube = FrankaOrcaObjectCfg().cube
        # MuJoCo scales contact stiffness with the bodies' masses, so the few-gram fingertips sink ~5 mm into the cube
        # at ~2 N; the cube's own, higher-priority contact parameters stiffen all of its contacts in one place
        cube = replace(cube, collision_props=[cube.collision_props, CUBE_CONTACT])
        self.scene.object = replace(self.scene.object, spawn=cube)

        # the curriculum ramps these from the start; without it they sit at their final values
        self.curriculum = None
        self.events.variable_gravity = None
        self.terminations.object_out_of_bound.params["in_bound_range"]["z"] = (0.0, 2.0)
        # indexes the action by joint id, which only the joint-space action matches
        self.rewards.finger_action_l2 = None

        # Newton reads only MuJoCo's gravcomp, not the asset's (PhysX) disable_gravity; at full gravity the arm sags
        # ~16 cm below its joint targets (joint 6 saturates), which differential IK has no integral term to take out
        # plain fingertip hulls, without taxels to sense
        remove_taxels(self)
        robot_spawn = self.scene.robot.spawn
        robot_spawn.rigid_props = [robot_spawn.rigid_props, MujocoRigidBodyCfg(gravcomp=1.0)]
        # a pad on a cube face is one flat convex pair; with a single contact point per pair that point jumps across
        # the face from step to step, the contact chatters and the pads slide up the cube when the hand lifts
        for physics in (self.sim.physics.newton_mjwarp, self.sim.physics.default):
            physics.solver_cfg.enable_multiccd = True
            physics.solver_cfg.cone = "elliptic"
            physics.solver_cfg.impratio = 10.0
        # keep the robot dynamics nominal, so tracking errors come from the IK
        self.events.joint_stiffness_and_damping = None
        self.events.joint_friction = None
        self.events.object_scale_mass = None

        # home pose and a cube resting on the table in front of the hand
        self.events.conditional_reset = None
        self.events.reset_robot = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": [0.0, 0.0],
                "velocity_range": [0.0, 0.0],
            },
        )
        self.events.reset_object = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [-0.07, -0.07]},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
