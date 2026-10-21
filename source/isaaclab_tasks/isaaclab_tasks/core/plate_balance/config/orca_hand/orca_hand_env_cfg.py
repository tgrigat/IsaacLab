# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the ORCA-hand plate-balancing environments.

The right ORCA hand is fixed with its wrist locked; its 16 finger joints hold a level plate on all five fingertips,
each touching it on its tactile pad. The hand's pose (:data:`HAND_ROOT_ROT`, :data:`FINGER_POSE`) was found by a
kinematic search over the finger joints and the plate's normal in the hand frame, with these constraints:

* all five fingertips touch the plate at the same height (within 0.2 mm),
* each fingertip's highest point is on its tactile pad, at least 2.5 mm in from the pad's edge (2.7-5.7 mm),
* every other part of the hand stays at least 4 mm below the plate (14 mm), and the fingers 2 mm from each other
  (4.5 mm),
* and, softer, every joint 15% of its range from its limits, and the fingertip contact Jacobians far from singular
  (smallest singular value 12.5 mm/rad, the largest of the poses found).

To turn all five pads up at once, the fingers lie nearly straight with their tips curled, and the palm faces up
tilted about 30 deg towards the wrist. The thumb's base joint is the binding constraint, at its 15% limit margin.
"""

from dataclasses import MISSING
from typing import ClassVar

from isaaclab_newton.sensors import TactileSensorCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.franka_orca import (
    ORCA_HAND_CFG,
    ORCA_HAND_JOINTS,
    TAXEL_METHODS,
    fingertip_pad_region,
    orca_hand_tactile_dir,
    orca_hand_usd,
)

from ... import mdp
from ...plate_balance_env_cfg import BALL_RADIUS, PlateBalanceEnvCfg, PlateBalanceSceneCfg

WRIST_JOINT = "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e"
FINGER_JOINTS = [joint for joint in ORCA_HAND_JOINTS if joint != WRIST_JOINT]
"""The 16 finger joints, the actions."""

FINGERTIPS = {
    "thumb": "T_DP_b7429e50",
    "index": "I_FingerTipAssembly_ec49c16c",
    "middle": "M_FingerTipAssembly_34afb748",
    "ring": "M_FingerTipAssembly_424a8e75",
    "pinky": "P_FingerTipAssembly_cd219176",
}
"""Fingertip body of each finger; each gets a contact sensor on the plate and, with taxels, a tactile sensor."""

PLATE_SENSORS = [f"{finger}_plate_s" for finger in FINGERTIPS]
"""Contact sensors of the fingertips, filtered to the plate."""

HAND_ROOT_POS = (0.0, 0.0, 0.3)
"""Position [m] of the hand's root, the forearm, above the ground."""

HAND_ROOT_ROT = (0.7071, 0.6976, -0.0061, -0.1157)
"""Orientation (x, y, z, w) of the forearm: turns the plate's normal in the hand frame to world +z, with the
forearm's long axis (+y) in the world xz plane, so the hand reaches along +x, the forearm 10 deg below level."""

FINGER_POSE = {
    "P_AP_f5e42b61_to_R_Carpals_8d1f1041": 0.0556,
    "P_PP_1d411b9b_to_P_AP_f5e42b61": -0.1330,
    "P_FingerTipAssembly_cd219176_to_P_PP_1d411b9b": 0.8002,
    "M_AP_6ec59111_to_R_Carpals_8d1f1041": 0.1547,
    "M_PP_8660a1eb_to_M_AP_6ec59111": -0.0253,
    "M_FingerTipAssembly_424a8e75_to_M_PP_8660a1eb": 0.4373,
    "M_AP_e04a96f2_to_R_Carpals_8d1f1041": 0.0393,
    "M_PP_08efa608_to_M_AP_e04a96f2": -0.1150,
    "M_FingerTipAssembly_34afb748_to_M_PP_08efa608": 0.8238,
    "I_AP_R_d95d02d1_to_R_Carpals_8d1f1041": 0.0617,
    "I_PP_bacbd481_to_I_AP_R_d95d02d1": 0.1685,
    "I_FingerTipAssembly_ec49c16c_to_I_PP_bacbd481": 0.4748,
    "T_TP_R_1c2b802d_to_R_Carpals_8d1f1041": -0.5550,
    "R_T_AP_a9723101_to_T_TP_R_1c2b802d": -0.0697,
    "T_PP_68395e98_to_R_T_AP_a9723101": -0.1103,
    "T_DP_b7429e50_to_T_PP_68395e98": 0.6223,
}
"""Finger joint positions [rad] the plate rests on."""

CONTACT_HEIGHT = 0.0732
"""Height [m] of the fingertip contacts above the hand's root, in :data:`FINGER_POSE`."""

PLATE_CENTER_XY = (0.1893, -0.0365)
"""Plate centre [m] relative to the hand's root: the centroid of the fingertip contacts. With the plate's centre of
mass there, the five fingertips, about equally stiff, share its weight about equally; centred on the contacts'
bounding box instead, it sits 1.8 cm from the thumb-pinky edge of the contacts, and the thumb and pinky carry it while
the ring finger lifts off."""

PLATE_SIZE = (0.255, 0.16, 0.004)
"""Plate size [m]: around :data:`PLATE_CENTER_XY`, out to the farthest fingertip contact plus 1.5 cm on each side."""

SPAWN_GAP = 0.0005
"""Gap [m] the plate is spawned above the fingertips, and the ball above the plate; both settle onto them."""

PLATE_POS = (*PLATE_CENTER_XY, HAND_ROOT_POS[2] + CONTACT_HEIGHT + SPAWN_GAP + PLATE_SIZE[2] / 2)
BALL_POS = (*PLATE_CENTER_XY, PLATE_POS[2] + PLATE_SIZE[2] / 2 + SPAWN_GAP + BALL_RADIUS)

ACTION_SCALE = 0.05
"""Joint position target offset [rad] from :data:`FINGER_POSE` per unit action, about 4 mm at a fingertip. Holding
five contacts of about 0.1 N each takes sub-millimetre precision (the fingertips give about 0.3 mm per 0.1 N), and
tilting the plate 1 deg takes under 2 mm; at 0.2 rad, the policy's exploration noise alone lifts fingertips off."""

PER_TAXEL_OBSERVATIONS = False
"""Whether the policy observes every taxel's force (3 per taxel, 1089 in all) instead of each fingertip's total taxel
force (3 per fingertip). Off to start from the total force per fingertip."""

PAD_TOLERANCE = 2.0e-3
"""Largest distance [m] past its pad's edge, as the plate sees it, of a fingertip's contact point for its force on the
plate to count, without taxels (see :func:`use_fingertips`). With the plate at rest, every contact is 2.7-5.7 mm in
from its pad's edge, but the contact points the simulator reports scatter by up to 1.7 mm."""

TACTILE_CONTACT_THRESHOLD = 0.05
"""Force [N] on a fingertip's pad above which the binary tactile student senses a contact: about half of each
fingertip's equal share of the plate's weight, as in the contact reward."""

PAD_COLLISION_GROUP = 2
"""Newton collision group shared by the fingertips' pad proxies (see ``TactileSensorCfg``)."""

SOLVER_CAPACITY = {"convex_taxels": (400, 1000)}
"""MJWarp contacts and constraint rows per environment (``nconmax``, ``njmax``) of the fingertip methods with taxels,
as in the FrankaOrca lift: every taxel touching the plate is its own contact."""


##
# Scene definition
##


@configclass
class OrcaHandPlateBalanceSceneCfg(PlateBalanceSceneCfg):
    """The ORCA hand at the plate pose, the plate on its fingertips and the ball on the plate."""

    robot: ArticulationCfg = replace(
        ORCA_HAND_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HAND_ROOT_POS, rot=HAND_ROOT_ROT, joint_pos={WRIST_JOINT: 0.0, **FINGER_POSE}
        ),
        actuators={
            "orca_hand_fingers": replace(ORCA_HAND_CFG.actuators["orca_hand_fingers"], joint_names_expr=FINGER_JOINTS),
            # locked: stiff enough that the plate's weight moves it by well under a milliradian
            "orca_hand_wrist": ImplicitActuatorCfg(
                joint_names_expr=[WRIST_JOINT],
                effort_limit_sim=20.0,
                stiffness=200.0,
                damping=5.0,
                armature=0.001,
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.plate = replace(
            self.plate,
            spawn=replace(self.plate.spawn, size=PLATE_SIZE),
            init_state=RigidObjectCfg.InitialStateCfg(pos=PLATE_POS),
        )
        self.ball = replace(self.ball, init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_POS))
        for finger, body in FINGERTIPS.items():
            setattr(
                self,
                f"{finger}_plate_s",
                ContactSensorCfg(
                    # the URDF converter nests every body below its parent link
                    prim_path="{ENV_REGEX_NS}/Robot/.*/" + body,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Plate"],
                ),
            )


##
# MDP settings
##


@configclass
class OrcaHandActionsCfg:
    """Absolute finger joint position targets, offset from the plate pose; the wrist holds still."""

    fingers = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=FINGER_JOINTS, scale=ACTION_SCALE, use_default_offset=True, preserve_order=True
    )


##
# Environment configuration
##


@configclass
class OrcaHandPlateBalanceEnvCfg(PlateBalanceEnvCfg):
    """ORCA-hand ball-on-plate balancing environment."""

    fingertips: ClassVar[str] = "no_taxels"
    """Fingertip collision method (``FINGERTIP_METHODS``), applied by :func:`use_fingertips`. A class variable, set
    by each task variant, as the method decides which sensors and observations the scene has."""

    scene: OrcaHandPlateBalanceSceneCfg = OrcaHandPlateBalanceSceneCfg(
        num_envs=4096, env_spacing=1.0, replicate_physics=True
    )
    actions: OrcaHandActionsCfg = OrcaHandActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        fingers = SceneEntityCfg("robot", joint_names=FINGER_JOINTS, preserve_order=True)
        self.observations.proprio.joint_pos.params["asset_cfg"] = fingers
        self.observations.proprio.joint_vel.params["asset_cfg"] = fingers
        self.observations.proprio.plate_forces.params["contact_sensor_names"] = PLATE_SENSORS
        self.rewards.fingers_share_plate.params["contact_sensor_names"] = PLATE_SENSORS
        self.terminations.fingertip_off_plate.params["contact_sensor_names"] = PLATE_SENSORS
        self.rewards.joint_vel.params["asset_cfg"] = fingers
        self.terminations.ball_off_plate.params["half_extents"] = (PLATE_SIZE[0] / 2, PLATE_SIZE[1] / 2)
        self.commands.ball_target.marker_height = PLATE_SIZE[2] / 2
        use_fingertips(self, self.fingertips)


def pad_params() -> dict:
    """Parameters of the pad-gated contact terms (see :func:`~.mdp.pad_contacts`): each fingertip's pad region, the
    tolerance :data:`PAD_TOLERANCE` and the fingertip bodies, in the order of :data:`PLATE_SENSORS`."""
    return {
        "pad_regions": [str(fingertip_pad_region(finger)) for finger in FINGERTIPS],
        "pad_tolerance": PAD_TOLERANCE,
        "fingertips_cfg": SceneEntityCfg("robot", body_names=list(FINGERTIPS.values()), preserve_order=True),
    }


def use_fingertips(env_cfg: OrcaHandPlateBalanceEnvCfg, fingertips: str):
    """Spawn the hand asset of a fingertip method and, if it has taxels, add the tactile sensors and their
    observations.

    Without taxels, the fingertips sense the plate only through the fingertip contact sensors, and as a tactile sensor
    would: a fingertip's force counts, in the force observation, the contact reward and the off-plate termination, only
    while its contact point is on its pad (:func:`~isaaclab_assets.robots.franka_orca.fingertip_pad_region`, within
    :data:`PAD_TOLERANCE`). With taxels, every taxel touching the plate is its own contact, so MJWarp gets more
    contacts per environment (:data:`SOLVER_CAPACITY`), and each fingertip gets a tactile sensor (Newton backend only)
    and an observation of its taxels' total force or, with :data:`PER_TAXEL_OBSERVATIONS`, of every taxel's force.

    Args:
        env_cfg: The environment configuration to modify.
        fingertips: Fingertip collision method, one of ``FINGERTIP_METHODS``.
    """
    robot = env_cfg.scene.robot
    env_cfg.scene.robot = replace(robot, spawn=replace(robot.spawn, usd_path=str(orca_hand_usd(fingertips))))
    if fingertips not in TAXEL_METHODS:
        pads = pad_params()
        for finger in FINGERTIPS:
            getattr(env_cfg.scene, f"{finger}_plate_s").track_contact_points = True
        env_cfg.observations.proprio.plate_forces.params.update(pads)
        env_cfg.rewards.fingers_share_plate.params.update(pads)
        env_cfg.terminations.fingertip_off_plate.params.update(pads)
        return
    # the presets are separate copies, so both Newton entries are set
    for physics in (env_cfg.sim.physics.newton_mjwarp, env_cfg.sim.physics.default):
        physics.solver_cfg.nconmax, physics.solver_cfg.njmax = SOLVER_CAPACITY[fingertips]
    for finger, body in FINGERTIPS.items():
        setattr(
            env_cfg.scene,
            f"{finger}_tactile",
            TactileSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*/" + body,
                sidecar_path=str(orca_hand_tactile_dir(fingertips) / f"{finger}.npz"),
                pad_collision_group=PAD_COLLISION_GROUP,
            ),
        )
        setattr(
            env_cfg.observations.proprio,
            f"tactile_{finger}",
            ObsTerm(
                func=mdp.taxel_forces_s if PER_TAXEL_OBSERVATIONS else mdp.taxel_total_force_s,
                params={"sensor_name": f"{finger}_tactile"},
                clip=(-10.0, 10.0),  # [N], per taxel or per pad; the plate and ball weigh 0.6 N
            ),
        )


@configclass
class OrcaHandPlateBalanceNoTaxelsEnvCfg(OrcaHandPlateBalanceEnvCfg):
    """ORCA-hand ball-on-plate balancing with convex fingertip hulls, without taxels."""

    fingertips: ClassVar[str] = "no_taxels"


@configclass
class OrcaHandPlateBalanceConvexTaxelsEnvCfg(OrcaHandPlateBalanceEnvCfg):
    """ORCA-hand ball-on-plate balancing with convex taxels over a convex fingertip hull."""

    fingertips: ClassVar[str] = "convex_taxels"


##
# Distillation to tactile students
##


@configclass
class StudentObsCfg(ObsGroup):
    """What a tactile student observes: the finger joints, the target, the last action and, added by each student
    variant, a tactile signal from each fingertip's pad; not the ball or the plate."""

    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": MISSING})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": MISSING})
    target = ObsTerm(func=mdp.generated_commands, params={"command_name": "ball_target"})
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


TACTILE_SIGNALS = {
    "force": ObsTerm(func=mdp.pad_contact_forces_b, clip=(-10.0, 10.0)),
    "position": ObsTerm(func=mdp.pad_contact_points_b),
    "direction": ObsTerm(func=mdp.pad_contact_directions_b),
    "binary": ObsTerm(func=mdp.pad_contact_binary, params={"force_threshold": TACTILE_CONTACT_THRESHOLD}),
}
"""Tactile signal of each student variant, per fingertip, all only from contacts on the pad: the contact force
vector [N] in the fingertip frame (3), the contact point [m] in the fingertip frame (3), the force's unit direction
in the fingertip frame (3), or contact above :data:`TACTILE_CONTACT_THRESHOLD` (1)."""


@configclass
class OrcaHandPlateBalanceStudentEnvCfg(OrcaHandPlateBalanceNoTaxelsEnvCfg):
    """ORCA-hand ball-on-plate balancing without taxels, with a ``student`` observation group for distilling the
    NoTaxels policy, the teacher observing ``policy`` and ``proprio`` as it was trained, to a tactile student."""

    tactile: ClassVar[tuple[str, ...]] = MISSING
    """Tactile signals of the student, keys of :data:`TACTILE_SIGNALS`, observed in this order; set by each student
    variant."""

    def __post_init__(self):
        super().__post_init__()
        fingers = SceneEntityCfg("robot", joint_names=FINGER_JOINTS, preserve_order=True)
        student = StudentObsCfg()
        student.joint_pos.params["asset_cfg"] = fingers
        student.joint_vel.params["asset_cfg"] = fingers
        for signal in self.tactile:
            tactile = replace(TACTILE_SIGNALS[signal])
            tactile.params = {**tactile.params, "contact_sensor_names": PLATE_SENSORS, **pad_params()}
            setattr(student, f"tactile_{signal}", tactile)
        self.observations.student = student


@configclass
class OrcaHandPlateBalanceStudentForceEnvCfg(OrcaHandPlateBalanceStudentEnvCfg):
    """Distillation to a student sensing each pad's contact force vector."""

    tactile: ClassVar[tuple[str, ...]] = ("force",)


@configclass
class OrcaHandPlateBalanceStudentPositionEnvCfg(OrcaHandPlateBalanceStudentEnvCfg):
    """Distillation to a student sensing each pad's contact point."""

    tactile: ClassVar[tuple[str, ...]] = ("position",)


@configclass
class OrcaHandPlateBalanceStudentBinaryEnvCfg(OrcaHandPlateBalanceStudentEnvCfg):
    """Distillation to a student sensing whether each pad is in contact."""

    tactile: ClassVar[tuple[str, ...]] = ("binary",)


@configclass
class OrcaHandPlateBalanceStudentPositionDirectionEnvCfg(OrcaHandPlateBalanceStudentEnvCfg):
    """Distillation to a student sensing each pad's contact point and the direction of its contact force."""

    tactile: ClassVar[tuple[str, ...]] = ("position", "direction")
