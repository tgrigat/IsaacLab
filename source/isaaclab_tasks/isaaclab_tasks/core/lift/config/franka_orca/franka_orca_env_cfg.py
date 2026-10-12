# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka + ORCA-hand lift and reorient environments.

Mirrors :mod:`..kuka_allegro.kuka_allegro_env_cfg`: the shared lift MDP with this robot's body and joint
names filled in.
"""

from isaaclab_newton.sensors import TactileSensorCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.franka_orca import (
    FRANKA_ORCA_ASSET_DIR,
    FRANKA_ORCA_CFG,
    FRANKA_ORCA_TACTILE_DIR,
    ORCA_FINGER_CHAINS,
    ORCA_HAND_JOINTS,
)

from ... import lift_env_cfg as lift
from ... import mdp
from . import events, observations, rewards

PALM_BODY = "R_Carpals_8d1f1041"
THUMB_TIP = "T_DP_b7429e50"
FINGERTIPS = [
    "I_FingerTipAssembly_ec49c16c",  # index
    "M_FingerTipAssembly_34afb748",  # middle
    "M_FingerTipAssembly_424a8e75",  # ring
    "P_FingerTipAssembly_cd219176",  # pinky
    THUMB_TIP,
]
HAND_BODIES = [PALM_BODY, *FINGERTIPS]

TACTILE_FINGERS = {
    "thumb": THUMB_TIP,
    "index": FINGERTIPS[0],
    "middle": FINGERTIPS[1],
    "ring": FINGERTIPS[2],
    "pinky": FINGERTIPS[3],
}
"""Fingers with baked taxels (``build_asset.py --fingers``) and their fingertip bodies; each gets a tactile sensor
and an observation of its taxels' normal and shear forces."""

PAD_COLLISION_GROUP = 2
"""Newton collision group shared by the fingertips' pad proxies, so fingertips meet pad to pad (and, in the default
group, unsensed hulls to unsensed hulls)."""

TACTILE_OBSERVATIONS = True
"""Whether the policy observes the taxels. Off for the ablation without tactile sensing: the tactile sensors stay in
the scene, as they set the taxel and pad-proxy collision groups, but nothing reads them. The fingertip contact
forces (``proprio.contact``) stay observed, as in Kuka-Allegro."""

NO_TAXELS_USD = FRANKA_ORCA_ASSET_DIR / "no_taxels" / "franka_orca.usda"
"""Robot USD without the baked taxels, with plain convex hulls on the fingertips. Built by
``build_asset.py --fingers --out-dir assets/franka_orca_mesh_fingertips/no_taxels``. Every taxel touching something is
its own contact, which overflows MJWarp's contact buffer in a grasp, and the dropped contacts let the fingers and the
cube sink into the cube and the table."""

THUMB_SENSOR = f"{THUMB_TIP}_object_s"
"""Contact sensor of the thumb."""

FINGER_SENSORS = [f"{name}_object_s" for name in FINGERTIPS if name != THUMB_TIP]
"""Contact sensors of the remaining fingers."""

ARM_JOINTS_NON_WRIST = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6"]

# Probed on the 2.3 port (probe_reset_randomization.py, 4096 resets): the lift defaults (+-0.5 rad on every
# joint, +-3 rad on the wrist) put the hand into the table in ~21% of resets; these gave 0/4096.
ARM_RESET_RANGE = 0.12  # rad
WRIST_RESET_RANGE = 0.25  # rad
HAND_RESET_RANGE = 0.5  # rad
# the four fingers' abduction joints span only about +-0.5 rad, and neighbouring fingers already touch at the
# ends of it, so they get a narrower range
HAND_ABDUCTION_RESET_RANGE = 0.15  # rad

FINGER_JOINTS = [joint for joint in ORCA_HAND_JOINTS if not joint.endswith("TopTower_Model_4a80d30e")]
"""The hand joints without the hand's wrist."""

FINGER_ACTION_L2_WEIGHT = -0.02
"""Extra action penalty on the finger joints, on top of ``action_l2`` (-0.01 on every joint). The relative joint
targets move by 0.1 rad per unit action and step, and fast finger motion drives the fingers through each other.

Reached at the maximum ADR difficulty. The penalty is off until the mean difficulty reaches 10%, which takes
successful grasps (``success`` needs the thumb and a finger on the object), so it does not hold the fingers still
before the policy has learned to close them on the object; from there it ramps in with the difficulty."""

HAND_ABDUCTION_JOINTS = [
    joint for joint in ORCA_HAND_JOINTS if "_AP_" in joint and joint.endswith("R_Carpals_8d1f1041")
]
"""Sideways spread of the index, middle, ring and pinky fingers."""

NJMAX = 1000
"""MJWarp constraint rows per environment. MJWarp drops the rows over it, which can blow the simulation up. The
lift default of 300 overflows in dense grasps, where every taxel on the object is its own contact. With the fingers
colliding with each other only pad to pad, random actions peak at about 560 rows."""

CAMERA_ENV = 0
"""Environment the visualizer (and so the recorded video) shows, from :data:`CAMERA_EYE` towards :data:`CAMERA_LOOKAT`,
both relative to its origin [m]; the robot base sits at the origin and reaches towards the table at -x."""
CAMERA_EYE = (-1.15, 0.55, 0.75)
CAMERA_LOOKAT = (-0.45, 0.0, 0.45)

OBJECT_SCALE = 0.75
"""The lift shapes are sized for the Allegro hand; scaled down uniformly for the smaller ORCA hand."""


def _scaled(shape):
    """Copy of a lift object shape, scaled by :data:`OBJECT_SCALE`."""
    if hasattr(shape, "size"):  # cuboids
        return replace(shape, size=tuple(s * OBJECT_SCALE for s in shape.size))
    if hasattr(shape, "height"):  # capsules, cones
        return replace(shape, radius=shape.radius * OBJECT_SCALE, height=shape.height * OBJECT_SCALE)
    return replace(shape, radius=shape.radius * OBJECT_SCALE)  # spheres


def _joint_offset_reset(joint_names: list[str], offset: float) -> EventTerm:
    """Reset term moving the given robot joints by a uniform offset in ``[-offset, offset]`` [rad]."""
    return EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names),
            "position_range": [-offset, offset],
            "velocity_range": [0.0, 0.0],
        },
    )


_LIFT_OBJECTS = lift.ObjectCfg()


##
# Scene definition
##


@configclass
class FrankaOrcaObjectCfg(lift.ObjectCfg):
    """The lift objects, scaled by :data:`OBJECT_SCALE`."""

    shapes = replace(_LIFT_OBJECTS.shapes, assets_cfg=[_scaled(s) for s in _LIFT_OBJECTS.shapes.assets_cfg])
    cube = _scaled(_LIFT_OBJECTS.cube)
    default = shapes


@configclass
class FrankaOrcaSceneCfg(lift.SceneCfg):
    """Franka + ORCA-hand scene for the lift and reorient tasks."""

    robot: ArticulationCfg = replace(FRANKA_ORCA_CFG, prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self):
        super().__post_init__()
        self.object = replace(self.object, spawn=FrankaOrcaObjectCfg())
        for link_name in FINGERTIPS:
            setattr(
                self,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    # the URDF converter nests every body below its parent link
                    prim_path="{ENV_REGEX_NS}/Robot/.*/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        # one tactile sensor per sensed fingertip (Newton backend only); the pad proxies share a collision group,
        # so fingertips collide pad to pad (and unsensed to unsensed), and the asset filters both against the taxels
        # (build_asset.py), so the taxels sense only the object, the table and the arm
        for finger, body in TACTILE_FINGERS.items():
            setattr(
                self,
                f"{finger}_tactile",
                TactileSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/.*/" + body,
                    sidecar_path=str(FRANKA_ORCA_TACTILE_DIR / f"{finger}.npz"),
                    pad_collision_group=PAD_COLLISION_GROUP,
                ),
            )


##
# MDP settings
##


@configclass
class FrankaOrcaObservationsCfg(lift.ObservationsCfg):
    """Lift observations with the ORCA fingertip contact forces, the taxels and the hand bodies filled in."""

    def __post_init__(self):
        super().__post_init__()
        self.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in FINGERTIPS]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        for finger in TACTILE_FINGERS if TACTILE_OBSERVATIONS else []:
            setattr(
                self.proprio,
                f"tactile_{finger}",
                ObsTerm(
                    func=observations.taxel_forces_s,
                    params={"sensor_name": f"{finger}_tactile"},
                    clip=(-10.0, 10.0),  # [N] per taxel
                ),
            )
        self.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = HAND_BODIES


@configclass
class FrankaOrcaRelJointPosActionCfg:
    """Relative joint position targets for all joints."""

    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class FrankaOrcaRewardCfg(lift.RewardsCfg):
    """Reward terms for the MDP, with the ORCA fingertip contact sensors filled in."""

    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=1.0,
        params={"threshold": 0.01, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=0.1,
        params={"threshold": 0.01, "sensor_names": FINGER_SENSORS + [THUMB_SENSOR]},
    )

    # off until FrankaOrcaCurriculumCfg.finger_action_l2 ramps it in
    finger_action_l2 = RewTerm(
        func=rewards.joint_action_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=FINGER_JOINTS)},
    )

    # Optional shaping from the 2.3 port (see .rewards); enable by setting a RewTerm, e.g.
    # RewTerm(func=rewards.finger_thumb_opposition, weight=0.01, params={...}).
    finger_thumb_opposition: RewTerm | None = None
    thumb_min_distance: RewTerm | None = None

    def __post_init__(self):
        super().__post_init__()
        # pushing the object out of bounds ends the episode for free otherwise, which the policy learns to prefer
        # over paying the action penalties of a longer episode
        self.early_termination.params["term_keys"] = ["abnormal_robot", "object_out_of_bound"]
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=HAND_BODIES)
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


THUMB_SHAPING_REWARDS = {
    "finger_thumb_opposition": RewTerm(
        func=rewards.finger_thumb_opposition,
        weight=0.01,  # the term is in N^2
        params={"thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    ),
    "thumb_min_distance": RewTerm(
        func=rewards.fingertip_min_distance,
        weight=1.0,
        params={"std": 0.03, "asset_cfg": SceneEntityCfg("robot", body_names=[THUMB_TIP])},
    ),
}
"""The two optional terms, ready to assign to :class:`FrankaOrcaRewardCfg` (2.3 port weights)."""


@configclass
class FrankaOrcaCurriculumCfg(lift.CurriculumCfg):
    """Lift curriculum with the finger action penalty ramped in once the hand grasps."""

    finger_action_l2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.finger_action_l2.weight",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": 0.0,
                "final_value": FINGER_ACTION_L2_WEIGHT,
                "difficulty_term_str": "adr",
            },
        },
    )


##
# Environment configuration
##


@configclass
class FrankaOrcaMixinCfg:
    """Franka + ORCA specific scene, observation, action and reward terms, mixed into the task configurations."""

    scene: FrankaOrcaSceneCfg = FrankaOrcaSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    rewards: FrankaOrcaRewardCfg = FrankaOrcaRewardCfg()
    observations: FrankaOrcaObservationsCfg = FrankaOrcaObservationsCfg()
    actions: FrankaOrcaRelJointPosActionCfg = FrankaOrcaRelJointPosActionCfg()
    curriculum: FrankaOrcaCurriculumCfg | None = FrankaOrcaCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # without the curriculum nothing ramps the finger action penalty in, so it applies from the start
        if self.curriculum is None:
            self.rewards.finger_action_l2.weight = FINGER_ACTION_L2_WEIGHT
        self.commands.object_pose.body_name = PALM_BODY
        # the goal ranges are in the robot root frame, which this robot turns 180 deg about z (see FRANKA_ORCA_CFG);
        # mirrored in x so the goals sit over the table at world -x, as for Kuka-Allegro, instead of behind the robot
        x_min, x_max = self.commands.object_pose.ranges.pos_x
        self.commands.object_pose.ranges.pos_x = (-x_max, -x_min)
        # the presets are separate copies, so both Newton entries are set
        for physics in (self.sim.physics.newton_mjwarp, self.sim.physics.default):
            physics.solver_cfg.njmax = NJMAX

        # show one environment up close; the lift camera sits at the world origin, between the environments
        self.sim.default_visualizer_cfg.visible_env_indices = [CAMERA_ENV]
        self.events.aim_camera = EventTerm(
            func=events.aim_camera_at_env,
            mode="startup",
            params={"env_index": CAMERA_ENV, "eye": CAMERA_EYE, "lookat": CAMERA_LOOKAT},
        )

        reset = self.events.conditional_reset.params
        terms = reset["terms"]
        terms["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ARM_JOINTS_NON_WRIST)
        terms["reset_robot_joints"].params["position_range"] = [-ARM_RESET_RANGE, ARM_RESET_RANGE]
        terms["reset_robot_wrist_joint"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="fr3_joint7")
        terms["reset_robot_wrist_joint"].params["position_range"] = [-WRIST_RESET_RANGE, WRIST_RESET_RANGE]
        # the arm-only joint reset no longer covers the fingers, so they get their own terms
        flexion_joints = [joint for joint in ORCA_HAND_JOINTS if joint not in HAND_ABDUCTION_JOINTS]
        terms["reset_hand_joints"] = _joint_offset_reset(flexion_joints, HAND_RESET_RANGE)
        terms["reset_hand_abduction"] = _joint_offset_reset(HAND_ABDUCTION_JOINTS, HAND_ABDUCTION_RESET_RANGE)
        # OmniReset-style reset modes, each with its own share of the bank: the robot terms above run for
        # every state, then one mode places the object. "in_hand" reads the hand pose, so it must run after
        # the robot terms, which the modes always do.
        in_hand = terms.pop("reset_object_to_target")
        in_hand.params["probability"] = 1.0
        in_hand.params["target_cfg"] = SceneEntityCfg("robot", body_names=PALM_BODY)
        # palm frame, fingers along +z, thumb at +y; this box sits between them
        in_hand.params["pose_range"] = {"x": [-0.02, 0.02], "y": [0.03, 0.06], "z": [0.05, 0.09]}
        reset["modes"] = {
            "table": {"reset_object": terms.pop("reset_object")},
            "in_hand": {"reset_object_to_target": in_hand},
        }
        # table/ground clearance: everything but the ground-mounted arm base
        reset["valid_criteria"]["robot_table_clearance"].body_names = "(?!fr3_link0$).*"
        # random hand-joint resets can start the fingers inside each other, which the solver resolves with
        # hundreds of contacts in the first steps
        finger_clearance = mdp.SelfClearanceCfg(asset_name="robot", body_groups=ORCA_FINGER_CHAINS)
        reset["valid_criteria"]["finger_clearance"] = finger_clearance
        # spread the reset bank over the grasp geometry, same bodies as fingers_to_object
        diversity_feature = reset.get("diversity_feature")
        if diversity_feature is not None:
            diversity_feature.body_names = HAND_BODIES


def remove_taxels(env_cfg: FrankaOrcaMixinCfg):
    """Swap the robot for :data:`NO_TAXELS_USD` and drop the tactile sensors and their observations.

    The fingertip contact sensors (``proprio.contact`` and the contact rewards) stay, as they sense whole bodies.
    """
    env_cfg.scene.robot = replace(
        env_cfg.scene.robot, spawn=replace(env_cfg.scene.robot.spawn, usd_path=str(NO_TAXELS_USD))
    )
    for finger in TACTILE_FINGERS:
        setattr(env_cfg.scene, f"{finger}_tactile", None)
        setattr(env_cfg.observations.proprio, f"tactile_{finger}", None)


@configclass
class FrankaOrcaReorientEnvCfg(FrankaOrcaMixinCfg, lift.ReorientEnvCfg):
    """Franka + ORCA-hand object reorientation environment."""


@configclass
class FrankaOrcaLiftEnvCfg(FrankaOrcaMixinCfg, lift.LiftEnvCfg):
    """Franka + ORCA-hand object lifting environment."""


@configclass
class FrankaOrcaLiftNoTaxelsEnvCfg(FrankaOrcaLiftEnvCfg):
    """Franka + ORCA-hand object lifting with plain fingertip hulls, without tactile sensing (:func:`remove_taxels`)."""

    def __post_init__(self):
        super().__post_init__()
        remove_taxels(self)
