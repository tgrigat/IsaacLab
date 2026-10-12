# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-hand variant of the Franka + ORCA-hand lift environment: carry a cube that starts in the open hand.

A simpler stepping stone to :class:`~.franka_orca_env_cfg.FrankaOrcaLiftEnvCfg`. The arm starts near one of several
fixed poses with the palm rolled about the fingers, so it faces up, sideways or down; the cube floats off the palm,
caged by the curled fingers and thumb, and the policy carries it to a goal in front of the robot. The rewards, the
gravity curriculum (no gravity at first, full gravity at the maximum ADR difficulty) and the out-of-bounds penalty are
the lift ones, as in Kuka-Allegro. Once gravity ramps in, a cube off a sideways or downturned palm falls unless the hand
holds it.
"""

from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass, replace

from ... import mdp
from . import events
from .franka_orca_env_cfg import FrankaOrcaLiftEnvCfg, FrankaOrcaObjectCfg

PALM_UP_JOINT_POS = {
    "fr3_joint1": -0.335,
    "fr3_joint2": -0.492,
    "fr3_joint3": 0.200,
    "fr3_joint4": -2.425,
    "fr3_joint5": 1.107,
    "fr3_joint6": 2.465,
    "fr3_joint7": 0.205,
    "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e": 0.024,
}
"""Arm and hand-wrist joints of the palm-up start pose [rad].

Solved with the palm pose IK of the fingertip IK task: the palm at (-0.55, 0.0, 0.45) m in the environment frame,
its normal (palm +y) up and the fingers (palm +z) level, turned 30 deg from straight ahead (world -x) towards -y.
Straight ahead runs ``fr3_joint7`` into its limit; this pose keeps every arm joint at least 0.59 rad from its limits.
"""

PALM_ROLL_JOINT_POS = {
    0: PALM_UP_JOINT_POS,
    45: {
        "fr3_joint1": -0.339,
        "fr3_joint2": -0.306,
        "fr3_joint3": -0.016,
        "fr3_joint4": -2.295,
        "fr3_joint5": 1.302,
        "fr3_joint6": 1.966,
        "fr3_joint7": 0.558,
        "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e": -0.051,
    },
    -45: {
        "fr3_joint1": -0.336,
        "fr3_joint2": -0.516,
        "fr3_joint3": 0.514,
        "fr3_joint4": -2.6,
        "fr3_joint5": 1.002,
        "fr3_joint6": 3.194,
        "fr3_joint7": -0.068,
        "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e": 0.153,
    },
    90: {
        "fr3_joint1": -0.332,
        "fr3_joint2": -0.191,
        "fr3_joint3": -0.121,
        "fr3_joint4": -2.36,
        "fr3_joint5": 1.639,
        "fr3_joint6": 1.71,
        "fr3_joint7": 0.965,
        "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e": -0.165,
    },
    -90: {
        "fr3_joint1": -0.515,
        "fr3_joint2": -0.073,
        "fr3_joint3": 0.797,
        "fr3_joint4": -2.696,
        "fr3_joint5": 0.628,
        "fr3_joint6": 4.269,
        "fr3_joint7": -0.24,
        "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e": 0.191,
    },
    135: {
        "fr3_joint1": -0.294,
        "fr3_joint2": -0.153,
        "fr3_joint3": -0.151,
        "fr3_joint4": -2.499,
        "fr3_joint5": 1.971,
        "fr3_joint6": 1.6,
        "fr3_joint7": 1.484,
        "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e": -0.205,
    },
    180: {
        "fr3_joint1": -0.199,
        "fr3_joint2": -0.182,
        "fr3_joint3": -0.135,
        "fr3_joint4": -2.666,
        "fr3_joint5": 2.388,
        "fr3_joint6": 1.586,
        "fr3_joint7": 2.035,
        "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e": -0.144,
    },
}
"""Start poses by the roll of the palm about the fingers from :data:`PALM_UP_JOINT_POS` [deg]: 0 faces up, +-90 sideways
and 180 down. The palm stays in place with the fingers level. Solved with the palm pose IK of the fingertip IK task,
the roll ramped in from the palm-up pose; each keeps every arm joint at least 0.25 rad from its limits. -135 deg runs
the arm into its limits, so the downturned half has only the +135 deg pose."""

HAND_CAGE_JOINT_POS = {
    # index, middle, ring and pinky: the proximal and distal joints curled alike
    "I_PP_bacbd481_to_I_AP_R_d95d02d1": 1.0,
    "I_FingerTipAssembly_ec49c16c_to_I_PP_bacbd481": 1.0,
    "M_PP_08efa608_to_M_AP_e04a96f2": 1.1,
    "M_FingerTipAssembly_34afb748_to_M_PP_08efa608": 1.1,
    "M_PP_8660a1eb_to_M_AP_6ec59111": 1.03,
    "M_FingerTipAssembly_424a8e75_to_M_PP_8660a1eb": 1.03,
    "P_PP_1d411b9b_to_P_AP_f5e42b61": 1.0,
    "P_FingerTipAssembly_cd219176_to_P_PP_1d411b9b": 1.0,
    # thumb, turned across the palm to oppose the fingers
    "T_TP_R_1c2b802d_to_R_Carpals_8d1f1041": -0.6,
    "R_T_AP_a9723101_to_T_TP_R_1c2b802d": 0.0,
    "T_PP_68395e98_to_R_T_AP_a9723101": 1.05,
    "T_DP_b7429e50_to_T_PP_68395e98": 1.05,
}
"""Finger joints of the start pose [rad], a loose cage around a cube centered on :data:`CUBE_IN_PALM_RANGE`.

Without gravity, a cube floating in an open hand drifts away at the first touch and never comes back, so no grasp,
and with it no contact reward or curriculum progress, ever happens. Curled, the fingertips sit in front of the cube
(along the fingers) and the thumb beside it, each 4.5-6.5 mm from the cube centered at (0, 0.06, 0.06) m in the palm
frame, measured with the reset's mesh clearance check; the abduction joints stay at zero. The finger joints are in
the palm frame, so the one cage serves every palm roll."""

# reset offsets around the start pose, small enough to keep the cage around the cube and the hand over the table
ARM_RESET_RANGE = 0.05  # rad
HAND_RESET_RANGE = 0.05  # rad
HAND_ABDUCTION_RESET_RANGE = 0.05  # rad

CUBE_IN_PALM_RANGE = {"x": [-0.004, 0.004], "y": [0.062, 0.068], "z": [0.056, 0.064]}
"""Cube position range in the palm frame [m]: on the palm normal (+y) and along the fingers (+z), inside the cage of
:data:`HAND_CAGE_JOINT_POS`. A cube dropped onto the upturned palm comes to rest at about y = 0.037 m; draws that
touch the hand are rejected by the reset."""

GOAL_RANGE = {"pos_x": (0.4, 0.7), "pos_y": (-0.25, 0.25), "pos_z": (0.45, 0.75)}
"""Goal positions in the robot root frame [m]. The root is turned 180 deg about z, so +x is in front of the robot,
over the table (world -x); the palm starts at x = 0.55 m, z = 0.45 m."""

PENALTY_INITIAL_SCALE = 0.01
"""Share of the ``action_l2`` and ``early_termination`` weights at zero ADR difficulty; the curriculum ramps them to the
full weights at the maximum.

At the full weights from the start, the penalties are the only rewards a policy sees before its first grasp: without
gravity, a touch tends to push the cube out of bounds, so the policy learns to keep still and away from the cube
instead of closing the fingers on it."""

IN_BOUND_RANGE = {"x": (-1.0, 0.0), "y": (-0.6, 0.6), "z": (0.3, 1.2)}
"""Workspace the cube must stay in, in the environment frame [m]. The lower z bound sits above the table top
(0.255 m), so dropping the cube ends the episode at every difficulty."""


@configclass
class FrankaOrcaInHandLiftEnvCfg(FrankaOrcaLiftEnvCfg):
    """Franka + ORCA-hand cube carrying environment, starting with the cube in the open hand at a random palm roll."""

    def __post_init__(self):
        super().__post_init__()
        robot = self.scene.robot
        joint_pos = {**robot.init_state.joint_pos, **PALM_UP_JOINT_POS, **HAND_CAGE_JOINT_POS}
        # the gravity curriculum is meant for the cube: PhysX keeps gravity off the robot (disable_gravity), but
        # Newton reads only MuJoCo's gravcomp, and at full gravity the arm sags ~15 cm and lowers the hand to the table
        rigid_props = [robot.spawn.rigid_props, MujocoRigidBodyCfg(gravcomp=1.0)]
        self.scene.robot = replace(
            robot,
            spawn=replace(robot.spawn, rigid_props=rigid_props),
            init_state=replace(robot.init_state, joint_pos=joint_pos),
        )
        self.scene.object = replace(self.scene.object, spawn=FrankaOrcaObjectCfg().cube)

        for key, value in GOAL_RANGE.items():
            setattr(self.commands.object_pose.ranges, key, value)

        # one of the palm roll poses with a small offset, and every episode in the hand
        reset = self.events.conditional_reset.params
        terms = reset["terms"]
        del terms["reset_robot_joints"], terms["reset_robot_wrist_joint"]
        terms["reset_robot_pose"] = EventTerm(
            func=events.reset_joints_to_random_pose,
            mode="reset",
            params={
                "poses": list(PALM_ROLL_JOINT_POS.values()),
                "position_range": [-ARM_RESET_RANGE, ARM_RESET_RANGE],
            },
        )
        terms["reset_hand_joints"].params["position_range"] = [-HAND_RESET_RANGE, HAND_RESET_RANGE]
        terms["reset_hand_abduction"].params["position_range"] = [
            -HAND_ABDUCTION_RESET_RANGE,
            HAND_ABDUCTION_RESET_RANGE,
        ]
        in_hand = reset["modes"]["in_hand"]
        in_hand["reset_object_to_target"].params["pose_range"] = CUBE_IN_PALM_RANGE
        reset["modes"] = {"in_hand": in_hand}

        # dropping the cube is a failure throughout, so the out-of-bounds floor stays above the table rather than
        # dropping to the ground with the difficulty
        self.terminations.object_out_of_bound.params["in_bound_range"] = dict(IN_BOUND_RANGE)
        if self.curriculum is not None:
            self.curriculum.oob_adr = None
            for term in ("action_l2", "early_termination"):
                weight = getattr(self.rewards, term).weight
                getattr(self.rewards, term).weight = weight * PENALTY_INITIAL_SCALE
                setattr(
                    self.curriculum,
                    term,
                    CurrTerm(
                        func=mdp.modify_term_cfg,
                        params={
                            "address": f"rewards.{term}.weight",
                            "modify_fn": mdp.initial_final_interpolate_fn,
                            "modify_params": {
                                "initial_value": weight * PENALTY_INITIAL_SCALE,
                                "final_value": weight,
                                "difficulty_term_str": "adr",
                            },
                        },
                    ),
                )
