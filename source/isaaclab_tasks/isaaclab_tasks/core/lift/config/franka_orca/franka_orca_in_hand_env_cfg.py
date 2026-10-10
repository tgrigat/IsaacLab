# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-hand variant of the Franka + ORCA-hand lift environment: carry a cube that starts in the open hand.

A simpler stepping stone to :class:`~.franka_orca_env_cfg.FrankaOrcaLiftEnvCfg`. The arm starts near a fixed pose
with the palm facing up and the hand open, the cube floats just above the palm, and the policy carries it to a goal
in front of the robot. The rewards, the gravity curriculum (no gravity at first, full gravity at the maximum ADR
difficulty) and the out-of-bounds penalty are the lift ones, as in Kuka-Allegro.
"""

from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg

from isaaclab.utils import configclass, replace

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
"""Arm and hand-wrist joints of the start pose [rad]; the fingers stay at zero, open and straight.

Solved with the palm pose IK of the fingertip IK task: the palm at (-0.55, 0.0, 0.45) m in the environment frame,
its normal (palm +y) up and the fingers (palm +z) level, turned 30 deg from straight ahead (world -x) towards -y.
Straight ahead runs ``fr3_joint7`` into its limit; this pose keeps every arm joint at least 0.59 rad from its limits.
"""

# reset offsets around the start pose, small enough to keep the palm facing up and the hand open
ARM_RESET_RANGE = 0.05  # rad
WRIST_RESET_RANGE = 0.1  # rad
HAND_RESET_RANGE = 0.1  # rad
HAND_ABDUCTION_RESET_RANGE = 0.05  # rad

CUBE_IN_PALM_RANGE = {"x": [-0.015, 0.015], "y": [0.05, 0.06], "z": [0.045, 0.07]}
"""Cube position range in the palm frame [m]: on the palm normal (+y) and along the fingers (+z), clear of the thumb.
A cube dropped onto the upturned palm comes to rest at about y = 0.037 m, z = 0.055-0.065 m; starting slightly
higher leaves room for the random orientations, and draws that still touch the hand are rejected by the reset."""

GOAL_RANGE = {"pos_x": (0.4, 0.7), "pos_y": (-0.25, 0.25), "pos_z": (0.45, 0.75)}
"""Goal positions in the robot root frame [m]. The root is turned 180 deg about z, so +x is in front of the robot,
over the table (world -x); the palm starts at x = 0.55 m, z = 0.45 m."""

IN_BOUND_RANGE = {"x": (-1.0, 0.0), "y": (-0.6, 0.6), "z": (0.3, 1.2)}
"""Workspace the cube must stay in, in the environment frame [m]. The lower z bound sits above the table top
(0.255 m), so dropping the cube ends the episode at every difficulty."""


@configclass
class FrankaOrcaInHandLiftEnvCfg(FrankaOrcaLiftEnvCfg):
    """Franka + ORCA-hand cube carrying environment, starting with the cube in the open, upturned hand."""

    def __post_init__(self):
        super().__post_init__()
        robot = self.scene.robot
        joint_pos = {**robot.init_state.joint_pos, **PALM_UP_JOINT_POS}
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

        # small offsets around the start pose and every episode in the hand
        reset = self.events.conditional_reset.params
        terms = reset["terms"]
        terms["reset_robot_joints"].params["position_range"] = [-ARM_RESET_RANGE, ARM_RESET_RANGE]
        terms["reset_robot_wrist_joint"].params["position_range"] = [-WRIST_RESET_RANGE, WRIST_RESET_RANGE]
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
