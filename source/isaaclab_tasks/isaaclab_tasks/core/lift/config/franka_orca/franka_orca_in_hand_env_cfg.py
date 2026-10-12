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

import copy

from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass, replace

from ... import mdp
from . import curriculums, events, rewards
from .franka_orca_env_cfg import FINGER_JOINTS, HAND_BODIES, FrankaOrcaLiftEnvCfg, FrankaOrcaObjectCfg, remove_taxels

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

GOAL_MARKER_RADIUS = 0.02
"""Radius of the goal marker [m], about the cube's half size."""

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
        # show the goal in training too, so the recorded videos show where the cube is meant to go: a sphere the
        # size of the cube, red until the cube is within 5 cm of it and green after; the cube gets a small one
        command = self.commands.object_pose
        command.debug_vis = True
        goal_markers = dict(command.goal_pose_visualizer_cfg.markers)  # a copy, the cube's markers may share it
        for name in ("position_far", "position_near"):
            goal_markers[name] = replace(goal_markers[name], radius=GOAL_MARKER_RADIUS)
        command.goal_pose_visualizer_cfg = replace(command.goal_pose_visualizer_cfg, markers=goal_markers)

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


@configclass
class FrankaOrcaInHandLiftNoTaxelsEnvCfg(FrankaOrcaInHandLiftEnvCfg):
    """In-hand cube carrying with plain fingertip hulls and no tactile sensing (see :func:`.remove_taxels`)."""

    def __post_init__(self):
        super().__post_init__()
        remove_taxels(self)


CUBE_NEAR_PALM_RANGE = {"x": [-0.04, 0.04], "y": [0.10, 0.14], "z": [0.0, 0.12]}
"""Cube position range in the palm frame [m] of the staged task's later stages: 3.5-7.5 cm further out along the palm
normal than :data:`CUBE_IN_PALM_RANGE`, over the curled fingers rather than in their cage, so the hand has to reach
for it. Draws that touch the hand are rejected by the reset."""

CUBE_FAR_FROM_PALM_RANGE = {"x": [-0.08, 0.08], "y": [0.14, 0.22], "z": [-0.04, 0.16]}
"""Cube position range in the palm frame [m] of the hardest floating stage: 7.5-15.5 cm further out along the palm
normal than :data:`CUBE_IN_PALM_RANGE` and spread wider across and along the hand, so the arm has to move to it."""

CUBE_ON_TABLE_RANGE = {"x": [-0.15, 0.15], "y": [-0.3, 0.1], "yaw": [-3.14, 3.14]}
"""Cube pose range on the table [m, rad], around the object's initial position (-0.55, 0.1) m in the environment frame,
so the cube lies on the table below the palm, flat at a random yaw. The palm starts at (-0.55, 0.0, 0.45) m."""

HAND_OPEN_JOINT_POS = {joint: 0.0 for joint in FINGER_JOINTS}
"""Finger joints of the open hand [rad]: fingers straight and together, thumb at rest."""

HAND_CLOSED_JOINT_POS = {
    **HAND_OPEN_JOINT_POS,
    # index, middle, ring and pinky: the proximal and distal joints curled alike
    "I_PP_bacbd481_to_I_AP_R_d95d02d1": 1.5,
    "I_FingerTipAssembly_ec49c16c_to_I_PP_bacbd481": 1.5,
    "M_PP_08efa608_to_M_AP_e04a96f2": 1.5,
    "M_FingerTipAssembly_34afb748_to_M_PP_08efa608": 1.5,
    "M_PP_8660a1eb_to_M_AP_6ec59111": 1.5,
    "M_FingerTipAssembly_424a8e75_to_M_PP_8660a1eb": 1.5,
    "P_PP_1d411b9b_to_P_AP_f5e42b61": 1.5,
    "P_FingerTipAssembly_cd219176_to_P_PP_1d411b9b": 1.5,
    # thumb half flexed: from about 0.8 rad on it runs into the curled index
    "T_PP_68395e98_to_R_T_AP_a9723101": 0.5,
    "T_DP_b7429e50_to_T_PP_68395e98": 0.5,
}
"""Finger joints of the closed hand [rad]: index, middle, ring and pinky curled into a fist, the thumb half flexed.
98% of the draws clear the reset's finger self-clearance check; a fully flexed thumb clears none."""

GRAVITY_START_FRAC = 0.5
"""Mean ADR difficulty fraction from which gravity ramps in, once every floating stage is in the mix."""

TABLE_START_FRAC = 0.75
"""Mean ADR difficulty fraction at which gravity is full and the cube stops floating and starts on the table."""

STAGE_FRACS = {
    "in_hand": (0.0, TABLE_START_FRAC),
    "near_hand": (0.15, TABLE_START_FRAC),
    "open_closed_hand": (0.3, TABLE_START_FRAC),
    "far_from_hand": (0.45, TABLE_START_FRAC),
    "on_table": (TABLE_START_FRAC, 1.0),
}
"""Reset modes of the staged task and the window of mean ADR difficulty fraction in which each is drawn:

* ``in_hand``: the cube in the caged hand (:data:`CUBE_IN_PALM_RANGE`),
* ``near_hand``: the cube a bit away from the caged hand (:data:`CUBE_NEAR_PALM_RANGE`),
* ``open_closed_hand``: the cube a bit away from a fully open or closed hand,
* ``far_from_hand``: the cube further away from a fully open or closed hand (:data:`CUBE_FAR_FROM_PALM_RANGE`),
* ``on_table``: the cube lying on the table at a random pose (:data:`CUBE_ON_TABLE_RANGE`), the hand open or closed.

The modes in the window are drawn uniformly, so the earlier floating stages stay in the mix. Gravity ramps in over the
floating stages, from :data:`GRAVITY_START_FRAC` to full at :data:`TABLE_START_FRAC`, so the policy learns to catch and
hold a falling cube before the cube starts on the table; from there it always does."""

REACH_REWARD_WEIGHT = 2.5
REACH_REWARD_STEP = 0.005  # m
"""Weight and step of the hand reach reward (:class:`.rewards.hand_object_progress`), which pays for every 5 mm the
hand gets closer to the cube than before in the episode, contact or not: about 0.17 per centimetre (the reward
manager scales by the 1/30 s step), 3.3 for the 20 cm to a far cube. The lift ``fingers_to_object`` term pays a
tenth of its 0.05 weight until the thumb and a finger touch the cube, too little to lead the hand to a cube 15-30 cm
away. A reward for being near the cube (a tanh kernel on the distance, weight 0.5) did lead the hand there, but the
policy then hovered next to the cube and never touched it, in every stage: a touch pushes the floating cube away and
costs that reward."""

STAGED_IN_BOUND_RANGE = {**IN_BOUND_RANGE, "z": (0.2, IN_BOUND_RANGE["z"][1])}
"""Workspace the cube must stay in, in the environment frame [m]: :data:`IN_BOUND_RANGE` with the floor below the table
top (0.255 m), so a cube on the table is in bounds and one that falls off the table ends the episode."""

OBJECT_RESET_BOUND_MARGIN = 0.03
"""Distance [m] a reset cube keeps from the faces of :data:`STAGED_IN_BOUND_RANGE`, so no episode starts on the verge of
the out-of-bounds termination."""

TABLE_CLEARANCE = 0.002
"""Gap [m] between the table and a cube reset onto it, which then settles onto the table."""


@configclass
class FrankaOrcaInHandStagedLiftEnvCfg(FrankaOrcaInHandLiftEnvCfg):
    """In-hand cube carrying with a staged curriculum: from the cube in the caged hand to a cube ever further from a
    hand that starts open or closed, with gravity ramping in, then a cube picked up from the table.

    The reset bank holds one reset mode per stage (:data:`STAGE_FRACS`), and a curriculum term draws the modes whose
    window holds the mean ADR difficulty; gravity ramps in from :data:`GRAVITY_START_FRAC` to full at
    :data:`TABLE_START_FRAC`, where the cube stops floating and starts on the table. A reach reward
    (:data:`REACH_REWARD_WEIGHT`) leads the hand to cubes it does not touch yet. The ADR difficulty rises with
    successes and falls with failures, so a stage that the policy cannot yet solve drops out again until it can. The
    success rate of each stage is logged as ``Metrics/success_rate/<mode>``.
    """

    def __post_init__(self):
        super().__post_init__()
        reset = self.events.conditional_reset.params
        in_hand = reset["modes"]["in_hand"]
        near_hand = copy.deepcopy(in_hand["reset_object_to_target"])
        near_hand.params["pose_range"] = CUBE_NEAR_PALM_RANGE
        far_from_hand = copy.deepcopy(in_hand["reset_object_to_target"])
        far_from_hand.params["pose_range"] = CUBE_FAR_FROM_PALM_RANGE

        # flat on the table: the range offsets the object's initial position, which floats above the table
        table = self.scene.table
        table_top = table.init_state.pos[2] + table.spawn.size[2] / 2
        cube_z = table_top + self.scene.object.spawn.size[2] / 2 + TABLE_CLEARANCE
        on_table = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {**CUBE_ON_TABLE_RANGE, "z": [cube_z - self.scene.object.init_state.pos[2]] * 2},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        def open_or_closed_hand() -> EventTerm:
            """Finger reset to the open or the closed hand, overriding the shared hand resets."""
            return EventTerm(
                func=events.reset_joints_to_random_pose,
                mode="reset",
                params={
                    "poses": [HAND_OPEN_JOINT_POS, HAND_CLOSED_JOINT_POS],
                    "position_range": [-HAND_RESET_RANGE, HAND_RESET_RANGE],
                },
            )

        # the hand terms come before the cube terms, which read the hand pose
        modes = {
            "in_hand": in_hand,
            "near_hand": {"reset_object_to_target": near_hand},
            "open_closed_hand": {
                "reset_hand_pose": open_or_closed_hand(),
                "reset_object_to_target": copy.deepcopy(near_hand),
            },
            "far_from_hand": {"reset_hand_pose": open_or_closed_hand(), "reset_object_to_target": far_from_hand},
            "on_table": {"reset_hand_pose": open_or_closed_hand(), "reset_object": on_table},
        }
        assert list(modes) == list(STAGE_FRACS), "every stage needs a reset mode, in the same order"
        reset["modes"] = modes
        reset["valid_criteria"]["object_in_bounds"] = ManagerTermBaseCfg(
            func=events.object_in_bounds,
            params={"in_bound_range": dict(STAGED_IN_BOUND_RANGE), "margin": OBJECT_RESET_BOUND_MARGIN},
        )
        # the robot keeps its clearance from the table, but the cube may rest on it
        reset["valid_criteria"]["robot_table_clearance"].object_min_clearance = 0.0
        self.terminations.object_out_of_bound.params["in_bound_range"] = dict(STAGED_IN_BOUND_RANGE)
        self.rewards.hand_to_object = RewTerm(
            func=rewards.hand_object_progress,
            weight=REACH_REWARD_WEIGHT,
            params={
                "min_improvement": REACH_REWARD_STEP,
                "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODIES),
            },
        )

        if self.curriculum is None:
            return  # every mode drawn uniformly, without gravity
        reset["mode_weights"] = [float(start <= 0.0) for start, _ in STAGE_FRACS.values()]
        self.curriculum.reset_stages = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "events.conditional_reset.params.mode_weights",
                "modify_fn": curriculums.stage_mode_weights,
                "modify_params": {"stage_fracs": list(STAGE_FRACS.values()), "difficulty_term_str": "adr"},
            },
        )
        gravity = self.curriculum.gravity_adr.params
        gravity["modify_fn"] = curriculums.delayed_interpolate_fn
        gravity["modify_params"]["start_frac"] = GRAVITY_START_FRAC
        gravity["modify_params"]["end_frac"] = TABLE_START_FRAC


@configclass
class FrankaOrcaInHandStagedLiftNoTaxelsEnvCfg(FrankaOrcaInHandStagedLiftEnvCfg):
    """Staged in-hand cube carrying with plain fingertip hulls and no tactile sensing (see :func:`.remove_taxels`)."""

    def __post_init__(self):
        super().__post_init__()
        remove_taxels(self)
