# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka FR3 arm + right ORCA hand, with halltaxel tactile fingertips.

The asset is a local build output of ``scripts/tools/franka_orca/build_asset.py`` (URDF -> USD with
per-taxel fingertip colliders) and is not shipped with Isaac Lab.

The following configurations are available:

* :obj:`FRANKA_ORCA_CFG`: FR3 arm + right ORCA hand with implicit actuators.
"""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

FRANKA_ORCA_ASSET_DIR = Path(
    os.environ.get(
        "FRANKA_ORCA_ASSET_DIR", Path(__file__).resolve().parents[4] / "assets" / "franka_orca_mesh_fingertips"
    )
)
"""Build output directory, ``<repo>/assets/franka_orca_mesh_fingertips`` unless ``$FRANKA_ORCA_ASSET_DIR`` is set.
Named after the fingertip collision method, so builds of the other methods can sit next to it."""

FRANKA_ORCA_USD = FRANKA_ORCA_ASSET_DIR / "franka_orca.usda"
"""Robot USD, including the baked taxel colliders and the collision filter pairs."""

FRANKA_ORCA_TACTILE_DIR = FRANKA_ORCA_ASSET_DIR / "tactile"
"""Per-finger tactile sidecars (``<finger>.npz``)."""

FR3_ARM_SHOULDER_JOINTS = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4"]
FR3_ARM_FOREARM_JOINTS = ["fr3_joint5", "fr3_joint6", "fr3_joint7"]

ORCA_HAND_JOINTS = [
    "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e",  # wrist
    "I_AP_R_d95d02d1_to_R_Carpals_8d1f1041",  # index MCP
    "I_PP_bacbd481_to_I_AP_R_d95d02d1",  # index PIP
    "I_FingerTipAssembly_ec49c16c_to_I_PP_bacbd481",  # index tip
    "M_AP_6ec59111_to_R_Carpals_8d1f1041",  # ring MCP
    "M_PP_8660a1eb_to_M_AP_6ec59111",  # ring PIP
    "M_FingerTipAssembly_424a8e75_to_M_PP_8660a1eb",  # ring tip
    "M_AP_e04a96f2_to_R_Carpals_8d1f1041",  # middle MCP
    "M_PP_08efa608_to_M_AP_e04a96f2",  # middle PIP
    "M_FingerTipAssembly_34afb748_to_M_PP_08efa608",  # middle tip
    "P_AP_f5e42b61_to_R_Carpals_8d1f1041",  # pinky MCP
    "P_PP_1d411b9b_to_P_AP_f5e42b61",  # pinky PIP
    "P_FingerTipAssembly_cd219176_to_P_PP_1d411b9b",  # pinky tip
    "T_TP_R_1c2b802d_to_R_Carpals_8d1f1041",  # thumb CMC
    "R_T_AP_a9723101_to_T_TP_R_1c2b802d",  # thumb MCP
    "T_PP_68395e98_to_R_T_AP_a9723101",  # thumb PIP
    "T_DP_b7429e50_to_T_PP_68395e98",  # thumb DP
]
"""The 17 actuated ORCA-hand joints."""

ORCA_FINGER_CHAINS = [
    ["I_AP_R_d95d02d1", "I_PP_bacbd481", "I_FingerTipAssembly_ec49c16c"],
    ["M_AP_e04a96f2", "M_PP_08efa608", "M_FingerTipAssembly_34afb748"],
    ["M_AP_6ec59111", "M_PP_8660a1eb", "M_FingerTipAssembly_424a8e75"],
    ["P_AP_f5e42b61", "P_PP_1d411b9b", "P_FingerTipAssembly_cd219176"],
    ["T_TP_R_1c2b802d", "R_T_AP_a9723101", "T_PP_68395e98", "T_DP_b7429e50"],
]
"""Bodies of each finger, base to tip (index, middle, ring, pinky, thumb)."""

##
# Configuration
##

FRANKA_ORCA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(FRANKA_ORCA_USD),
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=[
            # the USD filters the known permanent overlaps (within each finger, wrist vs. forearm);
            # self-collision stays on elsewhere, so fingers collide with each other
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        joint_drive_props=sim_utils.UsdPhysicsDriveCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # 180 deg about Z (XYZW), so the bent home pose reaches towards the lift table at world -X
        rot=(0.0, 0.0, 1.0, 0.0),
        joint_pos={
            "fr3_joint1": -0.026,
            "fr3_joint2": -0.467,
            "fr3_joint3": -0.006,
            "fr3_joint4": -2.006,
            "fr3_joint5": 0.018,
            "fr3_joint6": 2.274,
            "fr3_joint7": -1.823,
            **{name: 0.0 for name in ORCA_HAND_JOINTS},
        },
    ),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=FR3_ARM_SHOULDER_JOINTS,
            effort_limit_sim=87.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=FR3_ARM_FOREARM_JOINTS,
            effort_limit_sim=12.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "orca_hand_fingers": ImplicitActuatorCfg(
            joint_names_expr=ORCA_HAND_JOINTS,
            effort_limit_sim=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""FR3 arm + right ORCA hand, implicit actuators."""
