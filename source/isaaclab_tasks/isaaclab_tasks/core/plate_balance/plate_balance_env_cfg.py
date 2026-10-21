# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the plate-balancing environments.

A fixed hand holds a level plate on its fingertips, and a ball rests on the plate. The hand moves only its fingers,
tilting and shifting the plate, to roll the ball to a target point on the plate (:class:`mdp.PlateTargetCommand`).
An episode fails when the ball leaves the plate, the plate falls off the fingers, or a fingertip is off the plate
for more than 0.2 s.

Robot configurations fill in the robot, its pose, the plate and ball placement, the fingertip contact sensors and the
actions.
"""

from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

from . import mdp

PLATE_SIZE = (0.20, 0.15, 0.004)
PLATE_MASS = 0.058
"""Default plate size [m] and mass[kg]; robot configurations size it to their fingertip contacts."""

BALL_RADIUS = 0.035
BALL_MASS = 0.058
"""Ball radius [m] and mass[kg], a Tennis ball."""

##
# Scene definition
##


@configclass
class PlateBalanceSceneCfg(InteractiveSceneCfg):
    """Scene with a fixed hand, the plate on its fingertips and the ball on the plate."""

    robot: ArticulationCfg = MISSING

    plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.CuboidCfg(
            size=PLATE_SIZE,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=PLATE_MASS),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.8), opacity=0.6),
        ),
        init_state=MISSING,
    )
    """The plate, 50 g: light enough for the ORCA fingers' 0.5 Nm, heavy enough to sit firmly on the pads."""

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=BALL_MASS),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.05)),
        ),
        init_state=MISSING,
    )
    """The ball, 10 g."""

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
        collision_group=-1,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ball_target = mdp.PlateTargetCommandCfg(
        resampling_time_range=(5.0, 8.0),
        debug_vis=False,
        # within the plate, the ball's radius and a margin in from its edges
        ranges=mdp.PlateTargetCommandCfg.Ranges(pos_x=(-0.04, 0.04), pos_y=(-0.03, 0.03)),
        marker_height=PLATE_SIZE[2] / 2,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Task state: the ball, the plate, the target and the last action."""

        ball_pos = ObsTerm(func=mdp.ball_pos_plate)
        ball_vel = ObsTerm(func=mdp.ball_vel_plate, clip=(-5.0, 5.0))
        plate_pose = ObsTerm(func=mdp.plate_pose_root)
        target = ObsTerm(func=mdp.generated_commands, params={"command_name": "ball_target"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Proprioception: the finger joints and the fingertip forces on the plate; robot configurations add the
        taxels."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": MISSING})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": MISSING})
        plate_forces = ObsTerm(
            func=mdp.plate_contact_forces_root,
            params={"contact_sensor_names": MISSING},
            clip=(-10.0, 10.0),  # [N], the plate and ball weigh 0.6 N
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP, set by the robot configurations."""


@configclass
class EventCfg:
    """Configuration for events."""

    plate_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("plate"),
            "static_friction_range": [0.7, 1.2],
            "dynamic_friction_range": [0.6, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )

    ball_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.4, 0.8],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )

    # Gaussian scale of the masses: the real ball and plate are weighed, but only roughly
    plate_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("plate"),
            "mass_distribution_params": [1.0, 0.15],
            "operation": "scale",
            "distribution": "gaussian",
            "min_mass": 0.01,
        },
    )

    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": [1.0, 0.15],
            "operation": "scale",
            "distribution": "gaussian",
            "min_mass": 0.01,
        },
    )

    # the fingers start in the pose the plate rests on, the plate on them and the ball near the plate's centre
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": [0.0, 0.0], "velocity_range": [0.0, 0.0]},
    )

    # the hand is mounted a few degrees off, and the plate is put on the fingertips a few degrees turned
    reset_mount = EventTerm(
        func=mdp.reset_tilted_mount,
        mode="reset",
        params={
            "mount_std": (0.035, 0.035, 0.035),  # [rad], 2 deg
            "plate_yaw_std": 0.05,  # [rad], 3 deg
            "ball_pos_range": {"x": (-0.03, 0.03), "y": (-0.02, 0.02)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ball_target = RewTerm(func=mdp.ball_target_exp, weight=1.0, params={"std": 0.05, "command_name": "ball_target"})
    ball_target_fine = RewTerm(
        func=mdp.ball_target_exp, weight=1.0, params={"std": 0.01, "command_name": "ball_target"}
    )
    # 0.05 N is about half of each fingertip's equal share of the plate's weight
    fingers_share_plate = RewTerm(
        func=mdp.fingers_share_plate, weight=2.0, params={"contact_sensor_names": MISSING, "force_scale": 0.05}
    )
    plate_tilt = RewTerm(func=mdp.plate_tilt_l2, weight=-1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": MISSING})
    termination = RewTerm(func=mdp.is_terminated, weight=-50.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    ball_off_plate = DoneTerm(
        func=mdp.ball_off_plate,
        # past the edge, or with its centre below the plate's
        params={"half_extents": (PLATE_SIZE[0] / 2, PLATE_SIZE[1] / 2), "min_height": 0.0},
    )

    plate_dropped = DoneTerm(func=mdp.plate_dropped, params={"max_drop": 0.03, "max_tilt": 0.6})

    # every fingertip must keep the plate: without this, policies hold it on a tripod of fingers, whatever the contact
    # rewards pay, and the fingertips lifted off sense nothing
    fingertip_off_plate = DoneTerm(
        func=mdp.fingertip_off_plate,
        params={"contact_sensor_names": MISSING, "max_air_time": 0.2, "force_threshold": 0.01},
    )


##
# Physics backend presets
##


@configclass
class PhysicsCfg(PresetCfg):
    """Physics backend presets for the plate-balancing environments, the lift settings."""

    isaacsim_physx = PhysxCfg(bounce_threshold_velocity=0.01)
    ovphysx = OvPhysxCfg()
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            impratio=1.0,
            cone="pyramidal",
            update_data_interval=2,
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=4000000),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = newton_mjwarp


##
# Environment configuration
##


@configclass
class PlateBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Ball-on-plate balancing environment."""

    scene: PlateBalanceSceneCfg = PlateBalanceSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2  # 60 Hz
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.6, -0.5, 0.7), lookat=(0.12, 0.0, 0.35))

    def play_mode(self):
        super().play_mode()
        self.commands.ball_target.debug_vis = True
