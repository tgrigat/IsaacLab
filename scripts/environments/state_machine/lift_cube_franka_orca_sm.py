# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to pick and lift a cube with the Franka + ORCA hand through fingertip inverse kinematics.

The grasp is a template attached to the cube: contact points on its faces for the index, middle and thumb pads, and
the palm pose, both in the cube frame. A scripted state machine places the palm with one pose IK target and each
fingertip pad with one position IK target: it moves the hand above the grasp with the pads open (off the contacts
along the face normals), lowers it onto the grasp, presses the pads in along the normals and lifts. The ring and
pinky fingers hold their rest pose. No policy is involved, so this tests the fingertip IK on its own.

With a visualizer, the IK markers show the targets (green spheres, the larger palm frame) and the controlled points
(red spheres, the smaller palm frame); blue spheres show the contact points on the cube. In the Newton viewer, press
``R`` to reset the environments and restart the pick and lift.

.. code-block:: bash

    uv run python scripts/environments/state_machine/lift_cube_franka_orca_sm.py --num_envs 4 --viz newton_gl

"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Fingertip-IK pick and lift state machine for the Franka + ORCA hand.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument(
    "--num_steps", type=int, default=None, help="Environment steps to run; runs until the window closes if unset."
)
add_launcher_args(parser)
# parse the arguments, forwarding unrecognized ones as Hydra-style task config overrides
args_cli, hydra_overrides = parser.parse_known_args()

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.lift.config.franka_orca.franka_orca_env_cfg import FINGERTIPS, PALM_BODY
from isaaclab_tasks.core.lift.config.franka_orca.franka_orca_ik_env_cfg import FINGERTIP_PAD_OFFSETS
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-Lift-Cube-FrankaOrca-FingertipIK-Abs"

##
# Grasp template, in the cube frame (x, y, z [m]; z up), for the 3.75 cm cube of the task.
#
# Found by sweeping each finger's joints for the pad positions and pad normals (the fingertip's taxel side) it
# reaches in the palm frame, and searching for a pinch whose open and closed pad positions all lie in them with each pad
# facing into its contact face (all within about 20 deg). Pads that press with the fingertip ends or sides slide off
# the cube. On this hand the thumb opposes the curled index and middle pads from the base of the palm, so the pinch
# axis (cube x) runs back along the fingers; holding it level tilts the hand, with the palm about 11 cm above the cube.
##

GRASP_PALM_POS = (0.0382, 0.0, 0.1134)
"""Palm position in the cube frame [m]."""

GRASP_PALM_QUAT = (0.4325, 0.7921, -0.2983, -0.3105)
"""Palm orientation (x, y, z, w) in the cube frame."""

CONTACT_FINGERS = (0, 1, 4)
"""Fingers with a contact, as indices into the action's fingertip order (index, middle, ring, pinky, thumb)."""

CONTACT_POINTS = ((-0.0188, -0.0116, -0.0031), (-0.0188, 0.0116, 0.0031), (0.0188, 0.0, 0.0))
"""Contact points of the index, middle and thumb pads on the cube faces [m]."""

CONTACT_NORMALS = ((-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
"""Outward normals of the faces the contacts are on."""

OPEN_CLEARANCE = 0.015
"""Distance of the open pads from their contacts along the face normals [m]."""

SQUEEZE = 0.010
"""Depth of the closed pad targets below the cube faces [m], so the pads press on the cube."""

CLOSE_TIME = 0.6
"""Time the pads take to move from open to closed [s]; closing them at once lets the index and middle pads slap the
cube out from under the thumb before it arrives."""

APPROACH_HEIGHT = 0.08
"""Height of the palm above its grasp pose while approaching [m]."""

LIFT_HEIGHT = 0.15
"""Height the palm lifts the cube by [m]."""

LIFT_TIME = 1.5
"""Time the palm takes to lift the cube [s]; jumping the palm target up at once jerks the cube out of the pinch."""


class State:
    """States of the pick and lift state machine."""

    REST = 0
    APPROACH = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4


WAIT_TIME = {
    State.REST: 0.3,
    State.APPROACH: 0.3,
    State.DESCEND: 0.3,
    State.GRASP: CLOSE_TIME + 0.5,
    State.LIFT: float("inf"),
}
"""Time to hold each state once its palm target is reached [s]."""

TIMEOUT = {
    State.REST: 0.3,
    State.APPROACH: 4.0,
    State.DESCEND: 3.0,
    State.GRASP: CLOSE_TIME + 0.5,
    State.LIFT: float("inf"),
}
"""Time after which a state moves on whether or not its palm target is reached [s]."""

PALM_TOLERANCE = (0.005, 0.05)
"""Palm position [m] and orientation [rad] errors under which a palm target counts as reached."""


class GraspTemplatePickAndLiftSm:
    """Pick and lift state machine driven by a grasp template attached to the cube.

    Computes the absolute IK action ``[palm (x, y, z, qx, qy, qz, qw), 5 fingertips (x, y, z)]`` in the robot root
    frame. The pad targets are held in the palm frame, so they follow the palm on its way to the grasp: open, they
    are where the template puts them relative to the palm; closed, they are the contacts pressed in along the normals,
    placed on the cube where it actually is when the grasp starts.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str):
        """Initialize the state machine.

        Args:
            dt: The environment step [s].
            num_envs: The number of environments.
            device: The device of the environment.
        """
        self.dt = dt
        self.num_envs = num_envs
        self.device = device
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.state_time = torch.zeros(num_envs, device=device)
        self.reached_time = torch.zeros(num_envs, device=device)
        # captured while resting: the palm orientation (picks the cube yaw to grasp from) and the pad positions of
        # the fingers without a contact; captured on entering GRASP: the palm pose and the closed pad targets
        self.palm_quat_rest = torch.zeros(num_envs, 4, device=device)
        self.pads_rest = torch.zeros(num_envs, 5, 3, device=device)
        self.palm_pos_grasp = torch.zeros(num_envs, 3, device=device)
        self.palm_quat_grasp = torch.zeros(num_envs, 4, device=device)
        self.pads_closed = torch.zeros(num_envs, len(CONTACT_FINGERS), 3, device=device)

        self.palm_pos_c = torch.tensor(GRASP_PALM_POS, device=device).expand(num_envs, 3)
        self.palm_quat_c = torch.tensor(GRASP_PALM_QUAT, device=device).expand(num_envs, 4)
        contacts = torch.tensor(CONTACT_POINTS, device=device)
        normals = torch.tensor(CONTACT_NORMALS, device=device)
        self.contacts_c = contacts.expand(num_envs, -1, -1)
        self.pads_open_c = (contacts + OPEN_CLEARANCE * normals).expand(num_envs, -1, -1)
        self.pads_closed_c = (contacts - SQUEEZE * normals).expand(num_envs, -1, -1)
        # the cube's yaw symmetries, about its z axis
        angles = torch.arange(4, device=device) * math.pi / 2
        self.yaw_symmetries = math_utils.quat_from_angle_axis(angles, torch.tensor([0.0, 0.0, 1.0], device=device))

    def reset_idx(self, env_ids: Sequence[int] | torch.Tensor):
        """Restart the state machine of some environments."""
        self.state[env_ids] = State.REST
        self.state_time[env_ids] = 0.0
        self.reached_time[env_ids] = 0.0

    def grasp_frame(self, cube_pos: torch.Tensor, cube_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cube frame the template is applied in, in the root frame.

        The cube's yaw, turned by the quarter turn that brings the grasp palm orientation closest to the rest palm
        orientation, so the arm turns the hand as little as possible. Assumes the cube rests on a face.
        """
        candidates = math_utils.quat_mul(
            math_utils.yaw_quat(cube_quat)[:, None].expand(-1, 4, -1),
            self.yaw_symmetries[None].expand(self.num_envs, -1, -1),
        )
        palm_quats = math_utils.quat_mul(candidates, self.palm_quat_c[:, None].expand(-1, 4, -1))
        error = math_utils.quat_error_magnitude(palm_quats, self.palm_quat_rest[:, None].expand(-1, 4, -1))
        best = torch.argmin(error, dim=1)
        return cube_pos, candidates[torch.arange(self.num_envs, device=self.device), best]

    def contacts_b(self, cube_pos: torch.Tensor, cube_quat: torch.Tensor) -> torch.Tensor:
        """Contact points in the root frame. Shape is (num_envs, 3, 3)."""
        frame_pos, frame_quat = self.grasp_frame(cube_pos, cube_quat)
        return self._apply(frame_pos, frame_quat, self.contacts_c)

    def compute(
        self,
        palm_pos: torch.Tensor,
        palm_quat: torch.Tensor,
        cube_pos: torch.Tensor,
        cube_quat: torch.Tensor,
        pads_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the state machine and compute the IK action.

        Args:
            palm_pos: Palm position in the root frame. Shape is (num_envs, 3).
            palm_quat: Palm orientation (x, y, z, w) in the root frame. Shape is (num_envs, 4).
            cube_pos: Cube position in the root frame. Shape is (num_envs, 3).
            cube_quat: Cube orientation (x, y, z, w) in the root frame. Shape is (num_envs, 4).
            pads_pos: Fingertip pad positions in the root frame. Shape is (num_envs, 5, 3).

        Returns:
            The IK action. Shape is (num_envs, 22).
        """
        resting = self.state == State.REST
        self.palm_quat_rest[resting] = palm_quat[resting]
        self.pads_rest[resting] = self._to_frame(palm_pos, palm_quat, pads_pos)[resting]

        # palm targets: the template's palm pose on the cube, above it while approaching
        frame_pos, frame_quat = self.grasp_frame(cube_pos, cube_quat)
        grasp_pos, grasp_quat = math_utils.combine_frame_transforms(
            frame_pos, frame_quat, self.palm_pos_c, self.palm_quat_c
        )
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        lift_height = LIFT_HEIGHT * torch.clamp(self.state_time / LIFT_TIME, max=1.0)
        palm_pos_des = palm_pos.clone()
        palm_quat_des = grasp_quat.clone()
        for state, pos in (
            (State.REST, palm_pos),
            (State.APPROACH, grasp_pos + APPROACH_HEIGHT * up),
            (State.DESCEND, grasp_pos),
            # the grasp holds the palm where the descent left it, so closing pads do not drag it after the cube
            (State.GRASP, self.palm_pos_grasp),
            (State.LIFT, self.palm_pos_grasp + lift_height[:, None] * up),
        ):
            palm_pos_des[self.state == state] = pos[self.state == state]
        palm_quat_des[resting] = palm_quat[resting]
        holding = (self.state == State.GRASP) | (self.state == State.LIFT)
        palm_quat_des[holding] = self.palm_quat_grasp[holding]

        # pad targets in the palm frame: the fingers without a contact at rest, the others open where the template
        # puts them relative to the palm, then moving over CLOSE_TIME to closed on the cube where it was at the grasp
        pads_open = self._to_frame(self.palm_pos_c, self.palm_quat_c, self.pads_open_c)
        moving = (self.state == State.APPROACH) | (self.state == State.DESCEND)
        closing = torch.where(
            self.state == State.GRASP, torch.clamp(self.state_time / CLOSE_TIME, max=1.0), (self.state == State.LIFT)
        )[:, None, None]
        pads_contact = torch.where(moving[:, None, None], pads_open, self.pads_rest[:, CONTACT_FINGERS])
        pads_contact = torch.where(
            holding[:, None, None], (1.0 - closing) * pads_open + closing * self.pads_closed, pads_contact
        )
        pads_palm = self.pads_rest.clone()
        pads_palm[:, CONTACT_FINGERS] = pads_contact
        pads_des = self._apply(palm_pos, palm_quat, pads_palm)

        # advance: a state ends once its palm target has been reached for WAIT_TIME, or on its TIMEOUT
        self.state_time += self.dt
        reached = (torch.linalg.norm(palm_pos - palm_pos_des, dim=-1) < PALM_TOLERANCE[0]) & (
            math_utils.quat_error_magnitude(palm_quat, palm_quat_des) < PALM_TOLERANCE[1]
        )
        self.reached_time = torch.where(reached, self.reached_time + self.dt, torch.zeros_like(self.reached_time))
        wait = torch.tensor([WAIT_TIME[s] for s in range(5)], device=self.device)[self.state]
        timeout = torch.tensor([TIMEOUT[s] for s in range(5)], device=self.device)[self.state]
        done = (self.reached_time >= wait) | (self.state_time >= timeout)
        # the grasp closes the pads on the cube where it is, and holds and lifts the palm from where it is
        entering_grasp = done & (self.state == State.DESCEND)
        if entering_grasp.any():
            closed_b = self._apply(frame_pos, frame_quat, self.pads_closed_c)
            closed_palm = self._to_frame(palm_pos, palm_quat, closed_b)
            self.pads_closed[entering_grasp] = closed_palm[entering_grasp]
            self.palm_pos_grasp[entering_grasp] = palm_pos[entering_grasp]
            self.palm_quat_grasp[entering_grasp] = palm_quat[entering_grasp]
        self.state = torch.where(done, torch.clamp(self.state + 1, max=State.LIFT), self.state)
        self.state_time[done] = 0.0
        self.reached_time[done] = 0.0

        palm_action = torch.cat([palm_pos_des, palm_quat_des], dim=-1)
        return torch.cat([palm_action, pads_des.flatten(1)], dim=-1)

    @staticmethod
    def _apply(frame_pos: torch.Tensor, frame_quat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Points given in a frame, expressed in the frame's parent. Points have shape (num_envs, M, 3)."""
        quat = frame_quat[:, None].expand(-1, points.shape[1], -1)
        return frame_pos[:, None] + math_utils.quat_apply(quat, points)

    @staticmethod
    def _to_frame(frame_pos: torch.Tensor, frame_quat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Points given in a frame's parent, expressed in the frame. Points have shape (num_envs, M, 3)."""
        quat = frame_quat[:, None].expand(-1, points.shape[1], -1)
        return math_utils.quat_apply_inverse(quat, points - frame_pos[:, None])


def body_pose_b(env: ManagerBasedRLEnv, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Poses of robot bodies in the robot root frame, shapes (num_envs, len(body_ids), 3) and (..., 4)."""
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w.torch[:, None].expand(-1, len(body_ids), -1)
    root_quat = robot.data.root_quat_w.torch[:, None].expand(-1, len(body_ids), -1)
    return math_utils.subtract_frame_transforms(
        root_pos, root_quat, robot.data.body_pos_w.torch[:, body_ids], robot.data.body_quat_w.torch[:, body_ids]
    )


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, overrides=hydra_overrides)
    with launch_simulation(env_cfg, args_cli):
        env: ManagerBasedRLEnv = gym.make(TASK, cfg=env_cfg).unwrapped
        env.reset()
        robot = env.scene["robot"]
        cube = env.scene["object"]
        palm_id = robot.find_bodies(PALM_BODY)[0]
        tip_ids = robot.find_bodies(FINGERTIPS, preserve_order=True)[0]
        pad_offsets = torch.tensor(FINGERTIP_PAD_OFFSETS, device=env.device).expand(env.num_envs, -1, -1)
        contact_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/GraspContacts",
                markers={
                    "contact": sim_utils.SphereCfg(
                        radius=0.004, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.3, 1.0))
                    )
                },
            )
        )

        # the Newton viewer reports held keys; R resets on the press, not while held
        viewer = next((viz for viz in env.sim.visualizers if hasattr(viz, "is_key_down")), None)
        if viewer is not None:
            print("[INFO]: Press R in the viewer to reset the environments.")
        reset_key_down = False

        sm = GraspTemplatePickAndLiftSm(env.step_dt, env.num_envs, env.device)
        cube_start_z = torch.full((env.num_envs,), float("nan"), device=env.device)
        pads_des = palm_pos_des = None
        step = 0
        while env.sim.is_running() and (args_cli.num_steps is None or step < args_cli.num_steps):
            with torch.inference_mode():
                reset_key_was_down, reset_key_down = reset_key_down, viewer is not None and viewer.is_key_down("r")
                if reset_key_down and not reset_key_was_down:
                    env.reset()
                    sm.reset_idx(torch.arange(env.num_envs, device=env.device))
                    cube_start_z[:] = float("nan")
                    pads_des = palm_pos_des = None
                    print(f"[step {step:4d}] reset")
                root_pos, root_quat = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
                palm_pos, palm_quat = (x[:, 0] for x in body_pose_b(env, palm_id))
                pads_pos, tips_quat = body_pose_b(env, tip_ids)
                pads_pos = pads_pos + math_utils.quat_apply(tips_quat, pad_offsets)
                cube_pos, cube_quat = math_utils.subtract_frame_transforms(
                    root_pos, root_quat, cube.data.root_pos_w.torch, cube.data.root_quat_w.torch
                )
                cube_start_z = torch.where(cube_start_z.isnan(), cube_pos[:, 2], cube_start_z)

                # report every second: state, IK errors to the previous step's targets, cube lift
                if pads_des is not None and step % int(1.0 / env.step_dt) == 0:
                    palm_err = torch.linalg.norm(palm_pos_des - palm_pos, dim=-1) * 1000
                    pads_err = torch.linalg.norm(pads_des - pads_pos, dim=-1) * 1000
                    lift = (cube_pos[:, 2] - cube_start_z) * 100
                    for i in range(env.num_envs):
                        print(
                            f"[step {step:4d}] env {i} state {int(sm.state[i])}: IK error palm {palm_err[i]:.0f} mm,"
                            f" pads {[round(e) for e in pads_err[i].tolist()]} mm; cube lift {lift[i]:+.1f} cm"
                        )

                actions = sm.compute(palm_pos, palm_quat, cube_pos, cube_quat, pads_pos)
                palm_pos_des, pads_des = actions[:, :3], actions[:, 7:].view(-1, 5, 3)
                # contact markers, in the world frame
                contacts = sm.contacts_b(cube_pos, cube_quat)
                contacts_w = root_pos[:, None] + math_utils.quat_apply(
                    root_quat[:, None].expand(-1, contacts.shape[1], -1), contacts
                )
                contact_markers.visualize(contacts_w.reshape(-1, 3))
                env.step(actions)
                dones = env.termination_manager.dones
                if dones.any():
                    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    sm.reset_idx(done_ids)
                    cube_start_z[done_ids] = float("nan")
                step += 1

        env.close()


if __name__ == "__main__":
    main()
