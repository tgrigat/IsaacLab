# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a trained ORCA-hand plate-balancing policy on the real hand, through ``orca_core``.

The policy is either a tactile student of ``Isaac-PlateBalance-OrcaHand-NoTaxels-Student*`` or a centre-of-pressure
policy of ``Isaac-PlateBalance-OrcaHand-CoP*``, trained directly for the real hand as in Pan et al. (2026), "Beyond
Binary". Both observe the finger joints, the ball's target on the plate, their last action and a tactile signal from
each fingertip pad, and neither the ball nor the plate. The NoTaxels teacher observes the ball and the plate, which the
real hand cannot sense, so it is not supported.

Everything the policy was trained with comes from the run directory: the exported TorchScript policy
(``exported/policy.pt``, written by ``scripts/reinforcement_learning/play.py``) and ``params/env.yaml`` (the plate
pose, the action joints, type and scale, the control rate, the policy's observation terms and the target ranges). The
observation noise and delays of the centre-of-pressure policies are training-only and not applied here.

Sim to hardware conventions:

* Joints: Isaac Lab uses the URDF joints in radians, ``orca_core`` its joint ids in degrees. The four finger
  abduction joints are inverted (``sign: -1`` in ``orca_core/kinematics/data/v2_kinematics.yaml``: positive URDF
  abduction moves a finger towards the thumb, positive ``orca_core`` abduction towards the pinky); the others match.
* Tactile: ``orca_core`` streams each pad's taxel forces, rotated into the fingertip (distal link) frame, the same
  frame as the simulated fingertip body. A press reads as a force out of the pad, while the simulation observes the
  plate's force on the fingertip, into the pad, so the forces are negated (``--tactile-sign``). The contact point is
  the force-weighted mean of the taxel positions. The centre of pressure's normal force is the total force into the
  pad along the pad's normal at the pad point nearest the contact point, from the same pad mesh as in the simulation.
  With ``--taxel-calibration`` (from ``calibrate_taxels.py``), each taxel's force is first rotated by its calibrated
  rotation.

Physical setup, as in the simulation (``orca_hand_env_cfg.py``): the forearm fixed with the palm facing up, tilted
about 30 deg towards the wrist, the forearm's long axis 10 deg below level, so all five pads face straight up in the
plate pose; the wrist held at 0 deg. The target ``(x, y)`` is in the plate frame: ``x`` along the direction the
fingers reach, ``y`` to their left, from the plate's centre (the centroid of the five fingertip contacts).

Procedure: the hand moves to the plate pose; with nothing touching the pads the tactile sensors are zeroed; you place
the plate and then the ball on the fingertips and start the policy; Ctrl+C stops it, the hand returns to the plate
pose, and torque is released only once you have taken the plate off.

Examples:

.. code-block:: bash

    # without hardware: mock hand, no tactile data
    uv run python scripts/real_robot/plate_balance/play_orca_hand.py \\
        logs/rsl_rl/plate_balance_orca_hand_distillation/2026-10-17_09-07-25_student_binary --mock --duration 2

    # real hand, policy running but only holding the plate pose, to check the observations first
    uv run python scripts/real_robot/plate_balance/play_orca_hand.py <run_dir> \\
        --config ../orca_core/orca_core/models/v2/orcahand-touch-right/config.yaml --no-actuate

    # real hand
    uv run python scripts/real_robot/plate_balance/play_orca_hand.py <run_dir> \\
        --config ../orca_core/orca_core/models/v2/orcahand-touch-right/config.yaml --target 0.02 0.0 --log run.npz
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml
from orca_core import OrcaHandTouch, load_hand

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
"""Fingers in the order of the tactile observations (the plate contact sensors in ``orca_hand_env_cfg.py``)."""

WRIST_JOINT = "R_Carpals_8d1f1041_to_TopTower_Model_4a80d30e"

URDF_TO_ORCA = {
    WRIST_JOINT: ("wrist", 1.0),
    "I_AP_R_d95d02d1_to_R_Carpals_8d1f1041": ("index_abd", -1.0),
    "I_PP_bacbd481_to_I_AP_R_d95d02d1": ("index_mcp", 1.0),
    "I_FingerTipAssembly_ec49c16c_to_I_PP_bacbd481": ("index_pip", 1.0),
    "M_AP_e04a96f2_to_R_Carpals_8d1f1041": ("middle_abd", -1.0),
    "M_PP_08efa608_to_M_AP_e04a96f2": ("middle_mcp", 1.0),
    "M_FingerTipAssembly_34afb748_to_M_PP_08efa608": ("middle_pip", 1.0),
    "M_AP_6ec59111_to_R_Carpals_8d1f1041": ("ring_abd", -1.0),
    "M_PP_8660a1eb_to_M_AP_6ec59111": ("ring_mcp", 1.0),
    "M_FingerTipAssembly_424a8e75_to_M_PP_8660a1eb": ("ring_pip", 1.0),
    "P_AP_f5e42b61_to_R_Carpals_8d1f1041": ("pinky_abd", -1.0),
    "P_PP_1d411b9b_to_P_AP_f5e42b61": ("pinky_mcp", 1.0),
    "P_FingerTipAssembly_cd219176_to_P_PP_1d411b9b": ("pinky_pip", 1.0),
    "T_TP_R_1c2b802d_to_R_Carpals_8d1f1041": ("thumb_cmc", 1.0),
    "R_T_AP_a9723101_to_T_TP_R_1c2b802d": ("thumb_abd", 1.0),
    "T_PP_68395e98_to_R_T_AP_a9723101": ("thumb_mcp", 1.0),
    "T_DP_b7429e50_to_T_PP_68395e98": ("thumb_dip", 1.0),
}
"""``orca_core`` joint id of each URDF joint, and the sign from the URDF angle to the ``orca_core`` angle."""

TACTILE_SIZES = {"force": 3, "position": 3, "direction": 3, "binary": 1, "cop": 4}
"""Observation size per fingertip of each tactile signal (``TACTILE_SIGNALS`` and ``CONTACT_SIGNALS`` in
``orca_hand_env_cfg.py``)."""

TACTILE_FUNCS = {
    "pad_contact_forces_b": "force",
    "pad_contact_points_b": "position",
    "pad_contact_directions_b": "direction",
    "pad_contact_binary": "binary",
    "pad_contact_cop_b": "cop",
}
"""Tactile signal of each tactile observation function."""

PROPRIO_FUNCS = {
    "joint_pos_rel": "joint_pos",
    "joint_vel_rel": "joint_vel",
    "generated_commands": "target",
    "last_action": "actions",
}
"""Non-tactile observation of each observation function."""

PROPRIO_SIZES = {"joint_pos": 16, "joint_vel": 16, "target": 2, "actions": 16}
"""Observation size of each non-tactile observation."""


##
# Run loading
##


class _EnvYamlLoader(yaml.SafeLoader):
    """Loads ``params/env.yaml``, reading its Python-tagged nodes (tuples, slices, ...) as plain values."""


def _construct_python(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_EnvYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python)


@dataclass
class PolicySpec:
    """What the policy was trained with, read from its run directory."""

    task: str
    policy_path: Path
    step_dt: float
    """Control period [s]."""
    action_joints: list[str]
    """URDF joints of the actions, also the order of the joint observations."""
    action_scale: float
    """Joint target offset [rad] per unit action."""
    default_joint_pos: dict[str, float]
    """URDF joint positions [rad] of the plate pose, the action offset and the joint observations' zero."""
    obs_terms: list[str]
    """Observations, keys of :data:`PROPRIO_SIZES` or ``"tactile"``, in observation order."""
    tactile: list[str]
    """Tactile signals, keys of :data:`TACTILE_SIZES`, in observation order; together where ``obs_terms`` has
    ``"tactile"``."""
    contact_threshold: float
    """Force [N] above which the binary signal senses a contact."""
    target_ranges: tuple[tuple[float, float], tuple[float, float]]
    """Target ``x`` and ``y`` ranges [m] in the plate frame."""
    resampling_time_range: tuple[float, float]
    """Time [s] between targets."""
    action_type: str = "absolute"
    """``"absolute"``: targets offset from the plate pose; ``"ema"``: targets moved by smoothed increments."""
    ema_alpha: float = 1.0
    """Weight of the new increment of ``"ema"`` actions."""
    action_clip: float = float("inf")
    """Magnitude the raw ``"ema"`` actions are clipped to before scaling."""
    pad_regions: list[str] | None = None
    """Pad region mesh of each finger, in :data:`FINGERS` order, for the centre of pressure's pad normal."""

    @property
    def obs_dim(self) -> int:
        sizes = {**PROPRIO_SIZES, "tactile": len(FINGERS) * sum(TACTILE_SIZES[s] for s in self.tactile)}
        return sum(sizes[term] for term in self.obs_terms)


def load_spec(run: Path) -> PolicySpec:
    """Read the policy's training settings from a run directory, or from its ``exported/policy.pt``."""
    run_dir = run.parents[1] if run.is_file() else run
    policy_path = run if run.is_file() else run_dir / "exported" / "policy.pt"
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"no exported policy at {policy_path}; export it with"
            f" scripts/reinforcement_learning/rsl_rl/play.py --task <task> --checkpoint <run_dir>/model_<it>.pt"
        )
    task = json.loads((run_dir / "run.json").read_text())["task"]
    env = yaml.load((run_dir / "params" / "env.yaml").read_text(), Loader=_EnvYamlLoader)

    # tactile students observe the student group, centre-of-pressure policies the actor group, as they run here
    group_name = next((g for g in ("student", "actor") if env["observations"].get(g) is not None), None)
    if group_name is None:
        raise ValueError(
            f"{task} is neither a tactile student nor a centre-of-pressure policy: its policy observes the ball and"
            " the plate, which the real hand cannot sense. Use a run of Isaac-PlateBalance-OrcaHand-NoTaxels-Student*"
            " or Isaac-PlateBalance-OrcaHand-CoP*."
        )
    group = env["observations"][group_name]
    if group.get("history_length"):
        raise ValueError(f"{task} observes a history, which this script does not stack")
    terms = {name: term for name, term in group.items() if isinstance(term, dict) and "func" in term}
    obs_terms, tactile, contact_threshold, pad_regions = [], [], 0.05, None
    for name, term in terms.items():
        func = term["func"].rsplit(":", 1)[-1]
        if term.get("scale") is not None or term.get("modifiers"):
            raise ValueError(f"observation {name} has a scale or modifiers, which this script does not apply")
        if term.get("noise") is not None and group_name == "student":
            raise ValueError(f"student observation {name} has noise, which this script does not apply")
        if func in PROPRIO_FUNCS:
            obs_terms.append(PROPRIO_FUNCS[func])
        elif func in TACTILE_FUNCS:
            if "tactile" not in obs_terms:
                obs_terms.append("tactile")
            elif obs_terms[-1] != "tactile":
                raise ValueError("the tactile observations must be consecutive")
            tactile.append(TACTILE_FUNCS[func])
            if func == "pad_contact_binary":
                contact_threshold = float(term["params"]["force_threshold"])
            if func == "pad_contact_cop_b":
                pad_regions = list(term["params"]["pad_regions"])
        else:
            raise ValueError(f"unsupported observation {name} ({func})")

    actions = env["actions"]["fingers"]
    action_type, ema_alpha, action_clip = "absolute", 1.0, float("inf")
    if "EMARelativeJointPositionAction" in str(actions.get("class_type", "")):
        action_type, ema_alpha, action_clip = "ema", float(actions["alpha"]), float(actions["clip_value"])
    elif not actions["use_default_offset"] or actions.get("clip") is not None:
        raise ValueError("expected joint position actions offset from the default joint positions, without clipping")
    command = env["commands"]["ball_target"]
    return PolicySpec(
        task=task,
        policy_path=policy_path,
        step_dt=env["sim"]["dt"] * env["decimation"],
        action_joints=list(actions["joint_names"]),
        action_scale=float(actions["scale"]),
        default_joint_pos={k: float(v) for k, v in env["scene"]["robot"]["init_state"]["joint_pos"].items()},
        obs_terms=obs_terms,
        tactile=tactile,
        contact_threshold=contact_threshold,
        target_ranges=(tuple(command["ranges"]["pos_x"]), tuple(command["ranges"]["pos_y"])),
        resampling_time_range=tuple(command["resampling_time_range"]),
        action_type=action_type,
        ema_alpha=ema_alpha,
        action_clip=action_clip,
        pad_regions=pad_regions,
    )


def load_policy(spec: PolicySpec) -> torch.jit.ScriptModule:
    """Load the exported policy and check it takes the student observation."""
    policy = torch.jit.load(str(spec.policy_path), map_location="cpu").eval()
    obs_dim = policy.obs_normalizer._mean.shape[-1]
    if obs_dim != spec.obs_dim:
        raise ValueError(f"policy takes {obs_dim} observations, {spec.task} has {spec.obs_dim}")
    return policy


def reset_policy(policy: torch.jit.ScriptModule):
    """Zero the recurrent state, as at the start of an episode."""
    for name, buffer in policy.named_buffers():
        if name in ("hidden_state", "cell_state"):
            buffer.zero_()


##
# Joint conversion
##


def urdf_to_orca(joint_pos: dict[str, float]) -> dict[str, float]:
    """URDF joint positions [rad] to ``orca_core`` joint positions [deg]."""
    return {URDF_TO_ORCA[j][0]: URDF_TO_ORCA[j][1] * np.degrees(q) for j, q in joint_pos.items()}


def orca_to_urdf(joint_pos: dict[str, float | None], joints: list[str]) -> np.ndarray:
    """``orca_core`` joint positions [deg] to URDF joint positions [rad] of ``joints``."""
    values = []
    for joint in joints:
        orca_id, sign = URDF_TO_ORCA[joint]
        if joint_pos.get(orca_id) is None:
            raise RuntimeError(f"no position for {orca_id}: is the joint calibrated?")
        values.append(sign * np.radians(joint_pos[orca_id]))
    return np.array(values)


##
# Tactile observations
##


PAD_POINT_SPACING = 0.5e-3
"""Distance [m] between neighbouring pad points, as ``PAD_POINT_SPACING`` in the task's ``mdp/observations.py``."""


def pad_surfaces(spec: PolicySpec) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Points [m] covering each finger's pad region mesh and its outward unit normals there, as ``pad_surface`` in
    the task's ``mdp/observations.py``; ``None`` if the policy does not observe the centre of pressure."""
    if spec.pad_regions is None:
        return None
    surfaces = {}
    for finger, path in zip(FINGERS, spec.pad_regions):
        mesh = trimesh.load(path, force="mesh")
        mesh.merge_vertices()
        vertices, faces = trimesh.remesh.subdivide_to_size(
            mesh.vertices, mesh.faces, max_edge=PAD_POINT_SPACING, max_iter=20
        )
        mesh = trimesh.Trimesh(vertices, faces)
        _, keep = np.unique(np.floor(mesh.vertices / PAD_POINT_SPACING), axis=0, return_index=True)
        surfaces[finger] = (np.asarray(mesh.vertices[keep]), np.asarray(mesh.vertex_normals[keep]))
    return surfaces


def load_taxel_calibration(path: Path | None) -> dict[str, np.ndarray]:
    """Per-taxel rotations of each finger, shape (num_taxels, 3, 3), from ``calibrate_taxels.py``; none without a
    calibration file."""
    if path is None:
        return {}
    data = np.load(path)
    return {finger: data[f"{finger}_R"] for finger in FINGERS if f"{finger}_R" in data}


def read_tactile(
    hand,
    spec: PolicySpec,
    tactile_sign: float,
    pads: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    calibration: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    """The tactile observation, each finger's signals in :data:`FINGERS` order, and each finger's force [N] in its
    fingertip frame (``None`` without a tactile frame).

    As in the simulation, the force is the plate's force on the fingertip, and the contact point, force direction and
    centre of pressure are zero without a contact. With ``calibration``, each taxel's force is rotated by its
    calibrated rotation first.
    """
    data = hand.get_taxel_data(frame="fingertip") if isinstance(hand, OrcaHandTouch) else None
    forces, signals = {}, []
    for finger in FINGERS:
        taxels = data.get(finger) if data is not None else None
        if taxels is None:
            force, point = np.zeros(3), np.zeros(3)
        else:
            taxel_forces = tactile_sign * taxels.forces
            rotations = (calibration or {}).get(finger)
            if rotations is not None and len(rotations) == len(taxel_forces):
                taxel_forces = np.einsum("tij,tj->ti", rotations, taxel_forces)
            force = taxel_forces.sum(axis=0)
            weights = np.linalg.norm(taxel_forces, axis=1)
            point = weights @ taxels.positions / weights.sum() if weights.sum() > 0.0 else np.zeros(3)
        forces[finger] = force
        norm = np.linalg.norm(force)
        in_contact = norm > spec.contact_threshold
        for signal in spec.tactile:
            if signal == "force":
                signals.append(np.clip(force, -10.0, 10.0))
            elif signal == "position":
                signals.append(point if in_contact else np.zeros(3))
            elif signal == "direction":
                signals.append(force / norm if in_contact else np.zeros(3))
            elif signal == "binary":
                signals.append(np.array([float(in_contact)]))
            elif signal == "cop":
                if in_contact:
                    pad_points, pad_normals = pads[finger]
                    normal = pad_normals[np.argmin(np.linalg.norm(pad_points - point, axis=1))]
                    signals.append(np.array([max(-force @ normal, 0.0), *point]))
                else:
                    signals.append(np.zeros(4))
    # the observation is grouped by signal, then by finger
    per_finger = len(spec.tactile)
    grouped = [signals[f * per_finger + s] for s in range(per_finger) for f in range(len(FINGERS))]
    return np.concatenate(grouped) if grouped else np.zeros(0), (forces if data is not None else None)


##
# Main
##


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="Run directory of a tactile student, or its exported policy.pt.")
    parser.add_argument("--config", default=None, help="orca_core config.yaml of the hand; autodetected if omitted.")
    parser.add_argument("--mock", action="store_true", help="Use orca_core's mock hand, without hardware.")
    parser.add_argument(
        "--no-actuate", action="store_true", help="Run the policy but hold the plate pose, to check observations."
    )
    parser.add_argument("--target", type=float, nargs=2, default=(0.0, 0.0), help="Ball target (x, y) [m].")
    parser.add_argument(
        "--random-targets", action="store_true", help="Resample the target as in training, in its ranges and period."
    )
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds.")
    parser.add_argument(
        "--max-action",
        type=float,
        default=5.0,
        help="Clip the commanded actions to this magnitude (a safety limit; the policy observes its unclipped action).",
    )
    parser.add_argument(
        "--max-target-offset",
        type=float,
        default=0.5,
        help="Keep the joint targets of 'ema' actions within this offset [rad] of the plate pose (a safety limit).",
    )
    parser.add_argument(
        "--contact-threshold", type=float, default=None, help="Contact force [N]; defaults to the training value."
    )
    parser.add_argument("--tactile-sign", type=float, default=-1.0, help="Sign from sensor forces to plate forces.")
    parser.add_argument(
        "--taxel-calibration", type=Path, default=None, help="Per-taxel rotations from calibrate_taxels.py (.npz)."
    )
    parser.add_argument("--vel-filter", type=float, default=0.5, help="Low-pass weight of new joint velocity samples.")
    parser.add_argument("--move-time", type=float, default=3.0, help="Time [s] to move to the plate pose.")
    parser.add_argument("--print-every", type=float, default=0.5, help="Status print period [s].")
    parser.add_argument("--log", type=Path, default=None, help="Save the observations and actions to this .npz.")
    return parser.parse_args()


def prompt(message: str, args: argparse.Namespace):
    if args.mock:
        print(message)
    else:
        input(f"{message} Press Enter to continue.")


def main():
    args = parse_args()
    spec = load_spec(args.run)
    if args.contact_threshold is not None:
        spec.contact_threshold = args.contact_threshold
    policy = load_policy(spec)
    pads = pad_surfaces(spec)
    calibration = load_taxel_calibration(args.taxel_calibration)
    print(
        f"[INFO] {spec.task}: {spec.policy_path}, tactile {spec.tactile or 'none'}, {spec.action_type} actions,"
        f" {1.0 / spec.step_dt:.0f} Hz, {spec.obs_dim} observations"
        + (f", taxel calibration of {sorted(calibration)}" if calibration else "")
    )

    default_q = np.array([spec.default_joint_pos[j] for j in spec.action_joints])
    joint_limits = (default_q - args.max_target_offset, default_q + args.max_target_offset)
    plate_pose = urdf_to_orca({WRIST_JOINT: 0.0, **spec.default_joint_pos})

    hand = load_hand(config_path=args.config, mock=args.mock)
    if spec.tactile and not isinstance(hand, OrcaHandTouch):
        raise RuntimeError(f"{spec.task} needs tactile sensing, but the hand config has no 'sensors' block")
    ok, message = hand.connect()
    print(f"[INFO] connect: {message}")
    if not ok:
        raise RuntimeError(message)

    log = {"time": [], "obs": [], "action": [], "joint_pos": [], "joint_target": [], "target": [], "forces": []}
    try:
        if not hand.calibrated:
            raise RuntimeError("the hand is not calibrated; run orca_core's scripts/calibrate.py first")
        hand.init_joints(move_to_neutral=False)

        num_steps = max(1, int(args.move_time / 0.02))
        print("[INFO] moving to the plate pose")
        hand.set_joint_positions(plate_pose, num_steps=num_steps, step_size=0.02)

        if isinstance(hand, OrcaHandTouch):
            hand.start_tactile_stream(resultant=False, taxels=True, min_sensors=1)
            if not args.mock:
                prompt("Make sure nothing touches the fingertip pads, to zero the tactile sensors.", args)
                hand.zero_tactile_sensors(num_samples=100)
        prompt("Place the plate on the fingertips, then the ball on the plate.", args)

        reset_policy(policy)
        rng = np.random.default_rng()
        target = np.array(args.target, dtype=float)
        next_resample = 0.0
        last_action = np.zeros(len(spec.action_joints))
        joint_vel = np.zeros(len(spec.action_joints))
        # the smoothed increment and the joint targets of "ema" actions, from the plate pose
        increment = np.zeros(len(spec.action_joints))
        joint_target = default_q.copy()
        prev_q, prev_t = None, None
        num_clipped, num_overruns = 0, 0
        start = time.perf_counter()
        next_tick, next_print = start, start
        print("[INFO] running the policy, Ctrl+C to stop")
        while args.duration is None or time.perf_counter() - start < args.duration:
            now = time.perf_counter()
            t = now - start

            if args.random_targets and t >= next_resample:
                target = np.array([rng.uniform(*r) for r in spec.target_ranges])
                next_resample = t + rng.uniform(*spec.resampling_time_range)
                print(f"[INFO] target ({target[0]:+.3f}, {target[1]:+.3f}) m")

            q = orca_to_urdf(hand.get_joint_position().as_dict(), spec.action_joints)
            if prev_q is not None and now > prev_t:
                sample = (q - prev_q) / (now - prev_t)
                joint_vel = args.vel_filter * sample + (1.0 - args.vel_filter) * joint_vel
            prev_q, prev_t = q, now
            tactile, forces = read_tactile(hand, spec, args.tactile_sign, pads, calibration)

            values = {"joint_pos": q - default_q, "joint_vel": joint_vel, "target": target, "actions": last_action}
            values["tactile"] = tactile
            obs = np.concatenate([values[term] for term in spec.obs_terms]).astype(np.float32)
            with torch.inference_mode():
                action = policy(torch.from_numpy(obs).unsqueeze(0))[0].numpy().astype(float)
            last_action = action

            clipped = np.clip(action, -args.max_action, args.max_action)
            num_clipped += int(np.any(clipped != action))
            if spec.action_type == "ema":
                step = spec.action_scale * np.clip(clipped, -spec.action_clip, spec.action_clip)
                increment = spec.ema_alpha * step + (1.0 - spec.ema_alpha) * increment
                joint_target = np.clip(joint_target + increment, *joint_limits)
            else:
                joint_target = default_q + spec.action_scale * clipped
            if not args.no_actuate:
                hand.set_joint_positions(
                    urdf_to_orca({WRIST_JOINT: 0.0, **dict(zip(spec.action_joints, joint_target))})
                )

            if args.log is not None:
                for key, value in (
                    ("time", t),
                    ("obs", obs),
                    ("action", action),
                    ("joint_pos", q),
                    ("joint_target", joint_target),
                    ("target", target.copy()),
                    ("forces", np.stack([forces[f] for f in FINGERS]) if forces else np.zeros((len(FINGERS), 3))),
                ):
                    log[key].append(value)

            if now >= next_print:
                next_print += args.print_every
                touching = "".join(
                    f[0].upper() if forces and np.linalg.norm(forces[f]) > spec.contact_threshold else "."
                    for f in FINGERS
                )
                print(
                    f"t={t:6.2f}s |a|max={np.abs(action).max():5.2f} contacts[TIMRP]={touching}"
                    f" Fz={' '.join(f'{np.linalg.norm(forces[f]):.2f}' for f in FINGERS) if forces else '-'}"
                    f" clipped={num_clipped} overruns={num_overruns}"
                )

            next_tick += spec.step_dt
            delay = next_tick - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            else:
                num_overruns += 1
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        try:
            if hand.is_connected() and hand.calibrated:
                print("[INFO] returning to the plate pose")
                hand.set_joint_positions(plate_pose, num_steps=max(1, int(1.0 / 0.02)), step_size=0.02)
                prompt("Take the plate and the ball off; the hand's torque is released next.", args)
        except KeyboardInterrupt:
            pass
        finally:
            ok, message = hand.disconnect()
            print(f"[INFO] disconnect: {message}")
        if args.log is not None and log["time"]:
            np.savez(args.log, **{k: np.asarray(v) for k, v in log.items()}, joints=np.array(spec.action_joints))
            print(f"[INFO] saved {len(log['time'])} steps to {args.log}")


if __name__ == "__main__":
    main()
