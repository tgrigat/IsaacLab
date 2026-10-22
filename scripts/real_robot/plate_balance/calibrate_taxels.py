# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate the ORCA hand's taxel force directions against its joint torques, as in Pan et al. (2026), "Beyond
Binary".

A taxel's 3-axis force is measured in its own frame, which the sensor geometry only approximately aligns with the
fingertip frame. Each taxel's rotation into the fingertip frame is learned from the joint torques the contacts cause,
without ground-truth forces: while the finger holds still, an external force ``f`` on the fingertip at ``p`` needs the
joint torques ``tau = -J(p)^T f``, which the motors' currents measure. So, over many random presses on the pad,

.. math::

    -J(p_s)^T \\sum_i R_i f_{s,i} = k \\odot \\Delta c_s

with ``f_{s,i}`` taxel ``i``'s force in sample ``s`` (sign-corrected as in ``play_orca_hand.py``), ``R_i`` its rotation,
``p_s`` the force-weighted contact point, ``J`` the contact point's Jacobian in the fingertip frame and ``Delta c_s``
the motor currents above their no-contact baseline. The ORCA fingers are tendon driven, so unlike the paper's direct
drive the current to torque gain ``k`` of each joint is unknown and is fitted too. The rotations are parameterised by
their rotation vectors (the paper's 3x3 matrices projected by SVD have no gradient at the identity, where the fit
starts), fitted by Adam and regularised towards the identity.

Workflow:

.. code-block:: bash

    # check the fit on synthetic data, without hardware: it should recover the true rotations
    uv run python scripts/real_robot/plate_balance/calibrate_taxels.py synthetic

    # record: the hand holds the plate pose of a run; press each pad at random points and angles, for 2 minutes
    uv run python scripts/real_robot/plate_balance/calibrate_taxels.py record presses.npz --run <run_dir> \\
        --config ../orca_core/orca_core/models/v2/orcahand-touch-right/config.yaml

    # fit, and deploy with the result
    uv run python scripts/real_robot/plate_balance/calibrate_taxels.py fit presses.npz taxel_calibration.npz
    uv run python scripts/real_robot/plate_balance/play_orca_hand.py <run_dir> --taxel-calibration taxel_calibration.npz

The fit reports each finger's held-out torque R^2 with and without the rotations; ``play_orca_hand.py`` only applies
the rotations of the fingers saved, so drop a finger (``--fingers``) whose fit does not improve on the identity.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

JACOBIAN_STEP_DEG = 0.1
"""Joint step [deg] of the finite-difference contact Jacobians."""


##
# Kinematics
##


def finger_joints(kinematics, finger: str) -> list[str]:
    """``orca_core`` joints of a finger's kinematic chain, from its base."""
    return [entry.joint for entry in kinematics._finger_chains[finger] if entry.joint]


def contact_jacobian(kinematics, finger: str, joint_pos_deg: dict[str, float], point: np.ndarray) -> np.ndarray:
    """Jacobian [m/rad] of a point fixed in the fingertip frame with respect to the finger's joints, expressed in the
    fingertip frame, shape (3, num_joints), by central differences."""
    joints = finger_joints(kinematics, finger)
    pose = kinematics.fingertip_poses(joint_pos_deg, in_frame="palm", fingers=(finger,))[finger]
    rotation = pose.rotation
    columns = []
    for joint in joints:
        moved = []
        for sign in (1.0, -1.0):
            q = dict(joint_pos_deg)
            q[joint] = (q.get(joint) or 0.0) + sign * JACOBIAN_STEP_DEG
            tip = kinematics.fingertip_poses(q, in_frame="palm", fingers=(finger,))[finger]
            moved.append(tip.rotation @ point + tip.translation)
        columns.append(rotation.T @ (moved[0] - moved[1]) / (2.0 * np.radians(JACOBIAN_STEP_DEG)))
    return np.stack(columns, axis=1)


def contact_point(forces: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Force-magnitude-weighted mean of the taxel positions, as in ``play_orca_hand.py``, shape (3,)."""
    weights = np.linalg.norm(forces, axis=-1)
    return weights @ positions / max(weights.sum(), 1e-9)


##
# Fit
##


def to_rotation(rotvec: torch.Tensor) -> torch.Tensor:
    """Rotations of rotation vectors [rad], shape (..., 3) to (..., 3, 3)."""
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).view(*rotvec.shape[:-1], 3, 3)
    return torch.linalg.matrix_exp(skew)


def fit_finger(
    forces: np.ndarray,
    jacobians: np.ndarray,
    currents: np.ndarray,
    steps: int,
    lr: float,
    reg: float,
    shared: bool,
    holdout: float,
    seed: int = 0,
) -> dict:
    """Fit one finger's taxel rotations and current-to-torque gains.

    Args:
        forces: Taxel forces [N] in the fingertip frame, shape (num_samples, num_taxels, 3).
        jacobians: Contact Jacobians [m/rad] in the fingertip frame, shape (num_samples, 3, num_joints).
        currents: Motor currents above their baseline [mA], shape (num_samples, num_joints).
        steps: Adam steps.
        lr: Adam learning rate.
        reg: Weight of the rotations' squared angle [rad^2].
        shared: Whether all taxels share one rotation, instead of one each.
        holdout: Fraction of the samples held out to report the fit.

    Returns:
        The rotations, shape (num_taxels, 3, 3); the gains [Nm/mA], shape (num_joints,); and the held-out torque R^2
        with the identity and with the rotations.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(forces))
    num_test = int(holdout * len(forces))
    test, train = order[:num_test], order[num_test:]
    f = torch.tensor(forces, dtype=torch.float64)
    jac = torch.tensor(jacobians, dtype=torch.float64)
    cur = torch.tensor(currents, dtype=torch.float64)
    num_taxels = forces.shape[1]

    def predicted_torque(rotations: torch.Tensor, ids) -> torch.Tensor:
        total = torch.einsum("tij,stj->si", rotations, f[ids])
        return -torch.einsum("sij,si->sj", jac[ids], total)

    def best_gains(torque: torch.Tensor, ids) -> torch.Tensor:
        # least-squares k per joint: torque_j = k_j * current_j
        c = cur[ids]
        return (torque * c).sum(0) / (c.square().sum(0) + 1e-12)

    def r2(rotations: torch.Tensor, gains: torch.Tensor, ids) -> float:
        residual = predicted_torque(rotations, ids) - gains * cur[ids]
        target = gains * cur[ids]
        return float(1.0 - residual.square().sum() / ((target - target.mean(0)).square().sum() + 1e-12))

    eye = torch.eye(3, dtype=torch.float64)
    identity = eye.expand(num_taxels, 3, 3)
    gains_identity = best_gains(predicted_torque(identity, train), train).detach()

    raw = torch.zeros(1 if shared else num_taxels, 3, dtype=torch.float64, requires_grad=True)
    gains = gains_identity.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw, gains], lr=lr)
    scale = (gains_identity * cur[train]).square().mean().clamp(min=1e-12)
    for _ in range(steps):
        rotations = to_rotation(raw)
        full = rotations.expand(num_taxels, 3, 3)
        residual = predicted_torque(full, train) - gains * cur[train]
        loss = residual.square().mean() / scale + reg * raw.square().sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        full = to_rotation(raw).expand(num_taxels, 3, 3)
        eval_ids = test if num_test else train
        return {
            "R": full.numpy().copy(),
            "k": gains.detach().numpy().copy(),
            "r2_identity": r2(identity, gains_identity, eval_ids),
            "r2": r2(full, gains.detach(), eval_ids),
        }


def fit_recording(args: argparse.Namespace):
    """Fit every recorded finger and save the calibration."""
    from orca_core.kinematics import HandKinematics

    data = np.load(args.recording)
    kinematics = HandKinematics.load(str(data["hand_type"]))
    joint_names = [str(j) for j in data["joint_names"]]
    out = {}
    for finger in args.fingers:
        if f"{finger}_forces" not in data:
            warnings.warn(f"no recording of the {finger}")
            continue
        forces, positions = data[f"{finger}_forces"], data[f"{finger}_positions"]
        joints = finger_joints(kinematics, finger)
        cols = [joint_names.index(j) for j in joints]
        # only the samples pressing the pad
        pressed = np.linalg.norm(forces.sum(axis=1), axis=-1) > args.min_force
        jacobians = []
        for s in np.flatnonzero(pressed):
            q = dict(zip(joint_names, data["joint_pos"][s]))
            jacobians.append(contact_jacobian(kinematics, finger, q, contact_point(forces[s], positions)))
        currents = data["currents"][pressed][:, cols] - data["baseline"][cols]
        result = fit_finger(
            forces[pressed], np.stack(jacobians), currents, args.steps, args.lr, args.reg, args.shared, args.holdout
        )
        print(
            f"[INFO] {finger}: {pressed.sum()} samples, held-out torque R^2 {result['r2_identity']:.3f} (identity)"
            f" -> {result['r2']:.3f}, gains {np.array2string(result['k'], precision=5)} Nm/mA"
        )
        out.update({f"{finger}_R": result["R"], f"{finger}_k": result["k"], f"{finger}_r2": result["r2"]})
    np.savez(args.output, **out)
    print(f"[INFO] saved {args.output}")


def synthetic(args: argparse.Namespace):
    """Fit synthetic presses with known rotations and gains, and report how well they are recovered."""
    from orca_core.kinematics import HandKinematics

    rng = np.random.default_rng(args.seed)
    kinematics = HandKinematics.load("right")
    finger, num_taxels = "index", args.num_taxels
    joints = finger_joints(kinematics, finger)
    positions = rng.uniform([-0.008, -0.008, 0.035], [0.008, 0.008, 0.045], (num_taxels, 3))
    if args.shared:
        true_rotations = np.repeat(_random_rotations(rng, 1, args.true_angle), num_taxels, axis=0)
    else:
        true_rotations = _random_rotations(rng, num_taxels, args.true_angle)
    true_gains = rng.uniform(1e-4, 5e-4, len(joints))

    forces, jacobians, currents = [], [], []
    for _ in range(args.num_samples):
        q = {j: rng.uniform(5.0, 40.0) for j in joints}
        # a press on a few neighbouring taxels, into the pad, as the plate's force on the fingertip
        centre = positions[rng.integers(num_taxels)]
        weights = np.exp(-(np.linalg.norm(positions - centre, axis=1) ** 2) / (2 * 0.003**2))
        direction = np.array([*rng.normal(0.0, 0.3, 2), -1.0])
        f = weights[:, None] * direction * rng.uniform(0.2, 2.0) / weights.sum()
        measured = np.einsum("tji,tj->ti", true_rotations, f)  # what the taxels report, in their own frames
        jac = contact_jacobian(kinematics, finger, q, contact_point(f, positions))
        torque = -jac.T @ f.sum(axis=0)
        forces.append(measured)
        jacobians.append(jac)
        currents.append(torque / true_gains + rng.normal(0.0, args.current_noise, len(joints)))
    result = fit_finger(
        np.stack(forces),
        np.stack(jacobians),
        np.stack(currents),
        args.steps,
        args.lr,
        args.reg,
        args.shared,
        args.holdout,
        args.seed,
    )
    angle = _rotation_angle(np.einsum("tij,tkj->tik", result["R"], true_rotations))
    print(
        f"[INFO] held-out torque R^2 {result['r2_identity']:.3f} (identity) -> {result['r2']:.3f}; rotation error"
        f" {np.degrees(np.median(angle)):.1f} deg median (true rotations {np.degrees(args.true_angle):.0f} deg); gain"
        f" error {np.abs(result['k'] / true_gains - 1).max():.1%}"
    )


def _random_rotations(rng: np.random.Generator, n: int, angle: float) -> np.ndarray:
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = rng.uniform(0.0, angle, n)
    k = np.zeros((n, 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -axes[:, 2], axes[:, 1], -axes[:, 0]
    k -= k.transpose(0, 2, 1)
    return np.eye(3) + np.sin(angles)[:, None, None] * k + (1 - np.cos(angles))[:, None, None] * (k @ k)


def _rotation_angle(rotations: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip((np.trace(rotations, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))


##
# Recording
##


def record(args: argparse.Namespace):
    """Record random presses on the pads while the hand holds a pose."""
    from orca_core import OrcaHandTouch, load_hand

    hand = load_hand(config_path=args.config)
    if not isinstance(hand, OrcaHandTouch):
        raise RuntimeError("the hand config has no 'sensors' block")
    ok, message = hand.connect()
    print(f"[INFO] connect: {message}")
    if not ok:
        raise RuntimeError(message)
    joint_names = list(hand.config.joint_to_motor_map)
    motor_ids = [hand.config.joint_to_motor_map[j] for j in joint_names]
    log = {"joint_pos": [], "currents": [], "time": []}
    forces = {finger: [] for finger in args.fingers}
    positions = {}
    try:
        if not hand.calibrated:
            raise RuntimeError("the hand is not calibrated; run orca_core's scripts/calibrate.py first")
        hand.init_joints(move_to_neutral=args.run is None)
        if args.run is not None:
            from play_orca_hand import WRIST_JOINT, load_spec, urdf_to_orca

            spec = load_spec(args.run)
            print("[INFO] moving to the plate pose")
            hand.set_joint_positions(
                urdf_to_orca({WRIST_JOINT: 0.0, **spec.default_joint_pos}), num_steps=150, step_size=0.02
            )
        if args.max_current is not None:
            hand.set_max_current(args.max_current)
        hand.start_tactile_stream(resultant=False, taxels=True, min_sensors=1)
        input("Make sure nothing touches the fingertip pads, to zero the tactile sensors. Press Enter to continue.")
        hand.zero_tactile_sensors(num_samples=100)
        baseline = []
        for _ in range(int(2.0 * args.rate)):
            currents = hand.get_motor_current(as_dict=True)
            baseline.append([currents[m] for m in motor_ids])
            time.sleep(1.0 / args.rate)
        input(
            f"Press the pads of the {', '.join(args.fingers)} at random points and angles, from light to firm, for"
            f" {args.duration:.0f} s. Press Enter to start."
        )
        start = next_tick = time.perf_counter()
        while time.perf_counter() - start < args.duration:
            data = hand.get_taxel_data(frame="fingertip")
            q = hand.get_joint_position().as_dict()
            currents = hand.get_motor_current(as_dict=True)
            if data is not None and all(data.get(f) is not None for f in args.fingers):
                for finger in args.fingers:
                    forces[finger].append(args.tactile_sign * data[finger].forces)
                    positions[finger] = data[finger].positions
                log["joint_pos"].append([q.get(j) or 0.0 for j in joint_names])
                log["currents"].append([currents[m] for m in motor_ids])
                log["time"].append(time.perf_counter() - start)
            next_tick += 1.0 / args.rate
            time.sleep(max(0.0, next_tick - time.perf_counter()))
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        ok, message = hand.disconnect()
        print(f"[INFO] disconnect: {message}")
    np.savez(
        args.output,
        hand_type=hand.config.type if hasattr(hand.config, "type") else "right",
        joint_names=np.array(joint_names),
        baseline=np.mean(baseline, axis=0),
        **{k: np.asarray(v) for k, v in log.items()},
        **{f"{f}_forces": np.asarray(v) for f, v in forces.items() if v},
        **{f"{f}_positions": p for f, p in positions.items()},
    )
    print(f"[INFO] saved {len(log['time'])} samples to {args.output}")


##
# Main
##


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_fit_args(p: argparse.ArgumentParser):
        p.add_argument("--steps", type=int, default=500, help="Adam steps.")
        p.add_argument("--lr", type=float, default=0.02, help="Adam learning rate.")
        p.add_argument("--reg", type=float, default=1e-3, help="Weight of the rotations' squared angle.")
        p.add_argument("--shared", action="store_true", help="One rotation per finger instead of one per taxel.")
        p.add_argument("--holdout", type=float, default=0.2, help="Fraction of samples held out for the R^2.")

    rec = sub.add_parser("record", help="Record presses on the real hand.")
    rec.add_argument("output", type=Path)
    rec.add_argument("--config", default=None, help="orca_core config.yaml of the hand; autodetected if omitted.")
    rec.add_argument("--run", type=Path, default=None, help="Hold the plate pose of this run instead of neutral.")
    rec.add_argument("--fingers", nargs="+", default=list(FINGERS), choices=FINGERS)
    rec.add_argument("--duration", type=float, default=120.0, help="Recording time [s].")
    rec.add_argument("--rate", type=float, default=20.0, help="Sample rate [Hz].")
    rec.add_argument("--max-current", type=float, default=None, help="Motor current limit [mA] to hold stiffly.")
    rec.add_argument("--tactile-sign", type=float, default=-1.0, help="Sign from sensor forces to plate forces.")

    fit = sub.add_parser("fit", help="Fit a recording.")
    fit.add_argument("recording", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--fingers", nargs="+", default=list(FINGERS), choices=FINGERS)
    fit.add_argument("--min-force", type=float, default=0.1, help="Smallest total force [N] of a sample to fit.")
    add_fit_args(fit)

    syn = sub.add_parser("synthetic", help="Fit synthetic presses with known rotations.")
    syn.add_argument("--num-samples", type=int, default=2400)
    syn.add_argument("--num-taxels", type=int, default=40)
    syn.add_argument("--true-angle", type=float, default=0.3, help="Largest true rotation angle [rad].")
    syn.add_argument("--current-noise", type=float, default=1.0, help="Current noise [mA].")
    syn.add_argument("--seed", type=int, default=0)
    add_fit_args(syn)
    return parser.parse_args()


def main():
    args = parse_args()
    {"record": record, "fit": fit_recording, "synthetic": synthetic}[args.command](args)


if __name__ == "__main__":
    main()
