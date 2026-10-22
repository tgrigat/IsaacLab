* Added ``Isaac-PlateBalance-OrcaHand-CoP``, ORCA-hand ball balancing trained directly for the real hand after Pan et
  al. (2026), "Beyond Binary": a GRU actor observing the finger joints and each fingertip pad's centre of pressure
  (normal force and contact point, noisy and delayed), a GRU critic also observing the ball and the plate, joint
  targets moved by smoothed increments (:class:`~isaaclab_tasks.core.plate_balance.mdp.EMARelativeJointPositionAction`),
  the paper's rewards and actuator gain randomization, and the ``-Binary`` and ``-Base`` contact ablations.
  ``scripts/real_robot/plate_balance/play_orca_hand.py`` runs them on the real hand, and
  ``scripts/real_robot/plate_balance/calibrate_taxels.py`` calibrates the taxel force directions against the joint
  torques.
