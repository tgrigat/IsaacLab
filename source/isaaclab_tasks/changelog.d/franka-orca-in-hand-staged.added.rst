* Added ``Isaac-Lift-FrankaOrca-InHand-Staged`` and ``Isaac-Lift-FrankaOrca-InHand-Staged-NoTaxels``, the in-hand cube
  carry with a staged ADR curriculum: the cube starts caged in the hand, then ever further away from a caged, open or
  closed hand while gravity ramps in, and at full gravity it starts on the table: under the low, downturned hand,
  then just beside it, then further beside it, then anywhere. Each environment draws its stage by its own ADR level,
  and the arm moves at three quarters of the finger action scale, under the FR3 joint velocity limits. A reach
  reward that pays without contact leads the hand to the cube.
* Added ``mode_weights`` to the lift ``conditional_reset`` event, so a curriculum can set how often each reset mode is
  restored, shared by every environment or per environment.
* Added ``object_min_clearance`` to :class:`~isaaclab_tasks.core.lift.mdp.SlabClearanceCfg`, so a reset object may rest
  on a slab while the robot keeps its clearance.
