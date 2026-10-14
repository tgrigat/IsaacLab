* Added the ``Isaac-Lift-FrankaOrca`` environments, with a finger action penalty ramped in with the ADR difficulty
  once the hand grasps, and the ``early_termination`` penalty applied to the object going out of bounds. Every
  FrankaOrca task comes in one variant per fingertip collision method, as its suffix: ``-Primitives`` (every link one
  capsule or box, to learn the coarse motion cheaply), ``-NoTaxels`` (one convex hull per fingertip),
  ``-ConvexTaxels`` (one convex solid per taxel over a convex fingertip hull), ``-MeshTaxels`` (one triangle mesh per
  taxel over CoACD fingertip hulls) and ``-MeshFingertips`` (one triangle mesh per taxel over a triangle-mesh
  fingertip). The variants with taxels observe their per-taxel forces, and their fingertips touch each other pad to
  pad.
