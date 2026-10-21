* Added ``ORCA_HAND_CFG``, the right ORCA hand alone with its forearm fixed to the world, and its asset, built by
  ``build_asset.py --robot orca_hand`` into ``assets/orca_hand/`` for the ``no_taxels`` and ``convex_taxels``
  fingertip methods; :func:`robot_usd` gives either robot's USD per fingertip method, and
  :func:`fingertip_pad_region` each finger's tactile pad surface.
