"""Tactile sensing backends.

Each backend implements :class:`halltaxel.backends.base.TactileBackend`.

- :mod:`~halltaxel.backends.contact_patch`: one collision shape per taxel
  patch, read out through ``newton.sensors.SensorContact``.
- :mod:`~halltaxel.backends.spring_skin`: mass-spring-damper skin
  (not implemented).
"""
