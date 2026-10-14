"""Collision-filter helpers for shapes added to an existing body."""

from __future__ import annotations

from collections.abc import Sequence

import newton


def inherit_collision_filters(
    builder: newton.ModelBuilder, body: int, new_shapes: Sequence[int]
) -> None:
    """Copy the collision filters of a body's existing shapes to new shapes.

    Every shape on another body that is filtered against at least one
    existing shape of ``body`` is also filtered against each of
    ``new_shapes``.

    Parameters
    ----------
    builder : newton.ModelBuilder
        Builder to modify, before ``finalize()``.
    body : int
        Index of the body the new shapes were added to.
    new_shapes : Sequence[int]
        Indices of the shapes that receive the filters.

    Notes
    -----
    Needed for shapes added after, for example, ``add_urdf(...,
    enable_self_collisions=False)``, which only filters the shapes that
    exist at import time.
    """
    new = set(new_shapes)
    on_body = {s for s, b in enumerate(builder.shape_body) if b == body} - new
    partners = set()
    for a, b in builder.shape_collision_filter_pairs:
        if a in on_body and builder.shape_body[b] != body:
            partners.add(b)
        elif b in on_body and builder.shape_body[a] != body:
            partners.add(a)
    for shape in new:
        for partner in partners - new:
            builder.add_shape_collision_filter_pair(shape, partner)
