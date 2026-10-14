"""Surface geometry utilities for discretizing a mesh into taxel patches.

These functions operate on ``trimesh.Trimesh`` objects only and do not
require a ``newton.Model``. No function in this module transforms between
frames; all inputs are expected in a common frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

__all__ = [
    "SurfaceProjection",
    "SurfaceSegmentation",
    "load_surface_mesh",
    "project_to_surface",
    "segment_surface",
    "select_region",
]


@dataclass(frozen=True)
class SurfaceProjection:
    """Result of :func:`project_to_surface`.

    Attributes
    ----------
    points : ndarray, shape (N, 3)
        Closest point on the mesh surface to each input point.
    normals : ndarray, shape (N, 3)
        Unit normal of the face containing each closest point.
    distances : ndarray, shape (N,)
        Euclidean distance from each input point to its closest point.
    face_indices : ndarray, shape (N,)
        Index into ``mesh.faces`` of the face containing each closest
        point.
    """

    points: np.ndarray
    normals: np.ndarray
    distances: np.ndarray
    face_indices: np.ndarray


def load_surface_mesh(path: str | Path, *, scale: float = 1.0) -> trimesh.Trimesh:
    """Load a triangle mesh from a file.

    The mesh is loaded without processing, so vertices are not merged
    and faces keep their file order.

    Parameters
    ----------
    path : str or Path
        Mesh file, in any format trimesh can load.
    scale : float, default 1.0
        Uniform scale applied to the vertices, e.g. ``0.001`` to convert
        millimetres to metres.

    Returns
    -------
    trimesh.Trimesh
        The loaded mesh.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    TypeError
        If the file does not load as a single triangle mesh.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"surface mesh not found: {path}")

    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a single triangle mesh (got {type(mesh)})")

    if scale != 1.0:
        mesh.apply_scale(scale)
    return mesh


def project_to_surface(mesh: trimesh.Trimesh, points: np.ndarray) -> SurfaceProjection:
    """Find the closest point on a mesh surface to each of a set of points.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Surface to project onto.
    points : array_like, shape (N, 3)
        Points to project, in the frame of ``mesh``.

    Returns
    -------
    SurfaceProjection
        Closest points, their face normals and face indices, and the
        distances to them.

    Raises
    ------
    ValueError
        If ``points`` does not have shape (N, 3).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be an (N, 3) array, got shape {points.shape}")

    closest, distances, face_indices = mesh.nearest.on_surface(points)
    normals = mesh.face_normals[face_indices]
    return SurfaceProjection(
        points=np.asarray(closest, dtype=np.float64),
        normals=np.asarray(normals, dtype=np.float64),
        distances=np.asarray(distances, dtype=np.float64),
        face_indices=np.asarray(face_indices, dtype=np.int64),
    )


def select_region(
    mesh: trimesh.Trimesh, region_mesh: trimesh.Trimesh, *, tol: float | None = None
) -> np.ndarray:
    """Select the faces of a mesh that lie on the surface of a region mesh.

    A face of ``mesh`` is selected if all three of its vertices are
    closer than ``tol`` to the surface of ``region_mesh``. The two meshes
    do not need to share vertices or faces.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to select faces from.
    region_mesh : trimesh.Trimesh
        Mesh covering the region to select, lying on ``mesh``'s surface
        in the same frame.
    tol : float, optional
        Distance threshold. Defaults to the mean edge length of ``mesh``.

    Returns
    -------
    ndarray of bool, shape (len(mesh.faces),)
        True for each selected face.

    Raises
    ------
    ValueError
        If no face is selected.
    """
    if tol is None:
        tol = mesh.edges_unique_length.mean()
    face_vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    _, vertex_distances, _ = region_mesh.nearest.on_surface(face_vertices)
    face_distances = vertex_distances.reshape(-1, 3).max(axis=1)
    mask = face_distances < tol
    if not mask.any():
        raise ValueError(
            "no face of mesh lies within tol of region_mesh's surface - "
            "wrong mesh/region pairing, or the wrong frame"
        )
    return mask


@dataclass(frozen=True)
class SurfaceSegmentation:
    """Partition of a mesh's faces into patches.

    Every face belongs to exactly one patch.

    Attributes
    ----------
    labels : ndarray of int, shape (F,)
        Patch index, in ``0 .. num_regions - 1``, of each face of the
        mesh.
    seeds : ndarray, shape (num_regions, 3)
        Centroid of each patch's seed face.
    """

    labels: np.ndarray
    seeds: np.ndarray

    @property
    def num_regions(self) -> int:
        """int: Number of patches, ``len(seeds)``."""
        return len(self.seeds)

    def check_mesh(self, mesh: trimesh.Trimesh) -> None:
        """Check that the segmentation has one label per face of ``mesh``.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh the segmentation is expected to belong to.

        Raises
        ------
        ValueError
            If ``len(labels) != len(mesh.faces)``.
        """
        if len(self.labels) != len(mesh.faces):
            raise ValueError(
                f"segmentation has {len(self.labels)} face labels but mesh has "
                f"{len(mesh.faces)} faces - segmentation must come from segment_surface(mesh, ...) "
                "run on this exact mesh"
            )


def _assign_labels(centers: np.ndarray, seed_faces: np.ndarray) -> np.ndarray:
    """Label each face with the index of the seed face whose centroid is nearest to its own."""
    _, nearest = cKDTree(centers[seed_faces]).query(centers)
    return nearest.astype(np.int64)


def _farthest_point_seed_faces(centers: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Select ``k`` faces by farthest-point sampling of face centroids.

    The first face is drawn from ``rng``; each subsequent face is the one
    farthest from all faces selected so far.
    """
    n_faces = len(centers)
    seed_faces = [int(rng.integers(n_faces))]
    if k > 1:
        dist_to_nearest_seed = np.linalg.norm(centers - centers[seed_faces[0]], axis=1)
        for _ in range(k - 1):
            seed_faces.append(int(np.argmax(dist_to_nearest_seed)))
            dist_to_newest = np.linalg.norm(centers - centers[seed_faces[-1]], axis=1)
            dist_to_nearest_seed = np.minimum(dist_to_nearest_seed, dist_to_newest)
    return np.array(seed_faces, dtype=np.int64)


def _relax_seed_faces(
    centers: np.ndarray, face_areas: np.ndarray, labels: np.ndarray, seed_faces: np.ndarray
) -> np.ndarray:
    """Move each seed to the patch face nearest the patch's area-weighted centroid.

    One Lloyd-relaxation step. A patch with no faces keeps its seed.
    """
    new_seed_faces = seed_faces.copy()
    for region, previous_face in enumerate(seed_faces):
        member_faces = np.flatnonzero(labels == region)
        if len(member_faces) == 0:
            continue  # patch vanished this pass; keep growing it from where it was
        centroid = np.average(centers[member_faces], axis=0, weights=face_areas[member_faces])
        nearest = np.argmin(np.linalg.norm(centers[member_faces] - centroid, axis=1))
        new_seed_faces[region] = member_faces[nearest]
    return new_seed_faces


def _face_adjacency_edges(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Return all pairs of edge-adjacent faces, listed in both directions.

    Adjacency is computed on a copy of ``mesh`` with coincident vertices
    merged, so meshes loaded without vertex merging (e.g. STL files) are
    handled.

    Returns
    -------
    rows, cols : ndarray of int
        Face indices of each adjacent pair; ``(a, b)`` and ``(b, a)`` are
        both included.

    Raises
    ------
    ValueError
        If ``mesh`` has more than one face but no two faces share an
        edge.
    """
    welded = mesh.copy()
    welded.merge_vertices()
    if len(welded.face_adjacency) == 0 and len(mesh.faces) > 1:
        raise ValueError(
            "mesh has no shared edges between faces (even after welding coincident "
            "vertices) - smooth_iters needs face neighbors to vote with"
        )

    a, b = welded.face_adjacency[:, 0], welded.face_adjacency[:, 1]
    return np.concatenate([a, b]), np.concatenate([b, a])


def _smooth_labels(
    edges: tuple[np.ndarray, np.ndarray], labels: np.ndarray, seed_faces: np.ndarray, iters: int
) -> np.ndarray:
    """Smooth face labels by majority vote among each face and its neighbors.

    In each pass, every face except the seed faces takes the label most
    common among itself and its edge-adjacent faces, with ties going to
    the lowest label index. Stops early when a pass changes nothing.

    Parameters
    ----------
    edges : tuple of ndarray
        ``(rows, cols)`` from :func:`_face_adjacency_edges`.
    labels : ndarray of int, shape (F,)
        Current label of each face.
    seed_faces : ndarray of int, shape (k,)
        Seed face of each patch; these keep their label.
    iters : int
        Maximum number of passes.

    Returns
    -------
    ndarray of int, shape (F,)
        Smoothed labels.
    """
    n_faces = len(labels)
    num_regions = len(seed_faces)
    neighbor_rows, neighbor_cols = edges
    voter_rows = np.concatenate([np.arange(n_faces), neighbor_rows])  # self vote + one per neighbor
    for _ in range(iters):
        voted = np.concatenate([labels, labels[neighbor_cols]])
        counts = np.bincount(
            voter_rows * num_regions + voted, minlength=n_faces * num_regions
        ).reshape(n_faces, num_regions)
        new_labels = counts.argmax(axis=1)  # ties go to the lowest label index
        new_labels[seed_faces] = labels[seed_faces]
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels


def segment_surface(
    mesh: trimesh.Trimesh,
    k: int | None = None,
    *,
    seeds: np.ndarray | None = None,
    relax_iters: int = 0,
    smooth_iters: int = 1,
    rng: np.random.Generator | None = None,
) -> SurfaceSegmentation:
    """Partition the faces of a mesh into patches around seed points.

    Each seed is snapped to the face nearest to it. Every face is then
    assigned to the seed face whose centroid is nearest to its own
    centroid by Euclidean distance. The partition does not follow mesh
    connectivity, so a patch can extend across a fold or a narrow gap.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Surface to partition.
    k : int, optional
        Number of patches. Required if ``seeds`` is None; ignored
        otherwise.
    seeds : array_like, shape (k, 3), optional
        Points to center the patches on, in the frame of ``mesh``. If
        None, ``k`` seed faces are selected by farthest-point sampling of
        face centroids.
    relax_iters : int, default 0
        Number of Lloyd-relaxation passes. Each pass moves every seed to
        the face nearest its patch's area-weighted centroid and
        reassigns the faces, which equalizes patch areas at the cost of
        moving the seeds.
    smooth_iters : int, default 1
        Maximum number of majority-vote smoothing passes, applied after
        relaxation. Each pass relabels every non-seed face with the label
        most common among itself and its edge-adjacent faces. Use 0 for
        meshes without shared edges.
    rng : numpy.random.Generator, optional
        Generator for farthest-point sampling. Defaults to
        ``np.random.default_rng(0)``. Unused if ``seeds`` is given.

    Returns
    -------
    SurfaceSegmentation
        Patch label of every face and seed point of every patch.

    Raises
    ------
    ValueError
        If neither ``k`` nor ``seeds`` is given, ``seeds`` does not have
        shape (k, 3), ``k < 1``, ``k`` exceeds the number of faces, or
        ``smooth_iters > 0`` and no two faces of a multi-face mesh share
        an edge.
    """
    if seeds is not None:
        seeds = np.asarray(seeds, dtype=np.float64)
        if seeds.ndim != 2 or seeds.shape[1] != 3:
            raise ValueError(f"seeds must be a (k, 3) array, got shape {seeds.shape}")
        k = len(seeds)
    elif k is None:
        raise ValueError("must pass either k or seeds")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > len(mesh.faces):
        raise ValueError(f"k={k} exceeds the mesh's {len(mesh.faces)} faces")

    centers = mesh.triangles_center

    if seeds is None:
        rng = rng if rng is not None else np.random.default_rng(0)
        seed_faces = _farthest_point_seed_faces(centers, k, rng)
    else:
        _, _, seed_faces = mesh.nearest.on_surface(seeds)

    labels = _assign_labels(centers, seed_faces)
    face_areas = mesh.area_faces
    for _ in range(relax_iters):
        seed_faces = _relax_seed_faces(centers, face_areas, labels, seed_faces)
        labels = _assign_labels(centers, seed_faces)

    if smooth_iters > 0:
        edges = _face_adjacency_edges(mesh)
        labels = _smooth_labels(edges, labels, seed_faces, smooth_iters)

    return SurfaceSegmentation(labels=labels, seeds=centers[seed_faces])
