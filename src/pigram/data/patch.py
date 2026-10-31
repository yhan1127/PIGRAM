# src/pigram/data/patch.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_residue_centroids(protein_coords: np.ndarray) -> np.ndarray:
    """
    Compute one centroid per protein residue.

    Args:
        protein_coords:
            Array with shape [L, A, 3], where L is the number of residues
            and A is the number of atoms per residue.

    Returns:
        residue_centroids:
            Array with shape [L, 3].
    """
    protein_coords = np.asarray(protein_coords, dtype=np.float32)

    if protein_coords.ndim != 3 or protein_coords.shape[-1] != 3:
        raise ValueError(
            f"protein_coords should have shape [L, A, 3], got {protein_coords.shape}"
        )

    return protein_coords.mean(axis=1).astype(np.float32)


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean pairwise distances between 3D points.

    Args:
        points:
            Array with shape [N, 3].

    Returns:
        dist:
            Array with shape [N, N].
    """
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points should have shape [N, 3], got {points.shape}")

    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    return dist.astype(np.float32)


def build_knn_patches(
    residue_centroids: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """
    Build residue-centered KNN patches from protein residue centroids.

    Each residue is used as one patch center. The patch consists of the k
    nearest residues to that center, including the center itself.

    Args:
        residue_centroids:
            Array with shape [L, 3].
        k:
            Number of residues per patch.

    Returns:
        patch_indices:
            Array with shape [L, min(k, L)].
    """
    residue_centroids = np.asarray(residue_centroids, dtype=np.float32)

    if residue_centroids.ndim != 2 or residue_centroids.shape[-1] != 3:
        raise ValueError(
            f"residue_centroids should have shape [L, 3], got {residue_centroids.shape}"
        )

    num_residues = int(residue_centroids.shape[0])

    if num_residues <= 0:
        raise ValueError("Cannot build patches for an empty protein.")

    k_eff = min(int(k), num_residues)
    if k_eff <= 0:
        raise ValueError(f"k must be positive, got {k}")

    dist = pairwise_distances(residue_centroids)
    patch_indices = np.argsort(dist, axis=1)[:, :k_eff]

    return patch_indices.astype(np.int64)


def compute_patch_relative_distances(
    residue_centroids: np.ndarray,
    patch_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute distances from each patch center residue to residues in that patch.

    The p-th patch is assumed to be centered at residue p. This matches the
    construction in build_knn_patches, where one patch is created per residue.

    Args:
        residue_centroids:
            Array with shape [L, 3].
        patch_indices:
            Array with shape [N_patches, K].

    Returns:
        patch_relative_distances:
            Array with shape [N_patches, K].
    """
    residue_centroids = np.asarray(residue_centroids, dtype=np.float32)
    patch_indices = np.asarray(patch_indices, dtype=np.int64)

    if residue_centroids.ndim != 2 or residue_centroids.shape[-1] != 3:
        raise ValueError(
            f"residue_centroids should have shape [L, 3], got {residue_centroids.shape}"
        )

    if patch_indices.ndim != 2:
        raise ValueError(f"patch_indices should be 2D, got {patch_indices.shape}")

    num_patches, k = patch_indices.shape

    if num_patches > residue_centroids.shape[0]:
        raise ValueError(
            f"num_patches={num_patches} cannot exceed number of residues={residue_centroids.shape[0]}"
        )

    relative_distances = np.zeros((num_patches, k), dtype=np.float32)

    for patch_id in range(num_patches):
        center_idx = patch_id
        center_xyz = residue_centroids[center_idx]
        patch_xyz = residue_centroids[patch_indices[patch_id]]
        relative_distances[patch_id] = np.linalg.norm(
            patch_xyz - center_xyz[None, :],
            axis=-1,
        )

    return relative_distances.astype(np.float32)


def build_protein_patches(
    protein_coords: np.ndarray,
    k: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Full protein-side patch building pipeline.

    Args:
        protein_coords:
            Array with shape [L, A, 3].
        k:
            Number of residues per local patch.

    Returns:
        A dictionary with:
            residue_centroids: [L, 3]
            patch_indices: [L, K]
            patch_center_indices: [L]
            patch_relative_distances: [L, K]
    """
    residue_centroids = compute_residue_centroids(protein_coords)
    patch_indices = build_knn_patches(residue_centroids, k=k)

    patch_center_indices = np.arange(residue_centroids.shape[0], dtype=np.int64)
    patch_relative_distances = compute_patch_relative_distances(
        residue_centroids=residue_centroids,
        patch_indices=patch_indices,
    )

    return {
        "residue_centroids": residue_centroids.astype(np.float32),
        "patch_indices": patch_indices.astype(np.int64),
        "patch_center_indices": patch_center_indices.astype(np.int64),
        "patch_relative_distances": patch_relative_distances.astype(np.float32),
    }