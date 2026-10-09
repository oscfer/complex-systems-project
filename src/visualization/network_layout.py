"""Deterministic layout for transformation network figures."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _seeded_spring_layout(
    W: np.ndarray,
    layout_seed: int = 42,
    iterations: int = 100,
    k: Optional[float] = None,
) -> np.ndarray:
    """Deterministic spring layout of the undirected memory projection.

    Connected nodes attract, all nodes repel, and a seeded interior start avoids
    imposing a circular geometry on the network. Coordinates have no physical
    meaning and should be reused across frames of the same rollout.
    """
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]
    if N == 0:
        return np.empty((0, 2), dtype=np.float64)
    if N == 1:
        return np.zeros((1, 2), dtype=np.float64)
    rng = np.random.default_rng(layout_seed)
    G = np.maximum(W + W.T, 0.0)
    np.fill_diagonal(G, 0.0)
    strength = G.sum(axis=1)
    # Symmetric normalization prevents high-degree rows from dominating.
    A = G / np.maximum(np.sqrt(np.outer(strength, strength)), 1e-12)
    pos = rng.uniform(-0.5, 0.5, size=(N, 2))

    if k is None:
        k = 1.0 / np.sqrt(N)

    temperature = 0.1
    cooling = temperature / (iterations + 1)
    for _ in range(iterations):
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt((delta ** 2).sum(-1))
        np.fill_diagonal(dist, 1.0)
        repulsion = (k * k) / np.maximum(dist * dist, 1e-12)
        attraction = A * dist / k
        np.fill_diagonal(repulsion, 0.0)
        np.fill_diagonal(attraction, 0.0)
        # delta points away from the other node: repulsion adds it and
        # attraction subtracts it. Both forces are symmetric in each pair.
        displacement = np.einsum("ijk,ij->ik", delta, repulsion - attraction)
        magnitude = np.linalg.norm(displacement, axis=1)
        step = np.minimum(magnitude, temperature)[:, np.newaxis]
        pos += step * displacement / np.maximum(magnitude[:, np.newaxis], 1e-12)
        pos -= pos.mean(axis=0, keepdims=True)
        temperature -= cooling

    extent = np.max(np.abs(pos)) or 1.0
    return pos / extent * 1.1
