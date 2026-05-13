"""
lumq.algorithms.gbs.graph
==========================
Encode graph problems into GBS circuits and extract features from samples.
"""
from __future__ import annotations
import numpy as np

__all__ = ["adjacency_to_gbs", "gbs_to_graph_features"]


def adjacency_to_gbs(
    A: np.ndarray,
    scale: float = 0.5,
    max_squeezing: float = 1.5,
) -> tuple:
    """Encode a graph adjacency matrix into a GBS circuit.

    The edge weights of the graph are encoded via the Takagi decomposition.
    High-probability GBS samples correspond to dense subgraphs (related to
    the planted dense subgraph problem and max-clique heuristics).

    Parameters
    ----------
    A : 2D real symmetric array, shape (N, N)
        Adjacency matrix.  Edge weights in [0, 1].
    scale : float
        Controls the mean photon number.
    max_squeezing : float
        Maximum squeezing parameter in radians.

    Returns
    -------
    (circuit, r, U, info_dict)
    """
    from lumq.algorithms.gbs.circuit import gbs_circuit_from_graph

    A = np.asarray(A, dtype=float)
    N = A.shape[0]

    # Symmetrise and zero diagonal
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)

    circuit, r, U = gbs_circuit_from_graph(A, scale=scale, max_squeezing=max_squeezing)

    info = {
        "n_nodes":         N,
        "n_edges":         int(np.sum(A > 0) // 2),
        "mean_squeezing":  float(np.mean(r)),
        "max_squeezing":   float(np.max(r)),
        "mean_photon_est": float(np.sum(np.sinh(r)**2)),
    }

    return circuit, r, U, info


def gbs_to_graph_features(result, A: np.ndarray) -> dict:
    """Extract graph-theoretic features from GBS samples.

    For each click pattern, checks if the clicked nodes form a clique
    (fully connected subgraph) in A.  Returns statistics useful as
    a heuristic for dense subgraph / max-clique problems.

    Parameters
    ----------
    result : GBSResult
        Sampling result from GBSSampler.
    A : 2D array, shape (N, N)
        Adjacency matrix (same one used to generate the circuit).

    Returns
    -------
    dict with keys:
        max_clique_size   : largest clique found in any sample
        clique_candidates : list of node sets that are cliques
        density_scores    : mean edge density of click patterns
        top_patterns      : top-5 most frequent patterns with density
    """
    A = np.asarray(A)
    clicks = np.array(result.clicks, dtype=bool)   # (shots, N)
    patterns, counts = result.click_patterns

    clique_candidates = []
    max_clique = 0
    density_scores = []

    for pattern in patterns:
        nodes = np.where(pattern)[0].tolist()
        if len(nodes) < 2:
            density_scores.append(0.0)
            continue

        # Check if these nodes form a clique
        is_clique = True
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if A[nodes[i], nodes[j]] == 0:
                    is_clique = False
                    break
            if not is_clique:
                break

        if is_clique and len(nodes) > max_clique:
            max_clique = len(nodes)
            clique_candidates.append(set(nodes))

        # Subgraph edge density
        subA = A[np.ix_(nodes, nodes)]
        possible = len(nodes)*(len(nodes)-1)/2
        density = float(np.sum(subA>0)/2) / possible if possible > 0 else 0.0
        density_scores.append(density)

    top_5 = list(zip(patterns[:5].tolist(), counts[:5].tolist(), density_scores[:5]))

    return {
        "max_clique_size":   max_clique,
        "clique_candidates": clique_candidates,
        "density_scores":    density_scores,
        "top_patterns":      top_5,
    }

