"""lumq.algorithms.gbs - Gaussian Boson Sampling.

GBS encodes a graph adjacency matrix A into a photonic circuit via
    A = U diag(tanh r_1,...,tanh r_N) U^T
then samples photon-number patterns. The probability of detecting
pattern S is proportional to |Haf(A_S)|^2, where Haf is the hafnian.

Modules
-------
hafnian   - JAX hafnian (Ryser algorithm, differentiable)
circuit   - build a GBS circuit from adjacency matrix or squeezing params
sampling  - run GBS and return click / PNR samples
graph     - graph problem encodings (max clique, dense subgraph)
"""
from lumq.algorithms.gbs.hafnian import hafnian, hafnian_batch, torontonian
from lumq.algorithms.gbs.circuit import gbs_circuit, gbs_circuit_from_graph
from lumq.algorithms.gbs.sampling import GBSSampler, GBSResult
from lumq.algorithms.gbs.graph import adjacency_to_gbs, gbs_to_graph_features

__all__ = [
    "hafnian","hafnian_batch","torontonian",
    "gbs_circuit","gbs_circuit_from_graph",
    "GBSSampler","GBSResult",
    "adjacency_to_gbs","gbs_to_graph_features",
]

