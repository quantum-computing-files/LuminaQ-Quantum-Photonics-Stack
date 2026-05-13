"""
lumq.algorithms.gbs.sampling
==============================
Run GBS circuits and post-process samples.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["GBSSampler", "GBSResult"]


@dataclass
class GBSResult:
    """Container for GBS sampling results.

    Attributes
    ----------
    samples : int array, shape (shots, n_modes)
        Photon-number samples per mode per shot.
    clicks : bool array, shape (shots, n_modes)
        Threshold (click/no-click) detection pattern.
    n_modes : int
    shots : int
    mean_photon_number : array, shape (n_modes,)
        Estimated mean photon number per mode from samples.
    total_photons : array, shape (shots,)
        Total photon count per shot.
    collision_rate : float
        Fraction of shots with at least one mode detecting > 1 photon.
        High collision rate signals near-classical behaviour.
    """
    samples: object          # (shots, n_modes)
    clicks: object           # (shots, n_modes) bool
    n_modes: int
    shots: int
    mean_photon_number: object = None
    total_photons: object = None
    collision_rate: float = 0.0

    def __post_init__(self):
        self.samples = jnp.asarray(self.samples, dtype=jnp.int32)
        self.clicks  = self.samples > 0
        self.total_photons = jnp.sum(self.samples, axis=1)
        self.mean_photon_number = jnp.mean(self.samples.astype(jnp.float32), axis=0)
        multi = jnp.any(self.samples > 1, axis=1)
        self.collision_rate = float(jnp.mean(multi.astype(jnp.float32)))

    @property
    def click_patterns(self):
        """Unique click patterns and their counts."""
        patterns, counts = np.unique(
            np.array(self.clicks, dtype=np.int32), axis=0, return_counts=True
        )
        order = np.argsort(-counts)
        return patterns[order], counts[order]

    def photon_number_distribution(self, mode: int):
        """Marginal photon-number distribution for a single mode."""
        s = np.array(self.samples[:, mode])
        max_n = int(s.max()) + 1
        counts = np.bincount(s, minlength=max_n)
        return np.arange(max_n), counts / counts.sum()

    def __repr__(self):
        return (f"GBSResult(n_modes={self.n_modes}, shots={self.shots}, "
                f"mean_total_photons={float(self.total_photons.mean()):.2f}, "
                f"collision_rate={self.collision_rate:.3f})")


class GBSSampler:
    """Run a GBS circuit on the Gaussian simulator and return samples.

    Parameters
    ----------
    backend : GaussianSimulator, optional
        Simulator instance.  Created with default config if not provided.
    seed : int
        Random seed.

    Usage
    -----
    >>> from lumq.algorithms.gbs import gbs_circuit_from_graph, GBSSampler
    >>> import numpy as np
    >>> A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
    >>> circuit, r, U = gbs_circuit_from_graph(A, scale=0.5)
    >>> sampler = GBSSampler(seed=42)
    >>> result = sampler.sample(circuit, shots=1000)
    >>> print(result)
    """

    def __init__(self, backend=None, seed: int = 0):
        self.seed = seed
        if backend is None:
            from lumq.backends import GaussianSimulator, GaussianSimulatorConfig
            self._backend = GaussianSimulator(
                GaussianSimulatorConfig(n_modes=32, seed=seed, track_state=True)
            )
        else:
            self._backend = backend

    def sample(self, circuit, shots: int = 1000) -> GBSResult:
        """Run the GBS circuit and return photon-number samples.

        Uses PNR measurements on all modes.  For Gaussian circuits the
        Gaussian simulator provides the marginal probability for each mode
        (thermal + displacement).  Full multi-mode correlations require
        the hafnian; this sampler uses the marginal approximation for speed
        and exact hafnian-based sampling is available via sample_exact().

        Parameters
        ----------
        circuit : PhotonicCircuit
            Must have PNR measurements on all modes.
        shots : int
            Number of samples.

        Returns
        -------
        GBSResult
        """
        from lumq.compiler import Compiler

        # Compile: expand interferometer into Clements mesh
        compiled, _ = Compiler(decompose_interferometers=True).compile(circuit)

        # Run the backend — samples photon numbers from marginals
        job = self._backend.run(compiled, shots=shots)

        if job.samples is None:
            raise RuntimeError(
                "No samples returned. Ensure the circuit has PNR measurements on all modes."
            )

        return GBSResult(
            samples=job.samples.astype(jnp.int32),
            clicks=job.samples > 0,
            n_modes=circuit.n_modes,
            shots=shots,
        )

    def sample_exact(
        self,
        r: np.ndarray,
        U: np.ndarray,
        shots: int = 100,
        max_photons: int = 8,
    ) -> GBSResult:
        """Exact GBS sampling using the hafnian for multi-mode correlations.

        This is the statistically correct GBS sampler.  It computes the
        full probability distribution over photon-number patterns using the
        hafnian and samples from it exactly.

        Practical limit: N <= 10 modes, max_photons <= 6 per mode.

        Parameters
        ----------
        r : array (N,) — squeezing parameters
        U : array (N,N) — interferometer unitary
        shots : int
        max_photons : int — maximum photon number per mode considered

        Returns
        -------
        GBSResult
        """
        from lumq.algorithms.gbs.hafnian import hafnian
        from lumq.algorithms.gbs.circuit import gbs_circuit

        r = np.asarray(r); N = len(r); U = np.asarray(U, dtype=complex)
        key = jax.random.PRNGKey(self.seed)

        # Build the Gaussian state covariance structure
        # Q matrix for GBS: Q = I + X sigma_Q where sigma_Q is the covariance
        # For simplicity use the Gaussian simulator to get the covariance
        circuit = gbs_circuit(r, U)
        from lumq.compiler import Compiler
        compiled, _ = Compiler(decompose_interferometers=True).compile(circuit)

        # Remove measurements to get the state
        from lumq.compiler.ir import PhotonicCircuit, CircuitMetadata
        state_circuit = PhotonicCircuit(n_modes=N, metadata=CircuitMetadata(name="state"))
        state_circuit.ops = compiled.ops[:]

        from lumq.backends import GaussianSimulator, GaussianSimulatorConfig
        sim = GaussianSimulator(GaussianSimulatorConfig(n_modes=N, seed=self.seed, track_state=True))
        job = sim.run(state_circuit, shots=1)
        state = job.state

        # Build probability table using hafnian
        # P(n_1,...,n_N) = |Haf(A_S)|^2 / (prod n_k! * sqrt(det Q))
        # where A_S is the submatrix of A = X(I - Q^{-1}) indexed by S
        cov = np.array(state.cov)
        hbar = state.hbar
        # Covariance in the convention matching GBS literature
        Q = cov / (hbar/2) + np.eye(2*N)   # Husimi covariance
        A_mat = _build_A_matrix(Q, N)

        # Sample using the exact distribution
        samples = _exact_sample(A_mat, Q, N, shots, max_photons, key)

        return GBSResult(samples=samples, clicks=samples>0, n_modes=N, shots=shots)


def _build_A_matrix(Q: np.ndarray, N: int) -> np.ndarray:
    """Build the GBS A matrix from the Q-function covariance."""
    # A = X (I - Q^{-1}) where X = [[0,I],[I,0]] (mode permutation)
    I = np.eye(2*N)
    Q_inv = np.linalg.inv(Q)
    X = np.zeros((2*N, 2*N))
    X[:N, N:] = np.eye(N)
    X[N:, :N] = np.eye(N)
    return X @ (I - Q_inv)


def _exact_sample(A, Q, N, shots, max_photons, key):
    """Sample photon-number patterns using the hafnian probability table."""
    from lumq.algorithms.gbs.hafnian import hafnian
    import itertools

    det_Q = np.linalg.det(Q)

    # Build probability table for all patterns up to max_photons total
    probs = {}
    patterns = []

    # Enumerate patterns with total photons <= max_photons
    for total in range(0, max_photons+1):
        for pattern in _patterns_with_sum(N, total, max_photons):
            # Build the submatrix A_S (repeat indices for multi-photon)
            idx = []
            for k, nk in enumerate(pattern):
                idx.extend([k]*nk)           # first half: creation
            for k, nk in enumerate(pattern):
                idx.extend([k+N]*nk)         # second half: annihilation
            if not idx:
                p = 1.0 / abs(det_Q)**0.5
            else:
                idx_arr = np.array(idx)
                A_sub = A[np.ix_(idx_arr, idx_arr)]
                haf_val = complex(hafnian(jnp.asarray(A_sub, dtype=jnp.complex128)))
                import math
                denom = math.prod(math.factorial(nk) for nk in pattern)
                p = abs(haf_val)**2 / (denom * abs(det_Q)**0.5)
            probs[pattern] = max(0.0, float(p.real) if hasattr(p, 'real') else float(p))
            patterns.append(pattern)

    # Normalise
    total_prob = sum(probs.values())
    if total_prob <= 0:
        # Fall back to vacuum (all zeros)
        return jnp.zeros((shots, N), dtype=jnp.int32)
    norm_probs = np.array([probs[p]/total_prob for p in patterns])

    # Sample
    key_val = int(jax.random.randint(key, (), 0, 2**31))
    rng = np.random.default_rng(key_val)
    chosen_idx = rng.choice(len(patterns), size=shots, p=norm_probs)
    result = np.array([patterns[i] for i in chosen_idx], dtype=np.int32)
    return jnp.asarray(result)


def _patterns_with_sum(N, total, max_per_mode):
    """Generate all N-tuples of non-negative ints summing to total, each <= max_per_mode."""
    if N == 0:
        yield ()
        return
    for k in range(0, min(total, max_per_mode)+1):
        for rest in _patterns_with_sum(N-1, total-k, max_per_mode):
            yield (k,) + rest
