"""Shared type aliases and constants."""
from __future__ import annotations
from typing import Union
import jax.numpy as jnp
import numpy as np
from jax import Array

ArrayLike = Union[Array, np.ndarray, list, float, int]
ModeIndex = int
ModePair = tuple[ModeIndex, ModeIndex]
HBAR: float = 2.0
VACUUM_VARIANCE: float = 0.5

def _ensure_jax(x):
    return jnp.asarray(x, dtype=jnp.float64)

def _ensure_complex_jax(x):
    return jnp.asarray(x, dtype=jnp.complex128)

def _check_mode(mode, n_modes, label="mode"):
    if not (0 <= mode < n_modes):
        raise ValueError(f"{label} index {mode} out of range for {n_modes}-mode system.")

def _check_modes(modes, n_modes):
    m0, m1 = modes
    _check_mode(m0, n_modes, "modes[0]")
    _check_mode(m1, n_modes, "modes[1]")
    if m0 == m1:
        raise ValueError(f"Two-mode gate requires distinct modes, got ({m0}, {m1}).")
