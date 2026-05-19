"""Symplectic matrix primitives for Gaussian gates."""
from __future__ import annotations
import jax.numpy as jnp

def embed_single(S1, mode, n_modes):
    S = jnp.eye(2*n_modes); i = 2*mode
    return S.at[i:i+2, i:i+2].set(S1)

def embed_two(S2, modes, n_modes):
    m0, m1 = modes
    idx = [2*m0, 2*m0+1, 2*m1, 2*m1+1]
    S = jnp.eye(2*n_modes)
    for i, ri in enumerate(idx):
        for j, cj in enumerate(idx):
            S = S.at[ri, cj].set(S2[i, j])
    return S

def S_phase_shifter(phi):
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([[c, -s],[s, c]])

def S_squeezer(r, phi):
    c, s = jnp.cos(phi/2), jnp.sin(phi/2)
    er, emr = jnp.exp(r), jnp.exp(-r)
    sq = jnp.array([[emr, 0.],[0., er]])
    Rp = jnp.array([[c, s],[-s, c]])
    Rm = jnp.array([[c,-s],[s, c]])
    return Rm @ sq @ Rp

def S_beamsplitter(theta, phi):
    t = jnp.cos(theta)
    rx = jnp.cos(phi)*jnp.sin(theta)
    rp = jnp.sin(phi)*jnp.sin(theta)
    return jnp.array([[t, 0., -rx, rp],
                      [0., t,  rp, rx],
                      [rx,-rp,  t, 0.],
                      [-rp,-rx, 0., t]])

def S_two_mode_squeeze(r, phi):
    c, s = jnp.cosh(r), jnp.sinh(r)
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([[c,    0.,   s*cp, -s*sp],
                      [0.,   c,    s*sp,  s*cp],
                      [s*cp, s*sp, c,     0.  ],
                      [-s*sp,s*cp, 0.,    c   ]])

def S_displacer_vec(alpha, hbar=2.0):
    sc = jnp.sqrt(2.0*hbar)
    a = jnp.asarray(alpha, dtype=jnp.complex128)
    return jnp.array([sc*jnp.real(a), sc*jnp.imag(a)])


