
import numpy as np

import jax
import jax.numpy as jnp

from functools import partial

def q1_mul_q2(q1,q2):
    
    r1,v1 = q1[...,0],q1[...,1:]
    r2,v2 = q2[...,0],q2[...,1:]
    
    q_res = jnp.zeros_like(q1)
    q_res = q_res.at[...,0].set(r1*r2 - (v1*v2).sum(axis=-1))
    q_res = q_res.at[...,1:].set(r1[...,jnp.newaxis]*v2 + r2[...,jnp.newaxis]*v1 + jnp.cross(v1,v2))

    return q_res

def conj_q(q):
    
    qc = jnp.zeros_like(q)
    qc = qc.at[...,0].set(q[...,0])
    qc = qc.at[...,1:].set(- q[...,1:])

    return qc

@jax.jit
def q_to_psi(q,eps=1e-4):

    a2 = jnp.where(q[...,3] > (-1 + eps), (1. - q[...,3]) / (1. + q[...,3]) , 0)[...,jnp.newaxis]

    return q[...,:3] * 0.5 * (a2 + 1)

@jax.jit
def psi_to_q(psi):

    a2 = jnp.square(psi).sum(axis=-1)
    shp = list(psi.shape[:-1])
    shp.append(4)
    #nw,_ = psi.shape

    out = jnp.zeros(shp)
    out = out.at[...,:3].set(2 * psi / (a2[...,jnp.newaxis] + 1))
    out = out.at[...,3].set((1. - a2) / (1. + a2))

    return out

@jax.jit
def RotMat_q(q):

    R00 = 2 * (q[...,0]**2 + q[...,1]**2) - 1
    R01 = 2 * (q[...,1] * q[...,2] - q[...,0] * q[...,3])
    R02 = 2 * (q[...,1] * q[...,3] + q[...,0] * q[...,2])
    R10 = 2 * (q[...,1] * q[...,2] + q[...,0] * q[...,3])
    R11 = 2 * (q[...,0]**2 + q[...,2]**2) - 1
    R12 = 2 * (q[...,2] * q[...,3] - q[...,0] * q[...,1])
    R20 = 2 * (q[...,1] * q[...,3] - q[...,0] * q[...,2])
    R21 = 2 * (q[...,2] * q[...,3] + q[...,0] * q[...,1])
    R22 = 2 * (q[...,0]**2 + q[...,3]**2) - 1

    return R00,R01,R02,R10,R11,R12,R20,R21,R22

@jax.jit
def RotMat_Vec(q,t,x,y,z,nx,ny,nz,Δx,Δy,Δz):

    R00,R01,R02,R10,R11,R12,R20,R21,R22 = RotMat_q(q)
    xr = (R00 * (x - Δx*nx/2. - t[...,0]) + R10 * (y - Δy*ny/2. - t[...,1]) + R20 * (z - Δz*nz/2. - t[...,2])) / Δx + nx/2.
    yr = (R01 * (x - Δx*nx/2. - t[...,0]) + R11 * (y - Δy*ny/2. - t[...,1]) + R21 * (z - Δz*nz/2. - t[...,2])) / Δy + ny/2.
    zr = (R02 * (x - Δx*nx/2. - t[...,0]) + R12 * (y - Δy*ny/2. - t[...,1]) + R22 * (z - Δz*nz/2. - t[...,2])) / Δz + nz/2.
    
    return xr,yr,zr

@jax.jit
def RotMat_T_Vec(q,t,x,y,z,nx,ny,nz,Δx,Δy,Δz):

    R00,R01,R02,R10,R11,R12,R20,R21,R22 = RotMat_q(q)
    xr = (R00 * (x - Δx*nx/2.) + R01 * (y - Δy*ny/2.) + R02 * (z - Δz*nz/2.) + t[...,0]) / Δx + nx/2.
    yr = (R10 * (x - Δx*nx/2.) + R11 * (y - Δy*ny/2.) + R12 * (z - Δz*nz/2.) + t[...,1]) / Δy + ny/2.
    zr = (R20 * (x - Δx*nx/2.) + R21 * (y - Δy*ny/2.) + R22 * (z - Δz*nz/2.) + t[...,2]) / Δz + nz/2.
    
    return xr,yr,zr




@jax.jit
def prox_tv():
    ...