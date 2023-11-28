import numpy as np

import jax
import jax.numpy as jnp

from functools import partial

from utils import q_to_psi, psi_to_q, RotMat_q, RotMat_Vec, RotMat_T_Vec
from jax.scipy.ndimage import map_coordinates as mc

from jax import vjp, custom_vjp, value_and_grad
from jax.numpy import (zeros, zeros_like, ones, array, square, sinc, log, exp, pi, where, meshgrid, arange, clip, atleast_1d )
from jax.lax import fori_loop
from jax.random import permutation
from jax.tree_util import tree_map
from jax.nn import one_hot

import equinox as eqx
from equinox import combine, partition, filter_jit, tree_at

class PARS(eqx.Module):

    R: jnp.ndarray
    α: jnp.ndarray
    Δn: jnp.ndarray
    ψ: jnp.ndarray
    t: jnp.ndarray
    nx_d: jnp.integer
    nx: jnp.integer
    ny: jnp.integer
    nz: jnp.integer
    Δx: jnp.ndarray
    Δy: jnp.ndarray
    Δz: jnp.ndarray
    n0: jnp.ndarray
    x_C: jnp.ndarray
    x_D: jnp.ndarray
    x_F: jnp.ndarray
    x_R: jnp.ndarray
    ratio: jnp.ndarray
    x_d: jnp.ndarray
    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray


def T(p, x):

    return 1. / (1. + (( x - p.x_F) / (2. * p.n0 * p.x_R))**2 )

def H(p, x):

    return (((sinc( (x - p.x_C) / (2. * p.x_D)  ))**2 * 
        exp(- p.ratio / (2 * log(2)) * (pi * (x - p.x_C) / (2. * p.x_D))**2 )))

def br(a, p):

    xr, yr, zr = RotMat_T_Vec(p.ψ, p.t, p.x[:,None,None], p.y[None,:,None], p.z[None,None], p.nx, p.ny, p.nz, p.Δx, p.Δy, p.Δz)

    return mc(a, (xr, yr, zr), order=1)


def ip(p, xim):

    xr, yr, zr = RotMat_Vec(p.ψ, p.t, xim, p.y[:,None], p.z[None], p.nx, p.ny, p.nz, p.Δx, p.Δy, p.Δz)
    #jax.debug.print('{s}', s=xr.shape)
    rot = lambda a: mc(a, (xr, yr, zr), order=1)
    Rim, αim, Δnim = rot(p.R), rot(p.α), rot(p.Δn)

    return Rim, αim, Δnim, (xr, yr, zr)

def att(p, Rim, αim, Aim, xi, xim):

    Ai = Aim * exp(- αim * (xi - xim))
    Iim = Rim * Ai * T(p, xi) * H(p, xi)

    return Iim, Ai

def step(p, Aim, xim):

    Rim, αim, Δnim, Rri = ip(p, xim)
    xi = xim + p.Δx / (p.n0 + Δnim)
    Iim, Ai = att(p, Rim, αim, Aim, xi, xim)
    
    return Iim, Ai, xi, Rri

def prop_Ai_xi_bwd(p, Ai, xi):

    Aim = Ai * exp(αim * (xi - xim))
    xim = xi - p.Δx / (p.n0 + Δnim)

    return Aim, xim

def _fwd(p):

    def bf(i, val):

        (Aim, xim, I) = val
        Iim, Ai, xi, _ = step(p, Aim, xim)
        I = I.at[i].set(Iim)
        return (Ai, xi, I)

    I = zeros((p.nx, p.ny, p.nz))

    A0 = ones((p.ny, p.nz))
    x0 = zeros((p.ny, p.nz))

    _start = (A0 ,x0, I)
    _result = fori_loop(0, p.nx_d, bf, _start)

    return _result[-1]   


def _grad(p, IM, loss_fn, filter_spec):
    
    def step_loc(op, st, Aim, xim):
            
        p = combine(op, st)

        return step(p, Aim, xim)
        
    def bf_fwd(i, val):

        A_arr, x_arr = val

        Aim, xim = A_arr[i-1], x_arr[i-1]
        _, Ai, xi, _ = step(p, Aim, xim)
        #jax.debug.print('A = {A}, x = {x}', A = Ai.max(), x = xi.max())
        A_arr = A_arr.at[i].set(Ai)
        x_arr = x_arr.at[i].set(xi)

        return A_arr, x_arr

    op, st = partition(p, filter_spec)
    
    nx_d,ny,nz = IM.shape
    #A0 = ones((p.ny, p.nz))
    #x0 = zeros((p.ny, p.nz))
    #A0 = 
    #x0 = zeros((ny, nz))
    A_arr = zeros((nx_d, ny, nz))
    A_arr = A_arr.at[0].set(ones((ny, nz)))
    x_arr = zeros((nx_d, ny, nz))
    _start_fwd = (A_arr, x_arr)
    A_arr, x_arr = fori_loop(1, nx_d, bf_fwd, _start_fwd)


    def bf_bwd(i, val):

        (err, par_bar, Aj_bar, xj_bar) = val
        j =  p.nx_d - i - 1
        Ajm, xjm = A_arr[j - 1], x_arr[j - 1]
        #jax.debug.print('A = {A}, x = {x}', A = Ajm.max(), x = xjm.max())
        _step = lambda p, A, x: step_loc(p, st, A, x)
        (Iim, Aj, xj, (xjr, yjr, zjr)), vjp_step = vjp(_step, op, Ajm, xjm)
        mask = ~((xjr < 0) | (xjr > (p.nx - 1)) | (yjr < 0) | (yjr > (p.ny - 1)) | (zjr < 0) | (zjr > (p.nz - 1)))

        err_loc, I_bar = value_and_grad(loss_fn)(Iim, IM[j].astype(jnp.float32) / (2**16 - 1) ) 
        #I_bar = Iim - IM[i]
        #err += square(I_bar).sum()
        #Ai_bar, xi_bar = zeros_like(Ai), zeros_like(xi)
        Rr_bar = (zeros_like(xj), zeros_like(xj), zeros_like(xj))
        par_bar_part, Ajm_bar, xjm_bar = vjp_step((I_bar * mask, Aj_bar, xj_bar, Rr_bar))
        par_bar = tree_map(lambda a, b: a + b, par_bar, par_bar_part)

        return (err + err_loc, par_bar, Ajm_bar, xjm_bar)

    Aj_bar = zeros((ny, nz))
    xj_bar = zeros((ny, nz))
    par_bar = tree_map(lambda tree: zeros_like(tree), op)

    _start_bwd = (0, par_bar, Aj_bar, xj_bar)
    err, par_bar, A0_bar, x0_bar = fori_loop(0, p.nx_d, bf_bwd, _start_bwd)

    return err, par_bar

def get_update(bounds, opt_par):

    def update(p, p_bar, lrs):

        #return tree_map(lambda p, g, lr, b: clip(p - lr * g, b[0], b[1]), p, p_bar, lrs, bounds)
        return eqx.tree_at(lambda tree: tuple(getattr(tree, a) for a in opt_par), 
                            p,
                            replace = tuple(clip(getattr(p, a) - getattr(p_bar, a) * getattr(lrs, a), 
                                                 getattr(bounds, a)[0], 
                                                 getattr(bounds, a)[1])
                                            for a in opt_par))
    return update

def get_sgd_step(loss_fn, bounds, reg_fn, filter_opt, opt_par):

    update = get_update(bounds, opt_par)
    grad_reg = value_and_grad(reg_fn, allow_int=True)

    def sgd_step(p, IMi, Mi, lrs):

        err, p_bar = _grad(p, IMi, loss_fn, filter_opt)
        Mi_r = br(Mi, p)
        err_rg, pr_bar = grad_reg(p, Mi_r)
        p_bar = tree_map(lambda a, b: a + b, p_bar, pr_bar)

        #p_bar = tree_at(lambda tree: tuple(getattr(tree, a) for a in ['R', 'α', 'Δn']), 
        #                p_bar, 
        #                replace_fn = lambda tree: tuple(getattr(tree, a) + ran for (a,ran) in zip(['R', 'α', 'Δn'], pr_bar)))
        
        return err + err_rg, update(p, p_bar, lrs)

    return sgd_step

## todo: make parameter updates better (e.g. flatten/unflatten)

def get_sgd_epoch(sgd_step):

    def sgd_epoch(p_arr, lrs, IM, M, key):
        
        ns = IM.shape[0]
        ind = permutation(key, ns)
        
        def bf(i, val):

            err, p_arr = val
            j = ind[i]
            p_i = tree_at(lambda tree: tuple(getattr(tree, a) for a in ['ψ', 't', 'x_F']), 
                          p_arr, 
                          replace_fn = lambda tree: tree[j])
            e, p_i = sgd_step(p_i, IM[j], M[j], lrs)
            p_arr = tree_at(lambda tree: tuple(getattr(tree, a) for a in ['ψ', 't', 'x_F']), 
                            p_i, 
                            replace = tuple(getattr(p_arr, a) * (1. - one_hot(j, ns)[:,None,None,None]) + getattr(p_i, a)[None] * one_hot(j, ns)[:,None,None,None]  for a in ['ψ', 't', 'x_F']))

            return err + e, p_arr
        
        return fori_loop(0, ns, bf, (0, p_arr))

    return sgd_epoch










