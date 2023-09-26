import numpy as np

import jax
import jax.numpy as jnp

from functools import partial

import importlib

import utils
importlib.reload(utils)
from utils import q_to_psi, psi_to_q, RotMat_q, RotMat_Vec, RotMat_T_Vec

import rt
importlib.reload(rt)
from rt import rk4_step, rk2_step

def T(x, x_f, x_R, n0):

    return 1./(1. + jnp.square((x - x_f) / (2. * n0 * x_R)))

def H(x, x_C, x_D, ratio):

    return (jnp.square(jnp.sinc( (x - x_C) / (2. * x_D)  )) * 
            jnp.exp(- ratio / (2 * jnp.log(2)) * jnp.square(jnp.pi * (x - x_C) / (2. * x_D))))

@jax.jit
def crd_step_fwd(xim,
                 yim,
                 zim,
                 Δx,
                 n0,
                 Δnim):

    xi = xim + Δx / (n0 + Δnim)
    yi = yim
    zi = zim

    return xi,yi,zi


@jax.jit
def distort_ip(R,
                  α,
                  Δn,
                  n0,
                  xim,
                  yim,
                  zim,
                  q,
                  t,
                  nx,
                  ny,
                  nz,
                  Δx,
                  Δy,
                  Δz):

    xr,yr,zr = RotMat_Vec(q,t,xim,yim,zim,nx,ny,nz,Δx,Δy,Δz)

    Δnim = jax.scipy.ndimage.map_coordinates(Δn, (xr, yr, zr), order=1)
    αim =  jax.scipy.ndimage.map_coordinates(α,  (xr, yr, zr), order=1)
    Rim =  jax.scipy.ndimage.map_coordinates(R,  (xr, yr, zr), order=1)
    
    return Rim, αim, Δnim, xr, yr, zr#, RotMat_Vec(q,t,xim,yim,zim,nx,ny,nz,Δx,Δy,Δz)

@jax.jit
def step(Rim, αim, Aimm, xi, xim, n0, x_C, x_D, x_f, x_R, ratio):

    Aim = Aimm * jnp.exp(- αim * (xi - xim))
    Iim = Rim * Aim * H(xim, x_C, x_D, ratio) * T(xim, x_f, x_R, n0)
    
    return Iim, Aim

def step_full(R, α, Aimm, Δn, n0, xim, yim, zim, q, t, x_C, x_D, x_f, x_R, ratio, nx, ny, nz, Δx, Δy, Δz):

    Rim, αim, Δnim, xr, yr, zr = distort_ip(R, α, Δn, n0, xim, yim, zim, q, t, nx, ny, nz, Δx, Δy, Δz)
    xi, yi, zi = crd_step_fwd(xim, yim, zim, Δx, n0, Δnim)
    Ii, Aim = step(Rim, αim, Aimm, xi, xim, n0, x_C, x_D, x_f, x_R, ratio)
    return Ii, Aim, xi

@partial(jax.jit,static_argnames=('nx','ny','nz'))
def grad_distort_RαΔnqt(R, α, Δn, A0, n0, x0, y0, z0, q, t, x_C, x_D, x_f, x_R, ratio, nx, ny, nz, Δx, Δy, Δz, I_GT):

    def bf_fwd(i,val):
        
        Aarr, xarr = val
        xim = jnp.where(i > 0, xarr[i-1], x0)
        Aim = jnp.where(i > 0, Aarr[i-1], A0)
        Ii, Ai, xi = step_full(R, α, Aim, Δn, n0, xim, y0, z0, q, t, x_C, x_D, x_f, x_R, ratio, nx, ny, nz, Δx, Δy, Δz)
        xarr = xarr.at[i].set(xi)
        Aarr = Aarr.at[i].set(Ai)

        return Aarr, xarr
    
    def bf_bwd(i,val):

        j = nx - i - 1

        Aarr, xarr, R_bar, α_bar, Δn_bar, q_bar, t_bar, Aj_bar, xj_bar, x_C_bar, x_D_bar, x_f_bar, x_R_bar, ratio_bar, err = val
        Ajm = jnp.where(j > 0, Aarr[j-1], A0)
        xjm = jnp.where(j > 0, xarr[j-1], x0)

        step_ranqt = lambda r, a, A, dn, x, q, t, x_C, x_D, x_f, x_R, ratio: step_full(r, a, A, dn, n0, x, y0, z0, q, t, x_C, x_D, x_f, x_R, ratio, nx, ny, nz, Δx, Δy, Δz)

        (Ij, Aj, xj), vjp_step_ranqt = jax.vjp(step_ranqt, R, α, Ajm, Δn, xjm, q, t, x_C, x_D, x_f, x_R, ratio)
        xjr, yjr, zjr = RotMat_Vec(q, t, xjm, y0, z0, nx, ny, nz, Δx, Δy, Δz)
        mask = ~((xjr < 0) | (xjr > (nx - 1)) | (yjr < 0) | (yjr > (ny - 1)) | (zjr < 0) | (zjr > (nz - 1)))
        
        Ij_bar = (Ij - I_GT[j]) * mask
        err += jnp.square(Ij_bar).sum()        
        R_bar_part, α_bar_part, Ajm_bar, Δn_bar_part, xjm_bar, q_bar_part, t_bar_part, x_C_bar_part, x_D_bar_part, x_f_bar_part, x_R_bar_part, ratio_bar_part  = vjp_step_ranqt((Ij_bar, Aj_bar, xj_bar))
        
        
        R_bar = R_bar.at[:].set(R_bar + R_bar_part)
        α_bar = α_bar.at[:].set(α_bar + α_bar_part)
        Δn_bar = Δn_bar.at[:].set(Δn_bar + Δn_bar_part)
        q_bar = q_bar.at[:].set(q_bar + q_bar_part)
        t_bar = t_bar.at[:].set(t_bar + t_bar_part)

        x_C_bar += x_C_bar_part
        x_D_bar += x_D_bar_part
        x_f_bar += x_f_bar_part
        x_R_bar += x_R_bar_part 
        ratio_bar += ratio_bar_part


        return Aarr, xarr, R_bar, α_bar, Δn_bar, q_bar, t_bar, Ajm_bar, xjm_bar, x_C_bar, x_D_bar, x_f_bar, x_R_bar, ratio_bar, err 
    
    Aarr = jnp.zeros((nx, ny, nz))
    xarr = jnp.zeros((nx, ny, nz))
    Aarr,xarr = jax.lax.fori_loop(0, nx, bf_fwd, (Aarr, xarr))

    R_bar = jnp.zeros((nx, ny, nz))
    α_bar = jnp.zeros((nx, ny, nz))
    Δn_bar = jnp.zeros((nx, ny, nz))
    q_bar = jnp.zeros_like(q)
    t_bar = jnp.zeros_like(t)

    Aj_bar = jnp.zeros((ny,nz))
    xj_bar = jnp.zeros((ny,nz))

    x_C_bar = 0.
    x_D_bar = 0.
    x_f_bar = 0.
    x_R_bar = 0.
    ratio_bar = 0.

    err = 0.

    Aarr, xarr, R_bar, α_bar, Δn_bar, q_bar, t_bar, Aj_bar, xj_bar, x_C_bar, x_D_bar, x_f_bar, x_R_bar, ratio_bar, err  = jax.lax.fori_loop(0, nx, bf_bwd,
   (Aarr, xarr, R_bar, α_bar, Δn_bar, q_bar, t_bar, Aj_bar, xj_bar, x_C_bar, x_D_bar, x_f_bar, x_R_bar, ratio_bar, err))
    
    return err, R_bar, α_bar, Δn_bar, q_bar * 1./(nx * ny * nz), t_bar * 1./(nx * ny * nz),  x_C_bar, x_D_bar, x_f_bar, x_R_bar, ratio_bar

@partial(jax.jit,static_argnames=('nx','ny','nz'))
def distort_fwd(R, α, Δn, A0, n0, x0, y0, z0, q, t, x_C, x_D, x_f, x_R, ratio, nx, ny, nz, Δx, Δy, Δz):

    def body_fun(i,val):

        Iarr, Aim, xim = val
        
        Ii, Ai, xi = step_full(R, α, Aim, Δn, n0, xim, y0, z0, q, t, x_C, x_D, x_f, x_R, ratio, nx, ny, nz, Δx, Δy, Δz)
        Iarr = Iarr.at[i].set(Ii)

        return Iarr,Ai,xi
    
   # A0 = jnp.ones((ny,nz))
    Iarr = jnp.zeros((nx,ny,nz))
    
    Iarr,An,xn = jax.lax.fori_loop(0,nx,body_fun,(Iarr,A0,x0))

    return Iarr




def err_l2(a,mask):
    nx,ny,nz = a.shape
    return jnp.square( a * mask).sum() * 1. / (nx * ny * nz)

grad_l2 = jax.value_and_grad(err_l2, argnums=0)

def err_tvI(a,β):
    
    err_x = jnp.diff(a, axis=0, append=a[-1,jnp.newaxis])#(a[:-1,:-1,:-1] - a[1:,:-1,:-1]) 
    err_y = jnp.diff(a, axis=1, append=a[:,-1,jnp.newaxis])#(a[:-1,:-1,:-1] - a[:-1,1:,:-1])
    err_z = jnp.diff(a, axis=2, append=a[:,:,-1,jnp.newaxis])#(a[:-1,:-1,:-1] - a[:-1,:-1,1:])
    
    err = jnp.sqrt(err_x**2 + err_y**2 + err_z**2 + β).sum()
    
    nx,ny,nz = a.shape

    return err * 1. / (nx * ny * nz)

grad_tvI = jax.value_and_grad(err_tvI, argnums=0)

def err_tvII(a):
    
    err_x = jnp.diff(a, axis=0, append=a[-1,jnp.newaxis])#(a[:-1,:-1,:-1] - a[1:,:-1,:-1]) 
    err_y = jnp.diff(a, axis=1, append=a[:,-1,jnp.newaxis])#(a[:-1,:-1,:-1] - a[:-1,1:,:-1])
    err_z = jnp.diff(a, axis=2, append=a[:,:,-1,jnp.newaxis])#(a[:-1,:-1,:-1] - a[:-1,:-1,1:])
    
    err = (err_x**2 + err_y**2 + err_z**2).sum()
    
    nx,ny,nz = a.shape

    return err * 1. / (nx * ny * nz)

grad_tvII = jax.value_and_grad(err_tvII, argnums=0)


@partial(jax.jit, static_argnames=('nx', 'ny', 'nz'), donate_argnums=(0,1,2))
def optim_step_single(R,
                      α,
                      Δn,
                      A0,
                      n0,
                      x0,
                      y0,
                      z0,
                      q,
                      t, 
                      x_C, 
                      x_D, 
                      x_f, 
                      x_R, 
                      ratio,
                      xx,
                      yy,
                      zz,
                      nx,
                      ny,
                      nz,
                      Δx,
                      Δy,
                      Δz,
                      I_GT,
                      mask,
                      lr_R,
                      lr_α,
                      lr_Δn,
                      lr_psi,
                      lr_t,
                      lr_x_C,
                      lr_x_D,
                      lr_x_f,
                      lr_x_R,
                      lr_ratio,
                      λ_l2_R,
                      λ_tv_R,
                      λ_l2_α,
                      λ_tv_α,
                      λ_l2_Δn,
                      λ_tv_Δn,
                      R_min,
                      R_max,
                      α_min,
                      α_max,
                      Δn_min,
                      Δn_max,
                      t_min,
                      t_max,
                      x_C_min,
                      x_C_max,
                      x_D_min,
                      x_D_max,
                      x_f_min,
                      x_f_max,
                      x_R_min,
                      x_R_max,
                      ratio_min,
                      ratio_max):

    

    err = 0.
    β_R = 1e-10
    β_α = 1e-10
    β_Δn = 1e-10


    err_D, R_bar, α_bar, Δn_bar, q_bar, t_bar, x_C_bar, x_D_bar, x_f_bar, x_R_bar, ratio_bar = grad_distort_RαΔnqt(R,α,Δn,A0,n0,x0,y0,z0,q,t, x_C, x_D, x_f, x_R, ratio,nx,ny,nz,Δx,Δy,Δz,I_GT) 
    
    err += err_D

    mask_rot = jax.scipy.ndimage.map_coordinates(mask, RotMat_T_Vec(q,t,xx,yy,zz,nx,ny,nz,Δx,Δy,Δz), order=1, mode='constant', cval=1)

    err_l2_R, R_bar_l2 = grad_l2(R, mask_rot)
    err_tv_R, R_bar_tv = grad_tvI(R, β_R)
    err += err_l2_R * λ_l2_R
    err += err_tv_R * λ_tv_R 
    R_bar += λ_l2_R * R_bar_l2
    R_bar += λ_tv_R * R_bar_tv

    err_l2_α, α_bar_l2 = grad_l2(α, mask_rot)
    err_tv_α, α_bar_tv = grad_tvI(α, β_α)
    err += err_l2_α * λ_l2_α
    err += err_tv_α * λ_tv_α
    α_bar += λ_l2_α * α_bar_l2
    α_bar += λ_tv_α * α_bar_tv

    err_l2_Δn, Δn_bar_l2 = grad_l2(Δn, mask_rot)
    #err_tv_Δn, Δn_bar_tv = grad_tvI(Δn, β_Δn)
    err_tv_Δn, Δn_bar_tv = grad_tvII(Δn)
    err += err_l2_Δn * λ_l2_Δn
    err += err_tv_Δn * λ_tv_Δn
    Δn_bar += λ_l2_Δn * Δn_bar_l2
    Δn_bar += λ_tv_Δn * Δn_bar_tv

    R = R.at[:].set(jnp.clip(R - lr_R * R_bar, R_min, R_max))
    α = α.at[:].set(jnp.clip(α - lr_α * α_bar, α_min, α_max))
    Δn = Δn.at[:].set(jnp.clip(Δn - lr_Δn * Δn_bar, Δn_min, Δn_max))
    ψ = q_to_psi(q[jnp.newaxis])
    _,q_bar_to_psi_bar = jax.vjp(psi_to_q, ψ)
    ψ_bar = q_bar_to_psi_bar(q_bar[jnp.newaxis])[0]
    ψ = ψ.at[:].set(ψ - lr_psi * ψ_bar)
    q = psi_to_q(ψ)[0]
    t = t.at[:].set(jnp.clip(t - lr_t * t_bar, t_min, t_max))
    x_C = (jnp.clip(x_C - lr_x_C * x_C_bar, x_C_min, x_C_max))
    x_D = (jnp.clip(x_D - lr_x_D * x_D_bar, x_D_min, x_D_max))
    x_f = (jnp.clip(x_f - lr_x_f * x_f_bar, x_f_min, x_f_max))
    x_R = (jnp.clip(x_R - lr_x_R * x_R_bar, x_R_min, x_R_max))
    ratio = (jnp.clip(ratio - lr_ratio * ratio_bar, ratio_min, ratio_max))

    return R, α, Δn, q, t, x_C, x_D, x_f, x_R, ratio, err 

@partial(jax.jit, static_argnames=('nx', 'ny', 'nz'), donate_argnums=(0,1,2))
def optim_step_sgd(R,
                   α,
                   Δn,
                   A0,
                   n0,
                   x0,
                   y0,
                   z0,
                   q_arr,
                   t_arr,
                   x_C_arr, 
                   x_D_arr, 
                   x_f_arr, 
                   x_R_arr, 
                   ratio_arr,
                   xx,
                   yy,
                   zz,
                   nx,
                   ny,
                   nz,
                   Δx,
                   Δy,
                   Δz,
                   I_GT_arr,
                   mask_arr,
                   lr_R,
                   lr_α,
                   lr_Δn,
                   lr_psi,
                   lr_t,
                   lr_x_C,
                   lr_x_D,
                   lr_x_f,
                   lr_x_R,
                   lr_ratio,
                   λ_l2_R,
                   λ_tv_R,
                   λ_l2_α,
                   λ_tv_α,
                   λ_l2_Δn,
                   λ_tv_Δn,
                   R_min,
                   R_max,
                   α_min,
                   α_max,
                   Δn_min,
                   Δn_max,
                   t_min,
                   t_max,
                   x_C_min,
                   x_C_max,
                   x_D_min,
                   x_D_max,
                   x_f_min,
                   x_f_max,
                   x_R_min,
                   x_R_max,
                   ratio_min,
                   ratio_max,
                   rand_key,
                   allowed_indices):
    
    #n_angles = I_GT_arr.shape[0]
    n_angles = allowed_indices.shape[0]
    ind_arr = jax.random.permutation(rand_key, allowed_indices)

    def bf(i,val):

        R,α,Δn,q_arr,t_arr,x_C_arr,x_D_arr,x_f_arr,x_R_arr,ratio_arr,err = val
        ind_loc = ind_arr[i]
        R,α,Δn,q,t,x_C,x_D,x_f,x_R,ratio,err_loc = optim_step_single(R,
                                                                    α,
                                                                    Δn,
                                                                    A0,
                                                                    n0,
                                                                    x0,
                                                                    y0,
                                                                    z0,
                                                                    q_arr[ind_loc],
                                                                    t_arr[ind_loc],
                                                                    x_C_arr[ind_loc], 
                                                                    x_D_arr[ind_loc], 
                                                                    x_f_arr[ind_loc], 
                                                                    x_R_arr[ind_loc], 
                                                                    ratio_arr[ind_loc],
                                                                    xx,
                                                                    yy,
                                                                    zz,
                                                                    nx,
                                                                    ny,
                                                                    nz,
                                                                    Δx,
                                                                    Δy,
                                                                    Δz,
                                                                    I_GT_arr[ind_loc],
                                                                    mask_arr[ind_loc],
                                                                    lr_R,
                                                                    lr_α,
                                                                    lr_Δn,
                                                                    lr_psi,
                                                                    lr_t,
                                                                    lr_x_C,
                                                                    lr_x_D,
                                                                    lr_x_f,
                                                                    lr_x_R,
                                                                    lr_ratio,
                                                                    λ_l2_R,
                                                                    λ_tv_R,
                                                                    λ_l2_α,
                                                                    λ_tv_α,
                                                                    λ_l2_Δn,
                                                                    λ_tv_Δn,
                                                                    R_min,
                                                                    R_max,
                                                                    α_min,
                                                                    α_max,
                                                                    Δn_min,
                                                                    Δn_max,
                                                                    t_min,
                                                                    t_max,
                                                                    x_C_min,
                                                                    x_C_max,
                                                                    x_D_min,
                                                                    x_D_max,
                                                                    x_f_min,
                                                                    x_f_max,
                                                                    x_R_min,
                                                                    x_R_max,
                                                                    ratio_min,
                                                                    ratio_max)
        
        #q_arr = q_arr.at[ind_loc].set(jnp.where(ind_loc == 0, q_arr[0], q))
        #t_arr = t_arr.at[ind_loc].set(jnp.where(ind_loc == 0, t_arr[0], t))
        
        q_arr = q_arr.at[ind_loc].set(q)
        t_arr = t_arr.at[ind_loc].set(t)
        x_C_arr = x_C_arr.at[ind_loc].set(x_C)
        x_D_arr = x_D_arr.at[ind_loc].set(x_D)
        x_f_arr = x_f_arr.at[ind_loc].set(x_f)
        x_R_arr = x_R_arr.at[ind_loc].set(x_R)
        ratio_arr = ratio_arr.at[ind_loc].set(ratio)

        return R,α,Δn,q_arr,t_arr,x_C_arr,x_D_arr,x_f_arr,x_R_arr,ratio_arr,err+err_loc

    err = 0.
    return jax.lax.fori_loop(0,n_angles,bf,(R,α,Δn,q_arr,t_arr,x_C_arr,x_D_arr,x_f_arr,x_R_arr,ratio_arr,err))
