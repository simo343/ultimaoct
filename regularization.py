
from jax import value_and_grad
from jax.numpy import square, diff, sqrt

def err_l2(a,mask):
    nx,ny,nz = a.shape
    return square(a * mask).sum() * 1. / (nx * ny * nz)

grad_l2 = value_and_grad(err_l2, argnums=0)

def err_tvI(a,β):
    
    err_x = diff(a, axis=0, append=a[-1,None])#(a[:-1,:-1,:-1] - a[1:,:-1,:-1]) 
    err_y = diff(a, axis=1, append=a[:,-1,None])#(a[:-1,:-1,:-1] - a[:-1,1:,:-1])
    err_z = diff(a, axis=2, append=a[...,-1,None])#(a[:-1,:-1,:-1] - a[:-1,:-1,1:])
    
    err = sqrt(err_x**2 + err_y**2 + err_z**2 + β).sum()
    
    nx,ny,nz = a.shape

    return err * 1. / (nx * ny * nz)

grad_tvI = value_and_grad(err_tvI, argnums=0)

def err_tvII(a):
    
    err_x = diff(a, axis=0, append=a[-1,None])#(a[:-1,:-1,:-1] - a[1:,:-1,:-1]) 
    err_y = diff(a, axis=1, append=a[:,-1,None])#(a[:-1,:-1,:-1] - a[:-1,1:,:-1])
    err_z = diff(a, axis=2, append=a[:,:,-1,None])#(a[:-1,:-1,:-1] - a[:-1,:-1,1:])
    
    err = (err_x**2 + err_y**2 + err_z**2).sum()
    
    nx,ny,nz = a.shape

    return err * 1. / (nx * ny * nz)

grad_tvII = value_and_grad(err_tvII, argnums=0)