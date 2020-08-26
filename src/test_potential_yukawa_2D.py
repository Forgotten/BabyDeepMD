import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

from data_gen_2d import genDataYukawa2DPer
from data_gen_2d import computeDerPot2DPer

Np = 2 
mu = 10 
Nsamples = 2 
minDelta = 0.1
Ncells = 5# number of cells per dimension
Lcell = 1.0
L = Ncells * Lcell

radious = 0.75
maxNumNeighs = 8

NpointsPerCell = 1000
Nx = Ncells*NpointsPerCell + 1

xCenter = [0.0, 0.0]
nPointSmear = 10


x_grid, y_grid, pot, dpotdx, dpotdy = computeDerPot2DPer(Nx, mu, L)


plt.figure(1)
plt.imshow(np.fft.fftshift(pot))            


plt.figure(2) 
plt.plot(pot[0,:])
plt.plot(pot[:,0])
plt.legend(["y", "x"])

plt.figure(3)
plt.imshow(np.fft.fftshift(dpotdx))            

plt.figure(4)
plt.imshow(np.fft.fftshift(dpotdy))

plt.figure(5)
plt.plot(x_grid, dpotdx[0,:])
plt.plot(x_grid, dpotdx[:,0])
plt.legend(["y", "x"])

plt.figure(6)
plt.plot(x_grid, dpotdy[0,:])
plt.plot(x_grid, dpotdy[:,0])
plt.legend(["y", "x"])

mu = 1


x_grid, y_grid, pot, dpotdx, dpotdy = computeDerPot2DPer(Nx, mu, L)


plt.figure(11)
plt.imshow(np.fft.fftshift(pot))            


plt.figure(12) 
plt.plot(pot[0,:])
plt.plot(pot[:,0])
plt.legend(["y", "x"])

plt.figure(13)
plt.imshow(np.fft.fftshift(dpotdx))            

plt.figure(14)
plt.imshow(np.fft.fftshift(dpotdy))

plt.figure(15)
plt.plot(x_grid, dpotdx[0,:])
plt.plot(x_grid, dpotdx[:,0])
plt.legend(["y", "x"])

plt.figure(16)
plt.plot(x_grid, dpotdy[0,:])
plt.plot(x_grid, dpotdy[:,0])
plt.legend(["y", "x"])

plt.show()



