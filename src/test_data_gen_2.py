import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

from data_gen_2d import genDataPer2D
from data_gen_2d import potential
from data_gen_2d import forces

from data_gen_2d import potentialPer
from data_gen_2d import forces

from utilities import computInterList2Dv2
from utilities import genDistInvPerNlist2D

from utilities import computInterList2D
from utilities import genDistInvPerNlist2DSimple

from utilities import genDistInvPerNlist2Dwhere
 
from utilities import genDistInvPerNlist2Dwherev2

Ncells = 3
Np = 2 
mu = 10 
Nsamples = 2 
minDelta = 0.1
Lcell = 1.0 
L = Ncells*Lcell

radious = 0.75
maxNumNeighs = 8

Nx = 100
dx = Lcell/Nx
x = np.linspace(0,Lcell*Ncells, Nx*Ncells +1)[:-1]

XX, YY = np.meshgrid(x, x)
Z = np.concatenate([np.reshape(XX, (-1, 1)), np.reshape(YY, (-1, 1))], axis= -1 )
y = np.array([0.5004, 0.50004])
Pot = np.reshape(potential(Z, y, mu), (Nx*Ncells, Nx*Ncells))
plt.figure(1)
plt.imshow(Pot)


yPer = np.array([0.0013, 1.013])
PotPer = np.reshape(potentialPer(Z, yPer, mu, L), (Nx*Ncells, Nx*Ncells))
plt.figure(2)
plt.imshow(PotPer)

points, pot, forces = genDataPer2D(Ncells, Np, mu, Nsamples, minDelta, Lcell)

plt.figure(3)
plt.scatter(points[0,:,0], points[0, :, 1])

# we are testing the new interaction list 
# we repeat the index of the current particle
neighList = computInterList2Dv2(points, L,  radious, maxNumNeighs)
centerIdx = 3

Idx = neighList[0,centerIdx,:]
Idx = Idx[Idx!=centerIdx]

plt.scatter(points[0,centerIdx,0], points[0,centerIdx, 1], color='r')
plt.scatter(points[0,Idx,0], points[0,Idx, 1], color='g')

Nrender = 100
theta = np.linspace(0, 2*np.pi, Nrender+1)
xtheta = np.cos(theta)
ytheta = np.sin(theta)

circx = points[0,centerIdx,0] + radious*xtheta
circy = points[0,centerIdx,1] + radious*ytheta

# circx = circx - L*np.round(circx/L)
# circy = circy - L*np.round(circy/L)

plt.plot(circx, circy)

Rin = tf.Variable(points, dtype = tf.float32)
neighList = tf.Variable(neighList)

Npoints = Np*Ncells**2

R_diff = genDistInvPerNlist2D(Rin, Npoints, neighList, L)

# old computation of the list (this needs to be properly optimized 
# suing numba or something like that.)
neighListSimple = computInterList2D(points, L,  radious, maxNumNeighs)

neighListSimple = tf.Variable(neighListSimple)


R_diffSimple = genDistInvPerNlist2DSimple(Rin, Npoints, neighListSimple, L)

# R_diffwhere  = genDistInvPerNlist2Dwhere(Rin, Npoints, neighListSimple, L)

R_diffwhere2  = genDistInvPerNlist2Dwherev2(Rin, Npoints, neighListSimple, L)
