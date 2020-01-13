import numpy as np 
from data_gen_1d import *

Ncells = 4
Np = 2
mu = 10
minDelta = 0.01

sizeCell = 1/Ncells
midPoints = np.linspace(sizeCell/2.0,1-sizeCell/2.0, Ncells)
points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
points = np.sort(points.reshape((-1,1)), axis = 0)

# we want to check that the points are not too close 
while np.min(points[1:] - points[0:-1]) < minDelta:
	points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
	points = np.sort(points.reshape((-1,1)), axis = 0)

R = np.triu(potential(points,points.T, mu), 1)
potTotal = np.sum(R)

F = forces(points,points.T, mu)
Forces = np.sum(F, axis = 1) 
