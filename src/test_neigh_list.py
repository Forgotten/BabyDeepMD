import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 

from utilities import genDistInvPerNlist

Ls = 10.0 
Npoints = 5
Nsamples = 4 
threshold = 3.0
maxNumPoints = 4

Rinnumpy = Ls*(np.random.rand(Nsamples,Npoints))
Rinnumpy = np.sort(Rinnumpy, axis = 1 )

DistNumpy = np.abs(Rinnumpy.reshape(Nsamples,Npoints,1) - Rinnumpy.reshape(Nsamples,1,Npoints))

DistNumpy = DistNumpy - Ls*np.round(DistNumpy/Ls)

idex = np.where(DistNumpy<2.0)


Idex = []
for ii in range(0,Nsamples):
	sampleIdx = []
	for jj in range(0, Npoints):
		localIdx = []
		for kk in range(0, Npoints):
			if jj!= kk and np.abs(DistNumpy[ii,jj,kk]) < threshold:
				localIdx.append(kk)
		while len(localIdx) < maxNumPoints:
			localIdx.append(-1)
		sampleIdx.append(localIdx)

	Idex.append(sampleIdx)

Idx = np.array(Idex)
# samples, points and neighbors

Rin = tf.Variable(Rinnumpy)
# neighList = tf.Variable(Idx)
neighList = tf.Variable(Idx)

Rout = genDistInvPerNlist(Rin, Npoints, neighList, Ls)

LL = tf.reshape(Rout, (Nsamples, Npoints, maxNumPoints, 2))