# typical imports
# we have a simplified deep MD using only the radial information
# and the inverse of the radial information locally
# for the long range we only use
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import h5py
import sys
import json

from data_gen_1d import genDataYukawa
from utilities import genDistInv, train_step, genDistInvLongRange
from utilities import genDistLongRangeFull
from utilities import MyDenseLayer, pyramidLayer
from utilities import pyramidLayerNoBias, NUFFTLayer

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

nameScript = sys.argv[0].split('/')[-1]

# we are going to give all the arguments using a Json file
nameJson = sys.argv[1]
print("=================================================")
print("Executing " + nameScript + " following " + nameJson, flush = True)
print("=================================================")


# opening Json file # TODO:write a function to manipulate all this
jsonFile = open(nameJson) 
data = json.load(jsonFile)   

# loading the input data from the json file
Ncells = data["Ncells"]                  # number of cells
Np = data["Np"]                          # number of particules per cell
Nsamples = data["Nsamples"]              # number of samples 
Lcell = data["lengthCell"]               # lenght of each cell
mu = data["mu"]                          # the parameter mu of the potential
minDelta = data["minDelta"]
filterNet = data["filterNet"]
fittingNet = data["fittingNet"]
seed = data["seed"]
batchSize = data["batchSize"]
epochsPerStair = data["epochsPerStair"]
learningRate = data["learningRate"]
decayRate = data["decayRate"]
dataFolder = data["dataFolder"]
loadFile = data["loadFile"]
Nepochs = data["numberEpoch"]

# the ones not used yet
potentialType = data["potentialType"]

# we will save them for now
# we need to extract them from the kson file
NpointsFourier = 501
fftChannels = 1
sigmaFFT = 0.0001
xLims = [0.0, 10.0]


print("We are using the random seed %d"%(seed))
tf.random.set_seed(seed)

dataFile = dataFolder + "data_"+ potentialType + \
                        "_Ncells_" + str(Ncells) + \
                        "_Np_" + str(Np) + \
                        "_mu_" + str(mu) + \
                        "_minDelta_%.4f"%(minDelta) + \
                        "_Nsamples_" + str(Nsamples) + ".h5"

checkFolder  = "checkpoints/"
checkFile = checkFolder + "checkpoint_" + nameScript + \
                          "potential_"+ potentialType + \
                          "_Json_" + nameJson + \
                          "_Ncells_" + str(Ncells) + \
                          "_Np_" + str(Np) + \
                          "_mu_" + str(mu) + \
                          "_minDelta_%.4f"%(minDelta) + \
                          "_Nsamples_" + str(Nsamples)

print("Using data in %s"%(dataFile))

# if the file doesn't exist we create it
if not path.exists(dataFile):
  # TODO: encapsulate all this in a function
  print("Data file does not exist, we create a new one")

  pointsArray, \
  potentialArray, \
  forcesArray  = genDataYukawa(Ncells, Np, mu, Nsamples, minDelta, Lcell)
  
  hf = h5py.File(dataFile, 'w') 
  
  hf.create_dataset('points', data=pointsArray)   
  hf.create_dataset('potential', data=potentialArray) 
  hf.create_dataset('forces', data=forcesArray)
  
  hf.close()

# extracting the data
hf = h5py.File(dataFile, 'r')

pointsArray = hf['points'][:]
forcesArray = hf['forces'][:]
potentialArray = hf['potential'][:]

# normalization of the data

if loadFile: 
  # if we are loading a file, the normalization needs to be 
  # properly defined 
  forcesMean = data["forcesMean"]
  forcesStd = data["forcesStd"]
else: 
  forcesMean = np.mean(forcesArray)
  forcesStd = np.std(forcesArray)

print("mean of the forces is %.8f"%(forcesMean))
print("std of the forces is %.8f"%(forcesStd))

potentialArray /= forcesStd
forcesArray -= forcesMean
forcesArray /= forcesStd

# positions of the 
Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32)

#compute the statistics of the inputs in order to rescale 
#the descriptor computation 
genCoordinates = genDistInv(Rinput[0:1000,:], Ncells, Np)

# we need to compute the normalization coefficients
ExtCoords = genDistLongRangeFull(Rinput[0:1000,:], Ncells, Np, 
                                      [0.0, 0.0], [1.0, 1.0]) # this are hard coded

KernelGen0 = tf.reshape(tf.reduce_sum(tf.math.reciprocal(ExtCoords), axis = 2), (-1, 1))
KernelGen1 = tf.reshape(tf.reduce_sum(tf.sqrt(tf.math.reciprocal(ExtCoords)), axis = 2), (-1, 1))
KernelGen2 = tf.reshape(tf.reduce_sum(tf.square(tf.math.reciprocal(ExtCoords)), axis = 2), (-1, 1))
KernelGen3 = tf.reshape(tf.reduce_sum(tf.math.exp(-mu*ExtCoords), axis = 2), (-1, 1))

mean0, std0 =  tf.math.reduce_std(KernelGen0), tf.math.reduce_mean(KernelGen0)
mean1, std1 =  tf.math.reduce_std(KernelGen1), tf.math.reduce_mean(KernelGen1)
mean2, std2 =  tf.math.reduce_std(KernelGen2), tf.math.reduce_mean(KernelGen2)
mean3, std3 =  tf.math.reduce_std(KernelGen3), tf.math.reduce_mean(KernelGen3)

meanLongRange = np.array([mean0.numpy(), mean1.numpy(), mean2.numpy(), mean3.numpy()])
stdLongRange = np.array([std0.numpy(), std1.numpy(), std2.numpy(), std3.numpy()])

meanLongRange = meanLongRange.reshape((1,4))
stdLongRange = stdLongRange.reshape((1,4))
print("long range meand")
print(meanLongRange)

print("long range std")
print(stdLongRange)

if loadFile: 
  # if we are loadin a file we need to be sure that we are 
  # loding the correct mean and std for the inputs
  av = data["av"]
  std = data["std"]
  print("loading the saved mean and std of the generilized coordinates")
else:
  av = tf.reduce_mean(genCoordinates, 
                      axis = 0, 
                      keepdims =True ).numpy()[0]
  std = tf.sqrt(tf.reduce_mean(tf.square(genCoordinates - av), 
                               axis = 0, 
                               keepdims=True)).numpy()[0]

print("mean of the inputs are %.8f and %.8f"%(av[0], av[1]))
print("std of the inputs are %.8f and %.8f"%(std[0], std[1]))


class DeepMDsimpleForces(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               Np, 
               Ncells,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               mu = 2.0, 
               av = [0.0, 0.0],
               std = [1.0, 1.0],
               meanLongRange = [0.0 ,0.0, 0.0, 0.0],
               stdLongRange = [1.0 , 1.0, 1.0, 1.0],
               NpointsFourier = 500, 
               fftChannels = 4,
               sigmaFFT = 0.1, 
               xLims = [0.0, 10.0], 
               name='deepMDsimpleForces',
               **kwargs):
    super(DeepMDsimpleForces, self).__init__(name=name, **kwargs)

    print("xLims = %f, %f"%(xLims[0], xLims[1]) )

    # this should be done on the fly, for now we will keep it here
    self.Np = Np
    self.Ncells = Ncells
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.mu = mu
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]

    self.NpointsFourier = NpointsFourier
    self.fftChannels    = fftChannels 
    self.sigmaFFT = sigmaFFT
    self.meanLongRange = meanLongRange
    self.stdLongRange = stdLongRange
    # we may need to use the tanh here
    self.layerPyramid   = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)
    self.layerPyramidInv  = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)

    # layer to apply the fmm (this is already windowed)
    self.NUFFTLayer = NUFFTLayer(fftChannels, NpointsFourier, sigmaFFT, xLims)

    self.layerPyramidLongRange  = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)
    
    # we may need to use the tanh especially here
    self.fittingNetwork = pyramidLayer(fittingDim, 
                                       actfn = tf.nn.tanh)
    self.linfitNet      = MyDenseLayer(1)    

  @tf.function
  def call(self, inputs):

    with tf.GradientTape() as tape:
      # we watch the inputs 

      tape.watch(inputs)
      # (Nsamples, Ncells*Np)
      # in this case we are only considering the distances
      genCoordinates = genDistInv(inputs, self.Ncells, self.Np, 
                                  self.av, self.std)

      # (Nsamples*Ncells*Np*(3*Np - 1), 2)
      L1   = self.layerPyramid(genCoordinates[:,1:])*genCoordinates[:,1:]
      # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
      L2   = self.layerPyramidInv(genCoordinates[:,0:1])*genCoordinates[:,0:-1]
      # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)

      # we compute the FMM and the normalize by the number of particules
      longRangewCoord = self.NUFFTLayer(inputs)
      # (Nsamples, Ncells*Np, 1) # we are only using 4 kernels
      # we normalize the output of the fmm layer before feeding them to network
      longRangewCoord2 = tf.reshape(longRangewCoord, (-1, self.fftChannels))
      # (Nsamples*Ncells*Np, 1)
      L3   = self.layerPyramidLongRange(longRangewCoord2)
      # (Nsamples*Ncells*Np, descriptorDim)


      # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
      LL = tf.concat([L1, L2], axis = 1)
      # (Nsamples*Ncells*Np*(3*Np - 1), 2*descriptorDim)
      Dtemp = tf.reshape(LL, (-1, 3*self.Np-1, 
                              2*self.descriptorDim ))
      # (Nsamples*Ncells*Np, (3*Np - 1), 2*descriptorDim)
      D = tf.reduce_sum(Dtemp, axis = 1)
      # (Nsamples*Ncells*Np, 2*descriptorDim)

      DLongRange = tf.concat([D, L3], axis = 1)

      F2 = self.fittingNetwork(DLongRange)
      F = self.linfitNet(F2)

      Energy = tf.reduce_sum(tf.reshape(F, (-1, self.Ncells*self.Np)),
                              keepdims = True, axis = 1)

    Forces = -tape.gradient(Energy, inputs)

    return Forces


## Defining the model
model = DeepMDsimpleForces(Np, Ncells, 
                           filterNet, fittingNet, mu,
                            av, std, meanLongRange, stdLongRange, 
                            NpointsFourier, fftChannels, sigmaFFT, xLims)

# quick run of the model to check that it is correct.
# we use a small set 
F = model(Rinput[0:10,:])
model.summary()

# # Create checkpointing directory if necessary
# if not os.path.exists(checkFolder):
#     os.mkdir(checkFolder)
#     print("Directory " , checkFolder ,  " Created ")
# else:    
#     print("Directory " , checkFolder ,  " already exists :)")

# ## in the case we need to load an older saved model
# if loadFile: 
#   print("Loading the weights the model contained in %s"%(loadFile), flush = True)
#   model.load_weights(loadFile)

## We use a decent training or a custom one if necessary
if type(Nepochs) is not list:
  Nepochs = [200, 400, 800, 1600]
  #batchSizeArray = map(lambda x: x*batchSize, [1, 2, 4, 8]) 
  batchSizeArray = [batchSize*2**i for i in range(0,4)]  
else:  
  assert len(Nepochs) == len(batchSize)
  batchSizeArray = batchSize

print("Training cycles in number of epochs")
print(Nepochs)
print("Training batch sizes for each cycle")
print(batchSizeArray)

### optimization parameters ##
mse_loss_fn = tf.keras.losses.MeanSquaredError()

initialLearningRate = learningRate
lrSchedule = tf.keras.optimizers.schedules.ExponentialDecay(
             initialLearningRate,
             decay_steps=(Nsamples//batchSizeArray[0])*epochsPerStair,
             decay_rate=decayRate,
             staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedule)

loss_metric = tf.keras.metrics.Mean()

for cycle, (epochs, batchSizeL) in enumerate(zip(Nepochs, batchSizeArray)):

  print('++++++++++++++++++++++++++++++', flush = True) 
  print('Start of cycle %d' % (cycle,))
  print('Total number of epochs in this cycle: %d'%(epochs,))
  print('Batch size in this cycle: %d'%(batchSizeL,))

# we need to modify this one later
  weightE = 0.0
  weightF = 1.0

  x_train = (pointsArray, forcesArray)

  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batchSizeL)

  # Iterate over epochs.
  for epoch in range(epochs):
    print('============================', flush = True) 
    print('Start of epoch %d' % (epoch,))
  
    loss_metric.reset_states()
  
    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
      loss = train_step(model, optimizer, mse_loss_fn,
                        x_batch_train[0], 
                        x_batch_train[1], weightF)
      loss_metric(loss)
  
      if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))

    # mean loss saved in the metric
    meanLossStr = str(loss_metric.result().numpy())
    # learning rate using the decay 
    lrStr = str(optimizer._decayed_lr('float32').numpy())
    print('epoch %s: mean loss = %s  learning rate = %s'%(epoch,
                                                          meanLossStr,
                                                          lrStr))

  print("saving the weights")
  model.save_weights(checkFile+"_cycle_"+str(cycle)+".h5")


# print( "values of the trainable decay %f" %(model.NUFFTLayer.mu.numpy()))

##### testing ######
pointsTest, \
potentialTest, \
forcesTest  =  genDataYukawa(Ncells, Np, mu, 1000, minDelta, Lcell)

forcesTestRscl =  forcesTest - forcesMean
forcesTestRscl = forcesTestRscl/forcesStd

forcePred = model(pointsTest)

err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTestRscl)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))
print("Relative Error in the forces is " +str(err.numpy()))


# with tf.GradientTape() as tape:
#   # we watch the inputs 
#   tape.watch(Rinput)
#   # (Nsamples, Ncells*Np)
#   # in this case we are only considering the distances
#   genCoordinates = genDistInv(Rinput, model.Ncells, model.Np, 
#                               model.av, model.std)
#   # (Nsamples*Ncells*Np*(3*Np - 1), 2)
#   L1   = model.layerPyramid(genCoordinates[:,1:])*genCoordinates[:,1:]
#   # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
#   L2   = model.layerPyramidInv(genCoordinates[:,0:1])*genCoordinates[:,0:-1]
#   # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
#   # we compute the FMM and the normalize by the number of particules
#   longRangewCoord = model.NUFFTLayer(Rinput)
#   # (Nsamples, Ncells*Np, 1) # we are only using 4 kernels
#   # we normalize the output of the fmm layer before feeding them to network
#   longRangewCoord2 = tf.reshape(longRangewCoord, (-1, model.fftChannels))
#   # (Nsamples*Ncells*Np, 1)
#   L3   = model.layerPyramidLongRange(longRangewCoord2)
#   # (Nsamples*Ncells*Np, descriptorDim)

#   # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
#   LL = tf.concat([L1, L2], axis = 1)
#   # (Nsamples*Ncells*Np*(3*Np - 1), 2*descriptorDim)
#   Dtemp = tf.reshape(LL, (-1, 3*model.Np-1, 
#                           2*model.descriptorDim ))
#   # (Nsamples*Ncells*Np, (3*Np - 1), 2*descriptorDim)
#   D = tf.reduce_sum(Dtemp, axis = 1)
#   # (Nsamples*Ncells*Np, 2*descriptorDim)

#   DLongRange = tf.concat([D, L3], axis = 1)

#   F2 = model.fittingNetwork(DLongRange)
#   F = model.linfitNet(F2)

#   Energy = tf.reduce_sum(tf.reshape(F, (-1, model.Ncells*model.Np)),
#                           keepdims = True, axis = 1)

# Forces = -tape.gradient(Energy, inputs)

# return Forces


