# typical imports
# we have a simplified deep MD using only the radial information
# and the inverse of the radial information. We don't allow the particules to be
# too close, we allow biases in the pyramids and we multiply the outcome by 
# the descriptor income (in order to preserve the zeros)
# This version supports an inhomogeneous number of particules, however we need to 
# provide a neighboor list. 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import h5py
import sys
import json

from data_gen_1d import genDataYukawaPer
from utilities import genDistInvPerNlistLoop, trainStepList, computInterList
from utilities import MyDenseLayer, pyramidLayer, pyramidLayerNoBias

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

# These one are deprecated we will continue using them for 
# no to call old data sets
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

# we need to add a few new parameters
maxNumNeighs = data["maxNumNeighbors"]
radious = data["radious"]

# the ones not used yet
potentialType = data["potentialType"]

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

# TODO: add the path file for this one
assert potentialType == "Periodic"

# if the file doesn't exist we create it
if not path.exists(dataFile):
  # TODO: encapsulate all this in a function
  print("Data file does not exist, we create a new one")

  pointsArray, \
  potentialArray, \
  forcesArray  = genDataYukawaPer(Ncells, Np, mu, Nsamples, minDelta, Lcell)
  
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

# we only consider the first 100 
Rin = Rinput[:100,:]
#compute the statistics of the inputs in order to rescale 
#the descriptor

# this is the total lenght of the computational domain
L = Lcell*Ncells

# this is very ineficient, we need to rewrite it properly
# TODO: explore how to do this using scikitlearn
#####################
# computing the distances: 
Rinnumpy = Rin.numpy()

Idx = computInterList(Rinnumpy, L,  radious, maxNumNeighs)
# dimension are (Nsamples, Npoints and MaxNumneighs)
neighList = tf.Variable(Idx)
Npoints = Np*Ncells


genCoordinates = genDistInvPerNlistLoop(Rin, Npoints, neighList, L)
filter = tf.cast(tf.abs(genCoordinates)>0, tf.int32)
numNonZero =  tf.reduce_sum(filter, axis = 0).numpy()[0]
numTotal = genCoordinates.shape[0]  

if loadFile: 
  # if we are loadin a file we need to be sure that we are 
  # loding the correct mean and std for the inputs
  av = data["av"]
  std = data["std"]
  print("loading the saved mean and std of the generilized coordinates")
else:
  av = tf.reduce_sum(genCoordinates, 
                    axis = 0, 
                    keepdims =True).numpy()[0]/numNonZero
  std = np.sqrt((tf.reduce_sum(tf.square(genCoordinates - av), 
                             axis = 0, 
                             keepdims=True).numpy()[0] - av**2*(numTotal-numNonZero)) /numNonZero)

print("mean of the inputs are %.8f and %.8f"%(av[0], av[1]))
print("std of the inputs are %.8f and %.8f"%(std[0], std[1]))


class DeepMDsimpleEnergy(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               Npoints,
               maxNumNeighs = 4,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               av = [0.0, 0.0],
               std = [1.0, 1.0],
               name='deepMDsimpleEnergy',
               **kwargs):
    super(DeepMDsimpleEnergy, self).__init__(name=name, **kwargs)

    # this should be done on the fly, for now we will keep it here
    self.Npoints = Npoints
    # maximum number of neighbors
    self.maxNumNeighs = maxNumNeighs
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]
    # we may need to use the tanh here
    self.layerPyramid   = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)
    self.layerPyramidInv  = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)
    
    # we may need to use the tanh especially here
    self.fittingNetwork = pyramidLayer(fittingDim, 
                                       actfn = tf.nn.tanh)
    self.linfitNet      = MyDenseLayer(1)    

  @tf.function
  def call(self, inputs, neighList):
    with tf.GradientTape() as tape:
      # we watch the inputs 

      tape.watch(inputs)
      # (Nsamples, Npoints)
      # in this case we are only considering the distances
      genCoordinates = genDistInvPerNlistLoop(inputs, self.Npoints, 
                                          neighList, L, 
                                          self.av, self.std) # this need to be fixed
      # (Nsamples*Npoints*maxNumNeighs, 2)
      L1   = self.layerPyramid(genCoordinates[:,1:])*genCoordinates[:,1:]
      # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
      L2   = self.layerPyramidInv(genCoordinates[:,0:1])*genCoordinates[:,0:-1]
      # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
      LL = tf.concat([L1, L2], axis = 1)
      # (Nsamples*Npoints*maxNumNeighs, 2*descriptorDim)
      Dtemp = tf.reshape(LL, (-1, self.maxNumNeighs, 
                              2*self.descriptorDim ))
      # (Nsamples*Npoints, maxNumNeighs, descriptorDim)
      D = tf.reduce_sum(Dtemp, axis = 1)
      # (Nsamples*Npoints, descriptorDim)

      F2 = self.fittingNetwork(D)
      F = self.linfitNet(F2)

      Energy = tf.reduce_sum(tf.reshape(F, (-1, self.Npoints)),
                              keepdims = True, axis = 1)

    Forces = -tape.gradient(Energy, inputs)

    return Energy, Forces

## Defining the model
model = DeepMDsimpleEnergy(Npoints, maxNumNeighs,
                           filterNet, fittingNet, 
                            av, std)

# quick run of the model to check that it is correct.
# we use a small set 
E, F = model(Rin, neighList)
model.summary()

# Create checkpointing directory if necessary
if not os.path.exists(checkFolder):
    os.mkdir(checkFolder)
    print("Directory " , checkFolder ,  " Created ")
else:    
    print("Directory " , checkFolder ,  " already exists :)")

## in the case we need to load an older saved model
if loadFile: 
  print("Loading the weights the model contained in %s"(loadFile), flush = True)
  model.load_weights(loadFile)

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

  x_train = (pointsArray, potentialArray, forcesArray)

  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batchSizeL)

  # Iterate over epochs.
  for epoch in range(epochs):
    print('============================', flush = True) 
    print('Start of epoch %d' % (epoch,))
  
    loss_metric.reset_states()
  
    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):

      Rinnumpy = x_batch_train[0].numpy()
      Idx = computInterList(Rinnumpy, L,  radious, maxNumNeighs)
      neighList = tf.Variable(Idx)

      loss = trainStepList(model, optimizer, mse_loss_fn,
                        x_batch_train[0], neighList,
                        x_batch_train[1], 
                        x_batch_train[2], 
                        weightE, weightF)
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


##### testing ######
pointsTest, \
potentialTest, \
forcesTest  = genDataYukawaPer(Ncells, Np, mu, 1000, minDelta, Lcell)

forcesTestRscl =  forcesTest- forcesMean
forcesTestRscl = forcesTestRscl/forcesStd

potPred, forcePred = model(pointsTest)

err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTestRscl)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))
print("Relative Error in the forces is " +str(err.numpy()))


# # # ################# Testing each step inside the model#####
# with tf.GradientTape() as tape:
#   # we watch the inputs 

#   tape.watch(Rinput)
#   # (Nsamples, Ncells*Np)
#   # in this case we are only considering the distances
#   genCoordinates = genDistInvPer(Rinput, model.Ncells, model.Np)
#   # (Nsamples*Ncells*Np*(3*Np - 1), 2)
#   L1   = model.layerPyramid(genCoordinates[:,1:])*genCoordinates[:,1:]
#   # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
#   L2   = model.layerPyramidInv(genCoordinates[:,0:1])*genCoordinates[:,1:]
#   # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
#   LL = tf.concat([L1, L2], axis = 1)
#   # (Nsamples*Ncells*Np*(3*Np - 1), 2*descriptorDim)
#   Dtemp = tf.reshape(LL, (-1, 3*model.Np-1, 
#                           2*model.descriptorDim ))
#   # (Nsamples*Ncells*Np, (3*Np - 1), descriptorDim)
#   D = tf.reduce_sum(Dtemp, axis = 1)
#   # (Nsamples*Ncells*Np, descriptorDim)

#   F2 = model.fittingNetwork(D)
#   F = model.linfitNet(F2)

#   Energy = tf.reduce_sum(tf.reshape(F, (-1, model.Ncells*model.Np)),
#                           keepdims = True, axis = 1)

