# typical imports
# we have a simplified deep MD using only the radial information
# and the inverse of the radial information. We don't allow the particules to be
# too close, we allow biases in the pyramids and we multiply the outcome by 
# the descriptor income (in order to preserve the zeros)
# This version supports an inhomogeneous number of particules, however we need to 
# provide a neighboor list. 

# in this case we are not assuming rotational symmetry

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import h5py
import sys
import json
import time
import csv
from data_gen_2d import genDataPer2DMixed
from data_gen_2d import genDataYukawa2DPer
from nufft_layers import NUFFTLayerMultiChannel2Dmixed
from utilities import genDistInvPerNlistVec2D, trainStepList, computInterList2DOpt
from utilities import MyDenseLayer, pyramidLayer, pyramidLayerNoBias

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
nameScript = sys.argv[0].split('/')[-1]

# we are going to give all the arguments using a Json file
nameJson = sys.argv[1]
#nameJson = "C:/Users/DELL/Documents/GitHub/BabyDeepMD2d/run/Np20_Per_2D/Np20_Per_mu_10.json"
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
mu1 = data["mu1"]
mu2 = data["mu2"]
weight1 = data["weight1"]
weight2 = data["weight2"]                      
minDelta = data["minDelta"]
filterNet = data["filterNet"]
fittingNet = data["fittingNet"]
seed = data["seed"]
batchSize = data["batchSize"]
epochsPerStair = data["epochsPerStair"]
learningRate = data["learningRate"]
decayRate = data["decayRate"]
dataFolder = data["dataFolder"]
#loadFile = data["loadFile"]
Nepochs = data["numberEpoch"]

# we need to add a few new parameters
maxNumNeighs = data["maxNumNeighbors"]
radious = data["radious"]

NpointsFourier = 51
fftChannels = 2

# Limits of the super cell
xLims = [0.0, Ncells*Lcell]

# the ones not used yet
potentialType = data["potentialType"]

print("We are using the random seed %d"%(seed))
tf.random.set_seed(seed)

dataFile = dataFolder + "data_2D_"+ potentialType + \
                        "_Ncells_" + str(Ncells) + \
                        "_Np_" + str(Np) + \
                        "_mu1_" + str(mu1) + \
                        "_mu2_" + str(mu2) + \
                        "_weight1_" + str(weight1) + \
                        "_weight2_" + str(weight2) + \
                        "_minDelta_%.4f"%(minDelta) + \
                        "_Nsamples_" + str(Nsamples) + ".h5"

checkFolder  = "/"
checkFile = checkFolder + "checkpoint_2Dsimpleorigin_NUFFT_nonormal_" + \
                          "potential_"+ potentialType + \
                          "_Ncells_" + str(Ncells) + \
                          "_Np_" + str(Np) + \
                        "_mu1_" + str(mu1) + \
                        "_mu2_" + str(mu2) + \
                        "_weight1_" + str(weight1) + \
                        "_weight2_" + str(weight2) + \
                          "_minDelta_%.4f"%(minDelta) + \
                          "_Nsamples_" + str(Nsamples)

print("Using data in %s"%(dataFile))

# TODO: add the path file for this one
assert potentialType == "Periodic" or\
       potentialType == "YukawaPeriodic"

# if the file doesn't exist we create it
if not path.exists(dataFile):
  # TODO: encapsulate all this in a function
  print("Data file does not exist, we create a new one")

  if potentialType == "Periodic":

    print("Creating %s data"%(potentialType))
    pointsArray, \
    potentialArray, \
    forcesArray  = genDataPer2DMixed(Ncells, Np, 
                                mu1, mu2, Nsamples, 
                                minDelta, Lcell, weight1, weight2)

  elif potentialType == "YukawaPeriodic":

    print("Creating %s data"%(potentialType))
    pointsArray, \
    potentialArray, \
    forcesArray  = genDataYukawa2DPer(Ncells, Np, 
                                      mu, Nsamples, 
                                      minDelta, Lcell)
  
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

#if loadFile: 
#  # if we are loading a file, the normalization needs to be 
#  # properly defined 
#  forcesMean = data["forcesMean"]
#  forcesStd = data["forcesStd"]
#else: 
forcesMean = np.mean(forcesArray)
forcesStd = np.std(forcesArray)

print("mean of the forces is %.8f"%(forcesMean))
print("std of the forces is %.8f"%(forcesStd))

#potentialArray /= forcesStd
#forcesArray -= forcesMean
#forcesArray /= forcesStd


# positions of the 
Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32)

# we only consider the first 100 
Rin = Rinput[:100,:,:]
#compute the statistics of the inputs in order to rescale 
#the descriptor

# this is the total lenght of the computational domain
L = Lcell*Ncells

# this is very ineficient, we need to rewrite it properly
# TODO: explore how to do this using scikitlearn
#####################
# computing the distances: 
Rinnumpy = Rin.numpy()

Idx = computInterList2DOpt(Rinnumpy, L,  radious, maxNumNeighs)
# dimension are (Nsamples, Npoints and MaxNumneighs)
neighList = tf.Variable(Idx)
Npoints = Np*Ncells**2


genCoordinates = genDistInvPerNlistVec2D(Rin, neighList, L)
filter = tf.cast(tf.reduce_sum(tf.abs(genCoordinates), axis = -1)>0, tf.int32)
numNonZero =  tf.reduce_sum(filter, axis = 0).numpy()
numTotal = genCoordinates.shape[0]  


av = tf.reduce_sum(genCoordinates, 
                    axis = 0, 
                    keepdims =True).numpy()[0]/numNonZero
std = np.sqrt((tf.reduce_sum(tf.square(genCoordinates - av), 
                             axis = 0, 
                             keepdims=True).numpy()[0] - av**2*(numTotal-numNonZero)) /numNonZero)

#print("mean of the inputs are %.8f and %.8f"%(av[0], av[1], av[2]))
#print("std of the inputs are %.8f and %.8f"%(std[0], std[1], std[2]))


class DeepMDsimpleEnergy(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               Npoints,
               L, 
               maxNumNeighs = 4,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               av = [0.0, 0.0, 0.0],
               std = [1.0, 1.0, 1.0],
               NpointsFourier = 500,
               fftChannels = 4,
               xLims = [0.0,10.0],
               name='deepMDsimpleEnergy',
               **kwargs):

    super(DeepMDsimpleEnergy, self).__init__(name=name, **kwargs)

    self.L = L
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
    self.fftChannels = fftChannels
    # we may need to use the tanh here
    self.layerPyramid   = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)
    self.layerPyramidDir  = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)
    self.NUFFTLayer = NUFFTLayerMultiChannel2Dmixed(fftChannels, \
                                              NpointsFourier, 
                                              xLims, 
                                              mu1,mu2) 
    self.layerPyramidLongRange  = pyramidLayer(descripDim, 
                                       actfn = tf.nn.relu)    
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
      genCoordinates = genDistInvPerNlistVec2D(inputs, 
                                              neighList, self.L, 
                                              self.av, self.std) 
      # (Nsamples*Npoints*maxNumNeighs, 3)

      # the L1 and L2 functions only depends on the first entry
      L1   = self.layerPyramid(genCoordinates[:,:1])*genCoordinates[:,:1]
      # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
      L2   = self.layerPyramidDir(genCoordinates[:,1:])*genCoordinates[:,:1]
      # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
        
      LL = tf.concat([L1, L2], axis = 1)
      # (Nsamples*Npoints*maxNumNeighs, 2*descriptorDim)
      Dtemp = tf.reshape(LL, (-1, self.maxNumNeighs,
                              2*self.descriptorDim ))
      # (Nsamples*Npoints, maxNumNeighs, descriptorDim)
      D_short = tf.reduce_sum(Dtemp, axis = 1)
      # (Nsamples*Npoints, descriptorDim)

      long_range_coord = (self.NUFFTLayer(inputs) )#\
                       #   - self.av_long_range) \
                        #  / self.av_long_range
      # # (Nsamples, Ncells*Np, 1) # we are only using 4 kernels
      # we normalize the output of the fmm layer before feeding them to network
      long_range_coord2 = tf.reshape(long_range_coord, 
                                     (-1, self.fftChannels))
      # (Nsamples*Ncells*Np, 1)
      L3   = self.layerPyramidLongRange(long_range_coord2)
      
      DLongRange = tf.concat([D_short, L3], axis = 1)
      F2 = self.fittingNetwork(DLongRange)
      F = self.linfitNet(F2)

      Energy = tf.reduce_sum(tf.reshape(F, (-1, self.Npoints)),
                              keepdims = True, axis = 1)

    Forces = -tape.gradient(Energy, inputs)

    return Energy, Forces


# moving the mean and std to Tensorflow format 
avTF = tf.constant(av, dtype=tf.float32)
stdTF = tf.constant(std, dtype=tf.float32)

## Defining the model
model = DeepMDsimpleEnergy(Npoints, L, maxNumNeighs,
                           filterNet, fittingNet, 
                            avTF, stdTF,
                            NpointsFourier, fftChannels,
                            xLims)


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
#if loadFile: 
#  print("Loading the weights the model contained in %s"(loadFile), flush = True)
#  model.load_weights(loadFile)

## We use a decent training or a custom one if necessary
Nepochs = [200, 400, 800, 1600]
batchSizeArray = [8,16,32,32]  

print("Training cycles in number of epochs")
print(Nepochs)
print("Training batch sizes for each cycle")
print(batchSizeArray)

errorlist = []
losslist = []

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

if potentialType == "Periodic":
    pointsTest, \
    potentialTest, \
    forcesTest  = genDataPer2DMixed(Ncells, Np, 
                                mu1, mu2, 10, 
                                minDelta, Lcell, weight1, weight2)

if potentialType == "YukawaPeriodic":
    pointsTest, \
    potentialTest, \
    forcesTest  = genDataYukawa2DPer(Ncells, Np, 
                                      mu, 10, 
                                      minDelta, Lcell)
    
IdxTest = computInterList2DOpt(pointsTest, L,  radious, maxNumNeighs)
neighListTest = tf.Variable(IdxTest)

###################training loop ##################################

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
    start = time.time()
    print('============================', flush = True) 
    print('Start of epoch %d' % (epoch,))
  
    loss_metric.reset_states()
  
    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):

      Rinnumpy = x_batch_train[0].numpy()
      Idx = computInterList2DOpt(Rinnumpy, L,  radious, maxNumNeighs)
      neighList = tf.Variable(Idx)

      loss = trainStepList(model, optimizer, mse_loss_fn,
                           x_batch_train[0], neighList,
                           x_batch_train[1], 
                           x_batch_train[2], 
                        weightE, weightF)
      loss_metric(loss)
  
      if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))

    Idx = computInterList2DOpt(pointsArray[:10,:,:], L,  radious, maxNumNeighs)
    neighList = tf.Variable(Idx)

    pottrain, forcetrain = model(pointsArray[:10,:,:],neighList)
    errtrain = tf.sqrt(tf.reduce_sum(tf.square(forcetrain - forcesArray[:10,:,:])))/tf.sqrt(tf.reduce_sum(tf.square(forcetrain)))
    print("Relative Error in the trained forces is " +str(errtrain.numpy()))

    potPred, forcePred = model(pointsTest, neighListTest)
    
    err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTest)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))
    print("Relative Error in the forces is " +str(err.numpy()))
    end = time.time()
    print('time elapsed %.4f'%(end - start))
    errorlist.append(err.numpy())        
    with open('error'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist)
    # mean loss saved in the metric
    meanLossStr = str(loss_metric.result().numpy())
    # learning rate using the decay 
    lrStr = str(optimizer._decayed_lr('float32').numpy())
    print('epoch %s: mean loss = %s  learning rate = %s'%(epoch,
                                                          meanLossStr,
                                                          lrStr))
    losslist.append(loss_metric.result().numpy())
    with open('loss'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(losslist)

    print("saving the weights")
    model.save_weights(checkFile+".h5")



