# typical imports
# we have a simplified deep MD using only the radial information
# and the inverse of the radial information. We don't allow the particules to be
# too close, we allow biases in the pyramids and we multiply the outcome by 
# the descriptor income (in order to preserve the zeros)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import h5py
import sys
import json

from data_gen_1d import gen_data
from utilities import genDistInv, train_step
from utilities import MyDenseLayer, pyramidLayer, pyramidLayerNoBias

nameScript = sys.argv[0]

# we are going to give all the arguments using a Json file
nameJson = sys.argv[1]
print("=================================================")
print("Executing " + nameScript + " following " + nameJson)
print("=================================================")


# opening Json file # TODO:write a function to manipulate all this
jsonFile = open(nameJson) 
data = json.load(jsonFile)   

# loading the input data from the json file
Ncells = data["Ncells"]                  # number of cells
Np = data["Np"]                          # number of particules per cell
Nsamples = data["Nsamples"]              
mu = data["mu"]
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

# the ones not used yet
potentialType = data["potentialType"]

print("We are using the random seed %d"%(seed))
tf.random.set_seed(seed)

dataFile = dataFolder + "data_Ncells_" + str(Ncells) + \
                        "_Np_" + str(Np) + \
                        "_mu_" + str(mu) + \
                        "_minDelta_%.4f"%(minDelta) + \
                        "_Nsamples_" + str(Nsamples) + ".h5"

checkFolder  = "checkpoints/"
checkFile = checkFolder + "check_babyMDEnergyRecv3Body2_Ncells_" + str(Ncells) + \
                        "_Np_" + str(Np) + \
                        "_mu_" + str(mu) + \
                        "_minDelta_%.4f"%(minDelta) + \
                        "_Nsamples_" + str(Nsamples)

# if the file doesn't exist we create it
if not path.exists(dataFile):
  # TODO: encapsulate all this in a function
  pointsArray, \
  potentialArray, \
  forcesArray  = gen_data(Ncells, Np, mu, Nsamples, minDelta)
  
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
potMean = np.mean(potentialArray)
potentialArray -= potMean
potStd = np.std(potentialArray)
potentialArray /= potStd
forcesArray /= potStd

# positions of the 
Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32)

#compute the statistics of the inputs in order to rescale 
#the descriptor computation 
genCoordinates = genDistInv(Rinput, Ncells, Np)

av = tf.reduce_mean(genCoordinates, 
                    axis = 0, 
                    keepdims =True ).numpy()[0]
std = tf.sqrt(tf.reduce_mean(tf.square(genCoordinates - av), 
                             axis = 0, 
                             keepdims=True)).numpy()[0]

class DeepMDsimpleEnergy(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               Np, 
               Ncells,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               av = [0.0, 0.0],
               std = [1.0, 1.0],
               name='deepMDsimpleEnergy',
               **kwargs):
    super(DeepMDsimpleEnergy, self).__init__(name=name, **kwargs)

    # this should be done on the fly, for now we will keep it here
    self.Np = Np
    self.Ncells = Ncells
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]
    # we need to use the tanh here
    self.layerPyramid   = pyramidLayer(descripDim, 
                                       actfn = tf.nn.relu)
    self.layerPyramidInv  = pyramidLayer(descripDim, 
                                       actfn = tf.nn.relu)
    
    # we need to use the tanh especially here
    self.fittingNetwork = pyramidLayer(fittingDim, 
                                       actfn = tf.nn.relu)
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
      LL = tf.concat([L1, L2], axis = 1)
      # (Nsamples*Ncells*Np*(3*Np - 1), 2*descriptorDim)
      Dtemp = tf.reshape(LL, (-1, 3*self.Np-1, 
                              2*self.descriptorDim ))
      # (Nsamples*Ncells*Np, (3*Np - 1), descriptorDim)
      D = tf.reduce_sum(Dtemp, axis = 1)
      # (Nsamples*Ncells*Np, descriptorDim)

      F2 = self.fittingNetwork(D)
      F = self.linfitNet(F2)

      Energy = tf.reduce_sum(tf.reshape(F, (-1, self.Ncells*self.Np)),
                              keepdims = True, axis = 1)

    #Forces = -tape.gradient(Energy, inputs)

    return Energy#, Forces


model = DeepMDsimpleEnergy(Np, Ncells, 
                           filterNet, fittingNet, 
                            av, std)

E = model(Rinput)
model.summary()


### optimization ##
mse_loss_fn = tf.keras.losses.MeanSquaredError()


initialLearningRate = learningRate
lrSchedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initialLearningRate,
    decay_steps=(Nsamples//batchSize)*epochsPerStair,
    decay_rate=decayRate,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedule)

loss_metric = tf.keras.metrics.Mean()

x_train = (pointsArray, potentialArray)

# train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
# train_dataset = train_dataset.shuffle(buffer_size=10000).batch(50)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(16)


epochs = 100

# Iterate over epochs.
for epoch in range(epochs):
  print('============================') 
  print('Start of epoch %d' % (epoch,))

  loss_metric.reset_states()

  # Iterate over the batches of the dataset.
  for step, x_batch_train in enumerate(train_dataset):
    loss = train_step(model, optimizer, mse_loss_fn,
                      x_batch_train[0], 
                      x_batch_train[1])
    loss_metric(loss)

    if step % 100 == 0:
      print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))

  print('epoch %s: mean loss = %s' % (epoch, str(loss_metric.result().numpy())))


model.save_weights(checkFile+"_cycle_0.h5")

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(32)

epochs = 200

# Iterate over epochs.
for epoch in range(epochs):
  print('============================') 
  print('Start of epoch %d' % (epoch,))

  loss_metric.reset_states()

  # Iterate over the batches of the dataset.
  for step, x_batch_train in enumerate(train_dataset):
    loss = train_step(model, optimizer, mse_loss_fn,
                      x_batch_train[0], x_batch_train[1])
    loss_metric(loss)

    if step % 100 == 0:
      print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))

  print('epoch %s: mean loss = %s' % (epoch, str(loss_metric.result().numpy())))

model.save_weights(checkFile+"_cycle_1.h5")

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(64)

epochs = 400

# Iterate over epochs.
for epoch in range(epochs):
  print('============================') 
  print('Start of epoch %d' % (epoch,))

  loss_metric.reset_states()

  # Iterate over the batches of the dataset.
  for step, x_batch_train in enumerate(train_dataset):
    loss = train_step(model, optimizer, mse_loss_fn,
                      x_batch_train[0], x_batch_train[1])
    loss_metric(loss)

    if step % 100 == 0:
      print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))

  print('epoch %s: mean loss = %s' % (epoch, str(loss_metric.result().numpy())))

model.save_weights(checkFile+"_cycle_2.h5")

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(128)

epochs = 800

# Iterate over epochs.
for epoch in range(epochs):
  print('============================') 
  print('Start of epoch %d' % (epoch,))

  loss_metric.reset_states()

  # Iterate over the batches of the dataset.
  for step, x_batch_train in enumerate(train_dataset):
    loss = train_step(model, optimizer, mse_loss_fn,
                      x_batch_train[0], x_batch_train[1])
    loss_metric(loss)

    if step % 100 == 0:
      print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))

  print('epoch %s: mean loss = %s' % (epoch, str(loss_metric.result().numpy())))

model.save_weights(checkFile+"_cycle_3.h5")


##### testing ######
pointsTest, \
potentialTest, \
forcesTest  = gen_data(Ncells, Np, mu, 1000)

potentialTest -= potMean
potentialTest /= potStd
forcesTest /= potStd

potPred = model(pointsTest)
err = tf.sqrt(tf.reduce_sum(tf.square(potPred - potentialTest)))/tf.sqrt(tf.reduce_sum(tf.square(potPred)))
print(str(err.numpy()))

# # ################# Testing each step inside the model#####
with tf.GradientTape() as tape:
  # we watch the inputs 

  tape.watch(Rinput)
  # (Nsamples, Ncells*Np)
  # in this case we are only considering the distances
  genCoordinates = genDistInv(Rinput, model.Ncells, model.Np)
  # (Nsamples*Ncells*Np*(3*Np - 1), 2)
  L1   = model.layerPyramid(genCoordinates[:,1:])*genCoordinates[:,1:]
  # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
  L2   = model.layerPyramidInv(genCoordinates[:,0:1])*genCoordinates[:,1:]
  # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
  LL = tf.concat([L1, L2], axis = 1)
  # (Nsamples*Ncells*Np*(3*Np - 1), 2*descriptorDim)
  Dtemp = tf.reshape(LL, (-1, 3*model.Np-1, 
                          2*model.descriptorDim ))
  # (Nsamples*Ncells*Np, (3*Np - 1), descriptorDim)
  D = tf.reduce_sum(Dtemp, axis = 1)
  # (Nsamples*Ncells*Np, descriptorDim)

  F2 = model.fittingNetwork(D)
  F = model.linfitNet(F2)

  Energy = tf.reduce_sum(tf.reshape(F, (-1, model.Ncells*model.Np)),
                          keepdims = True, axis = 1)

