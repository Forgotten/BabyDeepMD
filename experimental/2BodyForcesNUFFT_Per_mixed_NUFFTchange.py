# typical imports
# we have a simplified deep MD using only the radial information
# and the inverse of the radial information locally
# for the long range we only use
# in this case we initialize the multipliers and we let the method evolve on its own
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import h5py
import sys
import json
import csv

from data_gen_1d import genDataYukawaPerMixed
from utilities import genDistInvPeradjusted, train_step
from utilities import MyDenseLayerNoBias, pyramidLayer
#from utilities import pyramidLayerNoBias#, NUFFTLayerMultiChannelInitMixed
def gaussianPer(x, tau, L = 2*np.pi):
  return tf.exp( -tf.square(x  )/(4*tau)) + \
         tf.exp( -tf.square(x-L)/(4*tau)) + \
         tf.exp( -tf.square(x+L)/(4*tau))
#@tf.function 
def gaussianDeconv(k, tau):
  return tf.sqrt(np.pi/tau)*tf.exp(tf.square(k)*tau)


class NUFFTLayerMultiChannelInitMixed(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, nChannels, NpointsMesh, tau, xLims, mu1 = 1.0, mu2=0.5):
    super(NUFFTLayerMultiChannelInitMixed, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 
    self.mu1 = tf.constant(mu1, dtype=tf.float32)
    self.mu2 = tf.constant(mu2, dtype=tf.float32)
    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1

    
    self.xLims = xLims
    self.L = np.abs(xLims[1] - xLims[0])
    self.tau = tf.constant(12*(self.L/(2*np.pi*NpointsMesh))**2, 
                           dtype = tf.float32)# the size of the mollifications
    self.kGrid = tf.constant((2*np.pi/self.L)*\
                              np.linspace(-(NpointsMesh//2), 
                                            NpointsMesh//2, 
                                            NpointsMesh), 
                              dtype = tf.float32)
    # we need to define a mesh betwen xLims[0] and xLims[1]
    self.xGrid =  tf.constant(np.linspace(xLims[0], 
                                          xLims[1], 
                                          NpointsMesh+1)[:-1], 
                              dtype = tf.float32)


  def build(self, input_shape):

    print("building the channels")

   
    self.shift = []
    for ii in range(2):
      self.shift.append(self.add_weight("std_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))
    self.amplitud = []
    for ii in range(2):
      self.amplitud.append(self.add_weight("bias_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))
      
#    print(self.amplitud[0].shape)
#    print(self.shift[0].shape)
#  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    # this needs to be perodic distance!!!
    # input (batch_size, Np*Ncells)
    Npoints = input.shape[-1]
    batch_size = input.shape[0]
    diff = tf.expand_dims(input, -1) - tf.reshape(self.xGrid, (1,1, self.NpointsMesh))
    # (batch_size, Np*Ncells, NpointsMesh)
    array_gaussian = gaussianPer(diff, self.tau, self.L)
     # (batch_size, Np*Ncells, NpointsMesh)
    array_Gaussian_complex = tf.complex(array_gaussian, 0.0)
    #  (batch_size, Np*Ncells, NpointsMesh)


    fftGauss = tf.signal.fftshift(tf.signal.fft(array_Gaussian_complex),axes=-1)
    ###Note: tf.signal.fftshift will shift all axes in default and we can change the axes of the shift
    ### tf.signal.fft will do fft over the inner-most dimension of input but we cannot change it
    ### the shape here is required to be checked carefully
    
    # (batch_size, Np*Ncells, NpointsMesh)
    Deconv = tf.complex(tf.expand_dims(tf.expand_dims(gaussianDeconv(self.kGrid, self.tau), 0),0),0.0)
    #(1, 1, NpointsMesh)

    rfft = tf.multiply(fftGauss, Deconv)
    #(batch_size, Np*Ncells,NpointsMesh)

    Rerfft = tf.math.real(rfft)
    Imrfft = tf.math.imag(rfft)

    #print("applying the multipliers")
    multiplier1 = tf.expand_dims(tf.expand_dims(self.amplitud[0]*4*np.pi*\
                                tf.math.reciprocal( tf.square(self.kGrid) + \
                                tf.square(self.mu1*self.shift[0])), 0),0)
    multiplierRe1 = tf.math.real(multiplier1)    
    multReRefft = tf.multiply(multiplierRe1,Rerfft)
    multImRefft = tf.multiply(multiplierRe1,Imrfft)
    multfft = tf.complex(multReRefft,multImRefft)

    multiplier2 = tf.expand_dims(tf.expand_dims(self.amplitud[1]*4*np.pi*\
                                tf.math.reciprocal( tf.square(self.kGrid) + \
                                tf.square(self.mu2*self.shift[1])), 0),0)
    multiplierRe2 = tf.math.real(multiplier2)   
    multReRefft2 = tf.multiply(multiplierRe2,Rerfft)
    multImRefft2 = tf.multiply(multiplierRe2,Imrfft)
    multfft2 = tf.complex(multReRefft2, multImRefft2)


    multfftDeconv1 = tf.multiply(multfft, Deconv)
    multfftDeconv2 = tf.multiply(multfft2, Deconv)
#    irfft = tf.math.real(tf.signal.ifft(tf.signal.ifftshift(multfftDeconv,axes=-1)))/(2*np.pi*self.NpointsMesh/self.L)/(2*np.pi)
    irfft1 = tf.math.real(tf.signal.ifft(tf.signal.ifftshift(multfftDeconv1,axes=-1)))/(2*np.pi*self.NpointsMesh/self.L)/(2*np.pi)
    irfft2 = tf.math.real(tf.signal.ifft(tf.signal.ifftshift(multfftDeconv2,axes=-1)))/(2*np.pi*self.NpointsMesh/self.L)/(2*np.pi)
    ##(batch_size, Np*Ncells, NpointsMesh)

    diag_sum1 = tf.reduce_sum(irfft1*array_gaussian,axis=-1)
    ##(batch_size,Np*Ncells) part energy
    local1 = tf.reduce_sum(tf.reduce_sum(irfft1,axis=1,keepdims=True)*array_gaussian,axis=-1)
    ##(batch_size,Np*Ncells) 
    energy1 = local1 - diag_sum1
    diag_sum2 = tf.reduce_sum(irfft2*array_gaussian,axis=-1)
    ##(batch_size,Np*Ncells) part energy
    local2 = tf.reduce_sum(tf.reduce_sum(irfft2,axis=1,keepdims=True)*array_gaussian,axis=-1)
    ##(batch_size,Np*Ncells) 
    energy2 = local2 - diag_sum2
    

    energy = tf.concat([tf.expand_dims(energy1,axis=-1),tf.expand_dims(energy2,axis=-1)],axis=-1)
    ##(batch_size, Np*Ncells,2)
#    print('energy',energy.shape)
    return energy


import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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
mu1 = data["mu1"]                          # the parameter mu of the potential
mu2 = data["mu2"]
weight1 = data["weight1"]                          # the parameter mu of the potential
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
loadFile = data["loadFile"]
Nepochs = data["numberEpoch"]

# the ones not used yet
potentialType = data["potentialType"]

# for now only periodic data
assert potentialType == "Periodic"
assert mu1 >= mu2
assert weight1 + weight2 == 1.0 
# we will save them for now
# we need to extract them from the kson file
NpointsFourier = 1001
fftChannels = 2
sigmaFFT = 0.0001
print('I have not adjusted xLims')
L = Lcell*Ncells
xLims = [0.0, L]


print("We are using the random seed %d"%(seed))
tf.random.set_seed(seed)

dataFile = dataFolder + "data_"+ potentialType + \
                        "_Ncells_" + str(Ncells) + \
                        "_Np_" + str(Np) + \
                        "_mu1_" + str(mu1) + \
                        "_mu2_" + str(mu2) + \
                        "_weight1_" + str(weight1) + \
                        "_weight2_" + str(weight2) + \
                        "_minDelta_%.4f"%(minDelta) + \
                        "_Nsamples_" + str(Nsamples) + ".h5"

checkFolder  = "/checkpointsNUFFTchange/"
checkFile = checkFolder + "checkpoint_NUFFTchange_" + \
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

# if the file doesn't exist we create it
if not path.exists(dataFile):
  # TODO: encapsulate all this in a function
  print("Data file does not exist, we create a new one")

  pointsArray, \
  potentialArray, \
  forcesArray  = genDataYukawaPerMixed(Ncells, Np, mu1,mu2,Nsamples, minDelta, Lcell, weight1,weight2)
  
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



# positions of the 
Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32)
Rin = Rinput
#compute the statistics of the inputs in order to rescale 
#the descriptor computation 
L = Lcell*Ncells

if Lcell == 0.25 and Np ==1 and minDelta == 0.1:
    leftindex = -5
    rightindex = 5
    print('the neghbor uses points within the radious 1.5')
else:
    print('no parameter matching it')
    print('no parameter matching it')
    exit()
    
genCoordinates = genDistInvPeradjusted(Rinput, Ncells, Np, L, leftindex,rightindex)


av = tf.reduce_mean(genCoordinates, 
                      axis = 0, 
                      keepdims =True ).numpy()[0]
std = tf.sqrt(tf.reduce_mean(tf.square(genCoordinates - av), 
                               axis = 0, 
                               keepdims=True)).numpy()[0]

print("mean of the inputs are %.8f and %.8f"%(av[0], av[1]))
print("std of the inputs are %.8f and %.8f"%(std[0], std[1]))

pointsTest, \
potentialTest, \
forcesTest  = genDataYukawaPerMixed(Ncells, Np, mu1,mu2,500, minDelta, Lcell, weight1,weight2)


class DeepMDsimpleForces(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               Np, 
               Ncells,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               mu1 = 1.0,
               mu2 = 1.0,
               leftindex = -1,
               rightindex = 2,
               av = [0.0, 0.0],
               std = [1.0, 1.0],
#               avNUFFT = [0.0,0.0],
#               stdNUFFT = [1.0,1.0],
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
#    self.avNUFFT = avNUFFT
#    self.stdNUFFT = stdNUFFT 
    self.leftindex = leftindex
    self.rightindex = rightindex
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.mu1 = mu1
    self.mu2 = mu2
    self.L = np.abs(xLims[1]-xLims[0])
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]

    self.NpointsFourier = NpointsFourier
    self.fftChannels    = fftChannels 
    self.sigmaFFT = sigmaFFT
    # we may need to use the tanh here
    self.layerPyramid   = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)
    self.layerPyramidInv  = pyramidLayer(descripDim, 
                                       actfn = tf.nn.tanh)

    # layer to apply the fmm (this is already windowed)
    self.NUFFTLayerMultiChannelInitMixed = NUFFTLayerMultiChannelInitMixed(fftChannels, \
      NpointsFourier, sigmaFFT, xLims, mu1,mu2)#

    self.layerPyramidLongRange  = pyramidLayer(descripDim, 
                                       actfn = tf.nn.relu)
    
    # we may need to use the tanh especially here
    self.fittingNetwork = pyramidLayer(fittingDim, 
                                       actfn = tf.nn.tanh)
    self.linfitNet      = MyDenseLayerNoBias(1)    

#  @tf.function
  def call(self, inputs):

    with tf.GradientTape() as tape:
      # we watch the inputs 

      tape.watch(inputs)
      # (Nsamples, Ncells*Np)
      # in this case we are only considering the distances
      genCoordinates = genDistInvPeradjusted(inputs, self.Ncells, self.Np,  self.L,self.leftindex,self.rightindex,
                                  self.av, self.std)

      # (Nsamples*Ncells*Np*(3*Np - 1), 2)
      L1   = self.layerPyramid(genCoordinates[:,1:])*genCoordinates[:,1:]
      # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
      L2   = self.layerPyramidInv(genCoordinates[:,0:1])*genCoordinates[:,0:-1]
      # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)

      # we compute the FMM and the normalize by the number of particules
      # here we are harcoding the normalization
      longRangewCoord = (self.NUFFTLayerMultiChannelInitMixed(inputs))# - self.avNUFFT)/ self.stdNUFFT
      # # (Nsamples, Ncells*Np, 2) # we are only using 4 kernels
      # we normalize the output of the fmm layer before feeding them to network
      longRangewCoord2 = tf.reshape(longRangewCoord, (-1, self.fftChannels))
      # (Nsamples*Ncells*Np, 1)
      L3   = self.layerPyramidLongRange(longRangewCoord2)
      # (Nsamples*Ncells*Np, descriptorDim)


      # (Nsamples*Ncells*Np*(3*Np - 1), descriptorDim)
      LL = tf.concat([L1, L2], axis = 1)
      # (Nsamples*Ncells*Np*(3*Np - 1), 2*descriptorDim)
      Dtemp = tf.reshape(LL, (-1, (self.rightindex - self.leftindex)*self.Np-1, 
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
                           filterNet, fittingNet, mu1, mu2, leftindex, rightindex, 
                            av, std, #avNUFFT,stdNUFFT,
                            NpointsFourier, fftChannels, sigmaFFT, xLims)

# quick run of the model to check that it is correct.
# we use a small set 
F = model(Rinput[0:10,:])
model.summary()


errorlist = []
losslist = []
# Create checkpointing directory if necessary
if not os.path.exists(checkFolder):
    os.mkdir(checkFolder)
    print("Directory " , checkFolder ,  " Created ")
else:    
    print("Directory " , checkFolder ,  " already exists :)")

## in the case we need to load an older saved model
#if loadFile: 
#  print("Loading the weights the model contained in %s"%(loadFile), flush = True)
#  model.load_weights(loadFile)

## We use a decent training or a custom one if necessary

Nepochs = [200, 400, 800, 1600]
  #batchSizeArray = map(lambda x: x*batchSize, [1, 2, 4, 8]) 
batchSizeArray = [8,16,32,64]  


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
                        x_batch_train[1], 
                        weightF)
      loss_metric(loss)
  
      if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))
    

    forcePred = model(pointsTest)
    err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTest)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))
    print("Relative Error in the forces is " +str(err.numpy()))
    # mean loss saved in the metric
    errorlist.append(err.numpy())        
    with open('error'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist)
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




