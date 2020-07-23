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
from utilities import MyDenseLayerNoBiasNUFFT, pyramidLayer
from utilities import pyramidLayerNoBias#, NUFFTLayerMultiChannelInitMixed
#####################
def train_step(model, optimizer, loss, 
               inputs, outputsE,
               weightE):
# funtion to perform one training step
  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predE= model(inputs, training=True)

    # fidelity loss usin mse
    total_loss = weightE*loss(predE, outputsE)

  # compute the gradients of the total loss with respect to the trainable variables
  gradients = tape.gradient(total_loss, model.trainable_variables)
  # update the parameters of the network
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return total_loss  
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
    # we initialize the channel multipliers
    # we need to add a parametrized family in here

    xExp = tf.expand_dims(tf.expand_dims(4*np.pi*tf.math.reciprocal(tf.square(self.kGrid) + \
                                  tf.square(self.mu1)), 0),0)
    #Im =  tf.zeros([1,1,NpointsMesh])
    initKExp = tf.keras.initializers.Constant(xExp.numpy())
    xExp2 = tf.expand_dims(4*np.pi*tf.math.reciprocal(tf.square(self.kGrid) + \
                                  tf.square(self.mu2)), 0)
    initKExp2 = tf.keras.initializers.Constant(xExp2.numpy())
    self.multipliersRe = []
    self.multipliersIm = []

    self.multipliersRe.append(self.add_weight("multRe_0",
                       initializer=initKExp, shape = (1, 1, self.NpointsMesh)))
    self.multipliersIm.append(self.add_weight("multIm_0",
                       initializer=tf.initializers.zeros(), 
                       shape = (1, 1,self.NpointsMesh)))

    self.multipliersRe.append(self.add_weight("multRe_1",
                       initializer=initKExp2, shape = (1, 1,self.NpointsMesh)))
    self.multipliersIm.append(self.add_weight("multIm_1",
                       initializer=tf.initializers.zeros(), 
                        shape = (1, 1,self.NpointsMesh)))
    # this needs to be properly initialized it, otherwise it won't even be enough
    self.Weight = []
    self.Weight.append(self.add_weight("weight_0",
                       initializer=tf.initializers.constant(0.5), shape = (1,1,1)))
    self.Weight.append(self.add_weight("weight_1",
                       initializer=tf.initializers.constant(0.5), shape = (1,1,1)))
    ###for two channels
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

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft = tf.multiply(self.multipliersRe[0],Rerfft)
    multReImfft = tf.multiply(self.multipliersIm[0],Rerfft)
    multImImfft = tf.multiply(self.multipliersIm[0],Imrfft)
    multImRefft = tf.multiply(self.multipliersRe[0],Imrfft)

    multfft = tf.complex(multReRefft-multImImfft, \
                                        multReImfft+multImRefft)
    ##(batch_size, Np*Ncells, NpointsMesh)
    # an alternative method:   
    #    fft = tf.complex(self.multipliersRe[0],self.multipliersIm[0])
    #    multFFT = tf.multiply(rfft,fft)
    multReRefft2 = tf.multiply(self.multipliersRe[1],Rerfft)
    multReImfft2 = tf.multiply(self.multipliersIm[1],Rerfft)
    multImImfft2 = tf.multiply(self.multipliersIm[1],Imrfft)
    multImRefft2 = tf.multiply(self.multipliersRe[1],Imrfft)

    multfft2 = tf.complex(multReRefft2-multImImfft2, \
                          multReImfft2+multImRefft2)
    multFFT = tf.multiply(tf.complex(self.Weight[0],0.0),multfft)+\
              tf.multiply(tf.complex(self.Weight[1],0.0),multfft2)    
    ### add them together 


    multfftDeconv = tf.multiply(multFFT, Deconv)
    ##(batch_size, Np*Ncells, NpointsMesh)

    irfft = tf.math.real(tf.signal.ifft(tf.signal.ifftshift(multfftDeconv,axes=-1)))/(2*np.pi*self.NpointsMesh/self.L)/(2*np.pi)
    ##(batch_size, Np*Ncells, NpointsMesh)
    ## the factor is right w.r.t. the energy computed in gendata
    diag_sum = tf.reduce_sum(tf.reduce_sum(irfft*array_gaussian,axis=-1),axis=-1)
    ##(batch_size) the diagonal sum of part energy
    local = tf.reduce_sum(tf.reduce_sum(irfft,axis=1)*tf.reduce_sum(array_gaussian,axis=1),axis=-1)
    ##(batch_size) the total sum of part energy
    energy = (local - diag_sum)/2
    ##(batch_size)
    
    return energy



######################
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
xLims = [0.0, 10.0]


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

checkFolder  = "/"
checkFile = checkFolder + "checkpoint_onlynoadjustNUFFTlinear_" + \
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
L = Lcell*Ncells

Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32)
Rin = Rinput[0:1000,:]
#compute the statistics of the inputs in order to rescale 
#the descriptor computation 
L = Lcell*Ncells
pointsTest, \
potentialTest, \
forcesTest  = genDataYukawaPerMixed(Ncells, Np, mu1,mu2,1000, minDelta, Lcell, weight1,weight2)

#forcesTestRscl =  forcesTest- np.mean(forcesTest)
#forcesTestRscl = forcesTestRscl/np.std(forcesTestRscl)

class DeepMDsimpleForces(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               Np, 
               Ncells,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               mu1 = 1.0,
               mu2 = 1.0,
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


    self.mu1 = mu1
    self.mu2 = mu2
    self.L = np.abs(xLims[1]-xLims[0])
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]

    self.NpointsFourier = NpointsFourier
    self.fftChannels    = fftChannels 


    # layer to apply the fmm (this is already windowed)
    self.NUFFTLayerMultiChannelInitMixed = NUFFTLayerMultiChannelInitMixed(fftChannels, \
      NpointsFourier, sigmaFFT, xLims, mu1,mu2)
#    self.linfitNet      = MyDenseLayerNoBiasNUFFT(1)   

#  @tf.function
  def call(self, inputs):

    with tf.GradientTape() as tape:
      # we watch the inputs 

      tape.watch(inputs)
      # (Nsamples, Ncells*Np)
      # in this case we are only considering the distances
      Energy = self.NUFFTLayerMultiChannelInitMixed(inputs)
    Forces = -tape.gradient(Energy, inputs)

    return Forces#,Energy


## Defining the model
model = DeepMDsimpleForces(Np, Ncells, 
                           filterNet, fittingNet, mu1, mu2, 
                            NpointsFourier, fftChannels, sigmaFFT, xLims)

# quick run of the model to check that it is correct.
# we use a small set 
F= model(Rinput[0:10,:])
Energy = potentialArray[0:10,:]

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
if loadFile: 
  print("Loading the weights the model contained in %s"%(loadFile), flush = True)
  model.load_weights(loadFile)

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
    

    forcePred= model(pointsTest)
    err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTest)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))
#    err_energy = tf.sqrt(tf.reduce_sum(tf.square(tf.squeeze(energyPred) - np.squeeze(potentialTest))))/tf.sqrt(tf.reduce_sum(tf.square(energyPred)))
    print("Relative Error in the forces is " +str(err.numpy()))
#    print("Relative Error in the energy is " +str(err_energy.numpy()))
#    if epoch % 20 == 0:
#        for layer in model.layers:
#            if layer.name == 'nufft_layer_multi_channel_init_mixed':
#                A = layer.variables
#        NpointsMeshplot = NpointsFourier 
#        kGridplot = tf.constant((2*np.pi/(L))*np.linspace(-(NpointsMeshplot//2), 
#                                                    NpointsMeshplot//2, 
#                                                    NpointsMeshplot), dtype = tf.float32)
#        plt.subplot(1,2,1)
#        plt.scatter(kGridplot,tf.squeeze(A[0],axis=0),s=10)
#        plt.title('real')
#        plt.subplot(1,2,2)
#        plt.scatter(kGridplot,tf.squeeze(A[1],axis=0),s=10)
#        plt.title('imag')
#        fntmp = ('%sNUFFT_cycle_'+str(cycle)+'epoch_'+str(epoch))%(FolderName)
#        fnm='%s.png'%(fntmp)
#        plt.savefig(fnm)
#        plt.close()


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




