import tensorflow as tf
import numpy as np 

# file with a few helper functions

# relative L2 misfit
@tf.function
def relMSElossfn(predF, outputsF):
  return tf.reduce_mean(tf.square(tf.subtract(predF,outputsF)))/tf.reduce_mean(tf.square(outputsF))


@tf.function
def genCoord(Rin, Ncells, Np):
    # function to generate the generalized coordinates
    # the input has dimensions (Nsamples, Ncells*Np)
    # the ouput has dimensions (Nsamples*Ncells*Np*(3*Np-1), 2)

    # add assert with respect to the input shape 
    Nsamples = Rin.shape[0]
    R = tf.reshape(Rin, (Nsamples, Ncells, Np))

    Diff_Array_Per = []
    # we compute the difference between points in the same
    # interaction list
    for i in range(Ncells):
        for j in range(Np):
            Diff_Array = []
            Dist_Array = []
            for l in range(-1,2):
                for k in range(Np):
                    if j != k or l != 0:
                        if i+l < 0 or i+l > Ncells-1:
                            Diff_Array.append(tf.expand_dims(tf.zeros_like(R[:,i,j]),-1))
                        else :
                            Diff_Array.append(tf.expand_dims(tf.subtract(R[:,i+l,k], R[:,i,j]),-1))
            Diff_Array_Per.append(tf.expand_dims(tf.concat(Diff_Array, axis = 1), 1))

    R_Diff_list = tf.concat(Diff_Array_Per, axis = 1)

    # computing the norm and direction
    R_Diff_abs = tf.abs(R_Diff_list)
    # or using the reciprocal 
    #R_Diff_abs = tf.math.reciprocal(tf.abs(R_Diff_list) + 0.00001)

    R_Diff_sign = tf.sign(R_Diff_list)

    R_Diff_total = tf.concat([tf.reshape(R_Diff_sign, (-1,1)), 
                              tf.reshape(R_Diff_abs, (-1,1))], axis = 1)

    # asserting the final size of the tensor
    assert R_Diff_total.shape[0] == Nsamples*Ncells*Np*(3*Np-1)

    return R_Diff_total

@tf.function
def genDistInv(Rin, Ncells, Np, av = [0.0, 0.0], std = [1.0, 1.0]):
    # function to generate the generalized coordinates
    # the input has dimensions (Nsamples, Ncells*Np)
    # the ouput has dimensions (Nsamples*Ncells*Np*(3*Np-1), 2)

    # add assert with respect to the input shape 
    Nsamples = Rin.shape[0]
    R = tf.reshape(Rin, (Nsamples, Ncells, Np))

    Abs_Array = []
    Abs_Inv_Array = []
    # we compute the difference between points in the same
    # interaction list
    for i in range(Ncells):
        for j in range(Np):
            absDistArray = []
            absInvArray = []
            for l in range(-1,2):
                for k in range(Np):
                    if j != k or l != 0:
                        if i+l < 0 or i+l > Ncells-1:
                            absDistArray.append(tf.expand_dims(
                                                tf.zeros_like(R[:,i,j]),-1))
                            absInvArray.append(tf.expand_dims(
                                               tf.zeros_like(R[:,i,j]),-1))
                        else :
                            absDistArray.append((tf.abs(tf.expand_dims(
                                                tf.subtract(R[:,i+l,k], R[:,i,j]),-1))-av[1])/std[1] )
                            absInvArray.append((tf.abs(tf.math.reciprocal(
                                               tf.expand_dims(
                                               tf.subtract(R[:,i+l,k], R[:,i,j]),-1))) -av[0])/std[0])
            Abs_Array.append(tf.expand_dims(
                             tf.concat(absDistArray, axis = 1), 1))
            Abs_Inv_Array.append(tf.expand_dims(
                                tf.concat(absInvArray, axis = 1), 1))

    # concatenating the lists of tensors to a large tensor
    absDistList = tf.concat(Abs_Array, axis = 1)
    absInvList = tf.concat(Abs_Inv_Array, axis = 1)


    # R_Diff_abs = absDistList)-
    # R_Diff_inv = (tf.abs(absInvList)-av[0])/std[0]
    # or using the reciprocal 
    #R_Diff_abs = tf.math.reciprocal(tf.abs(absDistList) + 0.00001)

    R_Diff_total = tf.concat([tf.reshape(absInvList, (-1,1)), 
                              tf.reshape(absDistList, (-1,1))], axis = 1)

    # asserting the final size of the tensor
    assert R_Diff_total.shape[0] == Nsamples*Ncells*Np*(3*Np-1)

    return R_Diff_total



@tf.function
def genDistInvLongRange(Rin, Ncells, Np, av = [0.0, 0.0], std = [1.0, 1.0]):
    # function to generate the generalized coordinates
    # the input has dimensions (Nsamples, Ncells*Np)
    # the ouput has dimensions (Nsamples*Ncells*Np*(3*Np-1), 2)

    # add assert with respect to the input shape 
    Nsamples = Rin.shape[0]
    R = tf.reshape(Rin, (Nsamples, Ncells*Np))

    Abs_Array = []
    Abs_Inv_Array = []
    # we compute the difference between points in the same
    # interaction list
    for i in range(Ncells*Np):
      absInvArray = []
      for j in range(Ncells*Np):
        if i != j :
                # absDistArray.append((tf.abs(tf.expand_dims(
                #                      tf.subtract(R[:,i+l,k], R[:,i,j]),-1))-av[1])/std[1] )
          absInvArray.append((tf.abs(tf.math.reciprocal(
                            tf.expand_dims(
                            tf.subtract(R[:,j], R[:,i]),-1))) - av[0])/std[0])
      
      Abs_Inv_Array.append(tf.expand_dims(
                                tf.concat(absInvArray, axis = 1), 1))


    # concatenating the lists of tensors to a large tensor
    absInvList = tf.reduce_sum(tf.concat(Abs_Inv_Array, axis = 1), axis = 2)

    return absInvList

@tf.function
def genDistLongRangeFull(Rin, Ncells, Np, av = [0.0, 0.0], std = [1.0, 1.0]):
    # function to generate the generalized coordinates
    # the input has dimensions (Nsamples, Ncells*Np)
    # the ouput has dimensions (Nsamples*Ncells*Np*(3*Np-1), 2)

    # add assert with respect to the input shape 
    Nsamples = Rin.shape[0]
    R = tf.reshape(Rin, (Nsamples, Ncells*Np))

    absArrayGlobal = []
    # we compute the difference between points in the same
    # interaction list
    for i in range(Ncells*Np):
      absArray = []
      for j in range(Ncells*Np):
        if i != j :
                # absDistArray.append((tf.abs(tf.expand_dims(
                #                      tf.subtract(R[:,i+l,k], R[:,i,j]),-1))-av[1])/std[1] )
          absArray.append((tf.abs(tf.expand_dims(
                            tf.subtract(R[:,j], R[:,i]),-1))-av[0])/std[0])
      
      absArrayGlobal.append(tf.expand_dims(
                                tf.concat(absArray, axis = 1), 1))


    # concatenating the lists of tensors to a large tensor 
    # we don't perform a reduction in the coordinates
    absInvList = tf.concat(absArrayGlobal, axis = 1)
    # (batchSize, Ncells*Np, Ncells*Np -1 ) 

    return absInvList

@tf.function
def genDistInvLongRangeWindow(Rin, Ncells, Np, widht_tran = 0.5, width = 3.0, av = [0.0, 0.0], std = [1.0, 1.0]):
    # function to generate the generalized coordinates
    # the input has dimensions (Nsamples, Ncells*Np)
    # the ouput has dimensions (Nsamples*Ncells*Np*(3*Np-1), 2)

    # add assert with respect to the input shape 
    Nsamples = Rin.shape[0]
    R = tf.reshape(Rin, (Nsamples, Ncells*Np))

    Abs_Array = []
    Abs_Inv_Array = []
    # we compute the difference between points in the same
    # interaction list
    for i in range(Ncells*Np):
      absInvArray = []
      for j in range(Ncells*Np):
        if i != j :
                # absDistArray.append((tf.abs(tf.expand_dims(
                #                      tf.subtract(R[:,i+l,k], R[:,i,j]),-1))-av[1])/std[1] )
          A = 1 - 0.5*tf.math.erfc(tf.abs(tf.expand_dims(tf.subtract(R[:,j], R[:,i]),-1))/widht_tran-width)
          absInvArray.append(tf.multiply(A, (tf.abs(tf.math.reciprocal(
                                            tf.expand_dims(
                                            tf.subtract(R[:,j], 
                                                        R[:,i]),-1)))-av[0])/std[0]))
      
      Abs_Inv_Array.append(tf.expand_dims(
                                tf.concat(absInvArray, axis = 1), 1))


    # concatenating the lists of tensors to a large tensor
    absInvList = tf.reduce_sum(tf.concat(Abs_Inv_Array, axis = 1), axis = 2)

    return absInvList

@tf.function
def genDist(Rin, Ncells, Np):
    # function to generate the generalized coordinates only with the distance
    # the input has dimensions (Nsamples, Ncells*Np)
    # the ouput has dimensions (Nsamples*Ncells*Np*(3*Np-1), 2)

    # add assert with respect to the input shape 
    Nsamples = Rin.shape[0]
    R = tf.reshape(Rin, (Nsamples, Ncells, Np))

    Diff_Array_Per = []
    # we compute the difference between points in the same
    # interaction list
    for i in range(Ncells):
        for j in range(Np):
            Diff_Array = []
            Dist_Array = []
            for l in range(-1,2):
                for k in range(Np):
                    if j != k or l != 0:
                        if i+l < 0 or i+l > Ncells-1:
                            Diff_Array.append(tf.expand_dims(tf.zeros_like(R[:,i,j]),-1))
                        else :
                            Diff_Array.append(tf.expand_dims(tf.subtract(R[:,i+l,k], R[:,i,j]),-1))
            Diff_Array_Per.append(tf.expand_dims(tf.concat(Diff_Array, axis = 1), 1))

    R_Diff_list = tf.concat(Diff_Array_Per, axis = 1)

    # computing the norm and direction
    R_Diff_abs = tf.abs(R_Diff_list)
    # or using the reciprocal 
    #R_Diff_abs = tf.math.reciprocal(tf.abs(R_Diff_list) + 0.00001)

    R_Diff_total = tf.reshape(R_Diff_abs, (-1,1))

    # asserting the final size of the tensor
    assert R_Diff_total.shape[0] == Nsamples*Ncells*Np*(3*Np-1)

    return R_Diff_total



class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  initializer=tf.initializers.GlorotNormal(),
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    self.bias = self.add_weight("bias",
                                initializer=tf.initializers.zeros(),    
                                shape=[self.num_outputs,])
  @tf.function
  def call(self, input):
    return tf.matmul(input, self.kernel) + self.bias

# pyramid layer with bias 
class pyramidLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, actfn = tf.nn.relu):
    super(pyramidLayer, self).__init__()
    self.num_outputs = num_outputs
    self.actfn = actfn

  def build(self, input_shape):
    self.kernel = []
    self.bias = []
    self.kernel.append(self.add_weight("kernel",
                       initializer=tf.initializers.GlorotNormal(),
                       shape=[int(input_shape[-1]),
                              self.num_outputs[0]]))
    self.bias.append(self.add_weight("bias",
                       initializer=tf.initializers.GlorotNormal(),
                       shape=[self.num_outputs[0],]))

    for n, (l,k) in enumerate(zip(self.num_outputs[0:-1], \
                                  self.num_outputs[1:])) :

      self.kernel.append(self.add_weight("kernel"+str(n),
                         shape=[l, k]))
      self.bias.append(self.add_weight("bias"+str(n),
                         shape=[k,]))

  # @tf.function
  # def call(self, input):
  #   x = self.actfn(tf.matmul(input, self.kernel[0]) + self.bias[0])
  #   for ker, b in zip(self.kernel[1:], self.bias[1:]):
  #     x = self.actfn(tf.matmul(x, ker) + b)
  #   return x

  @tf.function
  def call(self, input):
    x = self.actfn(tf.matmul(input, self.kernel[0]) + self.bias[0])
    for k, (ker, b) in enumerate(zip(self.kernel[1:], self.bias[1:])):
      if self.num_outputs[k] == self.num_outputs[k+1]:
        x += self.actfn(tf.matmul(x, ker) + b)
      else :
        x = self.actfn(tf.matmul(x, ker) + b)
    return x


class pyramidLayerNoBias(tf.keras.layers.Layer):
  def __init__(self, num_outputs, actfn = tf.nn.tanh):
    super(pyramidLayerNoBias, self).__init__()
    self.num_outputs = num_outputs
    self.actfn = actfn

  def build(self, input_shape):
    self.kernel = []
    self.kernel.append(self.add_weight("kernel",
                       initializer=tf.initializers.GlorotNormal(),
                       shape=[int(input_shape[-1]),
                              self.num_outputs[0]]))

    for n, (l,k) in enumerate(zip(self.num_outputs[0:-1], \
                                  self.num_outputs[1:])) :

      self.kernel.append(self.add_weight("kernel"+str(n),
                         shape=[l, k]))

  @tf.function
  def call(self, input):
    x = self.actfn(tf.matmul(input, self.kernel[0]))
    for k, ker in enumerate(self.kernel[1:]):
      # if two layers hav ethe same dimension we use a resnet block
      if self.num_outputs[k] == self.num_outputs[k+1]:
        x += self.actfn(tf.matmul(x, ker))
      else :
        x = self.actfn(tf.matmul(x, ker))
    return x

class fmmLayer(tf.keras.layers.Layer):
  def __init__(self, Ncells, Np, width_tran = 0.5, width = 3):
    super(fmmLayer, self).__init__()
    self.Ncells = Ncells
    self.Np = Np
    self.width_tran = width_tran 
    self.width = width


  def build(self, input_shape):
    self.std = []
    for ii in range(4):
      self.std.append(self.add_weight("std_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))

    self.bias = []
    for ii in range(4):
      self.bias.append(self.add_weight("bias_"+str(ii),
                       initializer=tf.initializers.zeros(),
                       shape=[1,]))


  @tf.function
  def call(self, input):
    ExtCoords =  genDistLongRangeFull(input, self.Ncells, self.Np, 
                                      [0.0, 0.0], [1.0, 1.0]) # this are hard coded

    window = 1-0.5*tf.math.erfc(ExtCoords/self.width_tran-self.width)
    
    kernelApp = []

    kernelApp.append(tf.abs(self.std[0])*(tf.expand_dims(tf.multiply(window,
                     tf.math.reciprocal(ExtCoords)), axis = -1) - self.bias[0]))
    kernelApp.append(tf.abs(self.std[1])*(tf.expand_dims(tf.multiply(window,
                     tf.sqrt(tf.math.reciprocal(ExtCoords))), axis = -1)- self.bias[1]))
    kernelApp.append(tf.abs(self.std[2])*(tf.expand_dims(tf.multiply(window,
                     tf.square(tf.math.reciprocal(ExtCoords))), axis = -1)- self.bias[2]))
    kernelApp.append(tf.abs(self.std[3])*(tf.expand_dims(tf.multiply(window,
                     tf.math.bessel_i0e(ExtCoords)), axis = -1)- self.bias[3]))

    fmm = tf.reduce_sum(tf.concat(kernelApp, axis = -1), axis = 2)


    return fmm 


class expSumLayer(tf.keras.layers.Layer):
  def __init__(self, Ncells, Np, mu = 2, winWidth = 3, winTrans = 0.5):
    super(fmmLayer, self).__init__()
    self.Ncells = Ncells
    self.Np = Np
    self.mu = mu
    # this is the width and the transition of the window
    self.winTrans = winTrans 
    self.winWidth = winWidth


  def build(self, input_shape):
    # we provide some freedom 
    self.std = []
    for ii in range(1):
      self.std.append(self.add_weight("std_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))

    self.bias = []
    for ii in range(1):
      self.bias.append(self.add_weight("bias_"+str(ii),
                       initializer=tf.initializers.zeros(),
                       shape=[1,]))


  @tf.function
  def call(self, input):
    ExtCoords =  genDistLongRangeFull(input, self.Ncells, self.Np, 
                                      [0.0, 0.0], [1.0, 1.0]) # this are hard coded

    window = 1-0.5*tf.math.erfc(ExtCoords/self.winTrans-self.winWidth)
    
    kernelApp = tf.abs(self.std[0])*(tf.multiply(window,
                     tf.math.exp(-self.mu*ExtCoords)) - self.bias[0])

    fmm = tf.reduce_sum(tf.concat(kernelApp, axis = -1), axis = )

    return fmm 


@tf.function
def train_step(model, optimizer, loss, inputs, outputsE):
# funtion to perform one training step
  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predE = model(inputs, training=True)

    # fidelity loss usin mse
    total_loss = loss(predE, outputsE)

  # compute the gradients of the total loss with respect to the trainable variables
  gradients = tape.gradient(total_loss, model.trainable_variables)
  # update the parameters of the network
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss

@tf.function
def train_step_2(model, optimizer, loss,
               inputs, outputsE, outputsF, 
               weightE, weightF):
# funtion to perform one training step when predicting both the
# potential and the forces
  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predE, predF = model(inputs, training=True)

    # fidelity loss usin mse
    lossE = loss(predE, outputsE)
    lossF = loss(predF, outputsF)/outputsF.shape[-1]

    if weightF > 0.0:
      total_loss = weightE*lossE + weightF*lossF
    else: 
      total_loss = weightE*lossE 

  # compute the gradients of the total loss with respect to the trainable variables
  gradients = tape.gradient(total_loss, model.trainable_variables)
  # update the parameters of the network
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss

def computeWeight(weightInit, weightLimit, lrT, lr0):
  return np.abs(weightLimit*(1-lrT/lr0) + weightInit*(lrT/lr0))


def computeNumStairs(Nepochs, batchSizeArray, Nsamples, epochsPerStair):
  decay_steps = np.round((Nsamples/batchSizeArray[0])*epochsPerStair)
  total_steps = 0
  for (epochs, batchSize) in zip(Nepochs, batchSizeArray):
    total_steps += epochs*(Nsamples/batchSize)

  return np.round(total_steps/decay_steps)

def computeLimitWeights(weights, decay, numStairs):
  limitWeight = weights[1]/(1.0 - decay**numStairs)
  return weights[0], limitWeight
