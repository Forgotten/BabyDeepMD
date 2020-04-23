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
def genDistInvPer(Rin, Ncells, Np, L, av = [0.0, 0.0], std = [1.0, 1.0]):
    # function to generate the generalized coordinates for periodic data
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
                        a = tf.subtract(R[:,(i+l)%Ncells,k], R[:,i,j])
                        b = a - L*tf.round(a/L)
                        absDistArray.append((tf.abs(tf.expand_dims(b,-1)) -av[1])/std[1] )
                        absInvArray.append((tf.abs(tf.math.reciprocal(
                                               tf.expand_dims(b,-1))) -av[0])/std[0])
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
def genDistLongRangeFull(Rin, Ncells, Np, \
                         av  = [0.0, 0.0], \
                         std = [1.0, 1.0]):
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
def genDistInvPerLongRange(Rin, Ncells, Np, L,\
                           av = [0.0, 0.0], \
                           std = [1.0, 1.0]):
    # function to generate the generalized coordinates
    # here we use the periodic distance
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
          a = tf.subtract(R[:,j], R[:,i]) 
          b = a - L*tf.round(a/L)
          absInvArray.append((tf.abs(tf.math.reciprocal(
                            tf.expand_dims(b,-1)))-av[0])/std[0])
      
      Abs_Inv_Array.append(tf.expand_dims(
                                tf.concat(absInvArray, axis = 1), 1))


    # concatenating the lists of tensors to a large tensor
    absInvList = tf.reduce_sum(tf.concat(Abs_Inv_Array, axis = 1), axis = 2)

    return absInvList

@tf.function
def genDistPerLongRangeFull(Rin, Ncells, Np, L, \
                            av  = [0.0, 0.0], \
                            std = [1.0, 1.0]):
    # function to generate the generalized coordinates
    # in this case we use the periodic distance
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
          a = tf.subtract(R[:,j], R[:,i]) 
          b = a - L*tf.round(a/L)
          absArray.append((tf.abs(tf.expand_dims(b,-1))-av[0])/std[0])
      
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
  # this layers uses a few kernels to approximate exp(-mu)
  def __init__(self, Ncells, Np):
    super(fmmLayer, self).__init__()
    self.Ncells = Ncells
    self.Np = Np


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
    
    kernelApp = []

    kernelApp.append(tf.abs(self.std[0])*(
                     tf.expand_dims(tf.math.reciprocal(ExtCoords), axis = -1) - self.bias[0]))
    kernelApp.append(tf.abs(self.std[1])*(
                     tf.expand_dims(tf.sqrt(tf.math.reciprocal(ExtCoords)), axis = -1)- self.bias[1]))
    kernelApp.append(tf.abs(self.std[2])*(
                     tf.expand_dims(tf.square(tf.math.reciprocal(ExtCoords)), axis = -1)- self.bias[2]))
    kernelApp.append(tf.abs(self.std[3])*(
                     tf.expand_dims(tf.math.bessel_i0e(ExtCoords), axis = -1)- self.bias[3]))

    fmm = tf.reduce_sum(tf.concat(kernelApp, axis = -1), axis = 2)

    return fmm 


class fmmLayerExact(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, Ncells, Np, mu):
    super(fmmLayerExact, self).__init__()
    self.Ncells = Ncells
    self.Np = Np 
    self.mu = mu

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
    
    kernelApp = []

    kernelApp.append(tf.abs(self.std[0])*(
                     tf.expand_dims(tf.math.reciprocal(ExtCoords), axis = -1) - self.bias[0]))
    kernelApp.append(tf.abs(self.std[1])*(
                     tf.expand_dims(tf.sqrt(tf.math.reciprocal(ExtCoords)), axis = -1)- self.bias[1]))
    kernelApp.append(tf.abs(self.std[2])*(
                     tf.expand_dims(tf.square(tf.math.reciprocal(ExtCoords)), axis = -1)- self.bias[2]))
    kernelApp.append(tf.abs(self.std[3])*(
                     tf.expand_dims(tf.math.exp(-self.mu*ExtCoords), axis = -1)- self.bias[3]))

    fmm = tf.reduce_sum(tf.concat(kernelApp, axis = -1), axis = 2)

    return fmm 

class fmmLayerExpTrainable(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # we add a wrong mu, but we let it evolve freely.
  def __init__(self, Ncells, Np):
    super(fmmLayerExpTrainable, self).__init__()
    self.Ncells = Ncells
    self.Np = Np 

  def build(self, input_shape):

    # we let the mu to be trained

    self.mu = self.add_weight("mu",
                       initializer=tf.initializers.ones(),
                       shape=[1,])

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
    
    kernelApp = []

    kernelApp.append(tf.abs(self.std[0])*(
                     tf.expand_dims(tf.math.reciprocal(ExtCoords), axis = -1) - self.bias[0]))
    kernelApp.append(tf.abs(self.std[1])*(
                     tf.expand_dims(tf.sqrt(tf.math.reciprocal(ExtCoords)), axis = -1)- self.bias[1]))
    kernelApp.append(tf.abs(self.std[2])*(
                     tf.expand_dims(tf.square(tf.math.reciprocal(ExtCoords)), axis = -1)- self.bias[2]))
    kernelApp.append(tf.abs(self.std[3])*(
                     tf.expand_dims(tf.math.exp(-self.mu*ExtCoords), axis = -1)- self.bias[3]))

    fmm = tf.reduce_sum(tf.concat(kernelApp, axis = -1), axis = 2)

    return fmm 


class fmmLayerExpTrainablePer(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # we add a wrong mu, but we let it evolve freely.
  # in this case we use the periodic distance
  def __init__(self, Ncells, Np, L):
    super(fmmLayerExpTrainablePer, self).__init__()
    # number of cells,
    self.Ncells = Ncells
    # number of points per cell
    self.Np = Np 
    # lenght of the domain 
    self.L = L

  def build(self, input_shape):

    # we let the mu to be trained

    self.mu = self.add_weight("mu",
                       initializer=tf.initializers.ones(),
                       shape=[1,])

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
    ExtCoords =  genDistPerLongRangeFull(input, self.Ncells, 
                                         self.Np, self.L,
                                         [0.0, 0.0], 
                                         [1.0, 1.0]) # this are hard coded
    
    kernelApp = []

    kernelApp.append(tf.abs(self.std[0])*(
                     tf.expand_dims(tf.math.reciprocal(ExtCoords), axis = -1) - self.bias[0]))
    kernelApp.append(tf.abs(self.std[1])*(
                     tf.expand_dims(tf.sqrt(tf.math.reciprocal(ExtCoords)), axis = -1)- self.bias[1]))
    kernelApp.append(tf.abs(self.std[2])*(
                     tf.expand_dims(tf.square(tf.math.reciprocal(ExtCoords)), axis = -1)- self.bias[2]))
    kernelApp.append(tf.abs(self.std[3])*(
                     tf.expand_dims(tf.math.exp(-self.mu*ExtCoords), axis = -1)- self.bias[3]))

    fmm = tf.reduce_sum(tf.concat(kernelApp, axis = -1), axis = 2)

    return fmm 

class fmmLayerWindow(tf.keras.layers.Layer):
  # in this case we window the close and long range using a window
  # we need a better window
  def __init__(self, Ncells, Np, width_tran = 0.5, width = 3):
    super(fmmLayerWindow, self).__init__()
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


class expSumLayerWindow(tf.keras.layers.Layer):
  def __init__(self, Ncells, Np, mu = 2, winWidth = 3, winTrans = 0.5):
    super(expSumLayerWindow, self).__init__()
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

    fmm = tf.reduce_sum(kernelApp, axis =2)

    return fmm 


class expSumLayer(tf.keras.layers.Layer):
  def __init__(self, Ncells, Np, mu = 2):
    super(expSumLayer, self).__init__()
    self.Ncells = Ncells
    self.Np = Np
    self.mu = mu

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
    
    kernelApp = tf.abs(self.std[0])*(tf.math.exp(-self.mu*ExtCoords)-self.bias[0])

    fmm = tf.reduce_sum(kernelApp, axis = 2)

    return fmm 


@tf.function
def train_step(model, optimizer, loss, 
               inputs, outputsE,
               weightE):
# funtion to perform one training step
  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predE = model(inputs, training=True)

    # fidelity loss usin mse
    total_loss = weightE*loss(predE, outputsE)

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

@tf.function
def trainStepList(model, optimizer, loss,
                    inputs, neighList, outputsE, outputsF, 
                    weightE, weightF):
# funtion to perform one training step when predicting both the
# potential and the forces
  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predE, predF = model(inputs, neighList, training=True)

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

def computePairWiseDist(points, Nsamples):
  tensorPoints = np.reshape(points, (Nsamples, -1, 1))
  Pairwise = tensorPoints - tensorPoints.transpose((0,2,1))

  return Pairwise
  

class FFTLayer(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, nChannels, NpointsMesh, sigma, xLims):
    super(FFTLayer, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 
    self.sigma = sigma  # the size of the mollifications
    self.xLims = xLims
    # we need to define a mesh (to be reshaped)
    self.mesh = tf.constant(np.linspace(xLims[0], xLims[1], NpointsMesh+1)[:-1], 
                dtype = tf.float32)
    self.h = (xLims[1]- xLims[0] +0.0)/NpointsMesh

  def build(self, input_shape):

    print("building the channels")
    # we initialize the channel multipliers
    # we need to add a parametrized family in here
    self.multChannelsRe= self.add_weight("multipliersRe",
                       initializer=tf.initializers.RandomUniform(),
                       shape=[1,1,self.nChannels,self.NpointsMesh//2+1])

    self.multChannelsIm= self.add_weight("multipliersIm",
                       initializer=tf.initializers.RandomUniform(),
                       shape=[1,1,self.nChannels,self.NpointsMesh//2+1])

    # this needs to be properly initialized it, otherwise it won't even be enough

  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    # this needs to be perodic distance!!!
    # (batch_size, Np*Ncells)
    diff = tf.expand_dims(input, -1) - tf.reshape(self.mesh, (1,1, self.NpointsMesh))
    # (batch_size, Np*Ncells, NpointsMesh)
    # we compute all the localized gaussians
    array_gaussian = gaussian(diff, self.sigma)
    # we add them together
    arrayReducGaussian = tf.reduce_sum(array_gaussian, axis = 1) 
    # (batch_size, NpointsMesh) (we sum the gaussians together)
    # we apply the fft
    # print("computing the FFT")
    rfft = tf.expand_dims(tf.signal.rfft(arrayReducGaussian), 1)
    # Fourier multipliers

    Rerfft = tf.math.real(rfft)
    Imrfft = tf.math.imag(rfft)

    # print("applying the multipliers")
    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft = tf.multiply(self.multChannelsRe,Rerfft)
    multReImfft = tf.multiply(self.multChannelsIm,Rerfft)
    multImImfft = tf.multiply(self.multChannelsIm,Imrfft)
    multImRefft = tf.multiply(self.multChannelsRe,Imrfft)

    multfft = tf.squeeze(tf.complex(multReRefft-multImImfft, \
                                    multReImfft+multImRefft), axis = 0)

    # print(multfft.shape)
    # print("inverse fft")
    irfft = tf.expand_dims(tf.signal.irfft(multfft), 2)

    local= irfft*tf.expand_dims(array_gaussian, 1)
    
    fmm = tf.reduce_sum(local, axis = -1)*self.h
    #mult = 

    return fmm 


@tf.function 
def gaussian(x, sigma):
  return tf.math.reciprocal(tf.sqrt(2*np.pi*sigma))*tf.exp( -0.5*tf.square(x/sigma))



class NUFFTLayer(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, nChannels, NpointsMesh, tau, xLims):
    super(NUFFTLayer, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 
    self.mu0 = tf.constant(mu0, dtype=tf.float32)

    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1

    
    self.xLims = xLims
    self.L = np.abs(xLims[1] - xLims[0])
    if tau == 0:
      self.tau = tf.constant(12*(self.L/(2*np.pi*NpointsMesh))**2, 
                            dtype = tf.float32)# the size of the mollifications
    else: 
      self.tau = tau

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
    self.shift = []
    for ii in range(1):
      self.shift.append(self.add_weight("std_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))

    self.amplitud = []
    ## TODO add initilizer 
    for ii in range(1):
      self.amplitud.append(self.add_weight("bias_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))



    # this needs to be properly initialized it, otherwise it won't even be enough

  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    # this needs to be perodic distance!!!
    # (batch_size, Np*Ncells)
    diff = tf.expand_dims(input, -1) - \
           tf.reshape(self.xGrid, (1,1, self.NpointsMesh))
    # (batch_size, Np*Ncells, NpointsMesh)
    # we compute all the localized gaussians
    array_gaussian = gaussianPer(diff, self.tau, self.L)
    # we add them together
    arrayReducGaussian = tf.complex(tf.reduce_sum(array_gaussian, \
                                                  axis=1), 0.0)
    # (batch_size, NpointsMesh) (we sum the gaussians together)
    # we apply the fft
    print("computing the FFT")

    fftGauss = tf.signal.fftshift(
               tf.signal.fft(arrayReducGaussian))/\
                             (2*np.pi*self.NpointsMesh/self.L)
    #(batch_size, NpointsMesh)
    Deconv = tf.complex(tf.expand_dims(gaussianDeconv(self.kGrid, \
                                                      self.tau), \
                                       0),0.0)
    #(1, NpointsMesh)

    rfft = tf.multiply(fftGauss, Deconv)
    #(batch_size, NpointsMesh)
    # we are only using one channel
    #rfft = tf.expand_dims(rfftDeconv, 1)
    # Fourier multipliers

    Rerfft = tf.math.real(rfft)
    Imrfft = tf.math.imag(rfft)

    print("applying the multipliers")

    multiplier = tf.expand_dims(self.amplitud[0]*4*np.pi*\
                                tf.math.reciprocal( tf.square(self.kGrid) + \
                                tf.square(5*self.shift[0])), 0)
    multiplierRe = tf.math.real(multiplier)
    multiplierIm = tf.math.imag(multiplier)

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft = tf.multiply(multiplierRe,Rerfft)
    multReImfft = tf.multiply(multiplierIm,Rerfft)
    multImImfft = tf.multiply(multiplierIm,Imrfft)
    multImRefft = tf.multiply(multiplierRe,Imrfft)

    multfft = tf.complex(multReRefft-multImImfft, \
                         multReImfft+multImRefft)

    multfftDeconv = tf.multiply(multfft, Deconv)

    print(multfft.shape)
    print("inverse fft")
    irfft = tf.math.real(tf.expand_dims( \
                         tf.signal.ifft( \
                         tf.signal.ifftshift(multfftDeconv)), 1))

    local = irfft*array_gaussian/(2*np.pi*self.NpointsMesh/self.L)
    
    fmm = tf.reduce_sum(local, axis = -1)
    #mult = 

    return fmm 

class NUFFTLayerMultiChannel(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, nChannels, NpointsMesh, tau, xLims):
    super(NUFFTLayerMultiChannel, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 
    assert NpointsMesh % 2 == 1
    self.tau = tau  # the size of the mollifications
    self.xLims = xLims
    # we need the number of points to be odd 
    self.kGrid = tf.constant(np.linspace(-(NpointsMesh//2), NpointsMesh//2, NpointsMesh), 
                dtype = tf.float32)
    self.L = np.abs(xLims[1] - xLims[0])
    # we need to define a mesh betwen 0 and 2pi
    self.xGrid = 2*np.pi*tf.constant(np.linspace(xLims[0], xLims[1], NpointsMesh+1)[:-1], 
                dtype = tf.float32)/(self.L)


  def build(self, input_shape):

    print("building the channels")
    # we initialize the channel multipliers
    # we need to add a parametrized family in here
    self.shift = []
    for ii in range(self.nChannels):
      self.shift.append(self.add_weight("std_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))

    self.amplitud = []
    for ii in range(self.nChannels):
      self.amplitud.append(self.add_weight("bias_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))



    # this needs to be properly initialized it, otherwise it won't even be enough

  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    # this needs to be perodic distance!!!
    # (batch_size, Np*Ncells)
    diff = tf.expand_dims(input*2*np.pi/self.L, -1) - tf.reshape(self.xGrid, (1,1, self.NpointsMesh))
    # (batch_size, Np*Ncells, NpointsMesh)
    # we compute all the localized gaussians
    array_gaussian = gaussian(diff, self.tau)
    # we add them together
    arrayReducGaussian = tf.complex(tf.reduce_sum(array_gaussian, axis = 1), 0.0)
    # (batch_size, NpointsMesh) (we sum the gaussians together)
    # we apply the fft
    print("computing the FFT")

    fftGauss = tf.signal.fftshift(tf.signal.fft(arrayReducGaussian))*self.L/(2*np.pi)
    #(batch_size, NpointsMesh)
    Deconv = tf.complex(tf.expand_dims(gaussianDeconv(self.kGrid, self.tau), 0),0.0)
    #(1, NpointsMesh)

    rfft = tf.multiply(fftGauss, Deconv)
    #(batch_size, NpointsMesh)
    # we are only using one channel
    #rfft = tf.expand_dims(rfftDeconv, 1)
    # Fourier multipliers

    Rerfft = tf.math.real(rfft)
    Imrfft = tf.math.imag(rfft)

    print("applying the multipliers")

 
    multiplier = tf.expand_dims(-self.amplitud[0]*4*np.pi*\
                                  tf.math.reciprocal(tf.square(self.kGrid) + \
                                  tf.square(5*self.shift[0])), 0)
      
    multiplierRe = tf.math.real(multiplier)
    multiplierIm = tf.math.imag(multiplier)

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft = tf.multiply(multiplierRe,Rerfft)
    multReImfft = tf.multiply(multiplierIm,Rerfft)
    multImImfft = tf.multiply(multiplierIm,Imrfft)
    multImRefft = tf.multiply(multiplierRe,Imrfft)

    multfft = tf.expand_dims(tf.complex(multReRefft-multImImfft, \
                                        multReImfft+multImRefft),1)

    multiplier2 = tf.expand_dims(self.amplitud[1]*4*np.pi*\
                                  tf.math.pow(tf.math.reciprocal(tf.square(self.kGrid) + \
                                  tf.square(5*self.shift[1])), 2), 0) 
      
    multiplierRe2 = tf.math.real(multiplier2)
    multiplierIm2 = tf.math.imag(multiplier2)

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft2 = tf.multiply(multiplierRe2,Rerfft)
    multReImfft2 = tf.multiply(multiplierIm2,Rerfft)
    multImImfft2 = tf.multiply(multiplierIm2,Imrfft)
    multImRefft2 = tf.multiply(multiplierRe2,Imrfft)

    multfft2 = tf.expand_dims(tf.complex(multReRefft2-multImImfft2, \
                          multReImfft2+multImRefft2), 1)

    multFFT = tf.concat([multfft, multfft2], axis = 1)


    multfftDeconv = tf.multiply(multFFT, tf.expand_dims(Deconv,1))

    print(multfft.shape)
    print("inverse fft")
    irfft = tf.math.real(tf.expand_dims(tf.signal.ifft(tf.signal.ifftshift(multfftDeconv)), 1))

    local = irfft*tf.expand_dims(array_gaussian, 2)
    
    fmm = tf.reduce_sum(local, axis = -1)/self.NpointsMesh
    #mult = 

    return fmm 

class NUFFTLayerMultiChannelOneSided(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, nChannels, NpointsMesh, sigma, xLims, mu0 = 1.0):
    super(NUFFTLayerMultiChannelOneSided, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 
    self.mu0 = tf.constant(mu0, dtype=tf.float32)

    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1

    
    self.xLims = xLims
    self.L = np.abs(xLims[1] - xLims[0])
    # this one needs to be a one-dimensional numpy array 
    self.sigma = sigma
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

    initSigma = tf.keras.initializers.Constant([self.sigma])

    self.sigmaVar = self.add_weight("sigma",
                       initializer=initSigma, shape = (1,))

    ## TODO: add the correct parametrization 
    print("building the channels")
    # we initialize the channel multipliers
    # we need to add a parametrized family in here
    self.shift = []
    for ii in range(self.nChannels):
      self.shift.append(self.add_weight("std_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))

    self.amplitud = []
    for ii in range(self.nChannels):
      self.amplitud.append(self.add_weight("bias_"+str(ii),
                       initializer=tf.initializers.ones(),
                       shape=[1,]))



    # this needs to be properly initialized it, otherwise it won't even be enough

  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    # this needs to be perodic distance!!!
    # (batch_size, Np*Ncells)
    diff = tf.expand_dims(input, -1) - tf.reshape(self.xGrid, (1,1, self.NpointsMesh))
    # (batch_size, Np*Ncells, NpointsMesh)
    # we compute all the localized gaussians
    array_gaussian = gaussian(diff, self.tau)
    # we add them together
    array_normal   = normalPer(diff, self.sigmaVar, self.L)
    # we add them together
    arrayReducGaussian = tf.complex(tf.reduce_sum(array_normal, axis = 1), 0.0)

    print("computing the FFT") 

    fftGauss = tf.signal.fftshift(tf.signal.fft(arrayReducGaussian))
    #(batch_size, NpointsMesh)
    Deconv = tf.complex(tf.expand_dims(gaussianDeconv(self.kGrid, self.tau), 0),0.0)
    #(1, NpointsMesh)
    #(batch_size, NpointsMesh)
    # we are only using one channel
    #rfft = tf.expand_dims(rfftDeconv, 1)
    # Fourier multipliers

    Rerfft = tf.math.real(fftGauss)
    Imrfft = tf.math.imag(fftGauss)

    print("applying the multipliers")

 
    multiplier = tf.expand_dims(-self.amplitud[0]*4*np.pi*\
                                  tf.math.reciprocal(tf.square(self.kGrid) + \
                                  tf.square(5*self.shift[0])), 0)
      
    multiplierRe = tf.math.real(multiplier)
    multiplierIm = tf.math.imag(multiplier)

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft = tf.multiply(multiplierRe,Rerfft)
    multReImfft = tf.multiply(multiplierIm,Rerfft)
    multImImfft = tf.multiply(multiplierIm,Imrfft)
    multImRefft = tf.multiply(multiplierRe,Imrfft)

    multfft = tf.expand_dims(tf.complex(multReRefft-multImImfft, \
                                        multReImfft+multImRefft),1)

    multiplier2 = tf.expand_dims(self.amplitud[1]*4*np.pi*\
                                  tf.math.pow(tf.math.reciprocal(tf.square(self.kGrid) + \
                                  tf.square(5*self.shift[1])), 2), 0) 
      
    multiplierRe2 = tf.math.real(multiplier2)
    multiplierIm2 = tf.math.imag(multiplier2)

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft2 = tf.multiply(multiplierRe2,Rerfft)
    multReImfft2 = tf.multiply(multiplierIm2,Rerfft)
    multImImfft2 = tf.multiply(multiplierIm2,Imrfft)
    multImRefft2 = tf.multiply(multiplierRe2,Imrfft)

    multfft2 = tf.expand_dims(tf.complex(multReRefft2-multImImfft2, \
                          multReImfft2+multImRefft2), 1)

    multFFT = tf.concat([multfft, multfft2], axis = 1)


    multfftDeconv = tf.multiply(multFFT, tf.expand_dims(Deconv,1))

    print(multfft.shape)
    print("inverse fft")
    irfft = tf.math.real(tf.expand_dims(tf.signal.ifft(tf.signal.ifftshift(multfftDeconv)), 1))

    local = irfft*tf.expand_dims(array_gaussian, 2)
    
    fmm = tf.reduce_sum(local, axis = -1)/self.NpointsMesh
    #mult = 

    return fmm 


class NUFFTLayerMultiChannelInit(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, nChannels, NpointsMesh, tau, xLims, mu0 = 1.0):
    super(NUFFTLayerMultiChannelInit, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 
    self.mu0 = tf.constant(mu0, dtype=tf.float32)

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

    xExp = tf.expand_dims(4*np.pi*tf.math.reciprocal(tf.square(self.kGrid) + \
                                  tf.square(self.mu0)), 0)

    initKExp = tf.keras.initializers.Constant(xExp.numpy())

    xExp2 = tf.expand_dims(4*np.pi*tf.math.reciprocal(tf.square(self.kGrid) + \
                                  tf.square(1.0)), 0)

    initKExp2 = tf.keras.initializers.Constant(xExp2.numpy())

    self.multipliersRe = []
    self.multipliersIm = []

    self.multipliersRe.append(self.add_weight("multRe_0",
                       initializer=initKExp, shape = (1, self.NpointsMesh)))
    self.multipliersIm.append(self.add_weight("multIm_0",
                       initializer=tf.initializers.zeros(), 
                       shape = (1, self.NpointsMesh)))

    self.multipliersRe.append(self.add_weight("multRe_1",
                       initializer=initKExp2, shape = (1, self.NpointsMesh)))
    self.multipliersIm.append(self.add_weight("multIm_1",
                       initializer=tf.initializers.zeros(), 
                        shape = (1, self.NpointsMesh)))


    # this needs to be properly initialized it, otherwise it won't even be enough

  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    # this needs to be perodic distance!!!
    # (batch_size, Np*Ncells)
    diff = tf.expand_dims(input, -1) - tf.reshape(self.xGrid, (1,1, self.NpointsMesh))
    # (batch_size, Np*Ncells, NpointsMesh)
    # we compute all the localized gaussians
    array_gaussian = gaussianPer(diff, self.tau, self.L)
    # we add them together
    arrayReducGaussian = tf.complex(tf.reduce_sum(array_gaussian, axis = 1), 0.0)
    # (batch_size, NpointsMesh) (we sum the gaussians together)
    # we apply the fft
    print("computing the FFT")

    fftGauss = tf.signal.fftshift(tf.signal.fft(arrayReducGaussian))
    #(batch_size, NpointsMesh)
    Deconv = tf.complex(tf.expand_dims(gaussianDeconv(self.kGrid, self.tau), 0),0.0)
    #(1, NpointsMesh)

    rfft = tf.multiply(fftGauss, Deconv)/(2*np.pi*self.NpointsMesh/self.L)
    #(batch_size, NpointsMesh)
    # we are only using one channel
    #rfft = tf.expand_dims(rfftDeconv, 1)
    # Fourier multipliers

    Rerfft = tf.math.real(rfft)
    Imrfft = tf.math.imag(rfft)

    print("applying the multipliers")

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft = tf.multiply(self.multipliersRe[0],Rerfft)
    multReImfft = tf.multiply(self.multipliersIm[0],Rerfft)
    multImImfft = tf.multiply(self.multipliersRe[0],Imrfft)
    multImRefft = tf.multiply(self.multipliersIm[0],Imrfft)

    multfft = tf.expand_dims(tf.complex(multReRefft-multImImfft, \
                                        multReImfft+multImRefft),1)

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft2 = tf.multiply(self.multipliersRe[1],Rerfft)
    multReImfft2 = tf.multiply(self.multipliersIm[1],Rerfft)
    multImImfft2 = tf.multiply(self.multipliersRe[1],Imrfft)
    multImRefft2 = tf.multiply(self.multipliersIm[1],Imrfft)

    multfft2 = tf.expand_dims(tf.complex(multReRefft2-multImImfft2, \
                          multReImfft2+multImRefft2), 1)

    multFFT = tf.concat([multfft, multfft2], axis = 1)


    multfftDeconv = tf.multiply(multFFT, tf.expand_dims(Deconv,1))

    print(multfft.shape)
    print("inverse fft")
    irfft = tf.math.real(tf.expand_dims(tf.signal.ifft(tf.signal.ifftshift(multfftDeconv)), 1))

    local = irfft*tf.expand_dims(array_gaussian, 2)
    
    fmm = tf.reduce_sum(local, axis = -1)/(2*np.pi*self.NpointsMesh/self.L)
    #mult = 

    return fmm 

class NUFFTLayerMultiChannelInitOneSided(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, nChannels, NpointsMesh, sigma, xLims, mu0 = 1.0):
    super(NUFFTLayerMultiChannelInitOneSided, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 
    self.mu0 = tf.constant(mu0, dtype=tf.float32)

    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1

    
    self.xLims = xLims
    self.L = np.abs(xLims[1] - xLims[0])
    # this one needs to be a one-dimensional numpy array 
    self.sigma = sigma
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


    initSigma = tf.keras.initializers.Constant([self.sigma])

    

    xExp = tf.expand_dims(4*np.pi*tf.math.reciprocal(tf.square(self.kGrid) + \
                                  tf.square(self.mu0)), 0)

    initKExp = tf.keras.initializers.Constant(xExp.numpy())

    xExp2 = tf.expand_dims(4*np.pi*tf.math.reciprocal(tf.square(self.kGrid) + \
                                  tf.square(1.0)), 0)

    initKExp2 = tf.keras.initializers.Constant(xExp2.numpy())

    self.sigmaVar = self.add_weight("sigma",
                       initializer=initSigma, shape = (1,))

    self.multipliersRe = []
    self.multipliersIm = []

    self.multipliersRe.append(self.add_weight("multRe_0",
                       initializer=initKExp, shape = (1, self.NpointsMesh)))
    self.multipliersIm.append(self.add_weight("multIm_0",
                       initializer=tf.initializers.zeros(), 
                       shape = (1, self.NpointsMesh)))

    self.multipliersRe.append(self.add_weight("multRe_1",
                       initializer=initKExp2, shape = (1, self.NpointsMesh)))
    self.multipliersIm.append(self.add_weight("multIm_1",
                       initializer=tf.initializers.zeros(), 
                        shape = (1, self.NpointsMesh)))


    # this needs to be properly initialized it, otherwise it won't even be enough

  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    # this needs to be perodic distance!!!
    # (batch_size, Np*Ncells)
    diff = tf.expand_dims(input, -1) - tf.reshape(self.xGrid, (1,1, self.NpointsMesh))
    # (batch_size, Np*Ncells, NpointsMesh)
    # we compute all the localized gaussians
    array_gaussian = gaussianPer(diff, self.tau, self.L)
    array_normal   = normalPer(diff, self.sigmaVar, self.L)
    # we add them together
    arrayReducGaussian = tf.complex(tf.reduce_sum(array_normal, axis = 1), 0.0)
    # (batch_size, NpointsMesh) (we sum the gaussians together)
    # we apply the fft
    print("computing the FFT")
    print(self.sigmaVar.shape)

    # FFT of the sum of the FFT
    fftGauss = tf.signal.fftshift(tf.signal.fft(arrayReducGaussian))
    #(batch_size, NpointsMesh)
    Deconv = tf.complex(tf.expand_dims(gaussianDeconv(self.kGrid, self.tau), 0),0.0)
    #(1, NpointsMesh)
    #(batch_size, NpointsMesh)
    # we are only using one channel
    #rfft = tf.expand_dims(rfftDeconv, 1)
    # Fourier multipliers

    Rerfft = tf.math.real(fftGauss)
    Imrfft = tf.math.imag(fftGauss)

    print("applying the multipliers")

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft = tf.multiply(self.multipliersRe[0],Rerfft)
    multReImfft = tf.multiply(self.multipliersIm[0],Rerfft)
    multImImfft = tf.multiply(self.multipliersRe[0],Imrfft)
    multImRefft = tf.multiply(self.multipliersIm[0],Imrfft)

    multfft = tf.expand_dims(tf.complex(multReRefft-multImImfft, \
                                        multReImfft+multImRefft),1)

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft2 = tf.multiply(self.multipliersRe[1],Rerfft)
    multReImfft2 = tf.multiply(self.multipliersIm[1],Rerfft)
    multImImfft2 = tf.multiply(self.multipliersRe[1],Imrfft)
    multImRefft2 = tf.multiply(self.multipliersIm[1],Imrfft)

    multfft2 = tf.expand_dims(tf.complex(multReRefft2-multImImfft2, \
                          multReImfft2+multImRefft2), 1)

    multFFT = tf.concat([multfft, multfft2], axis = 1)


    multfftDeconv = tf.multiply(multFFT, tf.expand_dims(Deconv,1))

    print(multfft.shape)
    print("inverse fft")
    irfft = tf.math.real(tf.expand_dims(tf.signal.ifft(tf.signal.ifftshift(multfftDeconv)), 1))

    local = irfft*tf.expand_dims(array_gaussian, 2)
    
    fmm = tf.reduce_sum(local, axis = -1)/(2*np.pi*self.NpointsMesh/self.L)
    #mult = 

    return fmm 


## TODO: RFFT using the fact that the output signa is real    


class testInit(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, dataInit):
    super(testInit, self).__init__()
    self.dataInit = dataInit


  def build(self, input_shape):

    init = tf.keras.initializer.Constant(self.dataInit)
  
    self.multipliersRe = []
    self.multipliersRe.append(self.add_weight("multRe_0",
                       initializer=init))



    # this needs to be properly initialized it, otherwise it won't even be enough

  @tf.function
  def call(self, input):
    
    fmm = tf.multiply(self.multipliersRe[0], input)
    #mult = 

    return fmm 



# periodic spreading of the functions (not sure if this works all the time)
@tf.function 
def gaussianPer(x, tau, L = 2*np.pi):
  return tf.exp( -tf.square(x  )/(4*tau)) + \
         tf.exp( -tf.square(x-L)/(4*tau)) + \
         tf.exp( -tf.square(x+L)/(4*tau))

@tf.function 
def gaussian(x, tau):
  return tf.exp( -tf.square(x)/(4*tau)) 

# normal distribution in tensorflow
@tf.function
def normalPer(x, tau, L):
  return tf.math.reciprocal(tf.sqrt(2*np.pi)*tau)*(\
          tf.exp( -0.5*tf.square(x)/tau**2 ) + \
          tf.exp( -0.5*tf.square(x - L)/tau**2 ) + \
          tf.exp( -0.5*tf.square(x + L)/tau**2 ))

@tf.function 
def gaussianDeconv(k, tau):
  return tf.sqrt(np.pi/tau)*tf.exp(tf.square(k)*tau)


############### variable number of particules per cell 

# write bining function input Rin, Ncells, L, max number particules per cell
# ouput the interaction list of each of the atoms involved

# loop over each atom, and compute the distances in tensorflow. 
# if the index of the interaction is -1 (zeros won't work here)
# add a zero in that point 

# make the 

# neigh List has a fixed lenght

@tf.function
def genDistInvPerNlist(Rin, Npoints, neighList, L, av = [0.0, 0.0], std = [1.0, 1.0]):
    # function to generate the generalized coordinates for periodic data
    # the input has dimensions (Nsamples, Ncells*Np)
    # the ouput has dimensions (Nsamples*Ncells*Np*(3*Np-1), 2)

    # neighList is a (Nsample, Npoints, maxNeigh)

    # add assert with respect to the input shape 
    Nsamples = Rin.shape[0]
    maxNumNeighs = neighList.shape[-1]
    R2 = tf.reshape(Rin, (Nsamples, Npoints))

    Abs_Array = []
    Abs_Inv_Array = []
    # we compute the difference between points in the same
    # interaction list

    Idx1 = tf.constant(np.linspace(0,Nsamples-1, Nsamples).astype(np.int64).reshape((-1,1)))

    for r in range(Npoints):
        absDistArrayPoint = []
        absInvArrayPoint =[]
        for j in range(maxNumNeighs) :
            # we consider the indices and the concatenat with the other set
            # in order to have a matrix having the coordinates in Rin
            Idx = tf.concat([Idx1, tf.reshape(neighList[:,r,j],(-1,1))], axis = 1) 
            # we create a filter for the unwanted data points 
            filter = tf.cast(Idx > -1, tf.int64)     
            # we substract
            a = tf.subtract(Rin[:,r], tf.gather_nd(Rin, Idx*filter)) 
            # apply filter         
            a = tf.multiply(a, tf.cast(filter[:,1], tf.float32))
            # applying the periodicity
            b = a - L*tf.round(a/L)

            # we apply the % TODO check when is the best place for the normalization
            bnorm = (tf.abs(tf.expand_dims(b,-1)) -av[1])/std[1] 
            # we need to smear it a little bit 
            binv = (tf.abs(tf.math.reciprocal(tf.expand_dims(b,-1)-0.000001)) -av[0])/std[0]

            # we filtered out the padding
            bnormFilt = tf.multiply(bnorm,
                                    tf.reshape(tf.cast(filter[:,1], tf.float32),(-1,1))) 
            binvFilt = tf.multiply(binv,
                                    tf.reshape(tf.cast(filter[:,1], tf.float32),(-1,1)))

            absDistArrayPoint.append(bnormFilt)
            absInvArrayPoint.append(binvFilt)
        Abs_Array.append(tf.expand_dims(
                         tf.concat(absDistArrayPoint, axis = 1), 1))
        Abs_Inv_Array.append(tf.expand_dims(
                             tf.concat(absInvArrayPoint, axis = 1), 1))

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
    # assert R_Diff_total.shape[0] == Nsamples*Ncells*Np*(3*Np-1)

    return R_Diff_total


@tf.function
def genDistInvPerNlistLoop(Rin, Npoints, neighList, L, av = [0.0, 0.0], std = [1.0, 1.0]):
    # function to generate the generalized coordinates for periodic data
    # the input has dimensions (Nsamples, Ncells*Np)
    # the ouput has dimensions (Nsamples*Ncells*Np*(3*Np-1), 2)

    # neighList is a (Nsample, Npoints, maxNeigh)

    # add assert with respect to the input shape 
    Nsamples = Rin.shape[0]
    maxNumNeighs = neighList.shape[-1]
    R2 = tf.reshape(Rin, (Nsamples, Npoints))


    # we compute the difference between points in the same
    # interaction list
    absSamplesArray = []
    invSamplesArray = []
    for k in range(Nsamples):
        Abs_Array = []
        Abs_Inv_Array = []
        for r in range(Npoints):
            absDistArrayPoint = []
            absInvArrayPoint =[]
            for j in range(maxNumNeighs) :

                if neighList[k,r,j] == -1:
                    bnorm = tf.Variable(0.0)
                    binv = tf.Variable(0.0)
                else: 
                  idx = neighList[k,r,j]
                  # we substract
                  a = tf.subtract(Rin[k,r], Rin[k,idx]) 
                  # applying the periodicity
                  b = a - L*tf.round(a/L)

                  # we apply the % TODO check when is the best place for the normalization
                  bnorm = (tf.abs(b) -av[1])/std[1] 
                  # we need to smear it a little bit 
                  binv = (tf.abs(tf.math.reciprocal(b)) -av[0])/std[0]

                absDistArrayPoint.append(tf.expand_dims(bnorm, axis = 0))
                absInvArrayPoint.append(tf.expand_dims(binv, axis = 0))
            Abs_Array.append(tf.expand_dims(
                             tf.concat(absDistArrayPoint, axis = 0), 0))
            Abs_Inv_Array.append(tf.expand_dims(
                                 tf.concat(absInvArrayPoint, axis = 0), 0))

        absSamplesArray.append(tf.expand_dims(
                                 tf.concat(Abs_Array, axis = 0), 0))
        invSamplesArray.append(tf.expand_dims(
                                 tf.concat(Abs_Inv_Array, axis = 0), 0))


    # concatenating the lists of tensors to a large tensor
    absDistList = tf.concat(absSamplesArray, axis = 0)
    absInvList = tf.concat(invSamplesArray, axis = 0)


    # R_Diff_abs = absDistList)-
    # R_Diff_inv = (tf.abs(absInvList)-av[0])/std[0]
    # or using the reciprocal 
    #R_Diff_abs = tf.math.reciprocal(tf.abs(absDistList) + 0.00001)

    R_Diff_total = tf.concat([tf.reshape(absInvList, (-1,1)), 
                              tf.reshape(absDistList, (-1,1))], axis = 1)

    # asserting the final size of the tensor
    # assert R_Diff_total.shape[0] == Nsamples*Ncells*Np*(3*Np-1)

    return R_Diff_total

def computInterList( Rinnumpy, L,  radious, maxNumNeighs):
  Nsamples, Npoints = Rinnumpy.shape

  # computing the distance
  DistNumpy = np.abs(Rinnumpy.reshape(Nsamples,Npoints,1) \
              - Rinnumpy.reshape(Nsamples,1,Npoints))

  # periodicing the distance
  DistNumpy = DistNumpy - L*np.round(DistNumpy/L)

  # loop to get the proper indices and padding the rest
  Idex = []
  for ii in range(0,Nsamples):
    sampleIdx = []
    for jj in range(0, Npoints):
      localIdx = []
      for kk in range(0, Npoints):
        if jj!= kk and np.abs(DistNumpy[ii,jj,kk]) < radious:
          # checking that we are not going over the mac number of
          # neighboors, if so we break the loop
          if len(localIdx) >= maxNumNeighs:
            print("Number of neighboors is larger than the max number allowed")
            break
          localIdx.append(kk)

      while len(localIdx) < maxNumNeighs:
        localIdx.append(-1)
      sampleIdx.append(localIdx)

    Idex.append(sampleIdx)

# converting the list of points to an numpy array.
  Idx = np.array(Idex)

  return Idx
