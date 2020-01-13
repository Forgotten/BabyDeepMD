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
      if self.num_outputs[k] == self.num_outputs[k+1]:
        x += self.actfn(tf.matmul(x, ker))
      else :
        x = self.actfn(tf.matmul(x, ker))
    return x
