import tensorflow as tf
import numpy as np 
from numba import jit 

@tf.function 
# periodic gaussian function
def gaussianPer(x, tau, L = 2*np.pi):
  return tf.exp( -tf.square(x  )/(4*tau)) + \
         tf.exp( -tf.square(x-L)/(4*tau)) + \
         tf.exp( -tf.square(x+L)/(4*tau))

@tf.function 
def gaussianDeconv2D(kx, ky, tau):
  return (np.pi/tau)*tf.exp((tf.square(kx) + tf.square(ky))*tau)

@tf.function 
def gaussianDeconv3D(kx, ky, kz, tau):
  return sqrt(np.pi/tau)**3*tf.exp((  tf.square(kx) \
                                    + tf.square(ky) \
                                    + tf.square(kz))*tau)


class NUFFTLayerMultiChannel2D(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, nChannels, NpointsMesh, xLims, 
               mu0 = 1.0, mu1 = 1.0):
    super(NUFFTLayerMultiChannel2D, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 

    # this is for the initial guess 
    self.mu0 = tf.constant(mu0, dtype=tf.float32)
    self.mu1 = tf.constant(mu1, dtype=tf.float32)
    
    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1

    self.xLims = xLims
    print(xLims)
    self.L = np.abs(self.xLims[1] - self.xLims[0])
    self.tau = tf.constant(12*(self.L/(2*np.pi*NpointsMesh))**2, 
                           dtype = tf.float32)# the size of the mollifications
    self.kGrid = tf.constant((2*np.pi/self.L)*\
                              np.linspace(-(NpointsMesh//2), 
                                            NpointsMesh//2, 
                                            NpointsMesh), 
                              dtype = tf.float32)
    self.ky_grid, self.kx_grid = tf.meshgrid(self.kGrid, 
                                             self.kGrid ) 

    # we need to define a mesh betwen xLims[0] and xLims[1]
    self.xGrid =  tf.constant(np.linspace(xLims[0], 
                                          xLims[1], 
                                          NpointsMesh+1)[:-1], 
                              dtype = tf.float32)

    self.y_grid, self.x_grid = tf.meshgrid(self.xGrid, 
                                           self.xGrid) 



  def build(self, input_shape):

    print("building the channels")
    # we initialize the channel multipliers
    # we need to add a parametrized family in here

    # TODO: add the symmetries in here
    # (1,NpointsMesh,NpointsMesh)
    xExp = 4*np.pi*tf.expand_dims(tf.math.reciprocal(tf.square(self.kx_grid) \
                                                  +  tf.square(self.ky_grid) \
                                                  +  tf.square(self.mu0)), 0)

    # QUESTION: can we do this without the numpy() call? 
    initKExp = tf.keras.initializers.Constant(xExp.numpy())

    # (1,NpointsMesh,NpointsMesh)
    xExp2 = 4*np.pi*tf.expand_dims(tf.math.reciprocal(tf.square(self.kx_grid)\
                                                  +  tf.square(self.ky_grid)\
                                                  +  tf.square(self.mu1)), 0)


    initKExp2 = tf.keras.initializers.Constant(xExp2.numpy())

    self.multipliersRe = []
    self.multipliersIm = []

    self.multipliersRe.append(self.add_weight("multRe_0",
                       initializer=initKExp, 
                       shape = (1, self.NpointsMesh,  self.NpointsMesh)))
    self.multipliersIm.append(self.add_weight("multIm_0",
                       initializer=tf.initializers.zeros(), 
                       shape = (1, self.NpointsMesh,  self.NpointsMesh)))

    self.multipliersRe.append(self.add_weight("multRe_1",
                              initializer=initKExp2, 
                              shape = (1, self.NpointsMesh,  
                                          self.NpointsMesh)))
    self.multipliersIm.append(self.add_weight("multIm_1",
                              initializer=tf.initializers.zeros(), 
                              shape = (1, self.NpointsMesh,  
                                          self.NpointsMesh)))


    # this needs to be properly initialized it, otherwise it won't even be enough

  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    # this needs to be perodic distance!!!
    # (batch_size, Np*Ncells, 2)
    diffx  = tf.expand_dims(tf.expand_dims(input[:,:,0], -1), -1) \
           - tf.reshape(self.x_grid, (1,1, 
                                      self.NpointsMesh, 
                                      self.NpointsMesh))

    diffy  = tf.expand_dims(tf.expand_dims(input[:,:,1], -1), -1) \
           - tf.reshape(self.y_grid, (1,1, 
                                      self.NpointsMesh, 
                                      self.NpointsMesh))

    # 2 x (batch_size, Np*Ncells, NpointsMesh, NpointsMesh)
    # we compute all the localized gaussians
    array_gaussian_x = gaussianPer(diffx, self.tau, self.L)
    array_gaussian_y = gaussianPer(diffy, self.tau, self.L)

    # we multiply the components
    array_gaussian = array_gaussian_x*array_gaussian_y
    # (batch_size, Np*Ncells, NpointsMesh, NpointsMesh)

    # we add them together
    arrayReducGaussian = tf.complex(tf.reduce_sum(array_gaussian, 
                                                  axis = 1), 0.0)
    # (batch_size, NpointsMesh, NpointsMesh)

    # (batch_size, NpointsMesh) (we sum the gaussians together)
    # we apply the fft
    print("computing the FFT")

    fftGauss = tf.signal.fftshift(tf.signal.fft2d(arrayReducGaussian))
    #(batch_size, NpointsMesh)

    # compute the deconvolution kernel 
    # BEWARE we can have overflowing issues here
    gauss_deconv = gaussianDeconv2D(self.kx_grid, 
                                    self.ky_grid, 
                                    self.tau)

    Deconv = tf.complex(tf.expand_dims(gauss_deconv, 0),0.0)
    #(1, NpointsMesh, NpointsMesh)

    # BEWARE: I'm not fully sure about the square in the denominator (check)
    # here we should expect a broadcasting
    rfft = tf.multiply(fftGauss, Deconv)
    #(batch_size, NpointsMesh, NpointsMesh)
    # we are only using one channel
    #rfft = tf.expand_dims(rfftDeconv, 1)
    # Fourier multipliers

    Rerfft = tf.math.real(rfft)
    Imrfft = tf.math.imag(rfft)

    print("applying the multipliers")

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft = tf.multiply(self.multipliersRe[0], Rerfft)
    multReImfft = tf.multiply(self.multipliersIm[0], Rerfft)
    multImRefft = tf.multiply(self.multipliersRe[0], Imrfft)
    multImImfft = tf.multiply(self.multipliersIm[0], Imrfft)

    multfft = tf.expand_dims(tf.complex(multReRefft-multImImfft, \
                                        multReImfft+multImRefft),1)

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft2 = tf.multiply(self.multipliersRe[1], Rerfft)
    multReImfft2 = tf.multiply(self.multipliersIm[1], Rerfft)
    multImRefft2 = tf.multiply(self.multipliersRe[1], Imrfft)
    multImImfft2 = tf.multiply(self.multipliersIm[1], Imrfft)

    multfft2 = tf.expand_dims(tf.complex(multReRefft2-multImImfft2, \
                                         multReImfft2+multImRefft2), 1)
    #(batch_size, 1, NpointsMesh, NpointsMesh)

    multFFT = tf.concat([multfft, multfft2], axis = 1)
    #(batch_size, 2, NpointsMesh, NpointsMesh)

    multfftDeconv = tf.multiply(multFFT, tf.expand_dims(Deconv,1))
    #(batch_size, 2, NpointsMesh, NpointsMesh)

    # tf.print(multfft.shape)
    # tf.print("inverse fft")

    irfft = tf.math.real(tf.expand_dims(
                         tf.signal.ifft2d(
                         tf.signal.ifftshift(multfftDeconv)), 1))
    #(batch_size, 1, 2, NpointsMesh, NpointsMesh)

    local = irfft*tf.expand_dims(array_gaussian, 2)
    # dim array_gaussian = 
    # (batch_size, Np*Ncells, NpointsMesh, NpointsMesh)
    # dims local = 
    # (batch_size, Np*Ncells, 2, NpointsMesh, NpointsMesh)

    fmm = tf.reduce_sum(tf.reduce_sum(local, axis = 4), axis = 3)\
          /(2*np.pi*self.NpointsMesh/self.L)**2

    # tf.print(fmm.shape)
    # (batch_size, Np*Ncells, 2)

    return fmm 




class NUFFTLayerMultiChannel3D(tf.keras.layers.Layer):
  # this layers uses a few kernels to approximate exp(-mu)
  # and we add the exact mu to check if that becomes worse
  def __init__(self, nChannels, NpointsMesh, xLims, 
               mu0 = 1.0, mu1 = 1.0):
    super(NUFFTLayerMultiChannel3D, self).__init__()
    self.nChannels = nChannels
    self.NpointsMesh = NpointsMesh 

    # this is for the initial guess 
    self.mu0 = tf.constant(mu0, dtype=tf.float32)
    self.mu1 = tf.constant(mu1, dtype=tf.float32)
    
    # we need the number of points to be odd 
    assert NpointsMesh % 2 == 1

    self.xLims = xLims
    print(xLims)
    self.L = np.abs(self.xLims[1] - self.xLims[0])
    self.tau = tf.constant(12*(self.L/(2*np.pi*NpointsMesh))**2, 
                           dtype = tf.float32)# the size of the mollifications
    self.kGrid = tf.constant((2*np.pi/self.L)*\
                              np.linspace(-(NpointsMesh//2), 
                                            NpointsMesh//2, 
                                            NpointsMesh), 
                              dtype = tf.float32)
    self.ky_grid,\
    self.kx_grid,\
    self.kz_grid = tf.meshgrid(self.kGrid, 
                               self.kGrid,
                               self.kGrid ) 

    # we need to define a mesh betwen xLims[0] and xLims[1]
    self.xGrid =  tf.constant(np.linspace(xLims[0], 
                                          xLims[1], 
                                          NpointsMesh+1)[:-1], 
                              dtype = tf.float32)

    self.y_grid,\
    self.x_grid,\
    self.z_grid = tf.meshgrid(self.xGrid,
                              self.xGrid, 
                              self.xGrid) 



  def build(self, input_shape):

    print("building the channels")
    # we initialize the channel multipliers
    # we need to add a parametrized family in here

    # TODO: add the symmetries in here
    # (1,NpointsMesh,NpointsMesh)
    xExp = 4*np.pi*tf.expand_dims(tf.math.reciprocal(tf.square(self.kx_grid) \
                                                  +  tf.square(self.ky_grid) \
                                                  +  tf.square(self.kz_grid) \
                                                  +  tf.square(self.mu0)), 0)

    # QUESTION: can we do this without the numpy() call? 
    initKExp = tf.keras.initializers.Constant(xExp.numpy())

    # (1,NpointsMesh,NpointsMesh)
    xExp2 = 4*np.pi*tf.expand_dims(tf.math.reciprocal(tf.square(self.kx_grid)\
                                                  +  tf.square(self.ky_grid)\
                                                  +  tf.square(self.kz_grid)\
                                                  +  tf.square(self.mu1)), 0)


    initKExp2 = tf.keras.initializers.Constant(xExp2.numpy())

    self.multipliersRe = []
    self.multipliersIm = []

    self.multipliersRe.append(self.add_weight("multRe_0",
                       initializer=initKExp, 
                       shape = (1, self.NpointsMesh,
                                self.NpointsMesh, self.NpointsMesh)))
    self.multipliersIm.append(self.add_weight("multIm_0",
                       initializer=tf.initializers.zeros(), 
                       shape = (1, self.NpointsMesh,
                                self.NpointsMesh, self.NpointsMesh)))

    self.multipliersRe.append(self.add_weight("multRe_1",
                              initializer=initKExp2, 
                              shape = (1, self.NpointsMesh,
                                          self.NpointsMesh,   
                                          self.NpointsMesh)))
    self.multipliersIm.append(self.add_weight("multIm_1",
                              initializer=tf.initializers.zeros(), 
                              shape = (1, self.NpointsMesh,
                                          self.NpointsMesh,   
                                          self.NpointsMesh)))


    # this needs to be properly initialized it, otherwise it won't even be enough

  @tf.function
  def call(self, input):
    # we need to add an iterpolation step
    # this needs to be perodic distance!!!
    # (batch_size, Np*Ncells, 2)
    diffx  = tf.expand_dims(tf.expand_dims(input[:,:,0], -1), -1) \
           - tf.reshape(self.x_grid, (1,1, 
                                      self.NpointsMesh, 
                                      self.NpointsMesh, 
                                      self.NpointsMesh))

    diffy  = tf.expand_dims(tf.expand_dims(input[:,:,1], -1), -1) \
           - tf.reshape(self.y_grid, (1,1, 
                                      self.NpointsMesh, 
                                      self.NpointsMesh, 
                                      self.NpointsMesh))

    diffz  = tf.expand_dims(tf.expand_dims(input[:,:,2], -1), -1) \
           - tf.reshape(self.z_grid, (1,1, 
                                      self.NpointsMesh, 
                                      self.NpointsMesh, 
                                      self.NpointsMesh))

    # 2 x (batch_size, Np*Ncells, NpointsMesh, NpointsMesh)
    # we compute all the localized gaussians
    array_gaussian_x = gaussianPer(diffx, self.tau, self.L)
    array_gaussian_y = gaussianPer(diffy, self.tau, self.L)
    array_gaussian_z = gaussianPer(diffz, self.tau, self.L)

    # we multiply the components
    array_gaussian = array_gaussian_x \
                    *array_gaussian_y \
                    *array_gaussian_z
    # (batch_size, Np*Ncells, NpointsMesh, NpointsMesh)

    # we add them together for each snapshot
    arrayReducGaussian = tf.complex(tf.reduce_sum(array_gaussian, 
                                                  axis = 1), 0.0)
    # (batch_size, NpointsMesh, NpointsMesh)

    # (batch_size, NpointsMesh) (we sum the gaussians together)
    # we apply the fft
    print("computing the FFT")

    fftGauss = tf.signal.fftshift(tf.signal.fft3d(arrayReducGaussian))
    #(batch_size, NpointsMesh)

    # compute the deconvolution kernel 
    # BEWARE we can have overflowing issues here
    gauss_deconv = gaussianDeconv3D(self.kx_grid, 
                                    self.ky_grid, 
                                    self.kz_grid, 
                                    self.tau)

    Deconv = tf.complex(tf.expand_dims(gauss_deconv, 0),0.0)
    #(1, NpointsMesh, NpointsMesh)

    # BEWARE: I'm not fully sure about the square in the denominator (check)
    # here we should expect a broadcasting
    rfft = tf.multiply(fftGauss, Deconv)
    #(batch_size, NpointsMesh, NpointsMesh)
    # we are only using one channel
    #rfft = tf.expand_dims(rfftDeconv, 1)
    # Fourier multipliers

    Rerfft = tf.math.real(rfft)
    Imrfft = tf.math.imag(rfft)

    print("applying the multipliers")

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft = tf.multiply(self.multipliersRe[0], Rerfft)
    multReImfft = tf.multiply(self.multipliersIm[0], Rerfft)
    multImRefft = tf.multiply(self.multipliersRe[0], Imrfft)
    multImImfft = tf.multiply(self.multipliersIm[0], Imrfft)

    multfft = tf.expand_dims(tf.complex(multReRefft-multImImfft, \
                                        multReImfft+multImRefft),1)

    # multfft = tf.multiply(self.multChannels*rfft)
    multReRefft2 = tf.multiply(self.multipliersRe[1], Rerfft)
    multReImfft2 = tf.multiply(self.multipliersIm[1], Rerfft)
    multImRefft2 = tf.multiply(self.multipliersRe[1], Imrfft)
    multImImfft2 = tf.multiply(self.multipliersIm[1], Imrfft)

    multfft2 = tf.expand_dims(tf.complex(multReRefft2-multImImfft2, \
                                         multReImfft2+multImRefft2), 1)
    #(batch_size, 1, NpointsMesh, NpointsMesh)

    multFFT = tf.concat([multfft, multfft2], axis = 1)
    #(batch_size, 2, NpointsMesh, NpointsMesh)

    multfftDeconv = tf.multiply(multFFT, tf.expand_dims(Deconv,1))
    #(batch_size, 2, NpointsMesh, NpointsMesh)

    # tf.print(multfft.shape)
    # tf.print("inverse fft")

    irfft = tf.math.real(tf.expand_dims(
                         tf.signal.ifft3d(
                         tf.signal.ifftshift(multfftDeconv)), 1))
    #(batch_size, 1, 2, NpointsMesh, NpointsMesh)

    local = irfft*tf.expand_dims(array_gaussian, 2)
    # dim array_gaussian = 
    # (batch_size, Np*Ncells, NpointsMesh, NpointsMesh)
    # dims local = 
    # (batch_size, Np*Ncells, 2, NpointsMesh, NpointsMesh)

    fmm = tf.reduce_sum(local, axis = [-3, -2, -1])\
          /(2*np.pi*self.NpointsMesh/self.L)**3

    # tf.print(fmm.shape)
    # (batch_size, Np*Ncells, 2)

    return fmm 