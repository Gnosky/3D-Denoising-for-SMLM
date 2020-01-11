import numpy as np
from keras import Input
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, LeakyReLU, UpSampling2D, MaxPooling2D, ZeroPadding2D, Cropping2D, Concatenate, Reshape, GlobalAveragePooling2D, BatchNormalization
from keras.initializers import Constant
import keras.backend as K
import tensorflow as tf

from keras.layers import Layer

class PoissonScaleLayer(Layer):
    def __init__(self, aval=0, afixed=False, useb=False, **kwargs):
        self.aval = aval
        self.afixed = afixed
        self.useb = useb
        super(PoissonScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='a', 
                                      shape=(),
                                      initializer=Constant(self.aval),
                                      trainable=not self.afixed)
        if self.useb:
            self.b = self.add_weight(name='b', 
                                        shape=(),
                                        initializer='zeros',
                                        trainable=True)
        super(PoissonScaleLayer, self).build(input_shape)

    def call(self, x):
        if not self.afixed:
          aloc = K.softplus(self.a-4)*x
        else:
          aloc = self.a*x
        if self.useb:
          aloc += self.b
        return aloc

    def compute_output_shape(self, input_shape):
        return input_shape

def laplace_loss(y,loc,scale):
  C = np.log(2.0)
  loss = K.abs((loc-y)/scale)+K.log(scale)+C
  return K.mean(loss)

def mean_abs_error_loss(y,loc):
  return K.mean(K.abs(y-loc))

def mean_squared_error_loss(y,loc):
  return K.mean((y-loc)**2)

def mean_var_loss(y,mean,var):
  loss = (y-mean)**2/var+K.log(var)
  return K.mean(loss)

def gaussian_loss(y,loc,log_var,log_noise_var,reg_weight=0.01):
  std = log_var
  var = log_var**2
  noise_var = log_noise_var**2
  #log_var = K.clip(log_var,-64,64)
  #log_noise_var = K.clip(log_noise_var,-64,64)
  #log_var = K.clip(log_var,-8,8)
  #log_noise_var = K.clip(log_noise_var,-8,8)
  #var = K.exp(log_var)
  #noise_var = K.exp(log_noise_var)
  #std = K.exp(0.5*log_var)
  total_var = var+noise_var+1e-3
  loss = (y-loc)**2 / total_var + tf.log(total_var)
  reg = reg_weight*K.abs(std)
  return K.mean(loss+reg)

def gaussian_posterior_mean(y,loc,log_var,log_noise_var):
  std = log_var
  var = log_var**2
  noise_var = log_noise_var**2
  #log_var = K.clip(log_var,-64,64)
  #log_noise_var = K.clip(log_noise_var,-64,64)
  #log_var = K.clip(log_var,-8,8)
  #log_noise_var = K.clip(log_noise_var,-8,8)
  #var = K.exp(log_var)
  #noise_var = K.exp(log_noise_var)
  total_var = var+noise_var+1e-3
  return (loc*noise_var + var*y)/total_var
  #return loc

"""
def gaussian_loss(y,loc,scale,noise_scale):
  total_var = scale**2+noise_scale**2+1e-3
  loss = (y-loc)**2 / total_var + tf.log(total_var)
  reg = 0.1*K.abs(scale)
  return K.mean(loss+reg)

def gaussian_posterior_mean(y,loc,scale,noise_scale):
  var = scale**2
  noise_var = noise_scale**2
  total_var = var+noise_var+1e-3
  return (loc*noise_var + var*y)/total_var
"""

def approx_poisson_loss(inputs,loc,aloc,var):
  #total_var = K.clip(aloc + var,1e-8,64)
  total_var = K.clip(aloc + var + 1e-3,1e-3,64)
  loss = (inputs-loc)**2 / total_var + tf.log(total_var)
  return K.mean(loss)

def approx_poisson_posterior_mean(inputs,loc,aloc,var):
  #total_var = K.clip(aloc + var,1e-8,64)
  total_var = K.clip(aloc + var + 1e-3,1e-3,64)
  return (loc*aloc + var*inputs) / total_var

class GammaLayer(Layer):
    def __init__(self, **kwargs):
        super(GammaLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='a', 
                                      shape=(),
                                      initializer='zeros',
                                      trainable=True)
        super(GammaLayer, self).build(input_shape)

    def call(self, x):
        z, log_alpha, log_beta = x
        log_alpha = K.clip(log_alpha,-64,64)
        log_beta = K.clip(log_beta,-64,64)
        alpha = K.exp(log_alpha)
        beta = K.exp(log_beta)
        a = K.softplus(self.a-4)
        loss = -alpha * log_beta + (alpha + z/a) * tf.log(1 + beta) - tf.lgamma(alpha+z/a) + tf.lgamma(alpha) + tf.lgamma(z/a+1)
        posterior_mean = (a*alpha+z)/(beta+1)
        return K.concatenate([loss,posterior_mean],axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + [2*input_shape[-1],]

def gamma_loss(z_over_a,log_alpha,alpha,log_beta,beta):
  loss = -alpha * log_beta + (alpha + z_over_a) * tf.log(1 + beta) - tf.lgamma(alpha+z_over_a) + tf.lgamma(alpha) + tf.lgamma(z_over_a+1)
  return K.mean(loss)
  
def gamma_posterior_mean(z,a_alpha,beta):
  return (a_alpha+z)/(beta+1)

def _conv(x, num_filters, name):
  """ 2d convolution """
  filter_size = [3,3]

  x = Conv2D(filters=num_filters, kernel_size=filter_size, padding='same', kernel_initializer='he_normal', name=name)(x)
  x = LeakyReLU(0.1)(x)

  return x

def _vshifted_conv(x, num_filters, name):
  """ Vertically shifted convolution """
  filter_size = [3,3]
  k = filter_size[0]//2

  x = ZeroPadding2D([[k,0],[0,0]])(x)
  x = Conv2D(filters=num_filters, kernel_size=filter_size, padding='same', kernel_initializer='he_normal', name=name)(x)
  x = LeakyReLU(0.1)(x)
  x = Cropping2D([[0,k],[0,0]])(x)

  return x

def _pool(x):
  """ max pooling"""
  x = MaxPooling2D(pool_size=2,strides=2,padding='same')(x)

  return x

def _vshifted_pool(x):
  """ Vertically shifted max pooling"""
  x = ZeroPadding2D([[1,0],[0,0]])(x)
  x = Cropping2D([[0,1],[0,0]])(x)

  x = MaxPooling2D(pool_size=2,strides=2,padding='same')(x)

  return x

def _vertical_blindspot_network(x):
  """ Blind-spot network; adapted from noise2noise GitHub
    Each row of output only sees input pixels above that row
  """
  skips = [x]

  n = x
  n = _vshifted_conv(n, 48, 'enc_conv0')
  n = _vshifted_conv(n, 48, 'enc_conv1')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv2')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv3')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv4')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv5')
  n = _vshifted_pool(n)
  n = _vshifted_conv(n, 48, 'enc_conv6')

  #-----------------------------------------------
  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv5')
  n = _vshifted_conv(n, 96, 'dec_conv5b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv4')
  n = _vshifted_conv(n, 96, 'dec_conv4b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv3')
  n = _vshifted_conv(n, 96, 'dec_conv3b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv2')
  n = _vshifted_conv(n, 96, 'dec_conv2b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv1a')
  n = _vshifted_conv(n, 96, 'dec_conv1b')

  # final pad and crop for blind spot
  n = ZeroPadding2D([[1,0],[0,0]])(n)
  n = Cropping2D([[0,1],[0,0]])(n)

  return n

def blindspot_network(inputs):
  b,h,w,c = K.int_shape(inputs)
  #if h != w:
    #raise ValueError('input shape must be square')
  if h % 32 != 0 or w % 32 != 0:
    raise ValueError('input shape (%d x %d) must be divisible by 32'%(h,w))

  #inputs = BatchNormalization()(inputs)

  # make vertical blindspot network
  vert_input = Input([h,w,c])
  vert_output = _vertical_blindspot_network(vert_input)
  vert_model = Model(inputs=vert_input,outputs=vert_output)

  # run vertical blindspot network on rotated inputs
  stacks = []
  for i in range(4):
      rotated = Lambda(lambda x: tf.image.rot90(x,i))(inputs)
      if i == 0 or i == 2:
          rotated = Reshape([h,w,c])(rotated)
      else:
          rotated = Reshape([w,h,c])(rotated)
      out = vert_model(rotated)
      out = Lambda(lambda x:tf.image.rot90(x,4-i))(out)
      stacks.append(out)

  # concatenate outputs
  x = Concatenate(axis=3)(stacks)

  # final 1x1 convolutional layers
  x = Conv2D(384, 1, kernel_initializer='he_normal', name='conv1x1_1')(x)
  x = LeakyReLU(0.1)(x)

  x = Conv2D(96, 1, kernel_initializer='he_normal', name='conv1x1_2')(x)
  x = LeakyReLU(0.1)(x)
  
  return x

def scale_network(x):
  skips = [x]

  n = x
  n = _conv(n, 48, 'scale_enc_conv0')
  n = _conv(n, 48, 'scale_enc_conv1')
  n = _pool(n)
  skips.append(n)

  n = _conv(n, 48, 'scale_enc_conv2')
  n = _pool(n)
  skips.append(n)

  n = _conv(n, 48, 'scale_enc_conv3')
  n = _pool(n)
  skips.append(n)

  n = _conv(n, 48, 'scale_enc_conv4')
  n = _pool(n)
  skips.append(n)

  n = _conv(n, 48, 'scale_enc_conv5')
  n = _pool(n)
  n = _conv(n, 48, 'scale_enc_conv6')

  #-----------------------------------------------
  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _conv(n, 96, 'scale_dec_conv5')
  n = _conv(n, 96, 'scale_dec_conv5b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _conv(n, 96, 'scale_dec_conv4')
  n = _conv(n, 96, 'scale_dec_conv4b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _conv(n, 96, 'scale_dec_conv3')
  n = _conv(n, 96, 'scale_dec_conv3b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _conv(n, 96, 'scale_dec_conv2')
  n = _conv(n, 96, 'scale_dec_conv2b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _conv(n, 96, 'scale_dec_conv1a')
  n = _conv(n, 96, 'scale_dec_conv1b')

  # final 1x1 convolutional layers
  x = Conv2D(96, 1, kernel_initializer='he_normal', name='scale_conv1x1_1')(x)
  x = LeakyReLU(0.1)(x)

  x = Conv2D(96, 1, kernel_initializer='he_normal', name='scale_conv1x1_2')(x)
  x = LeakyReLU(0.1)(x)

  x = Conv2D(1, 1, kernel_initializer='he_normal', name='scale_out')(x)
  x = Lambda(lambda x:K.softplus(x-4))(x)
  x = GlobalAveragePooling2D()(x)
  a = Reshape((1,1,1))(x)

  return a

def laplace_blindspot_network(input_shape,train_mean=0,train_std=1):
  # create input layer
  inputs = Input(input_shape)

  # apply normalization
  norm_input = Lambda(lambda x: (x-train_mean)/train_std)(inputs)
  
  # run blindspot network
  x = blindspot_network(norm_input)
  
  loc = Conv2D(1, 1, name='loc')(x)
  scale = Conv2D(1, 1, name='scale')(x)
  scale = Lambda(lambda x:K.softplus(x)+1e-3)(scale)
  
  output = Lambda(lambda x: x*train_std+train_mean)(loc)

  # create model
  model = Model(inputs=inputs,outputs=output)

  # create loss function
  loss = laplace_loss(norm_input,loc,scale)
  model.add_loss(loss)

  return model

def mae_blindspot_network(input_shape,train_mean=0,train_std=1):
  # create input layer
  inputs = Input(input_shape)

  # apply normalization
  norm_input = Lambda(lambda x: (x-train_mean)/train_std)(inputs)
  
  # run blindspot network
  x = blindspot_network(norm_input)
  
  loc = Conv2D(1, 1, name='loc')(x)
  
  output = Lambda(lambda x: x*train_std+train_mean)(loc)

  # create model
  model = Model(inputs=inputs,outputs=output)

  # create loss function
  loss = mean_abs_error_loss(norm_input,loc)
  model.add_loss(loss)

  return model

def mse_blindspot_network(input_shape,train_mean=0,train_std=1):
  # create input layer
  inputs = Input(input_shape)

  # apply normalization
  norm_input = Lambda(lambda x: (x-train_mean)/train_std)(inputs)
  
  # run blindspot network
  x = blindspot_network(norm_input)
  
  loc = Conv2D(1, 1, kernel_initializer='he_normal', name='loc')(x)
  
  output = Lambda(lambda x: x*train_std+train_mean)(loc)

  # create model
  model = Model(inputs=inputs,outputs=output)

  # create loss function
  loss = mean_squared_error_loss(norm_input,loc)
  model.add_loss(loss)

  return model

def mean_var_blindspot_network(input_shape):
  # create input layer
  inputs = Input(input_shape)

  # run blindspot network
  x = blindspot_network(inputs)
  
  mean = Conv2D(1, 1, name='mean')(x)
  var = Conv2D(1, 1, name='var')(x)
  scale = Lambda(lambda x:K.softplus(x)+1e-3)(var)
  
  # create model
  model = Model(inputs=inputs,outputs=mean)

  # create loss function
  loss = mean_var_loss(inputs,mean,var)
  model.add_loss(loss)

  return model
  
def gaussian_blindspot_network(input_shape,train_mean=0,train_std=1,reg_weight=0.01):
  # create input layer
  inputs = Input(input_shape)

  # apply normalization
  norm_input = Lambda(lambda x: (x-train_mean)/train_std)(inputs)
  
  # run blindspot network
  x = blindspot_network(norm_input)
  
  loc = Conv2D(1, 1, kernel_initializer='he_normal', name='loc')(x)
  log_scale = Conv2D(1, 1, kernel_initializer='he_normal', name='log_scale')(x)
  log_noise_scale = Conv2D(1, 1, kernel_initializer='he_normal', name='log_noise_scale')(x)

  posterior_mean = Lambda(lambda x:gaussian_posterior_mean(*x)*train_std+train_mean)([norm_input,loc,log_scale,log_noise_scale])
  #posterior_mean = Lambda(lambda x:gaussian_posterior_mean(*x))([inputs,loc,log_scale,log_noise_scale])

  # create model
  model = Model(inputs=inputs,outputs=posterior_mean)

  # create loss function
  # input is evaluated against output distribution
  loss = gaussian_loss(norm_input,loc,log_scale,log_noise_scale,reg_weight=reg_weight)
  #loss = gaussian_loss(inputs,loc,log_scale,log_noise_scale)
  model.add_loss(loss)

  return model

def approx_poisson_blindspot_network(input_shape,train_mean=0,train_std=1,a=0,afixed=False,use_anet=False,useb=False):
  # create input layer
  inputs = Input(input_shape)

  # apply normalization
  norm_input = Lambda(lambda x: (x-train_mean)/train_std)(inputs)
  
  x = blindspot_network(norm_input)
  
  log_loc = Conv2D(1, 1, kernel_initializer='he_normal', name='log_loc')(x)
  log_var = Conv2D(1, 1, kernel_initializer='he_normal', name='log_var')(x)

  clipped_exp = Lambda(lambda x:K.exp(K.clip(x,-64,64)))
  loc = clipped_exp(log_loc)
  var = clipped_exp(log_var)

  if use_anet:
    a = scale_network(norm_input)
    aloc = Lambda(lambda x:x[0]*x[1])([a,loc])
  else:
    aloc = PoissonScaleLayer(a,afixed,useb)(loc)

  posterior_mean = Lambda(lambda x:approx_poisson_posterior_mean(*x))([inputs,loc,aloc,var])

  # create model
  model = Model(inputs=inputs,outputs=posterior_mean)

  # create loss function
  # input is evaluated against output distribution
  loss = approx_poisson_loss(inputs,loc,aloc,var)
  model.add_loss(loss)

  return model

def gamma_blindspot_network(input_shape,train_mean=0,train_std=1,reg=1):
  # create input layer
  inputs = Input(input_shape)

  # apply normalization
  norm_input = Lambda(lambda x: (x-train_mean)/train_std)(inputs)
  
  # run blindspot network
  x = blindspot_network(norm_input)
  
  log_alpha = Conv2D(1, 1, name='log_alpha')(x)
  log_beta = Conv2D(1, 1, name='log_beta')(x)

  # create model
  model = Model(inputs=inputs,outputs=[log_alpha, log_beta])

  # create loss function
  # input is evaluated against output distribution
  loss = gamma_loss(255*inputs,log_alpha,log_beta,reg)
  model.add_loss(loss)

  return model

def gamma_blindspot_network_SMLM(input_shape,train_mean=0,train_std=1):
  # create input layer
  inputs = Input(input_shape)

  # apply normalization
  norm_input = Lambda(lambda x: (x-train_mean)/train_std)(inputs)

  # run blindspot network
  x = blindspot_network(norm_input)
  
  log_alpha = Conv2D(1, 1, name='log_alpha')(x)
  log_beta = Conv2D(1, 1, name='log_beta')(x)

  x = GammaLayer()([inputs,log_alpha,log_beta])
  loss = Lambda(lambda x:x[:,:,:,0:1])(x)
  posterior_mean = Lambda(lambda x:x[:,:,:,1:2])(x)

  # create model
  model = Model(inputs=inputs,outputs=posterior_mean)

  # add loss 
  model.add_loss(K.mean(loss))

  return model
  
