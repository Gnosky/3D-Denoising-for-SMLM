

import numpy as np
from keras import Input
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, LeakyReLU, UpSampling2D, MaxPooling2D, ZeroPadding2D, Cropping2D, Concatenate, Reshape, GlobalAveragePooling2D
from keras.initializers import Constant
import keras.backend as K
import tensorflow as tf

from keras.layers import Layer
from keras.layers import Conv3D, UpSampling3D, MaxPooling3D, ZeroPadding3D, Cropping3D
import imageio

from keras.optimizers import Adam, SGD
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau



def _vshifted_conv(x, num_filters, name):
    """ 
    Vertically shifted 3-d convolution
    """
    filter_size = [3,3,3]
    # Assumes the height is the second dimension
    k = filter_size[1]//2

    ### 2d code ###
#     x = ZeroPadding2D([[k,0],[0,0]])(x)
#     x = Conv2D(filters=num_filters, kernel_size=filter_size, padding='same', kernel_initializer='he_normal', name=name)(x)
#     x = LeakyReLU(0.1)(x)
#     x = Cropping2D([[0,k],[0,0]])(x)

    ### 3d adaptation ###
    
    # assumes first tuple is frame number, second is height, 3rd is width
    # padding on height
    x = ZeroPadding3D([[0,0],[k,0],[0,0]])(x)
    x = Conv3D(filters=num_filters, kernel_size=filter_size, padding='same', kernel_initializer='he_normal', name=name)(x)
    x = LeakyReLU(0.1)(x)
    x = Cropping3D([[0,0],[0,k],[0,0]])(x)

    return x

def _vshifted_pool(x):
    """ 
    Vertically shifted max pooling 3d
    """
    
    ### 2d code ###
#     x = ZeroPadding2D([[1,0],[0,0]])(x)
#     x = Cropping2D([[0,1],[0,0]])(x)

#     x = MaxPooling2D(pool_size=2,strides=2,padding='same')(x)

    ### 3d adaptation ###
    x = ZeroPadding3D([[0,0],[1,0],[0,0]])(x)
    x = Cropping3D([[0,0],[0,1],[0,0]])(x)
    
    x = MaxPooling3D(pool_size=(2,2,2),strides=2,padding='same')(x)

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
    n = UpSampling3D(2)(n)

    n = Concatenate(axis=4)([n, skips.pop()])
    n = _vshifted_conv(n, 96, 'dec_conv5')
    n = _vshifted_conv(n, 96, 'dec_conv5b')

    n = UpSampling3D(2)(n)

    n = Concatenate(axis=4)([n, skips.pop()])
    n = _vshifted_conv(n, 96, 'dec_conv4')
    n = _vshifted_conv(n, 96, 'dec_conv4b')

    n = UpSampling3D(2)(n)
    n = Concatenate(axis=4)([n, skips.pop()])
    n = _vshifted_conv(n, 96, 'dec_conv3')
    n = _vshifted_conv(n, 96, 'dec_conv3b')

    n = UpSampling3D(2)(n)
    n = Concatenate(axis=4)([n, skips.pop()])
    n = _vshifted_conv(n, 96, 'dec_conv2')
    n = _vshifted_conv(n, 96, 'dec_conv2b')

    n = UpSampling3D(2)(n)
    n = Concatenate(axis=4)([n, skips.pop()])
    n = _vshifted_conv(n, 96, 'dec_conv1a')
    n = _vshifted_conv(n, 96, 'dec_conv1b')

    # final pad and crop for blind spot
    n = ZeroPadding3D([[0,0],[1,0],[0,0]])(n)
    n = Cropping3D([[0,0],[0,1],[0,0]])(n)

    return n


def blindspot_network(inputs):
    # batch, height, width, depth, channel
    b,d,h,w,c = K.int_shape(inputs)
    #if h != w:
    #raise ValueError('input shape must be square')
    if h % 32 != 0 or w % 32 != 0: #or d % 32 != 0:
        raise ValueError('input shape (%d x %d x %d) must be divisible by 32'%(h,w,d))

    # make vertical blindspot network
    vert_input = Input([d,h,w,c])
    vert_output = _vertical_blindspot_network(vert_input)
    vert_model = Model(inputs=vert_input,outputs=vert_output)
        
    # run vertical blindspot network on rotated inputs
    stacks = []
    for i in range(4):
        # Rotate along width prior to network being run
        rotated = inputs
        for j in range(i):
            rotated = Lambda(lambda x: tf.transpose(x, perm = [0,1,3,2,4]))(rotated)
            rotated = Lambda(lambda x: tf.reverse(x, axis = [3]))(rotated)

        if i == 0 or i == 2:
            rotated = Reshape([d,w,h,c])(rotated)
        else:
            rotated = Reshape([d,h,w,c])(rotated)
        out = vert_model(rotated)
        
        # Undo the rotation after the network is run
        for j in range(i):
            out = Lambda(lambda x: tf.transpose(tf.reverse(x, axis = [3]),perm = [0,1,3,2,4]))(out)
            
        stacks.append(out)
        
    for i in [1,3]:
        rotated = inputs
        # Rotate along depth axis prior to network being run
        for j in range(i):
            rotated = Lambda(lambda x: tf.transpose(x, perm = [1,0,2,3,4]))(rotated) 
            rotated = Lambda(lambda x: tf.reverse(x, axis = [0]))(rotated)
        out = vert_model(rotated)
        
        # Undo rotation after network is run
        for j in range(i):
            out = Lambda(lambda x: tf.transpose(tf.reverse(x, axis = [0]),perm = [1,0,2,3,4]))(out)
        stacks.append(out)

    stacks = [vert_model(inputs) for i in range(6)]
    # concatenate outputs
    x = Concatenate(axis=4)(stacks)

    # final 1x1 convolutional layers
    x = Conv3D(384, 1, kernel_initializer='he_normal', name='conv1x1_1')(x)
    x = LeakyReLU(0.1)(x)

    x = Conv3D(96, 1, kernel_initializer='he_normal', name='conv1x1_2')(x)
    x = LeakyReLU(0.1)(x)

    return x


def mean_squared_error_loss(y,loc):
    return K.mean(0.5*K.pow(y-loc,2))

def mse_blindspot_network(input_shape,train_mean=0,train_std=1):
    # create input layer
    inputs = Input(input_shape)

    # apply normalization
    norm_input = Lambda(lambda x: (x-train_mean)/train_std)(inputs)

    # run blindspot network
    x = blindspot_network(norm_input)

    loc = Conv3D(1, 1, name='loc')(x)

    output = Lambda(lambda x: x*train_std+train_mean)(loc)

    # create model
    model = Model(inputs=inputs,outputs=output)

    # create loss function
    loss = mean_squared_error_loss(norm_input,loc)
    model.add_loss(loss)

    return model
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


def gaussian_blindspot_network(input_shape, train_mean=0,train_std=1, reg_weight=0.01):
	
	# Create input layer
	inputs = Input(input_shape)

	# Normalize
	norm_input = Lambda(lambda x: (x-train_mean)/train_std)(inputs)

	# run blindspot network
	x = blindspot_network(norm_input)

	loc = Conv3D(1,1,kernel_initializer='he_normal',name='loc')(x)
	log_scale = Conv3D(1,1,kernel_initializer='he_normal',name='log_scale')(x)
	log_noise_scale = Conv3D(1,1,kernel_initializer='he_normal',name='log_noise_scale')(x)

	posterior_mean = Lambda(lambda x: gaussian_posterior_mean(*x)*train_std+train_mean)([norm_input, loc, log_scale, log_noise_scale])

	model = Model(inputs=inputs, outputs=posterior_mean)
	
	loss = gaussian_loss(norm_input, loc, log_scale, log_noise_scale, reg_weight=reg_weight)
	model.add_loss(loss)

	return model

