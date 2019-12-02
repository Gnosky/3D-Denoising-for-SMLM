"""
TODO:
    - Add more samples to training and validation

"""

import sys
sys.path.append('../threedblindspot')
import random
import numpy as np
import skimage
from model import *

from os import listdir
from os.path import join
import imageio
import glob
from tqdm import trange

import argparse

import keras
from keras import backend as K
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


parser = argparse.ArgumentParser()
parser.add_argument('--loss',default='mse',help='loss function (mse,gamma, or softmax)')
parser.add_argument('--dataset',default='data_06',help='dataset name e.g. Confocal_MICE')
parser.add_argument('--noise',default='raw',help='noise level (raw, avg2, avg4, ...)')
parser.add_argument('--crop',type=int,default=0,help='crop size (0 for no crop)')
parser.add_argument('--batch',type=int,default=4,help='batch size')
parser.add_argument('--epoch',type=int,default=5,help='num epochs')
parser.add_argument('--reg',type=float,default=0,help='regularization weight')
parser.add_argument('--lr',type=float,default=0.0004,help='learning rate')
parser.add_argument('--decay',type=float,default=0,help='learning rate decay')
parser.add_argument('--depth',type=float,default=32,help='Length of z axis')
args = parser.parse_args()

""" Load dataset 
Make sure that no blocks have overlap between height
"""
train_images = []
path = '../../jventu09/3d-denoising-dataset/3Ddata06/'
for im_path in glob.glob(join(path,'noisy3*.tif')):
        im = imageio.volread(im_path).astype('float32')
        for i in range(im.shape[0]):
            train_images.append(im[i,:,:])
train_images = np.stack(train_images,axis=0)

val_images = []
for im_path in glob.glob(join(path,'noisy4*.tif')):
        im = imageio.volread(im_path).astype('float32')
        for i in range(im.shape[0]):
            val_images.append(im[i,:,:])
val_images = np.stack(val_images,axis=0)
# Use 90% as training and 10% as validation

#train_images = images[:train_split]
#val_images = images[train_split:]


"""The images are 8-bit (0-255 range) so we convert them to floating point, 0-1 range."""

norm = lambda im : (im / 255.0).reshape((512, 512, 1))
np_train_imgs = np.array([norm(im) for im in train_images])
np_train_imgs = np.expand_dims(np_train_imgs,0)

train_mean = np.mean(np_train_imgs)
train_std = np.std(np_train_imgs)
print(train_mean,train_std)

mean_std_path = 'meanstd.%s.%s.%s.%s_crop.npz'%(args.loss,args.dataset,args.noise,args.crop)
np.savez(mean_std_path,train_mean=train_mean,train_std=train_std)

del np_train_imgs
del train_images
""" Training """

""" Here we train on random crops of the training image.  We use center crops of the validation images as validation data. """

def random_crop_generator(file_names, crop_size, depth, batch_size):
    while True:
        file_name = random.sample(file_names,1)[0]
        data = imageio.volread(im_path).astype('float32')
        data = np.array([norm(im) for im in data])
        data = np.expand_dims(data,axis=0)

        y = np.random.randint(data.shape[2]-crop_size, size = batch_size)
        x = np.random.randint(data.shape[3]-crop_size, size = batch_size)
        z = np.random.randint(low=0,high=data.shape[1]-depth, size = batch_size)
        batch = np.zeros((batch_size,depth,crop_size,crop_size,1), dtype = data.dtype)
        for i, ind in enumerate(z):
            batch[i,:,:,:,:] = data[0,z[i]:z[i]+depth,y[i]:y[i]+crop_size,x[i]:x[i]+crop_size]
        
        yield batch, None


from keras.optimizers import Adam, SGD
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau

crop_size = 512 if args.crop == 0 else args.crop

if args.loss == 'mse':
    model = mse_blindspot_network((args.depth, crop_size, crop_size, 1), train_mean, train_std)
elif args.loss == 'gamma':
    model = gamma_blindspot_network((args.depth, crop_size, crop_size, 1), train_mean, train_std, args.reg)
elif args.loss == 'approx_poisson':
    model = approx_poisson_blindspot_network((args.depth, crop_size, crop_size, 1), train_mean, train_std)
else:
    raise ValueError('unknown loss %s'%args.loss)

#model = keras.utils.multi_gpu_model(model, gpus=4, cpu_merge=False)

model.compile(optimizer=Adam(args.lr))
#model.compile(optimizer=SGD(args.lr,momentum=0.9,decay=args.decay))

weights_path = 'weights.%s.%s.%s.%s_crop.latest.hdf5'%(args.loss,args.dataset,args.noise,args.crop)

#model.load_weights(weights_path)
callbacks = []
callbacks.append(ModelCheckpoint(filepath=weights_path, monitor='val_loss',save_best_only=1,verbose=1))
callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))

#model = keras.utils.multi_gpu_model(model)

if args.crop == 0:
    
    def no_crop_generator(file_names, depth, batch_size):
        while True:
            file_name = random.sample(file_names,1)[0]
            
            data = imageio.volread(im_path).astype('float32')
            data = np.array([norm(im) for im in data])
            data = np.expand_dims(data,axis=0)

            z = np.random.randint(low=0,high=data.shape[1]-depth, size = batch_size)
            batch = np.zeros((batch_size,depth,crop_size,crop_size,1), dtype = data.dtype)
            
            for i, ind in enumerate(z):
                batch[i,:,:,:,:] = data[0,z[i]:z[i]+depth,:,:]
            yield batch, None
    
    gen = no_crop_generator(glob.glob(path+'/noisy3*.tif'), args.depth, args.batch)
    val_gen = no_crop_generator(glob.glob(path+'/noisy4*.tif'), args.depth, args.batch)
    history = model.fit_generator(gen,
                        steps_per_epoch=2,
                        validation_data = val_gen,
                        validation_steps = 4,
                        epochs=args.epoch, 
                        verbose=1,
                        callbacks=callbacks)
else:
    gen = random_crop_generator(glob.glob(path+'/noisy3*.tif'),crop_size,args.depth, args.batch)
    val_gen = random_crop_generator(glob.glob(path+'/noisy4*.tif'),crop_size,args.depth,args.batch)

    history = model.fit_generator(gen,
                                  steps_per_epoch=2,
                                  validation_data=val_gen,
                                  validation_steps=2,
                                  epochs=args.epoch, 
                                  verbose=1,#,
                                  callbacks=callbacks)

