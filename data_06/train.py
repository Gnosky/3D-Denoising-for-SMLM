import sys
sys.path.append('../threedblindspot')

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
parser.add_argument('--crop',type=int,default=128,help='crop size (0 for no crop)')
parser.add_argument('--batch',type=int,default=4,help='batch size')
parser.add_argument('--epoch',type=int,default=5,help='num epochs')
parser.add_argument('--reg',type=float,default=0,help='regularization weight')
parser.add_argument('--lr',type=float,default=0.0004,help='learning rate')
parser.add_argument('--decay',type=float,default=0,help='learning rate decay')
parser.add_argument('--depth',type=float,default=32,help='Length of z axis')
args = parser.parse_args()

""" Load dataset """

images = []
path = '../../jventu09/3d-denoising-dataset/3Ddata06/'
for im_path in glob.glob(join(path,'noisy*.tif')):
	im = imageio.imread(im_path).astype('float32')
	images.append(im)

print('%d images'%len(images))

# Use 90% as training and 10% as validation
train_split = (len(images)//10)*9

train_images = images[:train_split]
val_images = images[train_split:]

print('%d training images'%len(train_images))
print('%d validation images'%len(val_images))

"""The images are 8-bit (0-255 range) so we convert them to floating point, 0-1 range."""

norm = lambda im : (im / 255.0).reshape((512, 512, 1))
np_train_imgs = np.array([norm(im) for im in train_images])
np_val_imgs = np.array([norm(im) for im in val_images])
np_train_imgs = np.expand_dims(np_train_imgs,0)
np_val_imgs = np.expand_dims(np_val_imgs,0)
print("np_train_imgs.shape",np_train_imgs.shape)
print("np_val_imgs.shape",np_val_imgs.shape)

train_mean = np.mean(np_train_imgs)
train_std = np.std(np_train_imgs)
print(train_mean,train_std)

mean_std_path = 'meanstd.%s.%s.%s.%s_crop.npz'%(args.loss,args.dataset,args.noise,args.crop)
np.savez(mean_std_path,train_mean=train_mean,train_std=train_std)

""" Training """

""" Here we train on random crops of the training image.  We use center crops of the validation images as validation data. """

def random_crop_generator(data, crop_size, depth, batch_size):
    while True:
        #inds = np.random.randint(data.shape[1], size=batch_size)
        y = np.random.randint(data.shape[2]-crop_size, size = batch_size)
        x = np.random.randint(data.shape[3]-crop_size, size = batch_size)
        z = np.random.randint(low=0,high=data.shape[1]-depth, size = batch_size)
        batch = np.zeros((batch_size,depth,crop_size,crop_size,1), dtype = data.dtype)
        for i, ind in enumerate(z):
            batch[i,:,:,:,:] = data[0,z[i]:z[i]+depth,y[i]:y[i]+crop_size,x[i]:x[i]+crop_size]
        
        yield batch, None

def center_crop_generator(data, crop_size, batch_size):
  n = 0
  y = data.shape[1]//2-crop_size//2
  x = data.shape[2]//2-crop_size//2
  while True:
    for n in range(0,len(data),batch_size):
        batch = data[n:n+batch_size,y:y+crop_size,x:x+crop_size]
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
callbacks.append(ModelCheckpoint(filepath=weights_path, monitor='loss',save_best_only=1,verbose=1))
callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))

#model = keras.utils.multi_gpu_model(model)

if args.crop == 0:
    history = model.fit(np_train_imgs, None,
                        validation_data=(np_val_imgs,None),
                        batch_size=args.batch,
                        epochs=args.epoch, 
                        verbose=1,
                        callbacks=callbacks)
else:
    """
    val_crops = []
    for img in np_val_imgs:
        print("img.shape",img.shape)
        for z in range(0,img.shape[0],args.depth):
            if z + args.depth > img.shape[0]: continue
            for y in range(0,img.shape[2],crop_size):
                if y + crop_size > img.shape[2]: continue
                for x in range(0,img.shape[3],crop_size):
                    if x+crop_size > img.shape[3]:
                        val_crops.append(img[z:z+args.depth,y:y+crop_size,x:x+crop_size,0])
    val_crops = np.stack(val_crops, axis=0)
    val_crops = np.expand_dims(val_crops, axis=4)
    """
    gen = random_crop_generator(np_train_imgs,crop_size,args.depth, args.batch)
    # val_gen = random_crop_generator(np_val_imgs,crop_size,args.depth,args.batch)
    
    """
    train_crops = []
    for ind in range(len(np_train_imgs)):
        y = np.random.randint(np_train_imgs.shape[1]-crop_size,size=80)
        x = np.random.randint(np_train_imgs.shape[2]-crop_size,size=80)
        for i in range(80):
            train_crops.append(np_train_imgs[[ind],y[i]:y[i]+crop_size,x[i]:x[i]+crop_size])
    train_data = np.concatenate(train_crops,axis=0)
    """

    """
    val_crops = []
    for y in range(0,512,crop_size):
        for x in range(0,512,crop_size):
            val_crops.append(np_val_imgs[:,y:y+crop_size,x:x+crop_size])
    val_data = np.concatenate(val_crops,axis=0)
    """

    history = model.fit_generator(gen,
                                  steps_per_epoch=np_train_imgs.shape[1]//args.batch,
                                 # validation_data=(val_crops,None),
    #history = model.fit(train_data, None, batch_size=args.batch,
                                  #validation_data=(val_data,None),
                                  epochs=args.epoch, 
                                  verbose=1,#,
                                  callbacks=callbacks)

