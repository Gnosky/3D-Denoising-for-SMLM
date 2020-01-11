"""
TODO:
"""

import random
import numpy as np
import skimage
import twodblindspot.model as twod_model
import threedblindspot.model as threed_model
from os import listdir
from os.path import join
import imageio
import glob
from tqdm import trange

import argparse

import keras
from keras import backend as K
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau

np.random.seed(1)
random.seed(1)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--loss',default='mse',help='loss function (mse,gamma, or softmax)')
parser.add_argument('--dataset',default='data_06',help='dataset name e.g. Confocal_MICE')
parser.add_argument('--noise',default='raw',help='noise level (raw, avg2, avg4, ...)')
parser.add_argument('--crop',type=int,default=512,help='crop size (512 for no crop)')
parser.add_argument('--batch',type=int,default=1,help='batch size')
parser.add_argument('--epoch',type=int,default=5,help='num epochs')
parser.add_argument('--reg',type=float,default=0,help='regularization weight')
parser.add_argument('--lr',type=float,default=0.0004,help='learning rate')
parser.add_argument('--decay',type=float,default=0,help='learning rate decay')
parser.add_argument('--depth',type=int,default=32,help='Length of z axis')
parser.add_argument('--network_dimension', type=int, default=3)
parser.add_argument('--train_names_glob', default = 'noisy[0-5][0-9].tif')
parser.add_argument('--test_names_glob', default = 'noisy6[0-9].tif')
parser.add_argument('--dir_path', default = '../../jventu09/3d-denoising-dataset/3Ddata06/')
args = parser.parse_args()

norm = lambda im : (im / 255.0).reshape((512, 512, 1))
class Network():

    def __init__(self, arguments):
        self.params = arguments
        self.generator = None
        self.network = None
        self.dir_path = args.dir_path
        self.train_names = glob.glob(join(self.dir_path,arguments.train_names_glob))
        self.test_names = glob.glob(join(self.dir_path,arguments.test_names_glob))
        self.train_mean = None
        self.train_std = None
    
    def _get_train_statistics(self):
        train_images = []
        for im_path in self.train_names:
            im = imageio.volread(im_path).astype('float32')
            for i in range(im.shape[0]):
                train_images.append(im[i,:,:])
        train_images = np.stack(train_images,axis=0)

        np_train_imgs = np.array([norm(im) for im in train_images])
        np_train_imgs = np.expand_dims(np_train_imgs,0)

        train_mean = np.mean(np_train_imgs)
        train_std = np.std(np_train_imgs)
        print("train_mean, train_std", train_mean,train_std)

        mean_std_path = 'data_06/meanstd.%s.%s.%s.%s.%s_crop.npz'%(self.params.network_dimension,self.params.loss,self.params.dataset,self.params.noise,self.params.crop)
        np.savez(mean_std_path,train_mean=train_mean,train_std=train_std)
        
        self.train_mean = train_mean
        self.train_std = train_std

        del np_train_imgs
        del train_images	
 

    def _create_2d_model(self):
        crop_size = self.params.crop
        reg = self.params.reg
        train_mean = self.train_mean
        train_std = self.train_std

        if self.params.loss == 'mse':
            model = twod_model.mse_blindspot_network((crop_size, crop_size, 1), train_mean, train_std)
        elif self.params.loss == 'gamma':
            model = twod_model.gamma_blindspot_network((crop_size, crop_size, 1), train_mean, train_std, reg)
        elif self.params.loss == 'approx_poisson':
            model = twod_model.approx_poisson_blindspot_network((crop_size, crop_size, 1), train_mean, train_std)
        elif self.params.loss == 'gaussian':
            model =twod_model.gaussian_blindspot_network((crop_size, crop_size, 1), train_mean, train_std)
        else:
            raise ValueError('unknown loss')
        return model

    def _create_3d_model(self):
        depth = self.params.depth
        crop_size = self.params.crop
        reg = self.params.reg
        train_mean = self.train_mean
        train_std = self.train_std

        if self.params.loss == 'mse':
            model = threed_model.mse_blindspot_network((depth, crop_size, crop_size, 1), train_mean, train_std)
        #elif self.params.loss == 'gamma':
#	model = gamma_blindspot_network((depth, crop_size, crop_size, 1), train_mean, train_std, reg)
        #elif self.params.loss == 'approx_poisson':
#	model = approx_poisson_blindspot_network((depth, crop_size, crop_size, 1), train_mean, train_std)
        elif self.params.loss == 'gaussian':
            model = threed_model.gaussian_blindspot_network((depth, crop_size, crop_size, 1), train_mean, train_std)
        else:
            raise ValueError('unknown loss')
        return model

    def _3d_random_crop_generator(self, file_names):
        depth = self.params.depth
        crop_size = self.params.crop
        batch_size = self.params.batch
        while True:
            im_path = random.sample(file_names,1)[0]
            data = imageio.volread(im_path).astype('float32')
            data = np.array([norm(im) for im in data])
            data = np.expand_dims(data,axis=0)
            z = np.random.randint(data.shape[1]-depth, size = batch_size)
            if crop_size != 512: 
                y = np.random.randint(data.shape[2]-crop_size, size = batch_size)
                x = np.random.randint(data.shape[3]-crop_size, size = batch_size)
            else:
                y = np.full(batch_size,0)
                x = np.full(batch_size,0)
            batch = np.zeros((batch_size,depth,crop_size,crop_size,1), dtype = data.dtype)
            for i, ind in enumerate(z):
                batch[i,:,:,:,:] = data[0,z[i]:z[i]+depth,y[i]:y[i]+crop_size,x[i]:x[i]+crop_size]
                    
            yield batch, None

    def _2d_random_crop_generator(self, file_names):
        crop_size = self.params.crop
        batch_size = self.params.batch

        while True:
            im_path = random.sample(file_names,1)[0]
            data = imageio.volread(im_path).astype('float32')
            data = np.array([norm(im) for im in data])
            z = np.random.randint(data.shape[0], size = batch_size)
            if crop_size != 512: 
                y = np.random.randint(data.shape[1]-crop_size, size = batch_size)
                x = np.random.randint(data.shape[2]-crop_size, size = batch_size)
                batch = np.zeros((batch_size,crop_size, crop_size,1), dtype = data.dtype)
                for i, ind in enumerate(z):
                    batch[i] = data[z[i],x[i]:x[i]+crop_size,y[i]:y[i]+crop_size]
            else:
                batch = np.zeros((batch_size,512,512,1), dtype = data.dtype)
                for i, ind in enumerate(z):
                    batch[i] = data[z[i],:,:]
                        
            yield batch, None


    def fit_model(self):
        # Create a genearator 
        if self.params.network_dimension == 3:
            gen = self._3d_random_crop_generator(self.train_names)
            val_gen = self._3d_random_crop_generator(self.test_names)
        elif self.params.network_dimension == 2:
            gen = self._2d_random_crop_generator(self.train_names)
            val_gen = self._2d_random_crop_generator(self.test_names)
        else:
            raise ValueError('unknown network dimension, only 2 or 3 are accepted')
        
        # Calculate training mean and std first
        self._get_train_statistics()

        # Instantiate model
        if self.params.network_dimension == 3:
            model = self._create_3d_model()
        elif self.params.network_dimension == 2:
            model = self._create_2d_model()
        else:
            raise ValueError('unknown network dimension, only 2 or 3 are accepted')
        
        model.compile(optimizer=Adam(self.params.lr))
        weights_path = 'data_06/weights.%s.%s.%s.%s.%s_crop.latest.hdf5'%(self.params.network_dimension, self.params.loss,self.params.dataset,self.params.noise,self.params.crop)

        callbacks = []
        callbacks.append(ModelCheckpoint(filepath=weights_path, monitor='val_loss',save_best_only=1,verbose=1))
        callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))

        print("Training %d dimensional model"%(self.params.network_dimension))
        
        # Fit model
        history = model.fit_generator(gen,
                          steps_per_epoch=2,
                          validation_data=val_gen,
                          validation_steps=1,
                          epochs=self.params.epoch, 
                          verbose=1,
                          callbacks=callbacks) 

net = Network(args)
net.fit_model()

""" Training """

""" Here we train on random crops of the training image.  We use center crops of the validation images as validation data. """

"""
crop_size = 512 if args.crop == 0 else args.crop

model.compile(optimizer=Adam(args.lr))
#model.compile(optimizer=SGD(args.lr,momentum=0.9,decay=args.decay))

weights_path = 'weights.%s.%s.%s.%s_crop.latest.hdf5'%(args.loss,args.dataset,args.noise,args.crop)

#model.load_weights(weights_path)
callbacks = []
callbacks.append(ModelCheckpoint(filepath=weights_path, monitor='val_loss',save_best_only=1,verbose=1))
callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))

#model = keras.utils.multi_gpu_model(model)

    gen = random_crop_generator(train_image_names,crop_size,args.depth, args.batch)
    val_gen = random_crop_generator(test_image_names,crop_size,args.depth,args.batch)

    history = model.fit_generator(gen,
                                  steps_per_epoch=2,
                                  validation_data=val_gen,
                                  validation_steps=1,
                                  epochs=args.epoch, 
                                  verbose=1,
                                  callbacks=callbacks)
"""
