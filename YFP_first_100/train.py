from keras import backend as K
from three_d_blindspot import *
import imageio
import numpy as np
from keras.optimizers import Adam, SGD
import argparse
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--crop', type=int, default=32)
parser.add_argument('--epochs',type=int, default=20)
parser.add_argument('--batch_size',type=int,default=1)
args = parser.parse_args()

file_path = "../data/12_first_100_frames_YFP.tif"

data = imageio.volread(file_path)

print("Total data shape", data.shape)

# Start with square crops for simplicity
train_images = data[:68, :, :]
val_images = data[68:, :, :]

print("train data shape", train_images.shape)
print("val_images shape", val_images.shape)

norm = lambda im: (im/255.0).reshape(data.shape[1],data.shape[2],1)
np_train_imgs = np.array([norm(im) for im in train_images])
np_val_imgs = np.array([norm(im) for im in val_images])
np_train_imgs = np.expand_dims(np_train_imgs,0)
np_val_imgs = np.expand_dims(np_val_imgs,0)

print("np_train_imgs shape", np_train_imgs.shape)
print("np_val_imgs shape", np_val_imgs.shape)

train_mean = np.mean(np_train_imgs)
train_std = np.std(np_train_imgs)

print("train mean",train_mean)
print("train std", train_std)


""" Training """
""" Here we train on random crops of the training image and use
center crops for validation """

def random_crop_generator(data, crop_size, batch_size):
    while True:
        print("data.shape[1]",data.shape[1])
        inds = np.random.randint(data.shape[1], size=batch_size)
        y = np.random.randint(data.shape[2]-crop_size, size = batch_size)
        x = np.random.randint(data.shape[3]-crop_size, size = batch_size)
        print(data.shape[1]-crop_size-1)
        z = np.random.randint(low=0,high=data.shape[1]-crop_size-2, size = batch_size)
        batch = np.zeros((batch_size, crop_size,crop_size,crop_size,1), dtype = data.dtype)
        for i, ind in enumerate(inds):
            batch[i,:,:,:,:] = data[0,z[i]:z[i]+crop_size,y[i]:y[i]+crop_size,x[i]:x[i]+crop_size]
        
        yield batch, None

def center_crop_generator(data, crop_size, batch_size):
    n = 0
    y = data.shape[2]//2 - crop_size//2
    x = data.shape[3]//2 - crop_size//2
    z = data.shape[1]//2 - crop_size//2

    while True:
        for n in range(0,data.shape[1], batch_size):
            batch = data[n:n+batch_size,z:z+crop_size,y:y+crop_size,x:x+crop_size]
            print(batch.shape)
            yield batch, None

crop_size = args.crop

model = mse_blindspot_network((crop_size,crop_size,crop_size,1), train_mean, train_std)

model.compile(optimizer = Adam(.0004))

weights_path = "weights_test_mse.hd5f"

callbacks = []
callbacks.append(ModelCheckpoint(filepath=weights_path,monitor='val_loss', \
        save_best_only=1, verbose=1))
callbacks.append(ReduceLROnPlateau(monitor='val_loss',factor=.1,patience=10, \
        verbose=1,mode='auto',min_delta=0.0001,cooldown=0,min_lr=0))
gen = random_crop_generator(np_train_imgs, crop_size, args.batch_size)
val_gen = random_crop_generator(np_val_imgs, crop_size, args.batch_size)


# Fit model
history = model.fit_generator(gen,validation_data=val_gen, \
        steps_per_epoch=np_train_imgs.shape[1]//args.batch_size, \
        verbose=1, validation_steps=max(np_val_imgs.shape[1]//args.batch_size,1), \
        epochs = args.epochs, callbacks = callbacks)
