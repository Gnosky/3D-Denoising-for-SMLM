import numpy as np
import skimage
from three_d_blindspot import *

from os import listdir
from os.path import join
import imageio
import glob
from tqdm import trange

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--loss',default='mse',help='loss function (mse,gamma, or softmax)')
parser.add_argument('--dataset',default='data_06',help='dataset name e.g. Confocal_MICE')
parser.add_argument('--noise',default='raw',help='noise level (raw, avg2, avg4, ...)')
parser.add_argument('--crop',type=int,default=32,help='crop size (0 for no crop)')
parser.add_argument('--depth',type=int,default=32)
args = parser.parse_args()

mean_std_path = 'meanstd.%s.%s.%s.npz'%(args.loss,args.dataset,args.noise)
mean_std = np.load(mean_std_path)
train_mean = mean_std['train_mean']
train_std = mean_std['train_std']
print(train_mean,train_std)

# Re-create the model and load the weights

if args.loss == 'mse':
    model = mse_blindspot_network((args.depth,args.crop, args.crop, 1),train_mean,train_std)
elif args.loss == 'gamma':
    model = gamma_blindspot_network((args.depth,args.crop, args.crop, 1),train_mean,train_std)
elif args.loss == 'approx_poisson':
    model = approx_poisson_blindspot_network((args.depth,args.crop, args.crop, 1),train_mean,train_std)
else:
    raise ValueError('unknown loss %s'%args.loss)

weights_path = 'weights.%s.%s.%s.latest.hdf5'%(args.loss,args.dataset,args.noise)

model.load_weights(weights_path)

images = []
path = '../../jventu09/3d-denoising-dataset/3Ddata06/'
for im_path in glob.glob(join(path,'noisy*.tif')):
	im = imageio.imread(im_path).astype('float32')
	images.append(im)

print('%d images'%len(images))
            
norm = lambda im : (im / 255.0).reshape((512, 512, 1))
np_test_imgs = np.array([norm(im) for im in images])
print("np_test_imgs.shape",np_test_imgs.shape)
# Take a depth, crop, crop size chunk from the image
np_test_imgs = np_test_imgs[16:48,(512//2)-args.crop//2:(512//2)+args.crop//2,(512//2)-args.crop//2:(512//2)+args.crop//2,:]
np_test_imgs = np.expand_dims(np_test_imgs,axis=0)
print("np_test_imgs.shape",np_test_imgs.shape)


im_path = '../../jventu09/3d-denoising-dataset/3Ddata06/clean.tif'
gt = imageio.imread(im_path).astype('float32')
            
results_path = 'results.%s.%s.%s.csv'%(args.loss,args.dataset,args.noise)
with open(results_path,'w') as f:
    f.write('Noisy PSNR,Mode PSNR,MAP PSNR\n')
    for im in np_test_imgs:
        im = np.expand_dims(im,axis=0)
        mode_out = np.zeros((args.depth,args.crop,args.crop),dtype='float32')
        map_out = np.zeros((args.depth,args.crop,args.crop),dtype='float32')
        pred = model.predict(im.reshape(1,args.depth,args.crop,args.crop,1))
        
        if args.loss == 'mse':
            x_mean = pred[0]*255
            x_posterior_mean = pred[0]*255
        elif args.loss == 'gamma':
            log_alpha = pred[0][0]
            log_beta = pred[1][0]
            """
            np.save('z.npy',im*255)
            np.save('pred.npy',pred)
            import sys
            sys.exit(0)
            print('log alpha range:',np.min(log_alpha),np.max(log_alpha))
            print('log beta range:',np.min(log_beta),np.max(log_beta))
            print('loss range:',np.min(loss),np.max(loss))
            """
            print('average theta value: ',np.mean(np.exp(-log_beta)))
            x_mean = gamma_mean(log_alpha,log_beta)
            x_posterior_mean = gamma_posterior_mean(im*255,log_alpha,log_beta)
            #x_posterior_mean = gamma_MAP(im*255,log_alpha,log_beta)
        elif args.loss == 'approx_poisson':
            log_loc = pred[0][0]
            log_scale = pred[1][0]
            print(np.mean(np.exp(log_scale)))
            x_mean = approx_poisson_mean(log_loc,log_scale)
            x_posterior_mean = approx_poisson_posterior_mean(im*255,log_loc,log_scale)
            
        mean_out = np.clip(np.squeeze(x_mean),0,255)
        posterior_mean_out = np.clip(np.squeeze(x_posterior_mean),0,255)

        random_frame = np.random.randint(low=0,high=31)
        # Randomly select a frame to compare to the clean image
        im = im[0,random_frame,:,:]
        gt = gt[(512//2)-16:(512//2)+16,(512//2)-16:(512//2)+16]
       # gt = np.expand_dims(gt,axis=2)
        noisy = np.squeeze(im)*255

        """
        Since we only have one clean image to compare to we are using only one 
        denoised frame from our 3d data set for comparison. This can later be generalized
        to using every possible frame
        """
        mean_out = mean_out[:,:,random_frame]
        posterior_mean_out = posterior_mean_out[:,:,random_frame]
        psnr_noisy = skimage.measure.compare_psnr(gt, noisy, data_range = 255)
        psnr_mean = skimage.measure.compare_psnr(gt, mean_out, data_range = 255)
        psnr_posterior_mean = skimage.measure.compare_psnr(gt, posterior_mean_out, data_range = 255)
            
        print(psnr_noisy,psnr_mean,psnr_posterior_mean)
        f.write('%.15f,%.15f,%.15f\n'%(psnr_noisy,psnr_mean,psnr_posterior_mean))
         
        imageio.imwrite('in.png',noisy.astype('uint8'))
        imageio.imwrite('mean.png',mean_out.astype('uint8'))
        imageio.imwrite('posterior.png',posterior_mean_out.astype('uint8'))
        imageio.imwrite('gt.png',gt.astype('uint8'))

results = np.loadtxt(results_path,delimiter=',',skiprows=1)
print('averages:')
print(np.mean(results,axis=0))
