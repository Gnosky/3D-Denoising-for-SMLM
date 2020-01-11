"""
TODO:
Fix the divisor for averaging
Visualize results



"""

import sys
sys.path.append('../threedblindspot')
from model import *

import numpy as np
import skimage

from os import listdir
from os.path import join
import imageio
import glob
from tqdm import trange
import argparse
from tifffile import imsave

parser = argparse.ArgumentParser()
parser.add_argument('--loss',default='mse',help='loss function (mse,gamma, or softmax)')
parser.add_argument('--dataset',default='data_06',help='dataset name e.g. Confocal_MICE')
parser.add_argument('--noise',default='raw',help='noise level (raw, avg2, avg4, ...)')
parser.add_argument('--crop',type=int,default=32,help='crop size (0 for no crop)')
parser.add_argument('--depth',type=int,default=32)
parser.add_argument('--bag',default="False")
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

weights_path = 'weights.%s.%s.%s.%s_crop.latest.hdf5'%(args.loss,args.dataset,args.noise,args.crop)

model.load_weights(weights_path)

images = []
path = '../../jventu09/3d-denoising-dataset/3Ddata06/'
for im_path in glob.glob(join(path,'noisy32.tif')):
    im = imageio.volread(im_path).astype('float32')
    for i in range(im.shape[0]):
        images.append(im[i,:,:])
images = np.stack(images)

print('%d images'%len(images))
            
norm = lambda im : (im / 255.0).reshape((512, 512, 1))
np_test_imgs = np.array([norm(im) for im in images])
print("np_test_imgs.shape",np_test_imgs.shape)

# Take a crop size, crop size chunk from the center of the image
np_test_imgs = np_test_imgs[:,(512//2)-args.crop//2:(512//2)+args.crop//2,(512//2)-args.crop//2:(512//2)+args.crop//2,:]
np_test_imgs = np.expand_dims(np_test_imgs,axis=0)

test_imgs = []
if args.bag is "True":
    # Get as many 'adjacent chunks' as possible. Could grab 'non-adjacent' chunks in the future
    # non-adjacent chunk - a chunk of images where not all images are stacked in the order gathered
    for i in range(np_test_imgs.shape[1]):
        if i + args.depth < np_test_imgs.shape[1]:
            curr_block = np_test_imgs[0,i:i+args.depth,:,:,:]
            test_imgs.append(curr_block)
            
else:
    # Split the test images in chunks of length args.depth with no overlapping images
    for i in range(0,np_test_imgs.shape[1]-1,args.depth):
        print("i",i)
        curr_block = np_test_imgs[0,i:i+args.depth,:,:,:]
        test_imgs.append(curr_block)

test_imgs = np.stack(test_imgs,axis=0)
print("test_imgs.shape",test_imgs.shape)

im_path = '../../jventu09/3d-denoising-dataset/3Ddata06/clean.tif'
gt = imageio.volread(im_path).astype('float32')
gt = gt[:32,(gt.shape[0]//2)-(args.crop//2):(gt.shape[0]//2)+(args.crop//2),(gt.shape[1]//2)-(args.crop//2):(gt.shape[1]//2)+(args.crop//2)]
print("gt.shape",gt.shape)
"""
mean_store and posterior_mean_store:
    These store the respective averages for each statistic

    i.e. the output posterior mean for each pixel in a frame is based on one 3d block's output from the network
    These values are added up over all blocks in the network and each frame's values are divided by the number
    of 3d blocks it appears in. This produces the average mean over all blocks for each frame.

    e.g. the first frame's output is divided by one because it appears in one 3d block and the second is divided
    by two.
"""
mean_store = np.zeros(shape=np_test_imgs.shape)
posterior_mean_store = np.zeros(shape=np_test_imgs.shape)


results_path = 'results.%s.%s.%s.csv'%(args.loss,args.dataset,args.noise)
with open(results_path,'w') as f:
    f.write('Noisy PSNR,Mode PSNR,MAP PSNR\n')
	# Iterate over each block in the test_imgs`
    for i, block in enumerate(test_imgs): 
        mode_out = np.zeros((args.depth,args.crop,args.crop),dtype='float32')
        map_out = np.zeros((args.depth,args.crop,args.crop),dtype='float32')

        pred = model.predict(block.reshape(1,args.depth,args.crop,args.crop,1))
        
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
            x_mean = approx_poisson_mean(log_loc,log_scale)
            x_posterior_mean = approx_poisson_posterior_mean(im*255,log_loc,log_scale)
            
        mean_out = np.clip(np.squeeze(x_mean),0,255)
        posterior_mean_out = np.clip(np.squeeze(x_posterior_mean),0,255)
        
        if args.bag is "True":
            for frame_index in range(mean_out.shape[0]):
                mean_out[frame_index,:,:] /= min(args.depth - abs(args.depth - (frame_index + i))+1,args.depth)
                posterior_mean_out[frame_index,:,:] /= min(args.depth - abs(args.depth - (frame_index + i))+1,args.depth)
                if min(args.depth - abs(args.depth - (frame_index + i)),args.depth) == 0:
                    print("frame_index",frame_index)
                    print("i",i)
            mean_store[0,i:i+args.depth,:,:,0] += (mean_out)
            posterior_mean_store[0,i:i+args.depth,:,:,0] += posterior_mean_out
			
			# Get results from each frame in the current block
            for frame_index in range(np_test_imgs.shape[1]):
                curr_img = np_test_imgs[0,frame_index,:,:]
                noisy = np.squeeze(curr_img)*255

                curr_mean_out = mean_store[0,frame_index,:,:,0]
                curr_posterior_mean_out = posterior_mean_store[0,frame_index,:,:,0]

                psnr_noisy = skimage.measure.compare_psnr(gt, noisy, data_range = 255)
                psnr_mean = skimage.measure.compare_psnr(gt, curr_mean_out, data_range = 255)
                psnr_posterior_mean = skimage.measure.compare_psnr(gt, curr_posterior_mean_out, data_range = 255)
                
                print(psnr_noisy,psnr_mean,psnr_posterior_mean)
                f.write('%.15f,%.15f,%.15f\n'%(psnr_noisy,psnr_mean,psnr_posterior_mean))
             
                imageio.imwrite(str(frame_index)+'in.png',noisy.astype('uint8'))
                imageio.imwrite(str(frame_index)+'mean.png',curr_mean_out.astype('uint8'))
                imageio.imwrite(str(frame_index)+'posterior.png',curr_posterior_mean_out.astype('uint8'))

        # Average over the images in each block
        else:
            psnr_noisy = 0
            psnr_mean = 0
            psnr_posterior_mean = 0
            for frame_index in range(block.shape[0]): 
                curr_img = block[frame_index,:,:]
                noisy = np.squeeze(curr_img)*255
                
                curr_gt = gt[frame_index,:,:]

                curr_mean_out = mean_out[frame_index,:,:]
                curr_posterior_mean_out = posterior_mean_out[frame_index,:,:]
                
                psnr_noisy += skimage.measure.compare_psnr(curr_gt, noisy, data_range = 255)
                psnr_mean += skimage.measure.compare_psnr(curr_gt, curr_mean_out, data_range = 255)
                psnr_posterior_mean += skimage.measure.compare_psnr(curr_gt, curr_posterior_mean_out, data_range = 255)
                
            print(psnr_noisy,psnr_mean,psnr_posterior_mean)
            f.write('%.15f,%.15f,%.15f\n'%(psnr_noisy,psnr_mean,psnr_posterior_mean))
             
            imsave(str(i)+'in.tiff',noisy.astype('uint8'),shape=noisy.shape)
            imsave(str(i)+'mean.tiff',curr_mean_out.astype('uint8'), shape=curr_mean_out.shape)
            imsave(str(i)+'posterior.tiff',curr_posterior_mean_out.astype('uint8'), shape=curr_posterior_mean_out.shape)

                
"""
if args.bag != "True":
    imageio.imwrite('gt.png',gt.astype('uint8'))
    results = np.loadtxt(results_path,delimiter=',',skiprows=1)
    print('averages:')
    print(np.mean(results,axis=0))
"""    
