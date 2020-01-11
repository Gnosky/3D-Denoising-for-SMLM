"""
TODO:

"""
import random
import sys
import twodblindspot.model as twod_model
import threedblindspot.model as threed_model
import numpy as np
import skimage

from os import listdir
from os.path import join
import imageio
import glob
from tqdm import trange
import argparse
from tifffile import imsave

np.random.seed(1)
random.seed(1)


parser = argparse.ArgumentParser()
parser.add_argument('--loss',default='mse',help='loss function (mse,gamma, or softmax)')
parser.add_argument('--dataset',default='data_06',help='dataset name e.g. Confocal_MICE')
parser.add_argument('--noise',default='raw',help='noise level (raw, avg2, avg4, ...)')
parser.add_argument('--crop',type=int,default=32,help='crop size (0 for no crop)')
parser.add_argument('--depth',type=int,default=32)
parser.add_argument('--bag',default="False")
parser.add_argument('--network_dimension',type=int,default=3)
parser.add_argument('--test_names_glob', default = 'noisy6[0-9].tif')
parser.add_argument('--dir_path', default = '../../jventu09/3d-denoising-dataset/3Ddata06/')
args = parser.parse_args()

norm = lambda im : (im / 255.0).reshape((512, 512, 1))
class Network():
    def __init__(self, arguments):
        self.params = arguments
        self.network = None
        self.dir_path = arguments.dir_path
        self.test_names = glob.glob(join(self.dir_path,arguments.test_names_glob))
        self.test_mean = None
        self.test_std = None
        self.train_mean = None
        self.train_std = None
        self.test_imgs = None
        self.gt = None
        self.model = None
    # Get mean and std for each image in the stack that will be denoised 
    def _get_test_statistics(self):
        test_images = []
        for im_path in self.test_names:
            im = imageio.volread(im_path).astype('float32')
            for i in range(im.shape[0]):
                test_images.append(im[i,:,:])
        test_images = np.stack(test_images,axis=0)

        np_test_imgs = np.array([norm(im) for im in test_images])
        np_test_imgs = np.expand_dims(np_test_imgs,0)

        test_mean = np.mean(np_test_imgs)
        test_std = np.std(np_test_imgs)
        print("test_mean, test_std", test_mean,test_std)

        self.test_mean = test_mean
        self.test_std = test_std

        del np_test_imgs
        del test_images
	
    # Get training mean and std from the batch that this data was trained on	
    def _get_train_statistics(self):
        mean_std_path = 'data_06/meanstd.%s.%s.%s.%s.%s_crop.npz'%(self.params.network_dimension,self.params.loss,self.params.dataset,self.params.noise,self.params.crop)
        mean_std = np.load(mean_std_path)
        self.train_mean = mean_std['train_mean']
        self.train_std = mean_std['train_std']
        print("train_mean,train_std",self.train_mean,self.train_std)

    def _create_2d_model(self):
        crop_size = self.params.crop
        train_mean = self.train_mean
        train_std = self.train_std

        if self.params.loss == 'mse':
            model = twod_model.mse_blindspot_network((crop_size, crop_size, 1), train_mean, train_std)
        elif self.params.loss == 'gamma':
            model = twod_model.gamma_blindspot_network((crop_size, crop_size, 1), train_mean, train_std)
        elif self.params.loss == 'approx_poisson':
            model = twod_model.approx_poisson_blindspot_network((crop_size, crop_size, 1), train_mean, train_std)
        elif self.params.loss == 'gaussian':
            model = twod_model.gaussian_blindspot_network((crop_size, crop_size, 1), train_mean, train_std)
        else:
            raise ValueError('unknown loss')
        
        weights_path = 'data_06/weights.%s.%s.%s.%s.%s_crop.latest.hdf5'%(self.params.network_dimension,self.params.loss,self.params.dataset,self.params.noise,self.params.crop)
        model.load_weights(weights_path)
        self.model = model

    def _create_3d_model(self):
        depth = self.params.depth
        crop_size = self.params.crop
        train_mean = self.train_mean
        train_std = self.train_std

        if self.params.loss == 'mse':
            model = threed_model.mse_blindspot_network((depth, crop_size, crop_size, 1), train_mean, train_std)
        #elif self.params.loss == 'gamma':
#	model = gamma_blindspot_network((depth, crop_size, crop_size, 1), train_mean, train_std)
        #elif self.params.loss == 'approx_poisson':
#	model = approx_poisson_blindspot_network((depth, crop_size, crop_size, 1), train_mean, train_std)
        elif self.params.loss == 'gaussian':
            model = threed_model.gaussian_blindspot_network((depth, crop_size, crop_size, 1), train_mean, train_std)
        else:
            raise ValueError('unknown loss')
        weights_path = 'data_06/weights.%s.%s.%s.%s.%s_crop.latest.hdf5'%(self.params.network_dimension,self.params.loss,self.params.dataset,self.params.noise,self.params.crop)
        model.load_weights(weights_path)
        self.model = model
    # Load images that will be denoised
    def _load_test_images(self):
        images = []
        for im_path in self.test_names:
            im = imageio.volread(im_path).astype('float32')
            for i in range(im.shape[0]):
                images.append(im[i,:,:])
        images = np.stack(images)

        print('%d images'%len(images))
        
        # Put images in correct format			
        norm = lambda im : (im / 255.0).reshape((512, 512, 1))
        np_test_imgs = np.array([norm(im) for im in images])

        # Take a crop size, crop size chunk from the center of the image
        np_test_imgs = np_test_imgs[:,(512//2)-self.params.crop//2:(512//2)+self.params.crop//2,(512//2)-self.params.crop//2:(512//2)+self.params.crop//2,:]
        
        if self.params.network_dimension == 3:
            np_test_imgs = np.expand_dims(np_test_imgs, axis=0)
            # Split images into chunks of length self.params.depth
            np_test_imgs = np.array_split(np_test_imgs, np_test_imgs.shape[1]//self.params.depth, axis=1)
        elif self.params.network_dimension == 2:
            # Split images into chunks of length 1
            np_test_imgs = np.array_split(np_test_imgs, np_test_imgs.shape[0], axis=0)
        self.test_imgs = np_test_imgs
        
    def get_ground_truth(self):	
        im_path = '../jventu09/3d-denoising-dataset/3Ddata06/clean.tif'
        gt = imageio.volread(im_path).astype('float32')
        gt = gt[:32,(gt.shape[1]//2)-(self.params.crop//2):(gt.shape[1]//2)+(self.params.crop//2),(gt.shape[2]//2)-(self.params.crop//2):(gt.shape[2]//2)+(self.params.crop//2)]
        
        return gt

    # Denoise on one block of images	
    def denoise(self, image_block):
        # Get network output
        pred = self.model.predict(image_block)
        # Apply denoising procedure if one is necessary for a given loss function
        if self.params.loss == 'mse':
            x_mean = pred[0]*255
            x_posterior_mean = pred[0]*255
            """
                elif self.params.loss == 'gamma':
            log_alpha = pred[0][0]
            log_beta = pred[1][0]
            x_mean = gamma_mean(log_alpha,log_beta)
            x_posterior_mean = gamma_posterior_mean(im*255,log_alpha,log_beta)
            #x_posterior_mean = gamma_MAP(im*255,log_alpha,log_beta)
        elif self.params.loss == 'approx_poisson':
            log_loc = pred[0][0]
            log_scale = pred[1][0]
            x_mean = approx_poisson_mean(log_loc,log_scale)
            x_posterior_mean = approx_poisson_posterior_mean(im*255,log_loc,log_scale)"""
        elif self.params.loss == 'gaussian':
            x_mean = pred[0]*255
            x_posterior_mean = pred[0]*255
        mean_out = np.squeeze(x_mean)
        posterior_mean_out = np.squeeze(x_posterior_mean)
        return mean_out, posterior_mean_out	
    
    def load_model_and_statistics(self):
        self._get_train_statistics()
        if self.params.network_dimension == 3:
            self._create_3d_model()
        elif self.params.network_dimension == 2:
            self._create_2d_model()
        else:
            raise ValueError('Unsupported model dimension, only 2 or 3 accepted')
    
    def get_test_images(self):
        self._load_test_images()
        return self.test_imgs
# Instantiate and prepare model, images, and statistics
net = Network(args)
net.load_model_and_statistics()
test_imgs = net.get_test_images()
gt = net.get_ground_truth()

results_path = 'data_06/results.%s.%s.%s.%s.csv'%(args.network_dimension,args.loss,args.dataset,args.noise)
with open(results_path,'w') as f:
    f.write('Results for images starting at stack %s\n'%('\n'.join(net.test_names))) 
    f.write('Frame Number, Noisy PSNR,Mode PSNR,MAP PSNR\n')
    print("len(test_imgs)",len(test_imgs))
    # Iterate over each block in the test_imgs`
    for i, block in enumerate(test_imgs):
        
        # Denoise images in block. If 3d network this will be of depth args.depth, if 2d it will be one frame 
        mean_out, posterior_mean_out = net.denoise(block)
        
        block = np.squeeze(block)
        block = np.expand_dims(block,axis=0) if args.network_dimension == 2 else block
        mean_out = np.expand_dims(mean_out,axis=0) if args.network_dimension == 2 else mean_out
        posterior_mean_out = np.expand_dims(posterior_mean_out,axis=0) if args.network_dimension == 2 else posterior_mean_out
         
        for frame_index in range(block.shape[0]):
            curr_img = block[frame_index,:,:]
            noisy = np.squeeze(curr_img)*255
            curr_gt = gt[frame_index,:,:]
            curr_mean_out = mean_out[frame_index,:,:]
            curr_posterior_mean_out = posterior_mean_out[frame_index,:,:]
            # Evalute denoising
            psnr_noisy = skimage.measure.compare_psnr(curr_gt, noisy, data_range = 255)
            psnr_mean = skimage.measure.compare_psnr(curr_gt, curr_mean_out, data_range = 255)
            psnr_posterior_mean = skimage.measure.compare_psnr(curr_gt, curr_posterior_mean_out, data_range = 255)
            
            f.write('%s,%.15f,%.15f,%.15f\n'%(frame_index+(args.depth*i),psnr_noisy,psnr_mean,psnr_posterior_mean))
        
        new_block = block*255
        imsave('data_06/%s_chunk_frame_%d_in.tiff'%(args.loss,i),new_block.astype('int16'), shape=new_block.shape)
        imsave('data_06/%s_chunk_frame_%d_mean.tiff'%(args.loss,i),mean_out.astype('int16'),shape=mean_out.shape)
        imsave('data_06/%s_chunk_frame_%d_posterior.tiff'%(args.loss,i),posterior_mean_out.astype('int16'),shape=posterior_mean_out.shape)

