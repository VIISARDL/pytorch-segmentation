import os
import numpy as np

from skimage import io, transform
from skimage import morphology as morph
import scipy.misc
from scipy import ndimage as ndi
import cv2

from torchlib.datasets import imageutl as imutl
from torchlib.datasets import utility as utl
from torchlib.datasets import weightmaps 
from torchlib import preprocessing as prep
from torchlib import utils


def preprocessing( image, label, imsize ):
    image, label, masks = utils.imrandomcrop(image, label, (imsize,imsize) )
    edges      = np.array([ morph.binary_dilation(prep.get_contour(x)) for x in masks ])       
    bmask      = utils.tobinary( masks ).astype( np.uint8 )    
    btouch     = prep.get_touchs( edges ).astype( np.uint8 )*255
    bcontour   = utils.tobinary( edges ).astype( np.uint8 ) 
    centers    = np.array([ morph.binary_dilation(prep.get_center(x)) for x in masks ]) 
    bcenters   = utils.tobinary( centers ).astype( np.uint8 )    
    weight = weightmaps.getunetweightmap( (bmask>0) + 2*(btouch>0), masks, w0=10, sigma=5, )
    return image, label, bmask, bcontour, btouch, bcenters, weight
    

def save_item(
    i, 
    pathname, 
    image_t, 
    label_t, 
    contours_t,
    centers_t,
    touchs_t,
    weight_t, 
    prefijo=''):   

    scipy.misc.imsave(os.path.join(pathname, 'images'   ,'{}{:06d}.png'.format(prefijo,i)), image_t )
    scipy.misc.imsave(os.path.join(pathname, 'labels'   ,'{}{:06d}.png'.format(prefijo,i)), label_t )
    scipy.misc.imsave(os.path.join(pathname, 'contours' ,'{}{:06d}.png'.format(prefijo,i)), contours_t )
    scipy.misc.imsave(os.path.join(pathname, 'centers'  ,'{}{:06d}.png'.format(prefijo,i)), centers_t )
    scipy.misc.imsave(os.path.join(pathname, 'touchs'   ,'{}{:06d}.png'.format(prefijo,i)), touchs_t )
    np.savetxt(os.path.join(pathname, 'weights', '{}{:06d}.txt'.format(prefijo,i)), weight_t, fmt="%2.3f", delimiter=",")

    print('>>', os.path.join(pathname, '{}{:06d}'.format(prefijo, i)) )
    

def create_dataset( data, files, g, num, ktrain, imsize ):
    
    for i in range( num ):
        k = i%(len(files))
        image, label = data[ files[k] ]
        print('>> ', data.getid()[:4] )
        image_t, label_t, bmask_t, bcontour_t, btouch_t, bcenters_t, weight_t = preprocessing( image, label, imsize )
        save_item(
            i,
            os.path.join(pathnameout, 'train' if i < int(ktrain*num/100) else 'test', g ), 
            image_t,
            bmask_t,
            bcontour_t,
            bcenters_t,
            btouch_t,
            weight_t,
            'd'
            )

if __name__ == '__main__':

    pathdataset = "~/.datasets/"
    namedataset = 'datasciencebowl'
    pathname = os.path.join( os.path.expanduser( pathdataset ), namedataset)
    pathnameout = '~/.datasets/'
    namedatasetout = 'datasciencebowlexsubclass'
    pathnameout = os.path.join( os.path.expanduser( pathnameout ) , namedatasetout)
    folders_images  = 'images'
    folders_labels  = 'masks'
    id_file_name   = 'stage1_train_labels.csv'
    sub_folder = 'train/images'
    ktrain=80 # in percent
    imsize=200
    num=100
    
    print('>> ', 'Create dir: ',  pathnameout)
    if os.path.exists(pathnameout) is not True:
        os.makedirs(pathnameout);
        os.makedirs(os.path.join(pathnameout,'train'));
        os.makedirs(os.path.join(pathnameout,'test'));

    print('>> ', 'loading dataset ...')
    data = imutl.dsxbProvide.create(
        pathname, 
        sub_folder, 
        id_file_name,
        folders_images, 
        folders_labels,
        )

    print('>> ','loader ok :)!!!')
    print('>> ',len(data))
    
    grups  = ['a','b','c','d','e','f','g']
        
    for g in grups:
        files = np.loadtxt('misc/grup_{}.txt'.format(g), delimiter=",")
        files = np.array(files).astype(np.uint32)
                
        if os.path.exists(os.path.join(pathnameout,'train',g)) is not True:
            #train
            os.makedirs(os.path.join(pathnameout,'train',g,'images'));
            os.makedirs(os.path.join(pathnameout,'train',g,'labels'));
            os.makedirs(os.path.join(pathnameout,'train',g,'weights'));
            os.makedirs(os.path.join(pathnameout,'train',g,'contours'));
            os.makedirs(os.path.join(pathnameout,'train',g,'centers'));
            os.makedirs(os.path.join(pathnameout,'train',g,'touchs'));
            #test
            os.makedirs(os.path.join(pathnameout,'test',g,'images'));
            os.makedirs(os.path.join(pathnameout,'test',g,'labels'));
            os.makedirs(os.path.join(pathnameout,'test',g,'weights'));
            os.makedirs(os.path.join(pathnameout,'test',g,'contours'));
            os.makedirs(os.path.join(pathnameout,'test',g,'centers'));
            os.makedirs(os.path.join(pathnameout,'test',g,'touchs'));
    
        
        print('Select files')
        print(files)
        
        create_dataset( data, files, g, num, ktrain, imsize )
    
    
    print('Done!!!!')