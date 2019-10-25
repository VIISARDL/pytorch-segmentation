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
    

def create_dataset( data, ntrain, imsize ):
    
    n = len(data)
    for i in range( n ):
        image, label = data[ i ]
        print('>> ', data.getid()[:4] )       
        image_t, label_t, bmask_t, bcontour_t, btouch_t, bcenters_t, weight_t = preprocessing( image, label, imsize )
        save_item( 
            i, 
            os.path.join(pathnameout,'train' if i < int(ntrain*n/100) else 'test' ), 
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
    namedatasetout = 'datasciencebowlex'
    pathnameout = os.path.join( os.path.expanduser( pathnameout ) , namedatasetout)
    folders_images  = 'images'
    folders_labels  = 'masks'
    id_file_name   = 'stage1_train_labels.csv'
    sub_folder = 'train/images'
    ntrain=80 # in percent
    imsize=200
    
    print('>> ', 'Create dir: ',  pathnameout)
    if os.path.exists(pathnameout) is not True:
        os.makedirs(pathnameout);
        os.makedirs(os.path.join(pathnameout,'train'));
        os.makedirs(os.path.join(pathnameout,'test'));
        #train
        os.makedirs(os.path.join(pathnameout,'train','images'));
        os.makedirs(os.path.join(pathnameout,'train','labels'));
        os.makedirs(os.path.join(pathnameout,'train','weights'));
        os.makedirs(os.path.join(pathnameout,'train','contours'));
        os.makedirs(os.path.join(pathnameout,'train','centers'));
        os.makedirs(os.path.join(pathnameout,'train','touchs'));
        #test
        os.makedirs(os.path.join(pathnameout,'test','images'));
        os.makedirs(os.path.join(pathnameout,'test','labels'));
        os.makedirs(os.path.join(pathnameout,'test','weights'));
        os.makedirs(os.path.join(pathnameout,'test','contours'));
        os.makedirs(os.path.join(pathnameout,'test','centers'));
        os.makedirs(os.path.join(pathnameout,'test','touchs'));

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
    
    create_dataset( data, ntrain, imsize )
    print('Done!!!!')