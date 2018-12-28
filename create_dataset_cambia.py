
import os
import numpy as np

from skimage import io, transform
import scipy.misc
from scipy import ndimage as ndi
import cv2

from torchlib.datasets import imageutl as imutl
from torchlib.datasets import utility as utl
from torchlib import preprocessing as prep



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


def create_dataset( data, ntrain ):
    
    n = len(data)
    for i in range( n ):
        image, label = data[ i ]
        print('>> ', data.getid() )          
        bmask_t, bcontour_t, btouch_t, bcenters_t, weight_t = prep.preprocessing( label )

        save_item( 
            i, 
            os.path.join(pathnameout,'train' if i < int(ntrain*n/100) else 'test' ), 
            image, 
            bmask_t, 
            bcontour_t, 
            bcenters_t, 
            btouch_t, 
            weight_t, 
            'k' 
            ) 
    

if __name__ == '__main__':

    pathdataset = "~/.datasets/"
    namedataset = 'cambia'
    pathname = os.path.join( os.path.expanduser(  pathdataset ), namedataset);
    pathnameout = '~/.datasets/'
    namedatasetout = 'cambiaext'
    pathnameout = os.path.join( os.path.expanduser( pathnameout ) , namedatasetout)
    folders_images  = 'images'
    folders_labels  = 'labels'
    sub_folder = ''
    ntrain=80 # in percent

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
    data = imutl.cambiaProvide.create(
        pathname, 
        sub_folder, 
        folders_images, 
        folders_labels,
        )

    print('>> ','loader ok :)!!!')
    print('>> ',len(data))
    
    create_dataset( data, ntrain )
    print('Done!!!!')