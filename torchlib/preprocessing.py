
import os
import numpy as np

import cv2
import random
import csv
import h5py

import scipy.misc
from scipy import ndimage
from skimage import io, transform, filters
from skimage import morphology as morph
from skimage import color
import skfmm

from .datasets import weightmaps 
from . import utils


def get_contour(img):    
    img = img.astype(np.uint8)
    edge = np.zeros_like(img)
    _,cnt,_ = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )
    cv2.drawContours( edge, cnt, -1, 1 , 1)
    edge = (edge>0).astype(np.uint8)    
    return edge

def get_center(img):
    cent = np.zeros_like(img).astype(np.uint8)
    y, x = ndimage.measurements.center_of_mass(img)
    cv2.circle(cent, (int(x), int(y)), 1, 1, -1)
    cent = (cent>0).astype(np.uint8) 
    return cent

def get_distance(x):
    return skfmm.distance((x).astype('float32') - 0.5) 

def get_touchs( edges ):       
    A = np.array([ morph.binary_dilation( c, morph.square(3) )  for c in edges ]) 
    A = np.sum(A,axis=0)>1  
    I = morph.remove_small_objects( A, 3 )
    I = morph.skeletonize(I)
    I = morph.binary_dilation( I, morph.square(3) )    
    return I


def create_groundtruth( masks ): 
    #processing mask
    masks = (masks > 0).astype(np.uint32)
    #masks =  ndimage.morphology.binary_fill_holes( masks )
    masks = utils.decompose( utils.tolabel( masks ) )
        
    #get features
    edges      = np.array([ morph.binary_dilation(get_contour(x)) for x in masks ])       
    bmask      = utils.tobinary(masks).astype( np.uint8 )    
    btouch     = get_touchs( edges ).astype( np.uint8 )*255
    bcontour   = utils.tobinary(edges).astype( np.uint8 ) 
    centers    = np.array([ morph.binary_dilation(get_center(x)) for x in masks ]) 
    bcenters   = utils.tobinary(centers).astype( np.uint8 ) 

    return masks, bmask, bcontour, btouch, bcenters


def preprocessing( label ):    
    # preprocessing
    masks, bmask, bcontour, btouch, bcenters = create_groundtruth( label )
    weight = weightmaps.getunetweightmap( (bmask>0) + 2*(btouch>0), masks, w0=10, sigma=5, )
    return bmask, bcontour, btouch, bcenters, weight


