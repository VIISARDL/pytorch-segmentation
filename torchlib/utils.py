

import numpy as np
from scipy import ndimage
from skimage import io, transform, filters
from skimage import morphology as morph
from itertools import product


def summary(data):
    print(np.min(data), np.max(data), data.shape)

def norm(x):
    x = x-x.min()
    x = x / x.max()
    return x

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def onehot2label( labels ):
    #classe 0 back
    m,n,c = labels.shape
    mlabel = np.zeros( (m,n) )
    for i in range(c):
        mlabel += labels[:,:,i]*(i+1)
    return mlabel

def tolabel(mask):
    labeled, nr_true = ndimage.label(mask)
    return labeled

def tobinary( x ):
    return (np.max(x,axis=0)>0)*255

def relabel(img):
    h, w = img.shape
    relabel_dict = {}
    for i, k in enumerate(np.unique(img)):
        if k == 0:
            relabel_dict[k] = 0
        else:
            relabel_dict[k] = i
    for i, j in product(range(h), range(w)):
        img[i, j] = relabel_dict[img[i, j]]
    return img

def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)
    if not masks: return np.array([labeled])
    else: return np.array(masks)

def drop_small(img, min_size):
    img = morph.remove_small_objects(img, min_size=min_size)
    return relabel(img)

def delete_black_layer( masks ):    
    newmasks = []
    for mask in masks:
        if mask.sum() != 0:
            newmasks.append(mask)             
    return newmasks








#---------------------------------------------------------------------------
# RunLen code and decoder 

# def rle_encode(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     pixels = img.flatten()
#     pixels[0] = 0
#     pixels[-1] = 0
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
#     runs[1::2] = runs[1::2] - runs[:-1:2]
#     return ' '.join(str(x) for x in runs)


def rle_encode(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    #if len(rle) != 0 and rle[-1] + rle[-2] == x.size:
    #    rle[-2] = rle[-2] - 1

    return rle


# def run_decode(rle, H, W, fill_value=255):
    
#     mask = np.zeros((H * W), np.uint8)
#     rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
#     for r in rle:
#         start = r[0]-1
#         end = start + r[1]
#         mask[start : end] = fill_value
#     mask = mask.reshape(W, H).T # H, W need to swap as transposing.
#     return mask


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle #mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


