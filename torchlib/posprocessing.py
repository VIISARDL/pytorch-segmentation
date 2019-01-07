
import numpy as np
import skimage.morphology as morph
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi

from . import utils


def fill_holes_per_blob(image):
    image_cleaned = np.zeros_like(image)
    for i in range(1, image.max() + 1):
        mask = np.where(image == i, 1, 0)
        mask = ndi.morphology.binary_fill_holes(mask)
        image_cleaned = image_cleaned + mask * i
    return image_cleaned


def postprocess(score):
    
    #prediction
    pred  = np.argmax( score, axis=2 ) == 1
    pred = morph.opening(pred, morph.disk(3))
    mk, _ = ndi.label(pred)
    
    score_prob = utils.sigmoid( score[:,:,1] )
    m_thresh = threshold_otsu(score_prob)
    m_back = score_prob > m_thresh
    
    distance = ndi.distance_transform_edt(pred)
    labels = morph.watershed(-distance,  mk, mask=m_back  )
    
    labels = fill_holes_per_blob(labels)
    labels = utils.decompose(labels)
    
    return labels