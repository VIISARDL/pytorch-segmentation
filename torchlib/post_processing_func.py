import numpy as np
from skimage.morphology import label,remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from scipy import ndimage
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from skimage.morphology import watershed

import matplotlib.pyplot as plt
import matplotlib as mpl
def myplot(image):
    fig, ax=plt.subplots()
    plt3=ax.matshow(image,cmap=plt.get_cmap('jet'))
    fig.colorbar(plt3)
    plt.show()

def prob42prob3(prob,assign='max'):
    if prob.shape[2]==4:
        if assign=='max':
            # assign gap to max
            map3=np.argmax( (prob[:,:,:3]*255).astype(int),axis=2)
            for m3i in range(3):
                prob[:,:,m3i]=prob[:,:,m3i]+prob[:,:,3]*(map3==m3i).astype(int)
            prob=prob[:,:,:3]
        elif assign=='back':
            # assign gap to back
            prob[:,:,0]=prob[:,:,0]+prob[:,:,3]
            prob=prob[:,:,:3]
        elif assign=='unif':
            # distribute gap to equally
            for m3i in range(3):
                prob[:,:,m3i]=prob[:,:,m3i]/(1 - prob[:,:,3] + np.finfo(float).eps)
            prob=prob[:,:,:3]      
    return prob

def sem2inst(output):

    prediction=np.array(output==1).astype(int)
    predictionlb=label(prediction,connectivity=1)   

    if output.max()>1:
        divis=np.array(output==2).astype(int)

        ccomp=np.array( (prediction+divis)>0 ).astype(int)
        ccomp=label(ccomp,connectivity=1)

        numcc=np.max(ccomp)
        numcell=np.max(predictionlb)

        for cp in range(1,numcc+1):
            o1=np.array((ccomp==cp) & (prediction==1)).astype(int)
            o2=np.array((ccomp==cp) & (divis==1)).astype(int)
            if o2.sum()>0:
                (idx_gamma_x,idx_gamma_y)   = ndimage.distance_transform_edt(1-o1,return_distances=False,return_indices=True)
                predictionlb[o2==1]= predictionlb[idx_gamma_x[o2==1],idx_gamma_y[o2==1]]

    cellscount=np.max(predictionlb)
    predictionlbnew=np.zeros_like(predictionlb)
    for i in range(1,cellscount+1):
        cell= np.array( (predictionlb ==i) ).astype(int)
        cell= binary_fill_holes(cell)
        lcell=label(cell)
        if lcell.max()>1:
            props=regionprops(lcell)
            marea = np.argmax( np.array([p.area for p in props ]) )
            cell= (lcell== (marea+1))
        predictionlbnew[cell]=i      

    return predictionlbnew
        
def inst2sem(label):
    touch=np.zeros_like(label)
    
    fore=np.array(label>0).astype(np.uint8)
    contour_in=fore-ndimage.binary_erosion(fore)

    touch=find_boundaries(label,mode='inner',background=0,connectivity=2).astype(np.uint8)*(1-contour_in).astype(np.uint8)
    touch=remove_small_objects(touch==1,10,connectivity=1).astype(np.uint8)
    touch=ndimage.binary_dilation(touch)*np.array(label!=0).astype(np.uint8)

    region3=np.array(label>0).astype(np.uint8)+ touch
    region=np.array(label>0).astype(np.uint8)- touch
    
    return region,region3

class TH_post(object):
    def __init__(self,cellclass=1,diviclass=2):
        self.cellclass=cellclass
        self.diviclass=diviclass

    def __call__(self,output,thresh=0.5,threshdiv=0.1,morph=0,prob4='max'):

        output=prob42prob3(output,prob4)

        if output.shape[2]>1:
            cellsprob=output[:,:,self.cellclass]
        else:
            cellsprob=output

        prediction=np.array(cellsprob>thresh).astype(int)
        diff= np.zeros_like(prediction)
        if morph>0:
            prediction1=ndimage.binary_opening(prediction,structure=np.ones((morph,morph)), iterations=2 ).astype(int)
            diff= prediction-prediction1
            prediction=prediction1

        if output.shape[2]>2:
            divis=output[:,:,self.diviclass]+diff
            divis=np.array(divis>=threshdiv).astype(int)

            prediction[divis.astype(bool)]=0
            prediction=remove_small_objects(prediction.astype(bool),50,connectivity=1).astype(int)
            prediction=binary_fill_holes(prediction).astype(int)

            divis[prediction.astype(bool)]=0
            prediction=self.cellclass*prediction+self.diviclass*divis

        predictionlb=sem2inst(prediction)
        region,prediction=inst2sem(predictionlb)
        
        return predictionlb, prediction,region,output


class MAP_post(object):
    def __init__(self,cellclass=1,diviclass=2):
        self.cellclass=cellclass
        self.diviclass=diviclass

    def __call__(self,output,morph=0,prob4='max'):
        output=prob42prob3(output,prob4)

        MAP=np.argmax(output, axis=2)
        if np.max(MAP)==3:
            MAP[MAP==3]=0

        prediction=(MAP==self.cellclass).astype(int)
        back=(MAP==0).astype(int)
        
        diff= np.zeros_like(prediction)
        if morph>0:
            prediction1=ndimage.binary_opening(prediction,structure=np.ones((morph,morph)), iterations=2 ).astype(int)
            diff= prediction-prediction1
            prediction=prediction1

        if output.shape[2]>2:
            divis=(MAP==self.diviclass).astype(int)+diff
            MAP[divis==1]=self.diviclass
        
        predictionlb=sem2inst(MAP)
        region,MAP=inst2sem(predictionlb)

        return predictionlb,MAP,region, output

from skimage.feature import peak_local_max
#from scipy.misc import imsave
class WTS_post(object):
    def __init__(self,cellclass=1,diviclass=2):
        self.cellclass=cellclass
        self.diviclass=diviclass

    def __call__(self,output,thresh_background=0.9,thresh_foreground=0.95,prob4='max'):
        output=prob42prob3(output,prob4)

        if output.shape[2]>1:
            cellsprob=output[:,:,self.cellclass]
            backprop=output[:,:,0]
            divisprop=output[:,:,self.diviclass]
        else:
            cellsprob=output
            backprop=1-output
            divisprop=np.zeros_like(cellsprob)

        foreground_seed=np.array(cellsprob>thresh_foreground).astype(int)
        background_seed=np.array(backprop>thresh_background).astype(int)
        markers= ndimage.label( foreground_seed )[0]

        distance= cellsprob-divisprop
        distance[distance<0]=0
        distance=(distance*255).astype(int)

        predictionlb=watershed(-distance, markers, mask=(background_seed!=1))
        cellscount=np.max(predictionlb)
        predictionlbnew=np.zeros_like(predictionlb)
        for i in range(1,cellscount+1):
            cell= np.array( (predictionlb ==i) ).astype(int)
            cell= binary_fill_holes(cell)
            lcell=label(cell)
            if lcell.max()>1:
                props=regionprops(lcell)
                marea = np.argmax( np.array([p.area for p in props ]) )
                cell= (lcell== (marea+1))
            predictionlbnew[cell]=i   
            
        region,prediction=inst2sem(predictionlbnew)
        
        return predictionlbnew, prediction,region, output
