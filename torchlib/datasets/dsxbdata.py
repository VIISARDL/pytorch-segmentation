

import os
import numpy as np

from torch.utils.data import Dataset

from pytvision.transforms.aumentation import  ObjectImageMaskAndWeightTransform, ObjectImageAndMaskTransform
from pytvision.datasets import utility

from .imageutl import dsxbExProvide, nucleiProvide2, TCellsProvide

import warnings
warnings.filterwarnings("ignore")



train = 'train'
validation = 'val'
test  = 'test'


class DSXBDataset(Dataset):
    '''
    Mnagement for Data Science Bowl image dataset
    https://www.kaggle.com/c/data-science-bowl-2018/data
    Args
    '''

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='touchs',
        folders_weights='weights',
        ext='png',
        num_channels=3,
        transform=None,
        ):
           
        self.data = dsxbExProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_contours,
                folders_weights,
                ext
                )

        self.transform = transform    
        self.num_channels = num_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   
        image, label, contours, weight = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )  
        label_t = np.zeros( (label.shape[0],label.shape[1],3) )
        label_t[:,:,0] = (label < 128)
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (contours > 128)

        weight_t = weight[:,:,np.newaxis]        

        obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()




class DSXBExDataset(Dataset):
    '''
    Mnagement for Data Science Bowl image dataset
    https://www.kaggle.com/c/data-science-bowl-2018/data
    '''

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='touchs',
        folders_weights='weights',
        ext='png',
        transform=None,
        count=1000,
        num_channels=3,
        ):
        """           
        """            
           
        self.data = dsxbExProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_contours,
                folders_weights,
                ext
                )


        self.transform = transform  
        self.count = count  
        self.num_channels = num_channels

    def __len__(self):
        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label, contours, weight = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )   

        label_t = np.zeros( (label.shape[0],label.shape[1],3) )
        label_t[:,:,0] = (label < 128)
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (contours > 128)

        weight_t = weight[:,:,np.newaxis]        

        obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()

    
    
class DSDataset(Dataset):
    '''Mnagement for Data Segmentation image dataset
    Args
    '''

    def __init__(self, 
        data,
        ext='png',
        num_channels=3,
        transform=None,
        ):
           
        self.data = data
        self.transform = transform    
        self.num_channels = num_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   
        image, label = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )  
        label_t = np.zeros( (label.shape[0],label.shape[1],2) )
        label_t[:,:,0] = (label < 1)
        label_t[:,:,1] = (label >= 1)            

        obj = ObjectImageAndMaskTransform( image_t, label_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()


class NucleiDataset(Dataset):

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='touchs',
        ext='png',
        transform=None,
        count=1000,
        num_channels=3,
        ):

        self.data = nucleiProvide2(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_contours,
                ext
                )


        self.transform = transform  
        self.count = count  
        self.num_channels = num_channels

    def __len__(self):
        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label, contours = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )   

        label_t = np.zeros( (label.shape[0], label.shape[1], 3) )
        label_t[:,:,0] = (label < 128)
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (contours > 128)     

        obj = ObjectImageAndMaskTransform( image_t, label_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()

    
class TCellsDataset(Dataset):

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='touchs',
        ext='png',
        transform=None,
        count=1000,
        num_channels=3,
        ):

        self.data = TCellsProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                ext
                )


        self.transform = transform  
        self.count = count  
        self.num_channels = num_channels

    def __len__(self):
        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )   

        label_t = (label > 127).astype(np.uint8)

        obj = ObjectImageAndMaskTransform( image_t, label_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()

class TCellsDataset2(Dataset):

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='touchs',
        ext='png',
        transform=None,
        count=1000,
        num_channels=3,
        ):

        self.data = TCellsProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                ext
                )


        self.transform = transform  
        self.count = count  
        self.num_channels = num_channels

    def __len__(self):
        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )   
        label_t = np.zeros_like( label )
        label_t[:,:,0] = (label[...,0] == 0)
        label_t[:,:,1] = (label[...,0] == 1)
        label_t[:,:,2] = (label[...,0] >= 2)     
        
        obj = ObjectImageAndMaskTransform( image_t, label_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()


class GenericDataset(Dataset):

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        ext='png',
        transform=None,
        count=1000,
        num_channels=3,
        use_weight=False,
        ):

        self.data = TCellsProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                ext,
                use_weight
                )


        self.transform    = transform  
        self.count        = count  
        self.num_channels = num_channels
        self.use_weight   = use_weight

    def __len__(self):
        if self.count is None:
            return len(self.data)

        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        data = self.data[idx]
        if self.use_weight:
            image, label, weight = data
        else:
            image, label = data
            
        label = (label == 255).astype(float)
        image_t = utility.to_channels(image, ch=self.num_channels )   
        
        
        if self.use_weight:
            obj = ObjectImageMaskAndWeightTransform(image_t, label, weight)
        else:
            obj = ObjectImageAndMaskTransform( image_t, label  )
        
        
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()