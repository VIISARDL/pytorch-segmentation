import os
import numpy as np

from torch.utils.data import Dataset

from pytvision.datasets.imageutl import dataProvide
from pytvision.transforms.aumentation import  ObjectImageMaskAndWeightTransform, ObjectImageAndMaskTransform
from pytvision.datasets import utility

import warnings
warnings.filterwarnings("ignore")


train = 'train'
validation = 'val'
test  = 'test'


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions, folder='images'):
    datas = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target, folder)
        if not os.path.isdir(d):
            continue
        for root, dirs, fnames in sorted(os.walk( d )):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(dir, target)
                    item = (path, fname, class_to_idx[target])
                    datas.append(item)
    return datas


class mixDSXBProvide( dataProvide ):
    '''
    Provide for Science Bowl dataset
    '''
    
    def __init__(self,
        base_folder,    
        sub_folder,     
        folders_image='images',
        folders_label='labels',
        folders_touchs='touchs',
        folders_weights='weights',
        pref_image='',
        pref_label='',
        ):
        super(mixDSXBProvide, self).__init__( )
        
        self.base_folder     = os.path.expanduser( base_folder )
        self.sub_folders     = sub_folder
        self.folders_image   = folders_image
        self.folders_label   = folders_label
        self.folders_touchs  = folders_touchs
        self.folders_weights = folders_weights
        self.data = []
        
        self.load_folders()


    def getimage(self,i):
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range')
        self.index = i
        path = self.data[i][0]
        name = self.data[i][1]
        
        #load image
        image = np.array(self._loadimage( os.path.join(path, self.folders_image ,name)  ), dtype=np.uint8)
        return image

    def getid(self): return self.data[self.index][0]
    

    def __getitem__(self, i):
                
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range')
        self.index = i

        #load data
        path = self.data[i][0]
        name = self.data[i][1]
        #y    = self.data[i][3]
        
        image  = np.array(self._loadimage( os.path.join(path, self.folders_image ,name)  ), dtype=np.uint8)
        label  = np.array(self._loadimage( os.path.join(path, self.folders_label ,name)  ), dtype=np.uint8)
        touch  = np.array(self._loadimage( os.path.join(path, self.folders_touchs ,name)  ), dtype=np.uint8)
        weight = np.loadtxt( os.path.join(path, self.folders_weights, name.split('.')[0] + '.txt' ), delimiter=",")
               
                
        return image, label, touch, weight
            

    def load_folders(self):
        '''load file patch for disk
        '''        
        self.data = []                
        pathname = os.path.join( self.base_folder, self.sub_folders )
        
        classes, class_to_idx = find_classes( pathname )
        data = make_dataset(pathname, class_to_idx, IMG_EXTENSIONS, folder='images' )
               
        self.data = data
        self.classes = classes
        self.class_to_idx = class_to_idx

        

        
class MIXDSXBExDataset(Dataset):
    '''
    Mnagement for Data Science Bowl image dataset
    https://www.kaggle.com/c/data-science-bowl-2018/data
    '''

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_touchs='touchs',
        folders_weights='weights',
        transform=None,
        count=1000,
        num_channels=3,
        ):
        """           
        """            
           
        self.data = mixDSXBProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_touchs,
                folders_weights
                )


        self.transform = transform  
        self.count = count  
        self.num_channels = num_channels

    def __len__(self):
        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label, touch, weight = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )   

        label_t = np.zeros( (label.shape[0],label.shape[1],3) )
        label_t[:,:,0] = (label < 128) #back
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (touch > 128)

        weight_t = weight[:,:,np.newaxis]        

        obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()

