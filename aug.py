
import cv2
from torchvision import transforms
from pytvision.transforms import transforms as mtrans


# transformations 
#normalize = mtrans.ToMeanNormalization(
#    #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
#    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
#    )

# cifar10
# normalize = mtrans.ToMeanNormalization(
#     mean = (0.4914, 0.4822, 0.4465), #[x / 255 for x in [125.3, 123.0, 113.9]],
#     std  = (0.2023, 0.1994, 0.2010), #[x / 255 for x in [63.0, 62.1, 66.7]],
#     )

# cifar100
#normalize = mtrans.ToMeanNormalization(
#    mean = [x / 255 for x in [129.3, 124.1, 112.4]],
#    std = [x / 255 for x in [68.2, 65.4, 70.4]],
#    )

# svhn
#normalize = mtrans.ToMeanNormalization(
#    mean = [x / 255 for x in [127.5, 127.5, 127.5]],
#    std = [x / 255 for x in [127.5, 127.5, 127.5]],
#    )

#resnet
normalize = mtrans.ToMeanNormalization(
    mean = (0.485, 0.456, 0.406),  
    std  = (0.229, 0.224, 0.225), 
    )


# normalize = mtrans.ToNormalization()

def get_transforms_aug( size_input=256, size_crop=512 ):        
    return transforms.Compose([
        
        #------------------------------------------------------------------
        #Resize
        #             
        mtrans.RandomCrop( (size_crop, size_crop), limit=10, padding_mode=cv2.BORDER_REFLECT_101  ),        
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REFLECT_101 ),
        #mtrans.ToPad( 5 , 5, padding_mode=cv2.BORDER_REFLECT_101 ),
                       
        #------------------------------------------------------------------
        #Geometric 
        
        mtrans.ToRandomTransform( mtrans.VFlip(), prob=0.5 ),
        mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 ),
        mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REFLECT_101 ), 
        mtrans.ToRandomTransform( mtrans.RandomGeometricalTransform( angle=45, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REFLECT_101 ), prob=0.5 ),
        mtrans.ToRandomTransform( mtrans.RandomElasticDistort( size_grid=32, deform=12, padding_mode=cv2.BORDER_REFLECT_101 ), prob=0.5 ),
        #mtrans.ToResizeUNetFoV(imsize, cv2.BORDER_REFLECT_101),
        
        
        #------------------------------------------------------------------
        #Colors 
        
        mtrans.ToRandomTransform( mtrans.RandomBrightness( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomContrast( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomGamma( factor=0.25 ), prob=0.50 ),
        #mtrans.ToRandomTransform( mtrans.RandomRGBPermutation(), prob=0.50 ),
        #mtrans.ToRandomTransform( mtrans.CLAHE(), prob=0.25 ),
        #mtrans.ToRandomTransform(mtrans.ToGaussianBlur( sigma=0.05 ), prob=0.25 ),
        
        #------------------------------------------------------------------
        mtrans.ToTensor(),
        normalize,
        
        ])    
    


def get_transforms_det(size_input=256, size_crop=512):    
    return transforms.Compose([   
        mtrans.RandomCrop( (size_crop, size_crop), limit=10, padding_mode=cv2.BORDER_REFLECT_101  ),
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REFLECT_101  ),
        #mtrans.ToPad( 5 , 5, padding_mode=cv2.BORDER_REFLECT_101 ),        
        mtrans.ToTensor(),
        normalize,
        ])


def get_transforms_test(size_input=256):    
    return transforms.Compose([   
        #mtrans.RandomCrop( (size_crop, size_crop), limit=10, padding_mode=cv2.BORDER_REFLECT_101  ),
        mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REFLECT_101  ),
        #mtrans.ToPad( 5 , 5, padding_mode=cv2.BORDER_REFLECT_101 ),        
        mtrans.ToTensor(),
        normalize,
        ])