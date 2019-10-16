import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
import scipy.ndimage as ndi
from .utils import decompose


def iou(gt, pred):
    gt[gt > 0] = 1.
    pred[pred > 0] = 1.
    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    union = np.sum(union)
    if union == 0:
        union = 1e-09
    return intersection / union

def compute_ious(gt, predictions):
    gt_ = gt
    predictions_ = predictions
    #gt_ = decompose(gt)
    #predictions_ = decompose(predictions)
    gt_ = np.asarray([el.flatten() for el in gt_])
    predictions_ = np.asarray([el.flatten() for el in predictions_])
    ious = pairwise_distances(X=gt_, Y=predictions_, metric=iou)
    return ious


def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)


def compute_eval_metric(gt, predictions):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return sum(precisions) / len(precisions)


def intersection_over_union(y_true, y_pred):
    ious = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        iou = compute_ious(y_t, y_p)
        iou_mean = 1.0 * np.sum(iou) / iou.shape[0]
        ious.append(iou_mean)

    return np.mean(ious)


def intersection_over_union_thresholds(y_true, y_pred):
    iouts = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        iouts.append(compute_eval_metric(y_t, y_p))
    return np.mean(iouts)

def pq_metric(y_true, y_pred):
    pq = PQ()
    
    y_true_np = y_true.cpu().detach().numpy()[0][1]
    y_pred_np = y_pred.cpu().detach().numpy()[0]
    y_pred_np = np.argmax(y_pred_np, axis=0) == 1;
    
    
    y_true_, _     = ndi.label(y_true_np)
    y_pred_, _     = ndi.label(y_pred_np)
    _, _, pq_val = pq(y_pred_, y_true_)
    #breakpoint()
    return pq_val

class PQ(object):
    def __init__(self):
        pass
        
    # Jaccard Index
    # obj1 and obj2 are binary masks of the segmentation and target respectively
    def iou(self,obj1,obj2):
        obj1=obj1.astype(int)
        obj2=obj2.astype(int)

        interection = ((obj1+obj2)==2).astype(int)
        union = ((obj1+obj2)>=1).astype(int)

        if union.sum()>0:
            return interection.sum()/union.sum()
        
        return 0

    # PQ metric
    # iou_array is the IOU matrix
    # th a threshold for defining True Postives
    def pq(self,iou_array,th):
        oiou_array=iou_array.copy()
        iou_array=(iou_array>=th).astype(int)
        
        # sum of IOU in TP
        soiu=(oiou_array[iou_array>0]).sum()

        # True Positives, False Negatives and False Positives
        sum_pred=iou_array.sum(1)
        sum_target=iou_array.sum(0)
        tp= ((sum_target>0).astype(int)).sum()
        fn= ((sum_target==0).astype(int)).sum()
        fp= ((sum_pred==0).astype(int)).sum()
        
        # SQ computation
        if tp>0:
            sq=soiu/tp
        else: 
            sq=0

        # RQ computation
        if (tp+0.5*fp+0.5*fn)>0:
            rq=tp/(tp+0.5*fp+0.5*fn)
        else:
            rq=0
        
        # PQ computation
        pq=sq*rq

        return sq,rq,pq

    # for comparing bbox
    def contained(self,p1,p2,p3):
        return (p3>=p1).all() and (p3<=p2).all()

    # PQ computation
    # input and target are instances maps
    # th is a threshold for defining TP
    def __call__(self,input,target,th=0.5):
        
        # number of instances in segmentation and target
        num_input=np.max(input)
        num_target=np.max(target)

        iou_array=np.zeros((num_input,num_target))
        input_box=np.zeros((num_input+1,4))
        target_box=np.zeros((num_target+1,4))

        # compute bbox for each instance
        for i in range(1,num_input+1):
            ind= np.where(input==i)
            input_box[i,:]= ind[0].min(), ind[1].min(), ind[0].max(), ind[1].max()

        for i in range(1,num_target+1):
            ind= np.where(target==i)
            target_box[i,:]= ind[0].min(), ind[1].min(), ind[0].max(), ind[1].max()

        # compute IOU for every pair of instance in input and target
        for i in range(1,num_input+1):
            for j in range(1,num_target+1):
                # we don't want to compare instances that doesn't have intersecting bboxes
                if self.contained(input_box[i,:2],input_box[i,2:],target_box[j,:2]) or self.contained(input_box[i,:2],input_box[i,2:],target_box[j,2:]) or self.contained(target_box[j,:2],target_box[j,2:],input_box[i,:2]) or  self.contained(target_box[j,:2],target_box[j,2:],input_box[i,2:]):
                    input_i =(input==i).astype(int)
                    target_j=(target==j).astype(int)
                    iou_array[i-1,j-1]= self.iou(input_i,target_j)

        return self.pq(iou_array,th)
