import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
import scipy.ndimage as ndi
from .utils import decompose
from torchlib import post_processing_func

class PQ(object):
    def __init__(self):
        pass
        
    def iou(self,obj1,obj2):
        obj1=obj1.astype(int)
        obj2=obj2.astype(int)

        interection = ((obj1+obj2)==2).astype(int)
        union = ((obj1+obj2)>=1).astype(int)

        if union.sum()>0:
            return interection.sum()/union.sum()
        
        return 0

    def prf(self,iou_array,th):
        oiou_array=iou_array.copy()
        iou_array=(iou_array>=th).astype(int)
        
        soiu=(oiou_array[iou_array>0]).sum()

        sum_pred=iou_array.sum(1)
        sum_target=iou_array.sum(0)
        tp= ((sum_target>0).astype(int)).sum()
        fn= ((sum_target==0).astype(int)).sum()
        fp= ((sum_pred==0).astype(int)).sum()
        if (tp+fp)>0:
            p= tp/(tp+fp)
        else:
            p=0
        if (tp+fn)>0:
            r= tp/(tp+fn)
        else:
            r=0
        if (p+r)>0:
            f= 2*p*r/(p+r)
        else:
            f=0
        
        if tp>0:
            sq=soiu/tp
        else: 
            sq=0
        if (tp+0.5*fp+0.5*fn)>0:
            rq=tp/(tp+0.5*fp+0.5*fn)
        else:
            rq=0
        pq=sq*rq

        return {'precision':p,'recall':r,'fmeasure':f,'sq':sq,'rq':rq,'pq':pq}

    def __call__(self,input,target,th=0.5):
        
        num_input=np.max(input)
        num_target=np.max(target)

        iou_array=np.zeros((num_input,num_target))
        input_box=np.zeros((num_input+1,4))
        target_box=np.zeros((num_target+1,4))
        for i in range(1,num_input+1):
            ind= np.where(input==i)
            input_box[i,:]= ind[0].min(), ind[1].min(), ind[0].max(), ind[1].max()

        for i in range(1,num_target+1):
            ind= np.where(target==i)
            target_box[i,:]= ind[0].min(), ind[1].min(), ind[0].max(), ind[1].max()

        for i in range(1,num_input+1):
            for j in range(1,num_target+1):
                x_left = max(input_box[i,0],target_box[j,0])
                y_top = max(input_box[i,1],target_box[j,1])
                x_right = min(input_box[i,2],target_box[j,2])
                y_bottom = min(input_box[i,3],target_box[j,3])

                if not (x_right < x_left or y_bottom < y_top):
                    input_i =(input==i).astype(int)
                    target_j=(target==j).astype(int)
                    iou_array[i-1,j-1]= self.iou(input_i,target_j)

        self.iou_array=iou_array
        
        return self.prf(iou_array,th)

    
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
    #breakpoint()
    pq = PQ()
    
    y_true_np = y_true.cpu().detach().numpy()[0][1]
    y_pred_np = y_pred.cpu().detach().numpy()[0]
    y_pred_np = np.argmax(y_pred_np, axis=0) == 1;
    
    
    y_true_, n_cells     = ndi.label(y_true_np)
    y_pred_, _     = ndi.label(y_pred_np)
    _, _, pq_val = pq(y_pred_, y_true_)
    #breakpoint()
    return pq_val, n_cells

map_post  = post_processing_func.MAP_post()
th_post   = post_processing_func.TH_post()
wts_post  = post_processing_func.WTS_post()
pq_metric = PQ()

def get_metrics_fidel(gt, outputs, post_label='map'):
    #gt = gt[0].cpu().numpy()
    out_np = outputs[0].cpu().numpy().transpose(1,2,0)
    
    if post_label == 'map':
        predictionlb, MAP, region, output = map_post(out_np)
    elif post_label == 'th':
        predictionlb, MAP, region, output = th_post(out_np, theshold=0.5)
    elif post_label == 'wts':
        predictionlb, MAP, region, output = wts_post(out_np)
    else:
        assert False, f"Get Metrics Fidel Error {post_label} -- expected map || th || tws"
    gt_, n_cells        = ndi.label(gt)
    
    results = pq_metric(predictionlb, gt_)
    results['n_cells'] = predictionlb.max()
    return results, n_cells, (predictionlb, MAP, region, output)
