import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class WeightedMCEloss(nn.Module):

    def __init__(self ):
        super(WeightedMCEloss, self).__init__()

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h )
        weight = centercrop(weight, w, h )
        
        y_pred_log =  F.log_softmax(y_pred, dim=1)
        logpy = torch.sum( weight * y_pred_log * y_true, dim=1 )
        #loss  = -torch.sum(logpy) / torch.sum(weight)
        loss  = -torch.mean(logpy)
        return loss


class WeightedMCEFocalloss(nn.Module):
    
    def __init__(self, gamma=2.0 ):
        super(WeightedMCEFocalloss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h )
        weight = centercrop(weight, w, h )
        
        y_pred_log =  F.log_softmax(y_pred, dim=1)

        fweight = (1 - F.softmax(y_pred, dim=1) ) ** self.gamma
        weight  = weight*fweight

        logpy = torch.sum( weight * y_pred_log * y_true, dim=1 )
        #loss  = -torch.sum(logpy) / torch.sum(weight)
        loss  = -torch.mean(logpy)
        
        return loss

class WeightedBCELoss(nn.Module):
    
    def __init__(self ):
        super(WeightedBCELoss, self).__init__()

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h )
     
        logit_y_pred = torch.log(y_pred / (1. - y_pred))
        loss = weight * (logit_y_pred * (1. - y_true) + 
                        torch.log(1. + torch.exp(-torch.abs(logit_y_pred))) + torch.clamp(-logit_y_pred, min=0.))
        loss = torch.sum(loss) / torch.sum(weight)

        return loss

class BCELoss(nn.Module):
    
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true ):        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        loss = self.bce(y_pred, y_true)
        return loss

class WeightedBDiceLoss(nn.Module):
    
    def __init__(self ):
        super(WeightedBDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h )
        y_pred = self.sigmoid(y_pred)
        smooth = 1.
        w, m1, m2 = weight, y_true, y_pred
        score = (2. * torch.sum(w * m1 * m2) + smooth) / (torch.sum(w * m1) + torch.sum(w * m2) + smooth)
        loss = 1. - torch.sum(score)
        return loss


class BDiceLoss(nn.Module):
    
    def __init__(self):
        super(BDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight=None ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        y_pred = self.sigmoid(y_pred)

        smooth = 1.
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        score = (2. * torch.sum(y_true_f * y_pred_f) + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return 1. - score


    
class BLogDiceLoss(nn.Module):
    
    def __init__(self, classe = 1 ):
        super(BLogDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classe = classe

    def forward(self, y_pred, y_true, weight=None ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        y_pred = self.sigmoid(y_pred)

        eps = 1e-15
        dice_target = (y_true[:,self.classe,...] == 1).float()
        dice_output = y_pred[:,self.classe,...]
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + eps

        return -torch.log(2 * intersection / union)

class WeightedMCEDiceLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.0  ):
        super(WeightedMCEDiceLoss, self).__init__()
        self.loss_mce = WeightedMCEFocalloss()
        self.loss_dice = BLogDiceLoss()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight ):        
        
        alpha = self.alpha
        weight = torch.pow(weight,self.gamma)
        loss_dice = self.loss_dice(y_pred, y_true)
        loss_mce = self.loss_mce(y_pred, y_true, weight)
        loss = loss_mce + alpha*loss_dice        
        return loss

class MCEDiceLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.5  ):
        super(MCEDiceLoss, self).__init__()
        self.loss_mce = BCELoss()
        self.loss_dice_fg = BLogDiceLoss( classe=1  )
        self.loss_dice_th = BLogDiceLoss( classe=2  )
        
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight=None ):        
        
        alpha = self.alpha  
        gamma = self.gamma

        # bce(all_channels) +  dice_loss(mask_channel) + dice_loss(border_channel)  
        loss_all  = self.loss_mce( y_pred[:,:2,...], y_true[:,:2,...]) 
        loss_fg   = self.loss_dice_fg( y_pred, y_true )         
        loss_th   = self.loss_dice_th( y_pred, y_true )  if y_true[:,2,... ].sum() > 0 else 0  
        #print(y_pred[0, 0], y_true[0,0])
        #print(loss_all, loss_fg, loss_th)  

        loss      = loss_all + alpha*loss_fg + gamma*loss_th     
        return loss


class Accuracy(nn.Module):
    
    def __init__(self, bback_ignore=True):
        super(Accuracy, self).__init__()
        self.bback_ignore = bback_ignore 

    def forward(self, y_pred, y_true ):
        
        n, ch, h, w = y_pred.size()        
        y_true = centercrop(y_true, w, h)

        prob = F.softmax(y_pred, dim=1).data
        prediction = torch.argmax(prob,1)

        accs = []
        for c in range( int(self.bback_ignore), ch ):
            yt_c = y_true[:,c,...]
            num = (((prediction.eq(c) + yt_c.data.eq(1)).eq(2)).float().sum() + 1 )
            den = (yt_c.data.eq(1).float().sum() + 1)
            acc = (num/den)*100
            accs.append(acc)
        
        accs = torch.stack(accs)
        return accs.mean()


class Dice(nn.Module):
    
    def __init__(self, bback_ignore=True):
        super(Dice, self).__init__()
        self.bback_ignore = bback_ignore       

    def forward(self, y_pred, y_true ):
        
        eps = 1e-15
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)

        prob = F.softmax(y_pred, dim=1)
        prob = prob.data
        prediction = torch.argmax(prob, dim=1)

        y_pred_f = flatten(prediction).float()
        dices = []
        for c in range(int(self.bback_ignore), ch ):
            y_true_f = flatten(y_true[:,c,...]).float()
            intersection = y_true_f * y_pred_f
            dice = (2. * torch.sum(intersection) / ( torch.sum(y_true_f) + torch.sum(y_pred_f) + eps ))*100
            dices.append(dice)
        
        dices = torch.stack(dices)
        return dices.mean()


def to_one_hot(mask, size):    
    n, c, h, w = size
    ymask = torch.FloatTensor(size).zero_()
    new_mask = torch.LongTensor(n,1,h,w)
    if mask.is_cuda:
        ymask = ymask.cuda(mask.get_device())
        new_mask = new_mask.cuda(target.get_device())
    new_mask[:,0,:,:] = torch.clamp(mask.data, 0, c-1)
    ymask.scatter_(1, new_mask , 1.0)    
    return Variable(ymask)

def centercrop(image, w, h):        
    nt, ct, ht, wt = image.size()
    padw, padh = (wt-w) // 2 ,(ht-h) // 2
    if padw>0 and padh>0: image = image[:,:, padh:-padh, padw:-padw]
    return image

def flatten(x):        
    x_flat = x.clone()
    x_flat = x_flat.view(x.shape[0], -1)
    return x_flat