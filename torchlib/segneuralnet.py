

import os
import math
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
from tqdm import tqdm

from . import models as nnmodels
from . import losses as nloss
from . import metrics

from pytvision.neuralnet import NeuralNetAbstract
from pytvision.logger import Logger, AverageFilterMeter, AverageMeter
from pytvision import graphic as gph
from pytvision import netlearningrate
from pytvision import utils as pytutils

#----------------------------------------------------------------------------------------------
# Neural Net for Segmentation


class SegmentationNeuralNet(NeuralNetAbstract):
    """
    Segmentation Neural Net 
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        """
        Initialization
            -patchproject (str): path project
            -nameproject (str):  name project
            -no_cuda (bool): system cuda (default is True)
            -parallel (bool)
            -seed (int)
            -print_freq (int)
            -gpu (int)
            -view_freq (in epochs)
        """

        super(SegmentationNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )
        self.view_freq = view_freq

 
    def create(self, 
        arch, 
        num_output_channels, 
        num_input_channels,  
        loss, 
        lr, 
        optimizer, 
        lrsch,    
        momentum=0.9,
        weight_decay=5e-4,      
        pretrained=False,
        size_input=388,
        ):
        """
        Create
        Args:
            -arch (string): architecture
            -num_output_channels, 
            -num_input_channels, 
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) : 
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
            
        """
        
        cfg_opt={ 'momentum':momentum, 'weight_decay':weight_decay } 
        cfg_scheduler={ 'step_size':100, 'gamma':0.1  }
        
        super(SegmentationNeuralNet, self).create( 
            arch, 
            num_output_channels, 
            num_input_channels, 
            loss, 
            lr, 
            optimizer, 
            lrsch, 
            pretrained,
            cfg_opt=cfg_opt, 
            cfg_scheduler=cfg_scheduler
        )
        self.size_input = size_input
        
        self.accuracy = nloss.Accuracy()
        self.dice = nloss.Dice()
       
        # Set the graphic visualization
        self.logger_train = Logger( 'Train', ['loss'], ['accs', 'dices'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss'], ['accs', 'dices', 'PQ'], self.plotter )

        self.visheatmap = gph.HeatMapVisdom(env_name=self.nameproject, heatsize=(100,100) )
        self.visimshow = gph.ImageVisdom(env_name=self.nameproject, imsize=(100,100) )

      
    def training(self, data_loader, epoch=0):        

        #reset logger
        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, sample in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            # get data (image, label, weight)
            inputs, targets, weights = sample['image'], sample['label'], None
            batch_size = inputs.shape[0]

            if self.cuda:
                inputs  = inputs.cuda() 
                targets = targets.cuda() 
            #print(inputs.shape, targets.shape)
                
            # fit (forward)            
            outputs = self.net(inputs)            

            # measure accuracy and record loss
            loss  = self.criterion(outputs, targets, weights)            
            accs  = self.accuracy(outputs, targets)
            dices = self.dice(outputs, targets)
            #pq    = metrics.pq_metric(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())
              
            # optimizer
            self.optimizer.zero_grad()
            (batch_size*loss).backward() #batch_size
            self.optimizer.step()
            
            # update
            self.logger_train.update(
                {'loss': loss.item() },
                {'accs': accs.item(), 
                 #'PQ': pq,
                 'dices': dices.item() },      
                batch_size,          
                )

            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )


    def evaluate(self, data_loader, epoch=0):
        
        # reset loader
        self.logger_val.reset()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate(data_loader):
                
                # get data (image, label)
                inputs, targets, weights = sample['image'], sample['label'], None 
                batch_size = inputs.shape[0]

                if self.cuda:
                    inputs  = inputs.cuda()
                    targets = targets.cuda()
                                 
                # fit (forward)
                outputs = self.net(inputs)

                # measure accuracy and record loss
                loss  = self.criterion(outputs, targets, weights)   
                accs  = self.accuracy(outputs, targets )
                dices = self.dice( outputs, targets )   

                pq    = metrics.pq_metric(targets, outputs)
              
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                #print(loss.item(), accs, dices, batch_size)
                self.logger_val.update( 
                    {'loss': loss.item() },
                    {'accs': accs.item(), 
                     'PQ': pq,
                     'dices': dices.item() },      
                    batch_size,          
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['accs'].avg
        pq = self.logger_val.info['metrics']['PQ'].avg
        

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )

        #vizual_freq
        if epoch % self.view_freq == 0:
            
            prob = F.softmax(outputs, dim=1)
            prob = prob.data[0]
            maxprob = torch.argmax(prob, 0)
            
            self.visheatmap.show('Label', targets.data.cpu()[0].numpy()[1,:,:] )
            #self.visheatmap.show('Weight map', weights.data.cpu()[0].numpy()[0,:,:])
            self.visheatmap.show('Image', inputs.data.cpu()[0].numpy()[0,:,:])
            self.visheatmap.show('Max prob',maxprob.cpu().numpy().astype(np.float32) )
            for k in range(prob.shape[0]):                
                self.visheatmap.show('Heat map {}'.format(k), prob.cpu()[k].numpy() )
        
        return pq


    def test(self, data_loader ):
        
        masks = []
        ids   = []
        k=0

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate( tqdm(data_loader) ):
                
                # get data (image, label)
                inputs, meta  = sample['image'], sample['metadata']    
                idd = meta[:,0]         
                x = inputs.cuda() if self.cuda else inputs    
                
                # fit (forward)
                yhat = self.net(x)
                yhat = F.softmax(yhat, dim=1)    
                yhat = pytutils.to_np(yhat)

                masks.append(yhat)
                ids.append(idd)       
                
        return ids, masks

    
    def __call__(self, image ):        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            yhat = self.net(x)
            yhat = F.softmax( yhat, dim=1 )
            #yhat = pytutils.to_np(yhat).transpose(2,3,1,0)[...,0]
        return yhat


    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained ):
        """
        Create model
            -arch (string): select architecture
            -num_classes (int)
            -num_channels (int)
            -pretrained (bool)
        """    

        self.net = None    

        #-------------------------------------------------------------------------------------------- 
        # select architecture
        #--------------------------------------------------------------------------------------------
        #kw = {'num_classes': num_output_channels, 'num_channels': num_input_channels, 'pretrained': pretrained}

        kw = {'num_classes': num_output_channels, 'in_channels': num_input_channels, 'pretrained': pretrained}
        self.net = nnmodels.__dict__[arch](**kw)
        
        self.s_arch = arch
        self.num_output_channels = num_output_channels
        self.num_input_channels = num_input_channels

        if self.cuda == True:
            self.net.cuda()
        if self.parallel == True and self.cuda == True:
            self.net = nn.DataParallel(self.net, device_ids= range( torch.cuda.device_count() ))

    def _create_loss(self, loss):

        # create loss
        if loss == 'wmce':
            self.criterion = nloss.WeightedMCEloss()
        elif loss == 'bdice':
            self.criterion = nloss.BDiceLoss()
        elif loss == 'wbdice':
            self.criterion = nloss.WeightedBDiceLoss()
        elif loss == 'wmcedice':
            self.criterion = nloss.WeightedMCEDiceLoss()
        elif loss == 'wfocalmce':
            self.criterion = nloss.WeightedMCEFocalloss()
        elif loss == 'mcedice':
            self.criterion = nloss.MCEDiceLoss()  
        else:
            assert(False)

        self.s_loss = loss





