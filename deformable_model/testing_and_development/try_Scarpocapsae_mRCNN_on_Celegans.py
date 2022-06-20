# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:15:56 2021

This script applies an mRCNN trained on Steinernema to the segmentation of C.
elegans

@author: Temmerman Lab
"""

import cv2
import os
import numpy as np
import torch
import copy
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
        hidden_layer, num_classes)
    return model


def prepare_model(model_file):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = get_instance_segmentation_model(num_classes) # needs this function defined above
    model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval();
    return model, device


def segment_full_frame(img, model, device, scale_factor):
    dim_orig = (img.shape[1],img.shape[0])
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    mask = np.zeros(np.shape(img),dtype = 'uint8')
    img = img.astype('float64'); img = img/255
    img = torch.tensor(img,dtype = torch.float)
    
    with torch.no_grad():
        img = torch.unsqueeze(img, dim=0)
        prediction = model([img.to(device)])
        for i in range(len(prediction[0]['scores'])):
            if prediction[0]['scores'][i] > 0.8:
                object_mask = Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
                mask = np.dstack((mask,object_mask))
        if np.ndim(mask) == 3:
            mask = np.amax(mask,2)
                    
    mask = cv2.resize(mask, dim_orig, interpolation = cv2.INTER_AREA)
    return mask




if __name__ == '__main__':
    try:
        # high contrast image
        img_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\Luca_T1_day30002 21-11-19 11-02-29_frame_1269.bmp'
        # low contrast (arena surface wet(?)) image
        img_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\Luca_T1_day60002 21-11-22 10-42-06_frame_4307.bmp'
        model_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20211128_full_frame_Steinernema_segmentation\20211130_full_frame_Sc_on_udirt_2.pt'        
        model, device = prepare_model(model_file)
        img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
        scale_factor = 0.5
        mask = segment_full_frame(img, model, device, scale_factor)
        del model, device
        thr = 10
        bw = copy.copy(mask); bw[bw>=thr] = 255; bw[bw<thr] = 0;
        
        
        # this line prevents "OMP: Error #15: Initializing libiomp5md.dll"
        os.environ['KMP_DUPLICATE_LIB_OK']='True' 
        import matplotlib.pyplot as plt
        plt.imshow(mask,cmap = 'gray'); plt.axis('off'); plt.show()
        plt.imshow(img,cmap = 'gray'); plt.axis('off'); plt.show()
        plt.imshow(bw,cmap = 'gray'); plt.axis('off'); plt.show()

    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)





