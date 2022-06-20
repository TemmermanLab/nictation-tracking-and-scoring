# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:01:58 2022

This script segments a single image of C. elegans dauers using the mask R-CNN.
It's purpose it to test whether inferencing can be done within an executable

@author: Temmerman Lab
"""

main_path = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\executable_test'

import cv2
import numpy as np
import os
import copy
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def prepare_model(model_file):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = get_instance_segmentation_model(num_classes) # needs this function defined above
    model.to(device)
    if device.type == 'cpu':
        model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_file))
    model.eval();
    return model, device


def segment_frame_by_tiling(img, model, device, tile_width, overlap):
    
    
    mask = np.zeros(np.shape(img),dtype = 'uint8')
    img = img.astype('float64'); img = img/255
    img = torch.tensor(img,dtype = torch.float)
    
    step_size = tile_width - overlap
    rows = np.arange(0,np.shape(img)[0],step_size)
    overrun = np.where(rows+tile_width > np.shape(img)[0])[0]
    if len(overrun) > 1:
        rows = rows[:-len(overrun)+1]
        rows[-1] =  np.shape(img)[0] - tile_width
    elif len(overrun) == 1:
        rows[-1] =  np.shape(img)[0] - tile_width

    
    cols = np.arange(0,np.shape(img)[1],step_size)
    overrun = np.where(cols+tile_width > np.shape(img)[1])[0]
    if len(overrun) > 1:
        cols = cols[:-len(overrun)+1]
        cols[-1] =  np.shape(img)[1] - tile_width
    elif len(overrun) == 1:
        cols[-1] =  np.shape(img)[1] - tile_width

    
    with torch.no_grad():
        for r in rows:
            for c in cols:
                print(r)
                tile = img[r:r+tile_width,c:c+tile_width]
                tile = torch.unsqueeze(tile, dim=0)
                prediction = model([tile.to(device)])
                #stack = np.zeros((np.shape(tile)[1],np.shape(tile)[2]),dtype = 'uint8')
                stack = copy.copy(mask[r:r+tile_width,c:c+tile_width]) # avoids overwriting masks on earlier, overlapping tiles
                for i in range(len(prediction[0]['scores'])):
                    if prediction[0]['scores'][i] > 0.8:
                        object_mask = Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
                        stack = np.dstack((stack,object_mask))
                if np.ndim(stack) == 3:
                    stack = np.amax(stack,2)
                mask[r:r+tile_width,c:c+tile_width] = stack
    
    
    # tile = img[500:1000,500:1000]
    # tile = torch.unsqueeze(tile, dim=0)
    
    # #Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # with torch.no_grad():
    #     prediction = model([tile.to(device)])
    
    # # stack the high scoring predictions together
    # stack = np.zeros((np.shape(tile)[1],np.shape(tile)[2]),dtype = 'uint8')
    # for i in range(len(prediction[0]['scores'])):
    #     if prediction[0]['scores'][i] > 0.9:
    #         object_mask = Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
    #         stack = np.dstack((stack,object_mask))
    # tile_mask = np.amax(stack,2)
    # #Image.fromarray(mask)
    
    return mask

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
            if prediction[0]['scores'][i] > 0.7:
                object_mask = Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
                mask = np.dstack((mask,object_mask))
        if np.ndim(mask) == 3:
            mask = np.amax(mask,2)
                    
    mask = cv2.resize(mask, dim_orig, interpolation = cv2.INTER_AREA)
    return mask


if __name__ == '__main__':
    try:
        # testing tiling version
        
        # img_file = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\full_frame.png'
        # model_file = r'C:\Users\PDMcClanahan\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\20211018_trained_MRCNN.pt'
        # img_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\full_frame.png'
        # model_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20211124_Steinernema_segmentation\202111XX_trained_MRCNN_Sc_XXX.pt'
        # tile_width = 500 # set equal to the size of the training images (assumed to be square)
        # overlap = 250 # set equal to max dimension of an object to be segmented
        # model, device = prepare_model(model_file)
        # img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
        # mask = segment_frame(img, model, device, tile_width, overlap)
        # import copy
        # thr = 10
        # bw = copy.copy(mask); bw[bw>=thr] = 255; bw[bw<thr] = 0;
        # del model, device
        
        
        # test full frame version
        model_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\Celegans\20220331_full_frame_Ce_on_udirt_2.pt'
        img_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\executable_test\test_frame.png'
        model, device = prepare_model(model_file)
        scale_factor = 1.0
        img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
        mask = segment_full_frame(img, model, device, scale_factor)
        
        
        # show result
        # this line prevents "OMP: Error #15: Initializing libiomp5md.dll"
        # os.environ['KMP_DUPLICATE_LIB_OK']='True' 
        # import matplotlib.pyplot as plt
        # plt.imshow(mask,cmap = 'gray'); plt.axis('off'); plt.show()
        # plt.imshow(img,cmap = 'gray'); plt.axis('off'); plt.show()
        cv2.imwrite(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\executable_test\test_frame_segmented.png',mask)
    
    except:
        import pdb
        import sys
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        
        
# model_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\Celegans\20220331_full_frame_Ce_on_udirt_2.pt'
# img_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\executable_test\test_frame.png'
# model, device = prepare_model(model_file)
# scale_factor = 1.0
# img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
# mask = segment_full_frame(img, model, device, scale_factor)
# cv2.imwrite(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\nictation\mask_R-CNN\executable_test\test_frame_segmented.png',mask)
    

