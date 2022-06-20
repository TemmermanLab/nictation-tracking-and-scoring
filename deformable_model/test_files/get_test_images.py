# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:07:18 2021

The script generates images for parameter optimization. Specifically:
    
    1. A 500x500 crop from a worm video that can be manually-segmented with 
    the segmentation_GUI.py to create a gold-standard mask
    
    2. A background-subtracted version of (1) that can be used for intensity-
    based segmentation and comparison of that to mask R-CNN-based segmentation
    
    3. A grayscale image consisting of high-scoring masks for objects
    predicted to exist in (1) by a mask R-CNN


@author: Temmerman Lab
"""

###### 1. 500x500 crop
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# frame, x, y of top right corner in imageJ
to_crop =  [
           [9500,1512,1228], # used for novel_test_img_small.png
           ]

# load video (on drive PDM 3)
vid_path = r'E:/20210727_nict_timecourse_4/'
vid_name = r'6d 21-08-02 13-35-18.avi'
vid = cv2.VideoCapture(vid_path + vid_name)

# create and save cropped images
img_path = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files'


if not os.path.exists(img_path+r'\images'):
    os.mkdir(img_path+r'\images')

for i in range(len(to_crop)):
    f = to_crop[i][0]
    x = to_crop[i][1]
    y = to_crop[i][2]
    vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    success,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crop = img[y:y+500,x:x+500]
    if len(to_crop) == 1:
        cv2.imwrite(img_path+r'\img.png', crop)
    else:
        cv2.imwrite(img_path+r'\img_'+str(i)+'.png', crop)
    print('cropped image '+str(i+1)+' of '+str(len(to_crop)))



###### 2. background-subtracted
# make background-subtracted version


def get_background(vid, bkgnd_nframes = 10, method = 'max_merge'):
    supported_meths = ['max_merge','min_merge']
    #pdb.set_trace()
    if method not in supported_meths:
        raise Exception('Background method not recognized, method must be one of ' + str(supported_meths))
    else:
        print('Calculating background image...')
        num_f = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        inds = np.round(np.linspace(0,num_f-1,bkgnd_nframes)).astype(int)
        for ind in inds:
            vid.set(cv2.CAP_PROP_POS_FRAMES, ind)
            success,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if ind == inds[0]:
                img = np.reshape(img,(img.shape[0],img.shape[1],1)) 
                stack = img
            else:
                img = np.reshape(img,(img.shape[0],img.shape[1],1))
                stack = np.concatenate((stack, img), axis=2)
        if method == 'max_merge':
            bkgnd = np.amax(stack,2)
        elif method == 'min_merge':
            bkgnd = np.amin(stack,2)
        print('Background image calculated')
    return bkgnd 


bkg = get_background(vid)

for i in range(len(to_crop)):
    f = to_crop[i][0]
    x = to_crop[i][1]
    y = to_crop[i][2]
    vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    success,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff = np.abs(np.int32(img) - np.int32(bkg))
    crop = diff[y:y+500,x:x+500]
    if len(to_crop) == 1:
        cv2.imwrite(img_path+r'\img_bkgnd.png', crop)
    else:
        cv2.imwrite(img_path+r'\img_bkgnd_'+str(i)+'.png', crop)
    print('cropped image '+str(i+1)+' of '+str(len(to_crop)))



###### 3. mask R-CNN output
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2

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

loadname = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\20211019_trained_MRCNN.pt'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
model_reloaded = get_instance_segmentation_model(num_classes) # needs this function defined above
model_reloaded.to(device)
model_reloaded.load_state_dict(torch.load(loadname))
model_reloaded.eval()


# use it to make an inference on an image it has never seen before
img_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\tracking\tracker_V2\testing_files\img.png'
img = cv2.imread(img_file)
img = img.astype('float64'); img = img/255 # for some reason pytorch inverts the image
img = torch.tensor(img,dtype = torch.float)
img = img.permute(2,0,1)
Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

with torch.no_grad():
    prediction = model_reloaded([img.to(device)])


# show the high scoring predictions together
stack = np.zeros((np.shape(img)[1],np.shape(img)[2]),dtype = 'uint8')
#stack = np.reshape(stack,(stack.shape[0],stack.shape[1],1)) 
for i in range(len(prediction[0]['scores'])):
    if prediction[0]['scores'][i] > 0.7:
        mask_new = Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
        stack = np.dstack((stack,mask_new))
mask_comb = np.amax(stack,2)
cv2.imwrite(img_path+r'\mRCNN_output.png', mask_comb)

