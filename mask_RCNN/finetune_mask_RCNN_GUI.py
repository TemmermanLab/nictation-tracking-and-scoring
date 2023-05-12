# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:38:39 2021

@author: Temmerman Lab
"""

# mostly copy / pasted from: https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=at-h4OWK0aoc

# (installing dependencies and downloaded dataset done before)



# DEFINE DATASET




import os
import cv2
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from PIL import Image
import sys


from tkinter import *
import tkinter as tk
from tkinter import filedialog

os.environ['KMP_DUPLICATE_LIB_OK']='True' # prevents crash when using pytorch and matplotlib



# Add the helper function directory to the path
import sys
sys.path.append(os.path.split(__file__)[0]+'\\helper_functions')
import pdb; pdb.set_trace()


# Get the path containing the training set
root = tk.Tk()
dataset_folder = tk.filedialog.askdirectory(initialdir = '/', \
    title = "Select the directory containing human-segmented <images> and <masks> directories for training...")
root.destroy()

# make a copy of a new mask RCNN in the training directory


# FILL IN the correct beginning and subfolder of mask_RCNN in this path
model_file = dataset_folder + r'\mask_rcnn.pt'
scale_factor = 1.0

train = False
test_inference = True


class WormsOnMicrodirt(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    
    
# TAKE A LOOK AT THE DATASET
# dataset = WormsOnMicrodirt(dataset_folder)
# dataset[0]

    
# (OPTION 1) FINETUNING A PRE-TRAINED MODEL
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
 
# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
 
# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (worm) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 


# (OPTION 2) CHANGING THE BACKBONE (SKIPPED FOR NOW)



# CREATE AN INSTANCE SEGMENTATION MODEL

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
    
    




# ADD A DATA-AUGMENTATION HELPER FUNCTION
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if train:
    # PUT EVERYTHING TOGETHER
    # use our dataset and defined transformations
    dataset = WormsOnMicrodirt(dataset_folder, get_transform(train=True))
    dataset_test = WormsOnMicrodirt(dataset_folder, get_transform(train=False))
    
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-2])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-2:])
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)   
    
    
    # INSTANTIATE MODEL AND OPTIMIZER
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    # our dataset has two classes only - background and person
    num_classes = 2
    
    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    learn_rate = 0.005
    optimizer = torch.optim.SGD(params, lr=learn_rate, #INCREASED 5 FOLD
                                momentum=0.5, weight_decay=0.0005) # DEC MOMENTUM TO 0.5 FROM 0.9
    
    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
        
    # RUN FOR 200 EPOCHS
    # let's train it for 200 epochs
    num_epochs = 200
    fuse = 20
    losses = []
    lrs = []
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        a = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        losses.append(a.meters['loss'].value)
        lrs.append(learn_rate)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        b = evaluate(model, data_loader_test, device=device)
        if fuse > 0:
            fuse = fuse-1
        else:
            fuse = 0
            if len(np.where(np.diff(losses[-10:])>0)[0]) >= 2: # considered to be plateuing if it increases at least twice in the last 10 epochs
                learn_rate = learn_rate/2.0    
                optimizer = torch.optim.SGD(params, lr=learn_rate, #INCREASED 5 FOLD
                                momentum=0.5, weight_decay=0.0005)
                fuse = 20
    
    torch.save(model.state_dict(), mode_file)

if test_inference:
    
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
                if prediction[0]['scores'][i] > 0.9:
                    object_mask = Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
                    mask = np.dstack((mask,object_mask))
            if np.ndim(mask) == 3:
                mask = np.amax(mask,2)
                        
        mask = cv2.resize(mask, dim_orig, interpolation = cv2.INTER_AREA)
        return mask
    
    
    
    test_file = r'E:\C. elegans\Luca_T2_Rep4_day140001 22-01-26 11-50-52.avi'
    model, device = prepare_model(model_file)
    vid = cv2.VideoCapture(test_file)
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret,img = vid.read(); img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = segment_full_frame(img, model, device, scale_factor)
    
    
    
    plt.imshow(img, cmap='gray'); plt.axis('off'); plt.show()
    plt.imshow(mask, cmap='gray'); plt.axis('off'); plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

