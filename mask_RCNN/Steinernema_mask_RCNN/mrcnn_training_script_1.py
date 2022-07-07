# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:38:39 2021

@author: Temmerman Lab
"""

# mostly copy / pasted from: https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=at-h4OWK0aoc

# (installing dependencies and downloaded dataset done before)



# DEFINE DATASET

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

dataset_folder = 'dataset_scaled'

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
dataset = WormsOnMicrodirt(dataset_folder)
dataset[0]

    
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
    
    

# PUT SOME EXTERNAL HELPER FUNCTIONS ON THE PATH
import sys
sys.path.append(r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\helper_functions')


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
optimizer = torch.optim.SGD(params, lr=0.005, #INCREASED 5 FOLD
                            momentum=0.5, weight_decay=0.0005) # DEC MOMENTUM TO 0.5 FROM 0.9

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
    
# RUN FOR 200 EPOCHS
# let's train it for 200 epochs
num_epochs = 20

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
    

savename = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20211128_full_frame_Steinernema_segmentation\20211128_full_frame_Sc_on_udirt.pt'
torch.save(model.state_dict(), savename)

# LOOK AT THE RESULTING SEGMENTATION
# pick one image from the test set

img, _ = dataset_test[0]

Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())


model.eval()
import time
t0 = time.time()
with torch.no_grad():
    prediction = model([img.to(device)])
stack = np.zeros((np.shape(img)[1],np.shape(img)[2]),dtype = 'uint8')
#stack = np.reshape(stack,(stack.shape[0],stack.shape[1],1))
max_w = 30
w = 0
for i in range(len(prediction[0]['scores'])):
    if prediction[0]['scores'][i] > 0.2 and w < max_w:
        mask_new = Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
        stack = np.dstack((stack,mask_new))
        w = w+1
mask_comb = np.amax(stack,2)
print('Inference time for one image = ' + str(time.time()-t0) +' s.')



Image.fromarray(mask_comb)
# what's in a prediction?
prediction

# show an image
i = 2
Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())


# show the high scoring masks
thr = 0.9
masknp = np.squeeze(np.array(prediction[0]['masks'].cpu()))


image1 = img.mul(255).permute(1, 2, 0).byte().numpy()



# saving and loading a trained model
# saving



# load the trained model under a different name and use it for inference
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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
model_reloaded = get_instance_segmentation_model(num_classes) # needs this function defined above
model_reloaded.to(device)
model_reloaded.load_state_dict(torch.load(savename))
model_reloaded.eval()


# use it to make an inference on an image it has never seen before
import cv2
img_test_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\novel_test_img_small.png'
img_test = cv2.imread(img_test_file)
img_test = img_test.astype('float64'); img_test = img_test/255 # for some reason pytorch inverts the image
img_test = torch.tensor(img_test,dtype = torch.float)
img_test = img_test.permute(2,0,1)
Image.fromarray(img_test.mul(255).permute(1, 2, 0).byte().numpy())

with torch.no_grad():
    prediction_test = model_reloaded([img_test.to(device)])

# show the predictions
for i in range(len(prediction_test[0]['scores'])):
    if prediction_test[0]['scores'][i] > 0.9:
        # NB: makes a .png in C:/Users/TEMMER~1/AppData/Local/Temp/ and shows it using the default program for .png
        Image.fromarray(prediction_test[0]['masks'][i, 0].mul(255).byte().cpu().numpy()).show()


# use it to make an inference on a large image
img_test_file = r'C:\Users\Temmerman Lab\Dropbox\Temmerman_Lab\code\mask_R-CNN\20210914_segmentation_attempt_1\novel_test_img_large.png'
img_test = cv2.imread(img_test_file)
img_test = img_test.astype('float64'); img_test = img_test/255 # for some reason pytorch inverts the image
img_test = torch.tensor(img_test,dtype = torch.float)
img_test = img_test.permute(2,0,1)
Image.fromarray(img_test.mul(255).permute(1, 2, 0).byte().numpy())

with torch.no_grad():
    prediction_test = model_reloaded([img_test.to(device)])

# show the predictions
for i in range(len(prediction_test[0]['scores'])):
    if prediction_test[0]['scores'][i] > 0.9:
        # NB: makes a .png in C:/Users/TEMMER~1/AppData/Local/Temp/ and shows it using the default program for .png
        Image.fromarray(prediction_test[0]['masks'][i, 0].mul(255).byte().cpu().numpy()).show()

# import cv2
# mask1 = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()
# mask2 = prediction[0]['masks'][1, 0].mul(255).byte().cpu().numpy()
# mask3 = prediction[0]['masks'][2, 0].mul(255).byte().cpu().numpy()
# mask4 = prediction[0]['masks'][3, 0].mul(255).byte().cpu().numpy()
# mask5 = prediction[0]['masks'][4, 0].mul(255).byte().cpu().numpy()
# cv2.imwrite('test_img_1.bmp',image1)
# cv2.imwrite('test_mask_1.bmp',mask1)
# cv2.imwrite('test_mask_2.bmp',mask2)
# cv2.imwrite('test_mask_3.bmp',mask3)
# cv2.imwrite('test_mask_4.bmp',mask4)
# cv2.imwrite('test_mask_5.bmp',mask5)




# # BEWARE OF CRASHING WHEN USING MATPLOTLIB
# import matplotlib.pyplot as plt
# import numpy as np
# mask = np.array(mask)

