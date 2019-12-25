#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable


# In[3]:


import pandas as pd
from pathlib import Path
from shutil import copyfile
import os
from PIL import Image
import matplotlib.pyplot as plt


# In[44]:


label_map = {'ANGER': 0,
 'CONTEMPT': 1,
 'DISGUST': 2,
 'FEAR': 3,
 'HAPPINESS': 4,
 'NEUTRAL': 5,
 'SADNESS': 6,
 'SURPRISE': 7}


# In[4]:


test_transforms = transforms.Compose([
                                       transforms.Resize(224,interpolation=Image.NEAREST),
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.ToTensor(),
                                       ])


# In[41]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),    
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 8),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)
model.load_state_dict(torch.load('../model/model_no_state.pth'))
model.eval()
#model


# In[6]:


def predict_image(image):   
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index 


# In[45]:


def output(input_image):
    to_pil = transforms.ToPILImage()
    #image = Image.open("../data/torch_data/ANGER/Angelina_Jolie_0005.jpg")
    image = test_transforms(input_image).float()
    image = torch.tensor(image, requires_grad=True)
    index = predict_image(to_pil(image))
    for k,v in label_map.items():
        if v==index:
            return k


# In[ ]:




