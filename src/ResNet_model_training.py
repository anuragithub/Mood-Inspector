#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pathlib import Path
from shutil import copyfile
import os
from PIL import Image
import matplotlib.pyplot as plt


# In[3]:


import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# In[4]:


legends = pd.read_csv("../data/facial_expressions/data/legend.csv")


# In[5]:


image_dir = "../data/facial_expressions/images/"


# In[6]:


data_dir = "../data/torch_data//"


# In[7]:


def separate_data_files(row,image_dir = image_dir):
    if Path(image_dir+row['image']).is_file():
        if not os.path.exists(data_dir+row['emotion']):
            os.makedirs(data_dir+row['emotion'])
        copyfile(image_dir+row['image'], data_dir+row['emotion']+"/"+row['image'])

def prepare_date():
    legends['emotion'] = legends['emotion'].str.upper()
    legends.apply(separate_data_files,axis=1)


# In[8]:


batch_size = 150


# In[9]:


def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([
                                       transforms.Resize(224,interpolation=Image.NEAREST),
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.ToTensor(),
                                       ])

    test_transforms = transforms.Compose([transforms.Resize(224,interpolation=Image.NEAREST),
                                          transforms.Grayscale(num_output_channels=3),
                                      transforms.ToTensor(),
                                      ])

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)


# In[19]:


testloader.dataset


# In[12]:


trainloader.dataset.class_to_idx


# In[20]:


trainloader.dataset


# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model


# In[22]:


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


# In[23]:


epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []


# In[24]:


for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            print(f"Step {steps} of {len(trainloader.dataset)/batch_size}..")
            running_loss = 0
            model.train()


# In[33]:


plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()


# In[28]:


torch.save(model.state_dict(),"../model/model.pth")


# In[36]:


torch.save(model,"../model/model_no_state.pth")


# In[54]:





# In[ ]:




