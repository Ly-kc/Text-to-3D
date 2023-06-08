import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Fourier_Features import Fourier_embedding
import huggingface_help

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_module = nn.Sequential(
            nn.Linear(64,128),nn.ReLU(),
            nn.Linear(128,256),nn.ReLU(),
            nn.Linear(256,512),nn.ReLU(),
            nn.Linear(512,1024),nn.ReLU(),
            nn.Linear(1024,256),nn.ReLU()            
        )
        self.sigma_module = nn.Sequential(
            nn.Linear(256,32),nn.ReLU(),
            nn.Linear(32,1),nn.ReLU() #应当大于零
        )
        self.color_module = nn.Sequential(
            nn.Linear(256+64,128),nn.ReLU(),
            nn.Linear(128,64),nn.ReLU(),
            nn.Linear(64,3),nn.Sigmoid() #应当归一化  
        )
        self.b = np.load("magic_fourier.npy") #会在Fourier_embedding中被统一转为tensor   
    def forward(self,x,dir):
        x = Fourier_embedding(x,self.b) #batch*64
        dir = Fourier_embedding(dir,self.b)
        x = self.x_module(x)
        sigma = self.sigma_module(x) #batch*1
        
        x = torch.cat((x,dir),dim=1)   #batch*(256+64)
        color = self.color_module(x) #batch*3
        
        return sigma,color
    
    
    
def clip_loss(color_img,trans_img,caption:str,clip_model,clip_processor):
    color_img *= 255
    inputs = clip_processor(text=[caption], images=color_img, 
                            return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    loss_clip = outputs.logits_per_image.sum(dim=1) # this is the image-text similarity score

    aver_trans = torch.mean(trans_img,dim=(0,1))
    loss_trans = -torch.min(0.5,aver_trans.item())

    return loss_clip + loss_trans*0.05