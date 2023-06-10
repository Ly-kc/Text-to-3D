import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Fourier_Features import Fourier_embedding
import clip
from PIL import Image

class SimpleNet(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.x_module = nn.Sequential(
            nn.Linear(64,128),nn.LeakyReLU(),
            nn.Linear(128,256),nn.LeakyReLU(),
            nn.Linear(256,512),nn.LeakyReLU(),
            nn.Linear(512,1024),nn.LeakyReLU(),
            nn.Linear(1024,256),nn.LeakyReLU()            
        )
        self.sigma_module = nn.Sequential(
            nn.Linear(256,32),nn.LeakyReLU(),
            nn.Linear(32,1),nn.LeakyReLU() #应当大于零
        )
        self.color_module = nn.Sequential(
            nn.Linear(256+64,128),nn.LeakyReLU(),
            nn.Linear(128,64),nn.LeakyReLU(),
            nn.Linear(64,3),nn.Sigmoid() #应当归一化  
        )
        self.b = np.load("magic_fourier.npy") #会在Fourier_embedding中被统一转为tensor  
        self.device=device 
    def forward(self,x,dir):
        x = Fourier_embedding(x,self.b,self.device) #batch*64
        dir = Fourier_embedding(dir,self.b,self.device)
        x = self.x_module(x)
        sigma = self.sigma_module(x) #batch*1
        
        x = torch.cat((x,dir),dim=1)   #batch*(256+64)
        color = self.color_module(x) #batch*3
        
        return sigma,color
    
    

#image:batch*resolution[0]*resolution[1]    
def calcu_clip_loss(color_img,trans_img,caption:str,clip_model,clip_processor,device):
    color_img = color_img*255
    color_img = color_img.permute(0,3,1,2)
    color_img = F.interpolate(color_img, size=(224,224))

    # print(color_img.shape)
    text = clip.tokenize([caption]).to(device)

    logits_per_image, logits_per_text = clip_model(color_img, text)
    loss_clip = -logits_per_image.sum(axis=0)[0]

    aver_trans = torch.mean(trans_img,dim=(0,1,2))
    loss_trans = -torch.min(torch.tensor(0.5),aver_trans)[0]

    print(loss_clip,loss_trans)
    return loss_clip + loss_trans*100
