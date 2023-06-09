import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Fourier_Features import Fourier_embedding
import clip
from PIL import Image

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
    
    
    
def calcu_clip_loss(color_img,trans_img,caption:str,clip_model,clip_processor,device):
    color_img = color_img*255
    color_img = color_img.permute(0,3,1,2)
    color_img = F.interpolate(color_img, size=(224,224))

    # print(color_img.shape)
    text = clip.tokenize([caption]).to(device)

    logits_per_image, logits_per_text = clip_model(color_img, text)
    print(logits_per_image)
    loss_clip = -logits_per_image

    # aver_trans = torch.mean(trans_img,dim=(0,1))
    # print(aver_trans.shape,"hahahah")
    # loss_trans = -torch.min(torch.tensor(0.5),aver_trans)
    loss_trans = 0

    return loss_clip + loss_trans*0.05
