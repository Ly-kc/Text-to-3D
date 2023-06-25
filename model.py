import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from Fourier_Features import Fourier_embedding
import clip
from PIL import Image

#依照论文设计的模型
class NewNet(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.dims = [64,256]+[512,256]*3
        self.x_layers = nn.ModuleList([nn.Linear(self.dims[i],self.dims[i+1]) for i in range(len(self.dims)-1)])
        self.sigma_module = nn.Sequential(
            nn.Linear(256,1),nn.ReLU() #应当大于零
        )
        self.color_module = nn.Sequential(
            nn.Linear(256+64,512),nn.SiLU(),
            nn.Linear(512,3),nn.Sigmoid()
        )
        self.b = torch.tensor(np.load("magic_fourier.npy"),device=device)   
        self.device=device 
        
    def forward(self,x,dir):
        x = Fourier_embedding(x,self.b,self.device) #batch*64
        dir = Fourier_embedding(dir,self.b,self.device)
        
        x = F.layer_norm(self.x_layers[0](x),normalized_shape=(256,)) #bacth*256
        #256-512-256-512
        prex = x
        for i in range(1,len(self.dims)-1):
            x = self.x_layers[i](x)
            if(i%2==0):
                x = F.layer_norm(F.silu(x + prex),normalized_shape=(256,)) #batch*256
                prex = x 
            else: x = F.silu(x) #batch*512
            
        sigma = self.sigma_module(x) #batch*1
        
        x = torch.cat((x,dir),dim=1)   #batch*(256+64)
        color = self.color_module(x) #batch*3
        
        return sigma,color

#最初的模型
class SimpleNet(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.residual = nn.Linear(64,256)
        self.x_module = nn.Sequential(
            nn.Linear(64,128),nn.ReLU(),
            nn.Linear(128,256),nn.ReLU(), 
            nn.Linear(256,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),          
            nn.Linear(256,512),nn.ReLU(),
            nn.Linear(512,1024),nn.ReLU(),
            nn.Linear(1024,256)           
        )
        self.sigma_module = nn.Sequential(
            nn.Linear(256,32),nn.ReLU(),
            nn.Linear(32,1),nn.ReLU() #应当大于零
        )
        self.color_module = nn.Sequential(
            nn.Linear(256+64,512),nn.ReLU(),
            nn.Linear(512,64),nn.ReLU(),
            nn.Linear(64,3),nn.Sigmoid()
        )
        self.b = torch.tensor(np.load("magic_fourier.npy"),device=device)   
        self.device=device 
        
    def forward(self,x,dir):
        x = Fourier_embedding(x,self.b,self.device) #batch*64
        dir = Fourier_embedding(dir,self.b,self.device)
        
        x = F.relu(self.x_module(x) + self.residual(x))
        sigma = self.sigma_module(x) #batch*1
        
        x = torch.cat((x,dir),dim=1)   #batch*(256+64)
        color = self.color_module(x) #batch*3
        
        return sigma,color
    
    

#image:batch*resolution[0]*resolution[1]*channel    
def calcu_clip_loss(color_img,trans_img,caption:str,clip_model,clip_processor,need_view,view_pos,device):
    #image preprocess
    color_img = color_img.permute(0,3,1,2)
    color_img = F.interpolate(color_img, size=(224,224))
    color_img = transforms.Normalize(mean=[0.4815,0.4578,0.4082],std=[0.2686,0.2613,0.2758])(color_img)
    #text process
    view_sentence = ['Front of','Left of','Back of','Right of']
    if(need_view): caption = view_sentence[view_pos] + caption 
    text = clip.tokenize([view_sentence[view_pos]+caption]).to(device)

    logits_per_image, logits_per_text = clip_model(color_img.to(torch.float32), text)
    loss_clip = -logits_per_image.sum(axis=0)[0]

    #同原论文的设计
    aver_trans = torch.mean(trans_img,dim=(0,1,2))
    loss_trans = -torch.min(torch.tensor(0.6),aver_trans)[0]

    print(aver_trans,loss_clip)
    return loss_clip + loss_trans*2
