import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from model import *
from util import *
import huggingface_help

caption = "A sculpture of a white cat"
intrinsics = (r,r,r)
resolution = (20,20)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)

net = SimpleNet().to(device)

optimizer = optim.SGD(net.parameters(), lr=5e-6, momentum=0.8)
schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.97)
clip_model,clip_processor = huggingface_help.get_clip_model_and_processor()

# optimizer=torch.optim.Adam(net.parameters(),
#                 lr=1e-5,
#                 betas=(0.9, 0.999),
#                 eps=1e-08,
#                 weight_decay=0,
#                 amsgrad=False)

epoch_losses = []
for epoch in tqdm(range(100)):
    #选定角度(弧度制)
    theta = -np.pi/12  
    phi_list = np.linspace(0,2*np.pi,24,endpoint=False)
    running_loss = 0.0
    for i,phi in enumerate(phi_list):
        optimizer.zero_grad()
        c2w = get_c2w(theta,phi)
        color_img,trans_img = render_image(net,c2w,intrinsics,resolution)
        loss = clip_loss(color_img,trans_img,caption,clip_model,clip_processor)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if i%1000 == 999:
        #     print(f"[{epoch+1} , {i+1:5d}] \\\
        #         loss: {running_loss / 1000:3f}")
        #     running_loss = 0.0
    print(f"[{epoch+1}] \\\
        loss: {running_loss / 1000:3f}")
    epoch_losses.append(running_loss)
    
            
torch.save(net.state_dict(),"cat.pth")
