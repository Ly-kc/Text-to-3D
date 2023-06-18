import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import time
import math
# torch.set_default_dtype(torch.float16)

from model import *
from util import *
import pretrained_model
# torch.autograd.set_detect_anomaly(True)
torch.manual_seed(1919810)
caption = "a sculpture of a yellow cat"
intrinsics = (r,r,r)
resolution = (128,128)
view_num = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


net = SimpleNet(device).to(device)

# optimizer = optim.SGD(net.parameters(), lr=5e-5, momentum=0.8)
clip_model,clip_processor = pretrained_model.get_clip_model_and_processor(device) 
clean_optimizer = optim.SGD(clip_model.parameters(), lr=5e-6)
optimizer=torch.optim.Adam(net.parameters(),
                lr=2e-4,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.94)


epoch_losses = []
#选定角度(弧度制)
theta = -np.pi/12  
phi_list = np.linspace(0,2*np.pi,view_num,endpoint=False) #24
print(phi_list)
c2w = get_c2w(theta,phi_list)
print(c2w.shape)
net.load_state_dict(torch.load("modi_norelu5000_2000" + ".pth"))
for epoch in tqdm(range(2000)):
    running_loss = 0.0
    optimizer.zero_grad()
    clean_optimizer.zero_grad()
    time1 = time.time()
    background = math.cos(epoch/500*2*math.pi)/3+2/3
    color_imgs,trans_imgs = render_image(net,c2w,intrinsics,resolution,background,device)
    time2 = time.time()
    loss = calcu_clip_loss(color_imgs,trans_imgs,caption,clip_model,clip_processor,device)
    time3 = time.time()
    loss.backward()
    time4 = time.time()
    optimizer.step()
    time5 = time.time()
    print(time2-time1,time3-time2,time4-time3,time5-time4)
    running_loss += loss.item()
    print(f"[{epoch+1}] \\\
        loss: {running_loss / view_num:3f}")
    epoch_losses.append(running_loss)
    scheduler.step()        
    # if(running_loss/view_num < -50): scheduler.step()
    if((epoch+1) % 1000 == 0):torch.save(net.state_dict(),"gradual_blend_"+str(epoch+1)+".pth")

plt.switch_backend('Agg') 
plt.plot(epoch_losses,'b',label = 'loss')        
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("gradual_blend" + ".jpg") 
