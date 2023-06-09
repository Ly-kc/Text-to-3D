import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import time
import os
import argparse
from visualize import visualize_scene
# torch.set_default_dtype(torch.float16)

from model import *
from util import *
import pretrained_model

#得到命令行参数
def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption",type=str,default="a sculpture of a yellow cat",help = "caption for the image")
    parser.add_argument("-f","--file_name",type=str,required=True,help = "file name")
    parser.add_argument("--model",type=str,default="new",help = "what nerf model to use")
    parser.add_argument("--load_model",type=str,default=None,help = "model to continue training")
    parser.add_argument("--remark",type=str,default='None',help = "model to continue training")
    parser.add_argument("--bg_intensity",type=float,default=1,help = "intensity of background")
    parser.add_argument("--lr",type=float,default=5e-4,help = "learning rate")
    parser.add_argument("--theta",type=float,default=-1/12,help = "Elevation Angle divided by pi")
    parser.add_argument("--sss",type=float,default=50,help = "scheduler step size")
    parser.add_argument("--gamma",type=float,default=0.95,help = "scheduler gamma")
    parser.add_argument("--radius",type=float,default=1.,help = "radius of sphere where the camera is traversing")
    parser.add_argument("--epoch",type=int,default=1000,help="epoch number")
    parser.add_argument("--save_num",type=int,default=1000,help="epoch number to save checkpoints")
    parser.add_argument("--view_batch",type=int,default=1,help="number of views in a batch")
    parser.add_argument("-wi","--width",type=int,default=128,help="width of image")
    parser.add_argument("-hi","--height",type=int,default=128,help="height of image")
    parser.add_argument("--row_batch",type=int,default=8,help="number of rows in a batch")
    parser.add_argument("--sample_num",type=int,default=32,help="sample times along one ray")
    parser.add_argument("--fine_num",type=int,default=8,help="times for additional sampling")
    parser.add_argument("--need_view",default=False,action='store_true',help = "flag to mark if additional message of view is needed by caption")
    parser.add_argument("--bg_gauss",default=False,action='store_true',help = "use gaussian background")
    parser.add_argument("--cpu",default=False,action='store_true',help = "whether to use cpu")
    parser.add_argument("--fix_view",default=False,action='store_true',help = "whether to use random view")
    args = parser.parse_args()
    
    return args


def train_scene(args): 
    torch.manual_seed(1919810)

    caption = args.caption
    file_name = args.file_name
    resolution = (args.height,args.width)
    need_view = args.need_view
    view_num = args.view_batch
    intrinsics = (r,r,r)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if(args.cpu): device = "cpu"
    print(device)

    net = NewNet(device).to(device) if args.model == "new" else SimpleNet(device).to(device)
    net.train()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.8)
    clip_model,clip_processor = pretrained_model.get_clip_model_and_processor(device) 
    clean_optimizer = optim.SGD(clip_model.parameters(), lr=5e-6)#只是用来清除clip上的梯度
    optimizer=torch.optim.Adam(net.parameters(),
                    lr=args.lr,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0,
                    amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.sss, gamma=args.gamma)

    epoch_losses = []
    
    #选定角度(弧度制)
    theta = -np.pi*args.theta
    
    root = f"./results/{file_name}"
    if not os.path.isdir(root): 
        os.mkdir(root)
    if(args.load_model is not None): net.load_state_dict(torch.load(f'results/{args.load_model[:-5]}'+"/"+args.load_model+".pth"))
    with open(root+'/'+'log.txt','w') as f:
        for stuff in vars(args): 
            if(vars(args)[stuff] is None): continue
            f.write(stuff+': ' + str(vars(args)[stuff]) + '\n')
    
    for epoch in tqdm(range(args.epoch)):
        running_loss = 0.0
        for i in range(4): #为了给caption加入视角信息，分为前左后右四种
            time0 = time.time()
            #得到一个view_batch中相机的位置
            if(args.fix_view): phi_list = np.array([np.pi*i/2])
            else: phi_list = np.pi/2*np.random.random(view_num) +(i/2-1/4)*np.pi
            c2w = get_c2w(theta,phi_list)    
            optimizer.zero_grad()
            clean_optimizer.zero_grad()
            time1 = time.time()
            #选择背景
            # background = 1
            if(args.bg_gauss):background = 9/10+1/10*torch.rand((view_num,resolution[0],resolution[1],3),device=device)
            else: background = background = math.cos(epoch/args.epoch*math.pi)/3 + 2/3
            background *= args.bg_intensity
            #渲染得到图片
            color_imgs,trans_imgs,weights = render_image(net,c2w,intrinsics,resolution,background,device,vars(args))
            time2 = time.time()
            loss = calcu_clip_loss(color_imgs,trans_imgs,caption,clip_model,clip_processor,need_view,i,device)
            time3 = time.time()
            loss.backward()
            time4 = time.time()
            optimizer.step()
            time5 = time.time()
            # print(time1-time0,time2-time1,time3-time2,time4-time3,time5-time4)
            running_loss += loss.item()
        print(f"[{epoch+1}] \\\
            loss: {running_loss / view_num / 4 :3f}")
        epoch_losses.append(running_loss/view_num/4)
        scheduler.step()        
        if((epoch+1) % args.save_num == 0):torch.save(net.state_dict(),f"./results/{file_name}/{file_name}_"+str(epoch+1)+".pth")
    #保存loss图像
    plt.switch_backend('Agg') 
    plt.plot(epoch_losses,'b',label = 'loss')        
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(f'./results/{file_name}/' + file_name + ".jpg") 
    #保存渲染结果
    i = args.save_num
    while(i <= args.epoch):
        visualize_scene('./results/'+file_name, file_name+'_'+str(i)+'.pth')
        i += args.save_num

if __name__ ==  "__main__":
    args = get_argparse()
    train_scene(args)