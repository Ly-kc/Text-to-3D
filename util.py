import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
# from main import r,sample_num,fine_sample_num

r = 1. #相机在哪个球面运动
sample_num = 32 #一次在光线上采样多少点
fine_sample_num = 8
#将相机移动旋转到指定地点
#定义theta为先绕x轴旋转的角度，phi为然后绕y轴旋转的角度
#phi:(view_num,)
def get_c2w(theta,phi):
    # 暂时认为相机一直是指向圆心的，所以只需要确定相机在圆周上的位置
    #移动到圆周上
    c2w = np.array([[1.0,0,0,0],
                    [0,1,0,0],
                    [0,0,1,r],
                    [0,0,0,1]],)
    #先绕x轴旋转,再绕y轴旋转
    rotx = np.array([[1.0,0,0,0],
                    [0,np.cos(theta),-np.sin(theta),0],
                    [0,np.sin(theta),np.cos(theta),0],
                    [0,0,0,1]])
    
    roty = np.zeros((phi.shape[0],4,4))
    for i,p in enumerate(phi):
        roty[i] = np.array([[np.cos(p),0,np.sin(p),0],
                        [0.0,1,0,0],
                        [-np.sin(p),0,np.cos(p),0],
                        [0,0,0,1]])
    c2w = roty@rotx@c2w

    return c2w #(n,4,4)
    
#pdf:(view_num,row_batch,col,sample_num)
#return: (view_num,row_batch,col,sample_num,1) 每个小区间中采样点到原点的距离
def pdf_sample(pdf,near,far):
    pdf = pdf*(1/(pdf.sum(axis=-1)[...,None]))
    ends = np.linspace(near,far,sample_num+1) #所有的端点
    cdf = torch.cumsum(pdf,dim=-1)
    cdf = torch.cat((torch.zeros_like(pdf[...,:1]),cdf),dim=-1) #(view_num,row_batch,col,sample_num+1)    
    random_num = torch.rand(list(cdf.shape[:-1])+[fine_sample_num])  #(view_num,row_batch,col,fine_sample_num)
    index = torch.searchsorted(cdf,random_num)-1 #采样第几个区间 (view_num,row_batch,col,fine_sample_num)
    samples = torch.zeros_like(index)
    #有待在选中的区间中平均采样
    return samples[...,None]
        
    
#平均采样
#返回从近到远 sample_num*1
def uni_sample(near,far):
    #现将near到far分为num段
    ends = np.linspace(near,far,sample_num+1)
    samples = torch.randn((sample_num,1))
    for i in range(sample_num):
        samples[i].uniform_(ends[i],ends[i+1])
    # print(samples)
    return np.array(samples)

#得到光线原点与方向 
#h，w为像平面上的坐标，原点为左下角   intrinsics：[H,W,focal]   c2w:(n,4,4)
def get_ray(c2w,hs,ws,intrinsics):
    H,W,focal = intrinsics
    #先计算出相机系中的光线
    direction = []
    for h in hs:
        for w in ws:
            direction.append([w-W/2, h-H/2, -focal])
    direction = np.array(direction)   #(reso[0]*len(hs),3)
    # print(direction.shape)
    direction *= (1/np.linalg.norm(direction,axis=1)[:,None]) #(reso,3)*(reso,1)
    #再转换到全局坐标系
    direction = (c2w[:,None,:3,:3]@direction[:,:,None])[...,0] #(n,1,3,3)@(reso,3,1) = (n,reso,3,1)
    origin = c2w[:,None,:3,3].repeat(len(ws)*len(hs),axis=1) #(n,reso,3)
    return direction.reshape(-1,3),origin.reshape(-1,3)   #(n*reso, 3)

#得到一条光线的颜色 
# W:(resolution[1],)
#返回color:(view_num,resolution[1],sample_num,3)
def render_one_ray(model,c2w,hs,ws,intrinsics,device):
    time0 = time.time()
    reso0,reso1 = len(hs),len(ws)
    view_num = c2w.shape[0]
    direction,origin = get_ray(c2w,hs,ws,intrinsics) #(n*reso[0]*reso[1] , 3) ,记reso=reso[0]*len(hs)
    time1 = time.time()
    samples = uni_sample(0.5*r, 1.5*r)*direction[:,None,:]  #(sample_num,1) * (n*reso,1,3)  = (n*reso,sample_num,3)
    samples = samples + origin[:,None,:]  #(n*reso,sample_num,3)
    directions = direction[:,None,:].repeat(sample_num, axis=1)  #(n*reso,sample_num,3)
    time2 = time.time()
    samples,directions = samples.reshape(-1,3),directions.reshape(-1,3) #(n*reso*sample_num,3)
    sigma,color = model(samples,directions)  #返回batch*1, batch*3
    sigma,color = sigma.view(view_num,reso0,reso1,sample_num,1),color.view(view_num,reso0,reso1,sample_num,3) 
    # print(sigma[0,60],color[0,60])
    time3 = time.time()
    # print(time1-time0,time2-time1,time3-time2)
    
    return sigma,color

# input: (view_num,resolution[0],resolution[1],,sample_num,3)
def ray_tracing(sigma_samples,color_samples,view_num,resolution,device):
    pixel_color = torch.zeros((view_num,resolution[0],resolution[1],3),device=device) #(view_num,3)
    total_alpha = torch.ones((view_num,resolution[0],resolution[1],1),device=device) #透明度 
    alphas = torch.exp(-sigma_samples*r/sample_num)  # (view_num,resolution[0],resolution[1],sample_num,3)
    weights = torch.zeros((view_num,resolution[0],resolution[1],1,sample_num),device=device)
    for k in range(sample_num):
        local_alpha = alphas[:,:,:,k,:] #(view_num,resolution[0],resolution[1],3)
        weights[...,k] = total_alpha*(1-local_alpha)
        pixel_color = pixel_color + total_alpha*(1-local_alpha)*color_samples[:,:,:,k,:]  #(n,reso[1],3)
        total_alpha = total_alpha*local_alpha  #(view_num,resolution[0],resolution[1],3)
    
    return pixel_color,total_alpha,weights.detach()

#渲染得到图片 
#resolution为图片长宽分辨率
#每次得到n个视角同一个位置像素的颜色 
# c2w:(n,4,4)
# color_img:(n,reso[0],reso[1],3) trans_img:(n,reso[0],reso[1],1)
def render_image(model,c2w,intrinsics,resolution,background,device):
    H,W,focal = intrinsics
    view_num = c2w.shape[0]
    row_batch_size = 8
    color_samples = torch.zeros((view_num,resolution[0],resolution[1],sample_num,3),device=device)
    sigma_samples = torch.zeros((view_num,resolution[0],resolution[1],sample_num,1),device=device) 
    color_img = torch.zeros((view_num,resolution[0],resolution[1],3),device=device)
    transparence_img = torch.zeros((view_num,resolution[0],resolution[1],1),device=device) 
    #一次处理一行
    ws = [j/(resolution[1]-1)*W for j in range(resolution[1])]
    time1 = time.time()
    assert(resolution[0]%row_batch_size == 0)
    for i in range(0,resolution[0],row_batch_size):
        hs = [(i+j)/(resolution[0]-1)*H for j in range(row_batch_size)]
        sigma_samples[:,i:i+row_batch_size],color_samples[:,i:i+row_batch_size] = render_one_ray(model,c2w,hs,ws,intrinsics,device)#(n,,reso[1],sample_num,1)  (n,sample_num,3)
    time2 = time.time()
    color_img,transparence_img,weights = ray_tracing(sigma_samples,color_samples,view_num,resolution,device)
    # color_img = color_img + torch.rand(1,device=device)*transparence_img #背景噪声
    color_img = color_img + transparence_img*background #背景色
    time3 = time.time()
    # print(time2-time1,time3-time2)
    
        
    return color_img,transparence_img,weights

         
            
