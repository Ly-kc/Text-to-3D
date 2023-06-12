import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

r = 1.0 #相机在哪个球面运动
sample_num = 32 #一次在光线上采样多少点

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
def get_ray(c2w,h,w,intrinsics):
    H,W,focal = intrinsics
    #先计算出相机系中的光线
    direction = np.array([w-W/2, h-H/2, -focal])
    direction /= np.linalg.norm(direction)
    #再转换到全局坐标系
    direction = c2w[:,:3,:3]@direction #(n,3)
    origin = c2w[:,:3,3] #(n,3)
    return direction,origin

#得到一条光线的颜色
def render_one_ray(model,c2w,h,w,intrinsics,device):
    time1 = time.time()
    view_num = c2w.shape[0]
    direction,origin = get_ray(c2w,h,w,intrinsics) #(n,3)
    samples = uni_sample(0.5*r, 1.5*r)*direction[:,None,:]  #(sample_num,1) * (n,1,3)  = n*sample_num*3
    samples = samples + origin[:,None,:]  #n*sample_num*3
    directions = np.ones_like(samples) #(n,sample_num,3)
    directions = directions*direction[:,None,:] #(n,sample_num,3)
    time2 = time.time()
    samples,directions = samples.reshape(-1,3),directions.reshape(-1,3)
    sigma,color = model(samples,directions)  #返回batch*1, batch*3
    sigma,color = sigma.view(view_num,-1,1),color.view(view_num,-1,3) #(n,sample_num,1)  (n,sample_num,3)

    return sigma,color

def ray_tracing(row_sigma_samples,row_color_samples,view_num,resolution1,device):
    pixel_color = torch.zeros((view_num,resolution1,3),device=device) #(view_num,3)
    total_alpha = torch.ones([view_num,resolution1,1],device=device) #透明度 
    for k in range(sample_num):
        local_alpha = torch.exp(-row_sigma_samples[:,:,k,:]) #(n,reso[1],1)
        pixel_color = pixel_color + total_alpha*(1-local_alpha)*row_color_samples[:,:,k,:]  #(n,reso[1],3)
        total_alpha = total_alpha*local_alpha  
    return pixel_color,total_alpha

#渲染得到图片 
#resolution为图片长宽分辨率
#每次得到n个视角同一个位置像素的颜色 
# c2w:(n,4,4)
# color_img:(n,reso[0],reso[1],3) trans_img:(n,reso[0],reso[1],1)
def render_image(model,c2w,intrinsics,resolution,device):
    H,W,focal = intrinsics
    view_num = c2w.shape[0]
    color_img = torch.zeros((view_num,resolution[0],resolution[1],3),device=device)
    transparence_img = torch.zeros((view_num,resolution[0],resolution[1],1),device=device) 
    #一次处理一行
    for i in range(resolution[0]):
        row_color_samples = torch.zeros((view_num,resolution[1],sample_num,3),device=device)
        row_sigma_samples = torch.zeros((view_num,resolution[1],sample_num,1),device=device)
        time1 = time.time()
        for j in range(resolution[1]):
            h,w = i/(resolution[0]-1)*H, j/(resolution[1]-1)*W
            row_sigma_samples[:,j],row_color_samples[:,j] = render_one_ray(model,c2w,h,w,intrinsics,device)#(n,sample_num,1)  (n,sample_num,3)
        time2 = time.time()
        pixel_color = torch.zeros((view_num,resolution[1],3),device=device) #(view_num,3)
        total_alpha = torch.ones((view_num,resolution[1],1),device=device) #透明度 
        for k in range(sample_num):
            local_alpha = torch.exp(-row_sigma_samples[:,:,k,:]) #(n,reso[1],1)
            pixel_color = pixel_color + total_alpha*(1-local_alpha)*row_color_samples[:,:,k,:]  #(n,reso[1],3)
            total_alpha = total_alpha*local_alpha  
            # print(total_alpha[0,0])
        color_img[:,i],transparence_img[:,i] = ray_tracing(row_sigma_samples,row_color_samples,view_num,resolution[1],device)
        time3 = time.time()
        # print(time2-time1,time3-time2)
        
    return color_img,transparence_img 

         
            
