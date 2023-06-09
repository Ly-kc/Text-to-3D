import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

r = 1.0 #相机在哪个球面运动
sample_num = 32 #一次在光线上采样多少点

#将相机移动旋转到指定地点
#定义theta为先绕x轴旋转的角度，phi为然后绕y轴旋转的角度
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
    roty = np.array([[np.cos(phi),0,np.sin(phi),0],
                     [0.0,1,0,0],
                     [-np.sin(phi),0,np.cos(phi),0],
                     [0,0,0,1]])
    c2w = roty@rotx@c2w

    return c2w
    
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
#h，w为像平面上的坐标，原点为左下角   intrinsics：[H,W,focal]
def get_ray(c2w,h,w,intrinsics):
    H,W,focal = intrinsics
    #先计算出相机系中的光线
    direction = np.array([w-W/2, h-H/2, -focal])
    direction /= np.linalg.norm(direction)
    #再转换到全局坐标系
    direction = c2w[:3,:3]@direction
    origin = c2w[:3,3]
    return direction,origin

#得到一条光线的颜色
def render_one_ray(model,c2w,h,w,intrinsics):
    direction,origin = get_ray(c2w,h,w,intrinsics) #(3,)
    samples = uni_sample(0.5*r, 1.5*r)*direction  #batch*3
    samples += origin
    directions = np.ones_like(samples) #batch*3
    directions *= direction 
    # print(directions.shape)
    sigma,color = model(samples,directions)  #batch*1, batch*3
    pixel_color = torch.zeros(3)
    total_alpha = 1 #透明度 
    delta = 1/sample_num 
    for i in range(sample_num):
        local_alpha = torch.exp(sigma[i].clone())
        pixel_color = pixel_color + total_alpha*(1-local_alpha)*color[i]
        total_alpha = total_alpha*local_alpha   #*=也不能用？
    
    return pixel_color,total_alpha
    
#渲染得到图片 
#resolution为图片长宽分辨率
def render_image(model,c2w,intrinsics,resolution):
    H,W,focal = intrinsics
    color_img = torch.zeros((resolution[0],resolution[1],3))
    transparence_img = torch.zeros((resolution[0],resolution[1],1)) 
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            h,w = i/(resolution[0]-1)*H, j/(resolution[1]-1)*W
            color_img[i,j],transparence_img[i,j] = render_one_ray(model,c2w,h,w,intrinsics)
    
    return color_img,transparence_img 

         
            
