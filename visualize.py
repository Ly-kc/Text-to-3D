import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import *
import os
from tqdm import tqdm
import json

from model import SimpleNet,NewNet

def visualize_frame():   
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d') 
    ax.set_xlabel('X label') # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    for j in range(3):
        phi = np.linspace(0,np.pi,9)
        theta = -np.pi/6/2*j
        c2ws = get_c2w(phi=phi,theta=theta)
        for i in range(9):
            c2w = c2ws[i]
            xs = c2w@(np.array([[j/5,0,0,1] for j in range(2)])[:,:,None])
            ys = c2w@(np.array([[0,j/5,0,1] for j in range(2)])[:,:,None])
            zs = c2w@(np.array([[0,0,j/5,1] for j in range(2)])[:,:,None])
            assert(xs[0,3,0] == 1)
            # assert(np.dot(xs[1,:,0] - xs[0,:,0],ys[1,:,0] - ys[0,:,0]) == 0 and 
            #        np.dot(xs[1,:,0] - xs[0,:,0],zs[1,:,0] - zs[0,:,0]) == 0)
            ax.plot(xs[:,0,0],xs[:,1,0],xs[:,2,0],color="red")
            ax.plot(ys[:,0,0],ys[:,1,0],ys[:,2,0],color="blue")
            ax.plot(zs[:,0,0],zs[:,1,0],zs[:,2,0],color="green")
    plt.axis("equal")
    plt.show()

def visualize_ray():   
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d') 
    ax.set_xlabel('X label') # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    resolution = (5,5)
    intrinsics = (r,r,r)
    for j in range(1):
        phi = np.linspace(0,np.pi,4)
        theta = -np.pi/6/2*j
        c2ws = get_c2w(phi=phi,theta=theta)
        for i in range(4):
            c2w = c2ws[i:i+1]
            xs = c2w@(np.array([[j/5,0,0,1] for j in range(2)])[:,:,None])
            ys = c2w@(np.array([[0,j/5,0,1] for j in range(2)])[:,:,None])
            zs = c2w@(np.array([[0,0,j/5,1] for j in range(2)])[:,:,None])
            assert(xs[0,3,0] == 1)
            ax.plot(xs[:,0,0],xs[:,1,0],xs[:,2,0],color="red")
            ax.plot(ys[:,0,0],ys[:,1,0],ys[:,2,0],color="blue")
            ax.plot(zs[:,0,0],zs[:,1,0],zs[:,2,0],color="green")
            H,W,focal = intrinsics
            for a in range(resolution[0]):
                for b in range(resolution[1]):
                    h,w = a/(resolution[0]-1)*H, b/(resolution[1]-1)*W
                    direction,origin = get_ray(c2w,h,w,intrinsics) #n*3
                    samples = uni_sample(0.5*r, 1.5*r)*direction[:,None,:]  #(sample_num,1) * (n,1,3)  = n*sample_num*3
                    samples = samples + origin[:,None,:]  #n*sample_num*3
                    ax.scatter(samples[0][::3,0],samples[0][::3,1],samples[0][::3,2],s=5)
                    # dot = (origin + direction*0.5)[0]
                    # origin = origin[0]
                    # ax.plot([origin[0],dot[0]],[origin[1],dot[1]],[origin[2],dot[2]])
            
    plt.axis("equal")
    plt.show()    
    
def visualize_scene(root,ckp):
    view_num = 8
    intrinsics = (r,r,r)
    print(r)
    resolution = (128,128)
    net = NewNet("cuda").to("cuda")
    # net.load_state_dict(torch.load("gauss_new_net_3000" + ".pth"))
    net.load_state_dict(torch.load(root+'/'+ckp))
    net.eval()
    theta = -np.pi/12  
    phi_list = np.linspace(0,2*np.pi,view_num,endpoint=False) 
    print(phi_list)
    c2w = get_c2w(phi=phi_list,theta=theta)
    with torch.no_grad():
        net.eval()
        color_img,trans_img,_ = render_image(net,c2w,intrinsics,resolution,0.2,"cuda")
        # print(color_img,trans_img)
        # color_img=trans_img.expand(1,128,128,3)
        color_img = color_img.permute(0,3,1,2)
        color_img = F.interpolate(color_img, size=(224,224))
        color_img = color_img.permute(0,2,3,1)
    file_name = root+'/'+ckp[:-4]+'_img'
    if not os.path.isdir(file_name):
        os.mkdir(file_name)
    for i in range(view_num):
        # print(i,color_img[i])
        img_pil = Image.fromarray(np.uint8(color_img[i].cpu().numpy()*255))
        img_pil.save(file_name+'/'+ckp[:-4]+f"_{i}.jpg")
        # print(np.asarray(image).shape)
        
def visualize_synthetic():
    with open('./data/nerf_synthetic/lego/transforms_train.json') as f:
        data = json.load(f)
    frames = data['frames']
    view_num = len(frames)
    transfrom_matrixs = []
    for i in range(view_num):
        transfrom_matrixs.append(frames[i]['transform_matrix'])
    transfrom_matrixs = np.array(transfrom_matrixs)
    print(transfrom_matrixs.shape)    #(100,4,4)
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d') 
    ax.set_xlabel('X label') # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    for i in range(9):
        c2w = transfrom_matrixs[i]
        xs = c2w@(np.array([[j/5,0,0,1] for j in range(2)])[:,:,None])
        ys = c2w@(np.array([[0,j/5,0,1] for j in range(2)])[:,:,None])
        zs = c2w@(np.array([[0,0,j/5,1] for j in range(2)])[:,:,None])
        assert(xs[0,3,0] == 1)
        # assert(np.dot(xs[1,:,0] - xs[0,:,0],ys[1,:,0] - ys[0,:,0]) == 0 and 
        #        np.dot(xs[1,:,0] - xs[0,:,0],zs[1,:,0] - zs[0,:,0]) == 0)
        ax.plot(xs[:,0,0],xs[:,1,0],xs[:,2,0],color="red")
        ax.plot(ys[:,0,0],ys[:,1,0],ys[:,2,0],color="blue")
        ax.plot(zs[:,0,0],zs[:,1,0],zs[:,2,0],color="green")
    plt.axis("equal")
    plt.show()
    
if __name__=="__main__":
    visualize_scene()
    # visualize_frame()
    # visualize_ray()
    # visualize_synthetic()