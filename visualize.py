import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import *
import os
from tqdm import tqdm

from model import SimpleNet

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
    
def visualize_scene():
    intrinsics = (r,r,r)
    resolution = (96,128)
    net = SimpleNet("cuda").to("cuda")
    net.load_state_dict(torch.load("new800scat2view" + ".pth"))
    net.eval()
    theta = -np.pi/12  
    phi_list = np.linspace(0,2*np.pi,2,endpoint=False) 
    c2w = get_c2w(phi=phi_list,theta=theta)
    with torch.no_grad():
        color_img,trans_img = render_image(net,c2w,intrinsics,resolution,"cuda")
    if not os.path.isdir("./result"):
        os.mkdir("result")
    for i in range(2):
        # print(i,color_img[i])
        img_pil = Image.fromarray(np.uint8(color_img[i].cpu().numpy()*255*5))
        img_pil.save (f"result/newview{i}.jpg")
        # print(np.asarray(image).shape)

if __name__=="__main__":
    visualize_scene()
    # visualize_frame()
    # visualize_ray()