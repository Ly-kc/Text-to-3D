import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import *
import os

from model import SimpleNet

def visualize_frame():   
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d') 
    ax.set_xlabel('X label') # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    for i in range(9):
        for j in range(3):
            phi = np.pi/8*i
            theta = -np.pi/6/2*j
            c2w = get_c2w(phi=phi,theta=theta)
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
    
def visualize_scene():
    intrinsics = (r,r,r)
    resolution = (32,32)
    net = SimpleNet("cpu")
    net.load_state_dict(torch.load("cat" + ".pth"))
    net.eval()
    print("ok")
    theta = -np.pi/9  
    phi_list = np.linspace(0,2*np.pi,9,endpoint=False) 
    c2w = get_c2w(phi=phi_list,theta=theta)
    with torch.no_grad():
        color_img,trans_img = render_image(net,c2w,intrinsics,resolution)
    if not os.path.isdir("./result"):
        os.mkdir("result")
    for i in range(9):
        # print(i,color_img[i])
        img_pil = Image.fromarray(np.uint8(color_img[i].cpu().numpy()*255*100))
        img_pil.save (f"result/color{i}.jpg")
        # print(np.asarray(image).shape)

if __name__=="__main__":
    visualize_scene()
    # visualize_frame()
