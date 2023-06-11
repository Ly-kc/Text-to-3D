import numpy as np
import torch
m = 32
# b = np.random.randn(m,3)   #m*3
# np.save("magic_fourier",b,allow_pickle=True)
# print(b)

'''
input: pos:batch*3   b:m*3
output:[cos(2pi*b_0x),sin(2pi*b_0x),...,cos(2pi*b_mx),sin(2pi*b_mx)] 2m*1
'''
def Fourier_embedding(pos,b,device="cpu"):
    degrees = 2*3.1415926*(b@pos[:,:,None])    #batch*m*1
    sins,coss = np.sin(degrees),np.cos(degrees)    #batch*m*1
    output = np.concatenate((sins,coss),axis=1)[:,:,0] #batch*2m
    output = torch.tensor(output, dtype=torch.float32, device=device)
    # print(output)
    return output

# print(Fourier_embedding(np.array([[1,2,3],[6,6,6]])).shape) 