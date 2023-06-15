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
    pos = torch.tensor(pos,device=device)
    # degrees = 2*3.1415926*(b@pos[:,:,None])    #(m*3)@(batch*3*1) = (batch,m,1)
    degrees = 2*3.1415926*(pos@b.transpose(0, 1))
    sins,coss = torch.sin(degrees),torch.cos(degrees)    #batch*m*1
    output = torch.cat((sins,coss),axis=1).to(torch.float32) #batch*2m
    # print(output)
    return output

# print(Fourier_embedding(np.array([[1,2,3],[6,6,6]])).shape) 

