import os
import cv2
#from training.py import ConvNet
import numpy as  np
import torch
from torchvision import transforms as trans
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import matplotlib.pyplot as p
from torch.utils.data import TensorDataset as dset
from torch.utils.data import DataLoader as dl
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
       
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(25 * 25 * 64, 16807)
        self.fc2 = nn.Linear(16807, 343)
        self.fc3 =nn.Linear(343,7)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out=self.fc3(out)
        return out
print("hello")
x=np.load("testdata.npy",allow_pickle=True)

m=torch.stack([(torch.from_numpy(i[0]))for i in x])
n=torch.stack([(torch.from_numpy(i[1]))for i in x])    
myset=dset(m,n)
#print(x.type)
test_loader=dl(myset,batch_size=64,shuffle=True)
model = ConvNet()
model.load_state_dict(torch.load("./model.pth"),strict=False)['State-Dict']
print("load")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
def webcam():
    h=0
def argmax(image):
    image= cv2.resize(image,(100,100))
    outputs = model(images.view(-1,1,100,100).type(torch.FloatTensor))#type error may exist with cuda change to torch.cuuda .Floattensor
    _, predicted = torch.max(outputs.data, 1)
    return predicted
