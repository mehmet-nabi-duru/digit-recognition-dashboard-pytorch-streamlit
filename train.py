# import the libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np
from torchvision import datasets,transforms
import torch.optim as optim
from model import CNN

# define the datasets
trainset = datasets.MNIST(download=True,train=True,root='./dataset',transform=transforms.ToTensor())
testset = datasets.MNIST(download=True,train=False,root='./dataset',transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset,batch_size=100,shuffle=True)
testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False)

# check if any gpu available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on gpu')
else:
    device = torch.device('cpu')
    print('running on cpu')

# defining model 
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)        

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*2*2, 256)
        self.dropout2 = nn.Dropout(0.75)
        self.fc2 = nn.Linear(256, 10)
        
        
    def forward(self, x):
      # cn layer 1
      x = self.conv1(x)
      x = self.batchnorm1(x)
      x = self.relu1(x)

      # cn layer 2
      x = self.conv2(x)
      x = self.batchnorm2(x)
      x = self.relu3(x)
      x = self.pool2(x)

      # cn layer 3
      x = self.conv3(x)
      x = self.batchnorm3(x)
      x = self.relu3(x)
      x = self.pool3(x)

      # cn layer 4
      x = self.conv4(x)
      x = self.batchnorm4(x)
      x = self.relu4(x)
      x = self.pool3(x)

      # fully connected layers
      x = x.view(-1, 64*2*2)
      x = self.dropout1(x)
      x = self.fc1(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      
      return x#torch.nn.functional.softmax(x, 1)  
      
model = CNN().to(device)

BATCH_SIZE = 100
EPOCHS = 25
optimizer = optim.Adam(model.parameters(), lr=0.001)
Loss = nn.CrossEntropyLoss()
batches = iter(trainloader)

# train the model
for e in range (EPOCHS):
    for batch in tqdm(batches):
        model.zero_grad()
        image, label = batch
        image = image.to(device)
        label = label.to(device)
        
        Y = model(image)
        loss = Loss(Y, label)
        loss.backward()
        optimizer.step()

    batches = iter(trainloader)

# testing
tests = iter(testloader)
correct = 0
total = 0
import matplotlib.pyplot as plt

with torch.no_grad():
    for batch in tqdm(tests):
        img, lbl = batch
        imag = img.to(device)
        lbl = lbl.to(device)
        Y = model(imag)
        predict = torch.argmax(Y,dim=1)
    plt.imshow(img[1].view(28,28), cmap='gray')
    # plt.title(Y)

    for p,l in zip(predict,lbl):
        #print(predict,lbl)
        if p == l:
            correct+=1
        total+=1

print("Accuracy: {}%".format((correct/total)*100))
print(correct,total)

# save the model with state dict
torch.save(model.state_dict(), "./model.pth")
#torch.save(optimizer.state_dict(), "./results/optimizer.pth")