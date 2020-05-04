import torch.nn as nn
import torch.nn.functional as F

#print("hello")

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        
        
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,256,3)
        self.conv5 = nn.Conv2d(256,512,1)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        
#        self.linear1 = nn.Linear(8192,1024)
#        self.linear1 = nn.Linear(43264,1024)
#        self.linear2 = nn.Linear(1024,512)
        
        self.linear1 = nn.Linear(512*6*6,1024)
        self.linear2 = nn.Linear(1024,512)
        self.linear3 = nn.Linear(512,136)
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        
    def forward(self,x):
        
#        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
#        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
#        x = self.dropout3(self.pool(F.relu(self.conv3(x))))
#        x = self.dropout4(self.pool(F.relu(self.conv4(x))))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.view(x.size(0), -1)
 
        x = self.dropout1(F.relu(self.linear1(x)))
        x = self.dropout1(F.relu(self.linear2(x)))
        x = self.linear3(x)

        return x
 
import torch.optim as optim
import torch
import numpy as np
from glob import glob 

def train_network(lr=0.001,n_epochs=1,train_loader=None,test_loader=None,train=True,extend=False):
    
    path = glob("models/testing*")
    
    if extend==True:
        pass
    
    if train==False:
        model = Net()
        model.load_state_dict(torch.load(path[-1]))
        
        return model,[]
    
    print("training started\n")
    
    device = torch.device("cuda:0")
    
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    loss_fn = nn.MSELoss()
    
    loss_graph = []
    
    for epoch in range(n_epochs):
        train_loss_array = [] 
        
        
        for i,sample in enumerate(train_loader):
            
            images = sample["image"]
            
            key_pts = sample["keypoints"].view(-1,136)
            images = images.to(device)
            key_pts = key_pts.to(device)
        
            optimizer.zero_grad()
        
            output = model(images)
        
            loss = loss_fn(output,key_pts)
        
            loss.backward()
        
            optimizer.step()
        
            train_loss_array.append(loss.item())
        
            print("\rEpochs={} Batch={} Training_loss={}".format(epoch,i,np.mean(train_loss_array)),end='\r')
        
        print("")
        validation_loss_array = [] 
        
        for i,sample in enumerate(test_loader):
            
            images = sample["image"]
            
            key_pts = sample["keypoints"].view(-1,136)
            images = images.to(device)
            key_pts = key_pts.to(device)
        
        
            output = model(images)
        
            loss = loss_fn(output,key_pts)
        
            validation_loss_array.append(loss.item())
        
            print("\rEpochs={} Batch={} Validation_loss={}".format(epoch,i,np.mean(validation_loss_array)),end='\r')
        
        torch.save(model.state_dict(), "models/testing"+str(epoch))
        print("")
        print('-'*20)
        loss_graph.append([epoch+1,np.mean(train_loss_array),np.mean(validation_loss_array)])
    
    
    return model,np.array(loss_graph)
        
        
        
        