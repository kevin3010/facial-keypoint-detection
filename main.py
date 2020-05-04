from torch.utils.data import DataLoader
from torchvision import transforms
from load_data import FacialKeypointsDataset,Rescale,RandomCrop,Normalize,ToTensor
import numpy as np

from utility.viz import viz_training,viz_data,viz_feature_maps

#data_transform = transforms.Compose([Rescale((224,224)),
#                                     Normalize(),ToTensor()])

data_transform = transforms.Compose([Rescale(250),RandomCrop(224),
                                     Normalize(),ToTensor()])

train_data = FacialKeypointsDataset("data/training_frames_keypoints.csv",
                                   "data/training/",transform=data_transform)

test_data = FacialKeypointsDataset("data/test_frames_keypoints.csv", 
                                   "data/test/",transform=data_transform)

test_data.prepare_data()
train_data.prepare_data()



batch_size = 16

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)


from models import train_network


model,loss_graph = train_network(0.001,10,train_loader,test_loader,train=False)

viz_feature_maps(model,test_data)


if len(loss_graph)!=0:
    viz_training(loss_graph)



viz_data(model,test_data)
    
    
#params = list(model.parameters())
#for i in range(len(params)):
#    print(params[i].size())

import gc

gc.collect()

#print(loss_graph)

#vis_














