import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def viz_data(model,test_loader):
    
    idx = np.random.randint(0,len(test_loader))
    
    sample = test_loader[idx]
    
    img = sample["image"].to(torch.device("cuda:0"))
    
    output = model(img.unsqueeze(0))
    output = output.to(torch.device("cpu"))
    
    
    
    output = output.squeeze(0).view(68,2)
    
    output = output.detach().numpy()
    output = output*50.0 + 100
    
    
    xpred = output[:,0]
    ypred = output[:,1]
    
#    for o,x,y in zip(output,xpred,ypred):
#        print(o,x,y)
      
    img = sample["image"][0].numpy()
    img = img*255

    keypoints = sample["keypoints"].numpy()
    keypoints = keypoints*50.0 + 100
    
    xtest = keypoints[:,0]
    ytest = keypoints[:,1]
    
#    fig,axs = plt.subplots(1,2,figure_size=(50,50))
    fi,axs = plt.subplots(1,2)
    axs = axs.flatten()
    
    axs[0].imshow(img,cmap='gray')
    axs[0].scatter(xtest,ytest,label="test")
    
    axs[1].imshow(img,cmap='gray')
    axs[1].scatter(xpred,ypred,label="predict",c='m')
    
#    plt.legend(loc="upper right")
    
    
    
def viz_training(loss_graph):
    
    epochs,training_loss,validation_loss = loss_graph[:,0],loss_graph[:,1],loss_graph[:,2]
    
    plt.plot(epochs,training_loss,label='training')
    plt.plot(epochs,validation_loss,label='validation')
    plt.legend(loc="upper right")
    
    
def viz_feature_maps(model,test_data):
    
    idx = np.random.randint(0,len(test_data))
    
    weights1 = model.conv1.weight.detach().numpy()
    
    sample = test_data[idx]
    
    img = sample["image"][0].numpy()
    
    fig,axs = plt.subplots(4,4)
    axs = axs.flatten()
    
    for i,ax in enumerate(axs):
        ax.imshow(cv2.filter2D(img.copy(),-1,weights1[i][0]),cmap='gray')
        
def viz_points(img,keypoints):
    pass 

def distance(p1 , p2):
    
    return np.int(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
    

def augment_objects(img,points):
    
    glasses_ori = cv2.imread("images/sunglasses.png",cv2.IMREAD_UNCHANGED)
    moustach_ori = cv2.imread("images/moustache.png",cv2.IMREAD_UNCHANGED)
    
    for face in points:
        
        #glasses
        width = distance(face[17],face[26]) + 50
        height = distance(face[27],face[33])
        
        angle = np.arctan(np.abs(face[17,1]-face[26,1])/np.abs(face[17,0]-face[26,0]))*180/np.pi
    
        left = np.int(face[27,0]-width/2)
        top = np.int(face[17,1])
        
        glasses = cv2.resize(glasses_ori,(width,height))
        
        M = cv2.getRotationMatrix2D((glasses.shape[1],0),angle,1)
        
        glasses = cv2.warpAffine(glasses,M,(width+50,height+50))
        
        img_patch = img[top:top+glasses.shape[0],left:left+glasses.shape[1]]
        
        ind = np.argwhere(glasses[:,:,3] > 0)
        
        for i in range(3):
            img_patch[ind[:,0],ind[:,1],i] = glasses[ind[:,0],ind[:,1],i] 
            
        img[top:top+glasses.shape[0],left:left+glasses.shape[1]] = img_patch
        
        #moustach
        width = distance(face[48],face[54]) + 25
        height = distance(face[33],face[51])
        
        left = np.int(face[51,0]-width/2)
        top = np.int(face[31,1])
        
        moustach = cv2.resize(moustach_ori,(width,height)) 
        M = cv2.getRotationMatrix2D((moustach.shape[1],0),angle,1)
        moustach = cv2.warpAffine(moustach,M,(width+50,height+50))
        
        img_patch = img[top:top+moustach.shape[0],left:left+moustach.shape[1]]
        
        ind = np.argwhere(moustach[:,:,3] > 0)
        
        for i in range(3):
            img_patch[ind[:,0],ind[:,1],i] = moustach[ind[:,0],ind[:,1],i] 
            
        img[top:top+moustach.shape[0],left:left+moustach.shape[1]] = img_patch
    return img
        
    
    