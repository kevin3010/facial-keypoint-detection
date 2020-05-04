from torch.utils.data import Dataset
from pandas import read_csv 
import os 
import matplotlib.image as mpimg
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from glob import glob


class FacialKeypointsDataset(Dataset):
    
    def __init__(self,csv_file,root_dir,transform = None):
        
        self.key_pts_frame = read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
    def prepare_data(self,force=False):
        
        filename = self.root_dir.split("/")
        filename = [word for word in filename if word!=''][-1]
        
        pickle_path = glob("data/*.pkl")
        
        if force==False and pickle_path[0].find(filename)!=-1:
            print('loading binary data')
            self.samples = pickle.load(open(pickle_path[0],"rb"))
            return
        elif force==False and pickle_path[1].find(filename)!=-1:
            print('loading binary data')
            self.samples = pickle.load(open(pickle_path[1],"rb"))
            return
#        path = glob("data/*.pkl")
        
        print("loading the fresh data\n")
        for i in tqdm(range(len(self.key_pts_frame))):
            image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.values[i, 0])
        
            image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
            if(image.shape[2] == 4):
                image = image[:,:,0:3]
        
            key_pts = self.key_pts_frame.values[i, 1:]
            key_pts = key_pts.astype('float').reshape(-1, 2)
            sample = {'image': image, 'keypoints': key_pts}
    
            self.samples.append(sample)
            
            
        
        pickle.dump(self.samples,open("data/"+filename+"_data.pkl",'wb'))

        
    def __len__(self):
        return len(self.key_pts_frame)
    
    def __getitem__(self, idx):
        
        if self.transform:
            sample = self.transform(self.samples[idx])
        else: 
            sample = self.samples[idx]
      
        return sample
    

    
import torch 

class Normalize(object):
    
    def __call__(self,sample):
        
        image,key_pts = sample["image"],sample["keypoints"]
        
        image_copy = image.copy()
        key_pts_copy = key_pts.copy()
        
        image_copy = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
        
        image_copy = image_copy / 255.0
        
        key_pts_copy = (key_pts_copy - 100)/50.0
        
        return {'image':image_copy , 'keypoints':key_pts_copy}
    
class VerticalFlip(object):
    
    def __call__(self,sample):
        
        if np.random.randint(0,2)%2 == 0:
            image,key_pts = sample["image"],sample["keypoints"]
            
            image_copy = image.copy()
            key_pts_copy = key_pts.copy()
            
            image_copy = cv2.flip(image_copy,1)
            
            width = image_copy.shape[1]
            
            key_pts_copy = key_pts_copy.reshape(136,)
            
            keypoints = [width-x if i%2==0 else x for i,x in enumerate(key_pts_copy)]
            
            keypoints = np.array(keypoints).reshape(68,2)
            
            return {'image':image_copy , 'keypoints':keypoints}
        
        return sample
    
class Rescale(object):
    
    def __init__(self,output_size):
        
        assert(isinstance(output_size,(tuple,int)))
        self.output_size = output_size
        
    def __call__(self,sample):
        
        image,key_pts = sample["image"],sample["keypoints"]
        
        image_copy = image.copy()
        key_pts_copy = key_pts.copy()
        
        h,w = image_copy.shape[0:2]
        
        if(isinstance(self.output_size,int)):
            
            if h > w:
                new_height,new_width = self.output_size*h/w ,self.output_size
            else:
                new_height,new_width = self.output_size, self.output_size*w/h
                
        else:
            
            new_height,new_width = self.output_size
            
        new_height = np.int(new_height)
        new_width = np.int(new_width)
            
        
        image_copy = cv2.resize(image_copy,(new_width,new_height))
        
        key_pts_copy = key_pts_copy * [new_width/w, new_height/h] 
        
        return {'image':image_copy , 'keypoints':key_pts_copy}
    
class RandomCrop(object):
    
    def __init__(self,output_size):
        
        if(isinstance(output_size,int)):
            self.output_size = (output_size,output_size)
        else:
            assert(len(output_size)==2)
            self.output_size = output_size
            
    def __call__(self,sample):
        
        image,key_pts = sample["image"],sample["keypoints"]
        
        h,w = image.shape[0:2]
        new_h,new_w = self.output_size
            
        top = np.random.randint(0,h-new_h)
        left = np.random.randint(0,w-new_w)
        
        image = image[top:top+new_h,left:left+new_w]
        
        key_pts = key_pts - [left,top]
        
        
        
        return {'image':image , 'keypoints':key_pts}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        
        return {'image': torch.as_tensor(image,dtype=torch.float32),
                'keypoints': torch.as_tensor(key_pts,dtype=torch.float32)}