import numpy as np 
from pandas import read_csv 
import matplotlib.pyplot as plt 
import cv2 

path = 'data/training_frames_keypoints.csv'

df = read_csv(path)
#kp1 = df.values[:,1:]
keypoint = df.values[:,1:].reshape(-1,68,2)
fname = df.values[:,0]

x = keypoint[0,:,0]
y = keypoint[0,:,1]

#img = cv2.imread('data/training/'+fname[0])
#
#plt.imshow(img)
#plt.scatter(x,y,c='m')
#
#print(img.shape)
#
#print(kp1.shape)

#kp1 = (kp1 - 100)/50

maximum = np.max(kp1,axis=1)
minimum = np.min(kp1)
mean = np.mean(kp1,axis=1)
std = np.std(kp1)
#
#print(maximum,minimum,mean,std)

for i,j in enumerate(train_data):
    
    if i==10:
        
        sample = j
        break

img = sample["image"][0].numpy()
img = img*255

keypoints = sample["keypoints"].numpy()

keypoints = keypoints*50.0 + 100

#keypoints = np.array(keypoints,dtype=np.int)

x = keypoints[:,0]
y = keypoints[:,1]



print(keypoints.shape)


plt.imshow(img,cmap='gray')
plt.scatter(x,y,c='m')


import matplotlib.pyplot as plt
import numpy as np 
from glob import glob

a=np.linspace(1,10)
b=np.linspace(11,20)
c=np.linspace(21,30)
plt.scatter(a,b,label='training')
plt.scatter(c,b,label='testing')
plt.legend(loc='upper right')


import pickle
import cv2

img = cv2.imread("data/training/Abdel_Aziz_Al-Hakim_00.jpg")

#print(img.shape)

a = {'img':img,"keypoint":10}

pickle.dump(a,open("testing.pkl",'wb'))

data = pickle.load(open("testing.pkl",'rb'))

print(type(img))

print(type(data['img']))







import numpy as np

a = np.random.randint(0,2,9).reshape(3,3)

print(a)

print("-"*20)

b = [[1,2],[0,0]]

non_zeroy = np.nonzero(a)[0]
non_zerox = np.nonzero(a)[1]

cordinates = np.transpose(np.vstack((non_zerox,non_zeroy)))

print(cordinates)

print("-"*20)

a[non_zeroy] = 10

print(a)