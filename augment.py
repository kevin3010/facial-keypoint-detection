import cv2
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.image as mpimg
import time

from pytorch_model import train_network

from utility.viz import viz_points,augment_objects
from utility.searching import find_faces,find_points

model,_ = train_network(train=False)
cap = cv2.VideoCapture(0)
i=0
while(True):
    
#    if i%1000==2:
#        i+=1
#        continue
    time.sleep(0.1)
    
    ret, frame = cap.read()
    image = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = find_faces(gray)
    points = find_points(model,faces)
    image = augment_objects(image,points)
    
    cv2.imshow('Fileters',image)
    
    i+=1    
        
    
        
    if len(faces)!=0:
        img = frame
        
        points = np.array(points,dtype=np.int)
        
        x1 = points[0][:,0]
        y1 = points[0][:,1]
        img[y1,x1] = [0,255,0]
        img[y1+1,x1] = [0,255,0]
        img[y1-1,x1] = [0,255,0]
        img[y1,x1+1] = [0,255,0]
        img[y1,x1-1] = [0,255,0]
#        cv2.imshow('Face',faces[0]["image"])
        cv2.imshow('keypoints',img)
        cv2.imshow('Face',faces[0]['image'])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#img = cv2.imread("images/augment_test.jpg")
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
##print(img.shape)
##img = cv2.resize(img,(759,500))
#gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#
#faces = find_faces(gray)
#
#
#points = find_points(model,faces)
#
#
#img = augment_objects(img,points)
#
#
#
#x1 = points[0][:,0]
#y1 = points[0][:,1]
#
#x2 = points[1][:,0]
#y2 = points[1][:,1]
#
#
#img[y1,x1] = [255,0,0]
#img[y1+1,x1] = [255,0,0]
#img[y1-1,x1] = [255,0,0]
#img[y1,x1+1] = [255,0,0]
#img[y1,x1-1] = [255,0,0]

#plt.imshow(img)

#
#plt.imshow(img,cmap='gray')
#plt.scatter(x1,y1,c='r')
#plt.scatter(x2,y2,c='m')




#plt.imshow(img)
#plt.imshow(img)

