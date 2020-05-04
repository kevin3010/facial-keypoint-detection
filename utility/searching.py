import cv2 
import sys ,os

import numpy as np
import torch

import matplotlib.pyplot as plt

def find_faces(img):
    face_detector = cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")

    faces = face_detector.detectMultiScale(img,1.3,5)
    
    padding = (40,100)
    face_images = []
    
    for x,y,w,h in faces:
        roi = img[y-padding[0]:y+h+padding[1],x-padding[0]:x+w+padding[0]]
        
        face_images.append({"image":roi,"dims":[x-padding[0],y-padding[0]]})
        
    return face_images


def find_points(model,faces):
    
    points = []
    
    
    
    for face in faces:
        
        face_img = cv2.resize(face['image'].copy(),(224,224))/255.0
        
        face_tensor = torch.as_tensor(face_img,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
#        print(face_tensor.size())
        output = model.forward(face_tensor)
        
        output = output.detach().numpy()
        
        output = output*50.0 + 100
        
        output = output.reshape(68,2)
        
        new_h,new_w = 224,224
        h,w = face["image"].shape[0],face["image"].shape[1]
        
        output = output * [w/new_w,h/new_h] 
    
        output = output + [face["dims"][0],face["dims"][1]]
        
        points.append(output)
        
    return np.array(points)
        


