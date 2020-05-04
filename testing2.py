import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./images/sunglasses.png",cv2.IMREAD_UNCHANGED)
#glass = cv2.resize(img,(64,64))
print(img.shape)

#img = cv2.imread("./images/augment_test.jpg")
diagonal = np.int(img.shape[1])
row = np.zeros(shape=(diagonal,diagonal,4))

center = np.int(diagonal/2 - img.shape[0]/2)

row[center:center+img.shape[0],0:img.shape[1]] = img
#print(img)

rows,cols = img.shape[0],img.shape[1]

M = cv2.getRotationMatrix2D((cols,0),20,1)
dst = cv2.warpAffine(img,M,(3263,3263))

non_zero = np.argwhere(dst[:,:,3]>0)

print(non_zero.shape)
ymax = np.max(non_zero[:,0])
xmax = np.max(non_zero[:,1])
ymin = np.min(non_zero[:,0])
xmin = np.min(non_zero[:,1])

dst = dst[ymin:ymax,xmin:xmax]

print(np.arctan(1)*180/np.pi)


a = np.random.randint(1,10,10).reshape(5,2)

print(a)

print(a[1],a[1][1])

#plt.imshow(dst)

