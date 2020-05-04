# facial-keypoint-detection-and-augmentation

## Introduction
This is a pytorch project for facial keypoint detection and I have tried to recreate the Intagram file by augmenting objects like hat and moustach on the detected keypoints.

Facial Recognition has large scale application which include: facial tracking, facial pose recognition, facial filters, and emotion recognition.

## Data-Discrption
This set of image data has been extracted from the YouTube Faces Dataset, which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

### Training and Testing Data
This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.

* 3462 of these images are training images, for you to use as you create a model to predict keypoints.
* 2308 are test images, which will be used to test the accuracy of your model.

The marking coordinates are given in the seperate csv files.
This can be visualized in the below image.


<center><img src="https://github.com/kevin3010/facial-keypoint-detection/blob/master/images/landmarks_numbered.jpg?raw=True" width=256></center>

<br><br>

## Data-preprocessing
[This](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) artice from pytorch documentation was of great help.Though there are few changes made in order to speedup the training process.
<br>
**Pre-processing on the the Images**<br>
Image is first resize converting the largest dimension to 250px maintaining the aspect ratio, then randomly cropped and conveted to 224 * 224 size image,then normalized such that value of each pixel is between [0,1], finally it is convert to torch tensor.

<br>

**Pre-processing on the the Images**<br>
Keypoints are normalized by subtracting the mean and dividing by the standard deviation.

### Key-points taken into consideration
* All the transformation are done on the fly during the training in order to carry out data augmentation.
* The transforms are available readily with the pytorch libray but as we need to transform the keypoints accordingly we have manually code them.
* Here the outputs are the values of the pixels which are large number and can lead to **exploding gradient** problem. Hence they are suppressed to smaller values

<br><br>

## Training 
The details about the model can be found [here](https://github.com/kevin3010/facial-keypoint-detection/blob/master/pytorch_model.py).
The architecture is inspired from NaimishNet.This is the link to the [research paper](https://arxiv.org/pdf/1710.00977.pdf)
Hyper-Parameters
* **Epochs     :** 10
* **Batch Size :** 16
* **lr         :** 0.001
* **optimizer  :** Adam

### Key-points taken into consideration
* The transformation that are done during the training is are CPU intensive and cannot to done before hand as randomity is the main point of data-augmentation.
* The Layers are initialized with weights from **Kaiming He** normal distribution which is widely considered good for non-symmetric activation funcition ReLU. And indeed it showed fast converence.
* Here there output is 138 values indicating the pixel coordinates and the MSELoss function give average loss accross each individual output(Well I didn't know that before!!)

### Training process
Here is the picture of the training process(I cannot understand why is the vlidation loss less :( ).<br>
<img style="display: block;margin-left: auto;margin-right: auto;width: 50%;" src="https://github.com/kevin3010/facial-keypoint-detection/blob/master/images/training_visualization.PNG?raw=True">

![alt text](https://github.com/kevin3010/facial-keypoint-detection/blob/master/images/training.png?raw=True)

### Visulizing the output
This is sample is taken from the test set which is not used for training. Left side images is the correct values while right side is that of the predicted values.
![alt text](https://github.com/kevin3010/facial-keypoint-detection/blob/master/images/train_sample_visualization.png?raw=True)

### Visulizing the CNN Filters
![alt text](https://github.com/kevin3010/facial-keypoint-detection/blob/master/images/viz_conv_filters.png?raw=True)
<br>(Ohh by the way that's me)
