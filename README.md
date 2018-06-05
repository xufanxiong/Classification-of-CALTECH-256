# ECE-285-Project 

This repository contains the code for Classification of CALTECH-256 based on DenseNet, ResNet. Moreover, we used transfer learning to imporve the performance of the model. 


# Introduction

In this project, we are using Caltech 256 as our data set to implement the object classification based on Convolution neural network. This data set includes 256 categories, and each category includes at least 80 labeled images. To achieve our goal, we find the new classification model Resnet and Densenet performs very well, when the layer goes deeper and deeper, by introducing the Residual. We try to train the model from the beginning, but the performances of the models were bad. Hence, to improve the model, transfer learning, pretrained by ImageNet, was introduced to develop out model. Because of it, the performances of our models were much better.

<img src="https://cloud.githubusercontent.com/assets/8370623/17981494/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg" width="480">

Figure 1: A dense block with 5 layers and growth rate 4. 

![densenet](https://cloud.githubusercontent.com/assets/8370623/17981496/fa648b32-6ad1-11e6-9625-02fdd72fdcd3.jpg)

Figure 2: A deep DenseNet with three dense blocks. 

# Implementation Detail

## Note:
* These codes ran on UCSD DSMLP. The CALTECH-256 dataset was imported from the server.
* Since the model files and the image set are too large to upload, the demonstraion can't run based on these code. It must import 'hdf5' model files when you want to try the demonstration code.

## Requirements:
* Python 2.7 or Python 3.6
* Tensorflow 
* Keras
* torchvision, Numpy, Pandas, matplotlib.pyplot, PIL, cv2
* glob, gc, math, time, tqdm

## Code Instruction:
* Demonstation.ipynb: The demonstraion code based on demo_set images. (Since the model files and the image set are too large to upload, the demonstraion can't run based on these code. It must use 'hdf5' model files when you want to try the demonstration code.)

* Compute per class accuracies.ipynb: This notebook aim to compute the accuracies of every classes.

* ### Preprocessing:
  Center Crop.ipynb: Aim to center crop all the images and resize the images to 224x224.
  
  Minibatch.ipynb: Create a list of random minibatch.
  
  Original Dataset.ipynb: Import CALTECH-256 dataset and convert the images to numpy array.
  
  Pick demo images.ipynb: This notebook aim to pick some demo images which prediction accuracy are higher than 99.99%. 
  
  Resize image.ipynb: Resize the images.
  
  Split Dataset.ipynb: Aim to split the dataset into training set, validation set and test set.
  
  label2onehot.py: This function aim to convert label to one hot matrix.
  
  label_dict.py: To build a dict to record the label.
  
  onehot2label.py: This function aim to convert onehot matrix to label vector.
  
* ### DenseNet:
  DenseNet.py: This function aim to build a complete DenseNet. (DenseNet-121, 169, 201, 264)
  
  Test Model.ipynb: Train the DenseNet model. If you change the parameter (eve_layers) of densenet function, you can train the four DenseNet models.
  
* ### Transfer learning - resize:
  Transfer learning 121.ipynb: This notebook aim to train DensNet-121 with transfer learning. The image size is 221x221.
  
  Transfer learning 169.ipynb: This notebook aim to train DensNet-169 with transfer learning. The image size is 221x221.
  
  Transfer learning 201.ipynb: This notebook aim to train DensNet-201 with transfer learning. The image size is 221x221.
  
  Transfer learning ResNet.ipynb: This notebook aim to train ResNet-50 with transfer learning. The image size is 221x221.
  
* ### Transfer learning - center crop:
  Transfer learning 121-crop.ipynb: This notebook aim to train DensNet-121 with transfer learning. The image size is 224x224.
  
  Transfer learning 169-crop.ipynb: This notebook aim to train DensNet-169 with transfer learning. The image size is 224x224.
  
  Transfer learning 201-crop.ipynb: This notebook aim to train DensNet-201 with transfer learning. The image size is 224x224.
  
  Transfer learning ResNet-crop.ipynb: This notebook aim to train ResNet-50 with transfer learning. The image size is 224x224.

# Result
| Model | Preprocessing | Transfer Learning | Training Set Accuracy | Validation Set Accuracy | Test Set Accuracy |
| --- | --- | --- | --- | --- | --- |
| DenseNet-121 | Resize to 128x128 | | 91.66% | 40.09% | 39.06% |
| DenseNet-169 | Resize to 128x128 | | 87.92% | 33.03% | 32.89% |
| DenseNet-201 | Resize to 128x128 | | 94.45% | 36.49% | 34.22% |
| DenseNet-264 | Resize to 128x128 | | 91.11% | 29.39% | 28.56% |
| DenseNet-121 | Resize to 221x221 | √ | 97.59% | 62.24% | 61.37% |
| DenseNet-169 | Resize to 221x221 | √ | 97.11% | 62.29% | 61.69% |
| DenseNet-201 | Resize to 221x221 | √ | 97.16% | 59.75% | 59.39% |
| ResNet-50 | Resize to 221x221 | √ | 97.49% | 54.03% | 54.21% |
| DenseNet-121 | Center Crop 224x224 | √ | 94.02% | 58.71% | 57.81% |
| DenseNet-169 | Center Crop 224x224 | √ | 93.76% | 56.87% | 57.35% |
| DenseNet-201 | Center Crop 224x224 | √ | 94.14% | 57.07% | 56.50% |
| ResNet-50 | Center Crop 224x224 | √ | 97.41% | 53.19% | 54.00% |
* CALTECH-256 dataset has 29780 usable images. 
* The ratio between training set, cross validation set and test set is 6 to 2 to 2. (Training set has 17868 images, corss validation set has 5956 images and test set has 5956 images) 
* The results of the first 4 models were obtained after 30 epochs. And the last 4 models were obtained after 15 epochs. Others were obtained after 30 epochs.
* The pretrained dataset of transfer learning is ImageNet.

![densenet.png](/images/demo.png) 

Figure 3: A demonstration of images prediction. 

![densenet.png](/images/Transfer-Resize.png) 

Figure 4: The relationship between loss, accuracy and epochs based on transfer learning. (The images were resized to 221x221)

# References
[1] [Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q.Weinberger. Densely Connected Convolutional Networks. cs.CV, 28 Jan2018](https://arxiv.org/pdf/1608.06993.pdf)

[2] [Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. cs.CV, 10 Dec 2015](https://arxiv.org/pdf/1512.03385.pdf)

[3] [Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning Transferable Architectures for Scalable Image Recognition. cs.CV., 11 Apr 2018](https://arxiv.org/pdf/1707.07012.pdf)
