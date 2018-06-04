# Classification of CALTECH-256

This repository contains the code for Classification of CALTECH-256 based on DenseNet, ResNet. Moreover, I used transfer learning to imporve the performance of the model. 

# Implementation Detail

## Note:
* These codes ran on UCSD DSMLP. The CALTECH-256 dataset was imported from the server.

## Requirements:
* Python 2.7 or Python 3.6
* Tensorflow 
* Keras
* torchvision, Numpy, Pandas, matplotlib.pyplot, PIL, cv2
* glob, gc, math, time, tqdm

## Code Instruction:
* ### Preprocessing:
  Center Crop.ipynb: Aim to center crop all the images and resize the images to 224x224.
  
  Minibatch.ipynb: Create a list of random minibatch.
  
  Original Dataset.ipynb: Import CALTECH-256 dataset and convert the images to numpy array.
  
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
| DenseNet-121 | Resize to 128x128 | | 91.66% | 40.09% | / |
| DenseNet-169 | Resize to 128x128 | | 87.92% | 33.03% | / |
| DenseNet-201 | Resize to 128x128 | | 94.45% | 36.49% | / |
| DenseNet-264 | Resize to 128x128 | | 91.11% | 29.39% | / |
| DenseNet-121 | Resize to 221x221 | √ | 97.59% | 62.24% | 61.37% |
| DenseNet-169 | Resize to 221x221 | √ | 97.11% | 62.29% | 61.69% |
| DenseNet-201 | Resize to 221x221 | √ | 97.16% | 59.75% | 59.39% |
| ResNet-50 | Resize to 221x221 | √ | 97.49% | 54.03% | 54.21% |
| DenseNet-121 | Center Crop 224x224 | √ | --- | --- | --- |
| DenseNet-169 | Center Crop 224x224 | √ | --- | --- | --- |
| DenseNet-201 | Center Crop 224x224 | √ | --- | --- | --- |
| ResNet-50 | Center Crop 224x224 | √ | --- | --- | --- |
* The results of the first 4 models were obtained after 30 epochs. And the last 4 models were obtained after 15 epochs. Others were obtained after 30 epochs.
* The pretrained dataset of transfer learning is ImageNet.

# References
[1] [Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q.Weinberger. Densely Connected Convolutional Networks. cs.CV, 28 Jan2018](https://arxiv.org/pdf/1608.06993.pdf)

[2] [Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. cs.CV, 10 Dec 2015](https://arxiv.org/pdf/1512.03385.pdf)

[3] [Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning Transferable Architectures for Scalable Image Recognition. cs.CV., 11 Apr 2018](https://arxiv.org/pdf/1707.07012.pdf)
