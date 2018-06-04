# Classification of CALTECH-256

This repository contains the code for Classification of CALTECH-256 based on DenseNet, ResNet. Moreover, we used transfer learning to imporve the performance of the model. 

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
* 

# Result
| Model | Preprocessing | Transfer Learning | Training Set Accuracy | Validation Set Accuracy | Test Set Accuracy |
| --- | --- | --- | --- | --- | --- |
| DenseNet-121 | Resize to 128x128 | | 91.66% | 40.09% | / |
| DenseNet-169 | Resize to 128x128 | | --- | --- | --- |
| DenseNet-201 | Resize to 128x128 | | 94.45% | 36.49% | / |
| DenseNet-264 | Resize to 128x128 | | --- | --- | --- |
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

