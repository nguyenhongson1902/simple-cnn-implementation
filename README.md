A simple way of implementing CNN architectures. The problem is to build a classification model.
*Tutorial from*: freeCodeCamp, Youtube

## Main requirements
- tensorflow 2.x
- matplotlib
- keras (available in the Tensorflow package)

## Version
Python 3.9.7
Tensorflow 2.6.0

## Environment
Anaconda, Windows 10

## How to use
In the terminal, run `python train.py` to train my model, run `python train.py -p` to train a model created from another pre-trained model (`-p` stands for the `pretrained` flag)

## Datasets
CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html). It contains 60,000 32x32 color images with 6000 images of each class.

The labels in this dataset are the following:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Documentation:
1. "Convolutional Neural Network (CNN) &nbsp;: &nbsp; TensorFlow Core." TensorFlow, www.tensorflow.org/tutorials/images/cnn.
2. "Transfer Learning with a Pretrained ConvNet &nbsp;: &nbsp; TensorFlow Core." TensorFlow, www.tensorflow.org/tutorials/images/transfer_learning.
3. Chollet François. Deep Learning with Python. Manning Publications Co., 2018.
4. Object Detection, https://github.com/tensorflow/models/tree/master/research/object_detection
