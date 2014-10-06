#!/share/imagedb/kamathv1/anaconda/bin/python

"""
First version of a convolutional network 
   - one that implements the LeNet architecture. 
Pooling - Max Pooling
Classification is implemented by means of a Logistic Regression Layer
Author: Vik Kamath
"""

#Tools
import cPickle
import gzip
import os
import sys
import time
#Numpy
import numpy as np
#Theano
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
#Code implemented in the utils folder
from utils.logisticRegression import LogisticRegression,load_data
from utils.hiddenLayer import HiddenLayer










