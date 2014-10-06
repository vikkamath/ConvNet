#!/share/imagedb/kamathv1/anaconda/bin/python

"""
First version of a convolutional network 
   - one that implements the LeNet architecture. 
Pooling - Max Pooling
Classification is implemented by means of a Logistic Regression Layer
Heavily based on the code from DeepLearning.net
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

class LeNetConvPoolLayer(object):
    """
    Pooling Layer of a Convolutional Network
    """

    def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2))
        """
        Allocate a LeNet Convolutional Pooling Layer with shared internal 
            parameters

        :param rng: Random Number Generator used to initialize weights
        :type rng: numpy.random.RandomState

        :param input: symbolic image tensor of shape image_shape
        :type input: theano.tensor.dtensor4

        :param filter_shape: (number_of_filters, 
                            number of input feature maps,
                            filter height,
                            filter width)
        :type filter_shape: list or tuple of length 4


        :param image_shape: (batch size, 
                            number of input feature maps, 
                            image height, 
                            image width)
        :type image_shape: tuple or list of length 4

        :param poolsize: the downsample (pooling factor)
                            (#rows,#cols)
        :type poolsize: tuple or list of length 2
        """

        #Ensure that the number of input feature maps are consistent
        #       in both the places that they're specified
        assert imageshape[1] == filtershape[1]
        
        self.input = input

        #There are (num_input_feature_maps * filter_height * filter_width)
        #   inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])

        #Each unit in a lower layer recieves a gradient from 
        #(num_output_feature_maps * filter_height * filter_width)/
        #               pooling_size units
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))/
                    np.prod(poolsize)

        #Initialize weights with random weights
        #Refer (Xavier, Bengio '10)
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = np.uniform(low = -W.bound,


        
        












