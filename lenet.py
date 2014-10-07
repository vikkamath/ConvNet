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
        self.W = theano.shared(np.asarray(
                        np.uniform(
                            low = -W.bound,
                            high = W.bound,
                            shape = filter_shape),
                        dtype=theano.config.floatX,
                        borrow=True))
        
        #The bias is a 1D tensor - i.e. One bias per 
        #           output feature map
        b_values = np.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values,borrow=True)

        #Convolve input feature maps with filters
        conv_out = conv.conv2d( input = input,
                            filters = self.W,
                            filter_shape = filter_shape,
                            image_shape = image_shape)

        #Downsample each feature map individually using maxpooling
        pooled_out = downsample.max_pool_2d(input = conv_out,
                                ds = poolsize,
                                ignore_border = True)

        #Add the bias term. 
        #Since the bias term is a 1D vector, we first reshape it to
        #       (1,n_filters,1,1)
        #Each of bias will be thus be broadcasted across feature maps
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        #NOTE: dimshuffle(0,'x',1) converts AxB to Ax1xB
        #       dimshuffle(1,'x',0) converts AxB to Bx1xA
        #   Therefore, dimshuffle('x',0,'x','x') converts Nx1 to 1xNx1x1

        #Store the parameters of this layer
        self.params = [self.W,self.b]

def evaluate_lenet5(learning_rate = 0.1,
                    n_epochs = 200,
                    dataset = 'mnist.pkl.gz',
                    nkerns = [20,50],
                    batch_size = 500):

    """
    Demonstrate LeNet on MNIST

    :param learning_rate : learning rate for gradient descent
    :type learning_rate: float

    :param n_epochs: Number of epochs that the optimizer runs
    :type n_epochs: int

    :param dataset: Path to the dataset being operated on
    :type dataset: String

    :param n_kerns: number of kernels on each layer
    :type n_kerns: list of integers

    :param batch_size: Size of each minibatch (for SGD)
    :type batch_size: int

    """

    rng = numpy.random.RandomState(1234)

    datasets = load_data(dataset)

    train_set_x,train_set_y = datasets[0]
    valid_set_x,valid_set_y = datasets[1]
    test_set_x,test_set_y = datasets[2]

    #Compute number of minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size 
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    #Allocate symbolic variables for the data
    index = T.lscalar() #Index to a minibatch
    x = T.matrix('x') #The data is presented as rasterized images
    y = T.ivector('y') #The labels are presented as a 1D vector of
                        #   integers

    ishape = (28,28) #This is the size of the MNIST images

    ###############
    #-BUILD MODEL-#
    ###############

    print '.... BUILDING MODEL ....'

    #Reshape matrix of rasterized images of shape (batch_size,28*28)
    #   into a 4D tensor compatible with our LeNetConvPool Layer
    layer0_input = x.reshape((batch_size,1,28,28))

    #Construct the first convolutional pooling layer
    #Filtering reduces the size of the layer to :
    #       (28-5+1,28-5+1) = (24,24)
    #Maxpooling reduces this further to :
    #   (24/2,24/2) = (12,12)
    #The output, a 4D tensor is therefore of the size:
    #       (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng,
                        input=layer0_input,
                        image_shape=(batch_size,1,28,28),
                        filter_shape=(n_kerns[0],1,5,5),
                        poolsize = (2,2))
    
    #The hiddenlayer being fully connected,operates on 2D matrices
    #of shape:
    #(batch_size, num_pixels) i.e. a matrix of rasterized images. 
    #NOTE: I don't understand this
    #This will generate a matrix of shape (20,32*4*4) = (20,512)
    #Refer: theano.tensor.flatten
    layer2_input = layer1.output.flatten(2)

    #Construct a fully connected sigmoidal layer
    layer2 = HiddenLayer(rng,input=layer2_input,
                        n_in = n_kerns[1]*4*4,
                        n_out = 500,
                        activation = T.tanh)

    #Classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input = layer2.output,
                                n_in = 500,
                                n_out = 10)

    #The cost we minimize during training is the 
    #       Negative Log Likelihood (NLL) of the model
    cost = layer3.negative_log_likelihood(y)

    #create a function to compute the errors made by the model
    test_model = theano.function([index],layer3.errors(y),
                    givens = {
                        x: test_set_x[index*batch_size:(index+1)*batch_size],
                        y: test_set_y[index*batch_size:(index+1)*batch_size]})

    validate_model = theano.function([index],layer3.errors(y),
                            givens = {
                                x : valid_set_x[index*batch_size:(index+1)*batch_size],
                                y : valid_set_y[index*batch_size:(index+1)*batch_size]})

    #create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    #Create a list of gradients for all the model parameters
    grads = T.grad(cost,params)

    #Train_model is a function that updates the model parameters
    #   by SGD. 
    #Since this model has so many parameters, it would be tedious
    #   to create update rules for each parameter. 
    #Therefore, we create the updates list by automatically looping
    #   over all (params[i],grad[i]) pairs. 
    updates = []
    for param_i , grad_i in zip(params,grads):
        updates.append((param_i,param_i - learning_rate*grad_i))

    train_model = theano.function([index],
                             cost,
                             updates = updates, 
                             givens = {
                                 x : train_set_x[index*batch_size:(index+1)*batch_size],
                                 y : train_set_x[index*batch_size:(index+1)*batch_size]})


    ###############
    #-TRAIN MODEL-#
    ###############

    print '......TRAINING.......'

    #early stopping parameters
    patience = 10000 #look at this many examples no matter what
    patience_increase = 2 #Wait this much longer when a new best is found

    improvement_threshold = 0.995 #A relative improvement of this much is considered 
                                #significant

    validation_frequency = min(n_train_batches, patience/2)
                        #Go through this many 
                        #minibatches before checking
                        #the error of the model on the 
                        #validation set. 
                        #Here, we check every epoch

    best_params = None
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_minibatches):

            iter = (epoch-1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter+1) % validation_frequency == 0: 

                #Compute zero-one loss on the validation set
                validation_losses = [validate_model(i) for i
                                    in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_loss)
                print('epcoh %i, minibatch %i/%i, validation error %f %%' %\
                        (epoch,minibatch_index+1,n_train_batches, \
                            this_validation_loss * 100.))

                #If this is the best validation loss so far
                if this_validation_loss < best_validation_loss:

                    #Improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \ 
                                                improvement_threshold
                            patience = max(patience, iter*patience_increase)

                    #Save the best validaiton score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    #Test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print('epoch %i, minibatch %i\%i, test error of best model %f %%' %\
                            (

                
    






    




        
        












