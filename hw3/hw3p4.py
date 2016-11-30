import numpy
import numpy as np
from math import floor
from random import randrange

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn
#from hw3_nn import drop, DropoutHiddenLayer
#from hw3_nn import BatchNormalization
#from hw3_nn import ConvLayer, PoolLayer
from hw3_nn import ConvLayer, DeConvLayer, MaxPooling, Unpooling2D
from hw3_nn import train_nn_restore

import matplotlib.pyplot as plt

from scipy.ndimage import shift
from scipy.ndimage import rotate
from numpy import fliplr


def drop(input, p=0.7): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.
    
    """            
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask


#Problem4
#Implement the convolutional neural network depicted in problem4 
def MY_CNN(learning_rate=0.1, n_epochs=0, batch_size=100):
    
    rng = numpy.random.RandomState(23455)
    
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    test_im = test_set_x.get_value(borrow=True)

    #train_set_x_drop = drop(train_set_x, p=0.7)
    #valid_set_x_drop = drop(valid_set_x, p=0.7)
    #test_set_x_drop = drop(test_set_x, p=0.7)
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    
    Input = x.reshape((batch_size, 3, 32, 32))
    
    ConvLayer1_input = drop(Input, p=0.7)

    ConvLayer1 = ConvLayer(
        rng, 
        input=ConvLayer1_input, 
        filter_shape=(64, 3, 3, 3), 
        image_shape=(batch_size, 3, 32, 32),
        padding='half'
    )
    
    ConvLayer2 = ConvLayer(
        rng, 
        input=ConvLayer1.output, 
        filter_shape=(64, 64, 3, 3), 
        image_shape=(batch_size, 64, 32, 32),
        padding='half'
    )
    
    MaxPoolLayer1 = MaxPooling(
        input=ConvLayer2.output, 
        poolsize=(2, 2), 
        ignore_border=False
    )
    
    ConvLayer3 = ConvLayer(
        rng, 
        input=MaxPoolLayer1.output, 
        filter_shape=(128, 64, 3, 3), 
        image_shape=(batch_size, 64, 16, 16),
        padding='half'
    )
    
    ConvLayer4 = ConvLayer(
        rng, 
        input=ConvLayer3.output, 
        filter_shape=(128, 128, 3, 3), 
        image_shape=(batch_size, 128, 16, 16),
        padding='half'
    )
    
    MaxPoolLayer2 = MaxPooling(
        input=ConvLayer4.output, 
        poolsize=(2, 2), 
        ignore_border=False
    )

    ConvLayer5 = ConvLayer(
        rng, 
        input=MaxPoolLayer2.output, 
        filter_shape=(256, 128, 3, 3), 
        image_shape=(batch_size, 128, 8, 8),
        padding='half'
    )
    
    UpPoolLayer2 = Unpooling2D(
        input=ConvLayer5.output, 
        poolsize=(2, 2)
    )
    
    DeconvLayer5 = ConvLayer(
        rng, 
        input=UpPoolLayer2.output, 
        filter_shape=(128, 256, 3, 3), 
        image_shape=(batch_size, 256, 16, 16),
        padding='half'
    )
    
    DeconvLayer4 = ConvLayer(
        rng, 
        input=DeconvLayer5.output, 
        filter_shape=(128, 128, 3, 3), 
        image_shape=(batch_size, 128, 16, 16),
        padding='half'
    )
    
    # ADD INPUTS
    UpPoolLayer1_input = ConvLayer4.output + DeconvLayer4.output
    
    UpPoolLayer1 = Unpooling2D(
        input=UpPoolLayer1_input, 
        poolsize=(2, 2)
    )
    
    DeconvLayer3 = ConvLayer(
        rng, 
        input=UpPoolLayer1.output, 
        filter_shape=(64, 128, 3, 3), 
        image_shape=(batch_size, 128, 32, 32),
        padding='half'
    )
    
    DeconvLayer2 = ConvLayer(
        rng, 
        input=DeconvLayer3.output, 
        filter_shape=(64, 64, 3, 3), 
        image_shape=(batch_size, 64, 32, 32),
        padding='half'
    )
    
    # ADD INPUTS
    DeconvLayer1_input = ConvLayer2.output + DeconvLayer2.output
    
    DeconvLayer1 = ConvLayer(
        rng, 
        input=DeconvLayer1_input, 
        filter_shape=(3, 64, 3, 3), 
        image_shape=(batch_size, 64, 32, 32),
        padding='half'
    )
    
    Output = DeconvLayer1.output
    
    # create a list of all model parameters to be fit by gradient descent
    params = (
        ConvLayer1.params 
        + ConvLayer2.params  
        + ConvLayer3.params
        + ConvLayer4.params
        + ConvLayer5.params
        + DeconvLayer1.params
        + DeconvLayer2.params
        + DeconvLayer3.params
        + DeconvLayer4.params
        + DeconvLayer5.params
    )
    
    #cost = T.mean((Output - y) ** 2)
    cost = T.mean(T.sqr(Output - Input) )

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        Output,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]#,
            #y: test_set_x[index * batch_size: (index + 1) * batch_size]       
        }
    )

    validate_model = theano.function(
        [index],
        cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]#,
            #y: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    #gparams = [T.grad(cost, param) for param in params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    momentum =theano.shared(numpy.cast[theano.config.floatX](0.5), name='momentum')
    updates = []
    for param in  params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param - learning_rate*param_update))
        updates.append(
            (param_update, 
             momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param))
        )
            
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]#,
            #y: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    train_nn_restore(
        train_model, 
        validate_model, 
        test_model,
        n_train_batches, 
        n_valid_batches,
        n_test_batches, 
        n_epochs,
        verbose = True
    )
    
    
    plt.figure(figsize=(16,6))
    # drop_input = T.dtensor4('drop_input')
    imdrop = theano.function([x], drop(x, p=0.7))
    drop_image = imdrop(test_im[0:8])
    restored = test_model(0)[0:8,:,:,:]
    # print restored.shape
    # print test_im[0]
    for i in range(8):
        plt.subplot(3,8,i+1)
        img_original = (np.reshape(test_im[i],(3,32,32))).transpose(1,2,0)
        plt.imshow(img_original)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Original Image')

        plt.subplot(3,8,i+9)
        img_drop = (np.reshape(drop_image[i],(3,32,32))).transpose(1,2,0)
        plt.imshow(img_drop)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Corrupted Image')

        plt.subplot(3,8,i+17)
        img_restored = (restored[i,:,:,:]).transpose(1,2,0)
        plt.imshow(img_restored)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Restored Image')
        
    #plt.suptitle('Translated pictures')    
    

