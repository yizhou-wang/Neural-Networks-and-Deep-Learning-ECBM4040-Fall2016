"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

This code contains implementation of some basic components in neural network.

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""

from __future__ import print_function

import timeit
import inspect
import sys
import numpy
import numpy as np
from theano.tensor.nnet import conv
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from theano.tensor.signal import pool

import math
import time

from theano.tensor.nnet.bn import batch_normalization


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    
def drop(input, p=0.5): 
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

class DropoutHiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, p=0.5):
        """
        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type is_train: theano.iscalar   
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
                           
        :type p: float or double
        :param p: probability of NOT dropping out a unit   
        """
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        
        output = activation(lin_output)
        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = drop(output,p)
        
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, p*output)
        
        # parameters of the model
        self.params = [self.W, self.b]


class BatchNormalization(object) :
    def __init__(self, input_shape, mode=0 , momentum=0.9) :
        '''
        # params :
        input_shape :
            when mode is 0, we assume 2D input. (mini_batch_size, # features)
            when mode is 1, we assume 4D input. (mini_batch_size, # of channel, # row, # column)
        mode : 
            0 : feature-wise mode (normal BN)
            1 : window-wise mode (CNN mode BN)
        momentum : momentum for exponential average
        '''
        self.input_shape = input_shape
        self.mode = mode
        self.momentum = momentum
        self.run_mode = 0 # run_mode : 0 means training, 1 means inference

        self.insize = input_shape[1]
        
        # random setting of gamma and beta, setting initial mean and std
        rng = np.random.RandomState(int(time.time()))
        self.gamma = theano.shared(
            np.asarray(
                rng.uniform(
                    low=-1.0/math.sqrt(self.insize), 
                    high=1.0/math.sqrt(self.insize), 
                    size=(input_shape[1])), 
                dtype=theano.config.floatX), 
            name='gamma', 
            borrow=True
        )
        self.beta = theano.shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name='beta', borrow=True)
        self.mean = theano.shared(np.zeros((input_shape[1]), dtype=theano.config.floatX), name='mean', borrow=True)
        self.var = theano.shared(np.ones((input_shape[1]), dtype=theano.config.floatX), name='var', borrow=True)

        # parameter save for update
        self.params = [self.gamma, self.beta]
        
        #print ('gamma.shape =')
        #print (self.gamma.get_value().shape)
        #print ('beta.shape =')
        #print (self.beta.get_value().shape)


    def set_runmode(self, run_mode) :
        self.run_mode = run_mode

    def get_result(self, input) :
        # returns BN result for given input.
        epsilon = 1e-06

        if self.mode==0 :
            if self.run_mode==0 :
                now_mean = T.mean(input, axis=0)
                now_var = T.var(input, axis=0)
                now_normalize = (input - now_mean) / T.sqrt(now_var+epsilon) # should be broadcastable..
                output = self.gamma * now_normalize + self.beta
                #print ('norm.shape =')
                #print (now_normalize.shape.eval({x: np.random.rand(2,2).astype(dtype=theano.config.floatX)}))

                # mean, var update
                self.mean = self.momentum * self.mean + (1.0-self.momentum) * now_mean
                self.var = self.momentum * self.var + (1.0-self.momentum) * (self.input_shape[0]/(self.input_shape[0]-1)*now_var)
            else : 
                output = self.gamma * (input - self.mean) / T.sqrt(self.var+epsilon) + self.beta

        else : 
            # in CNN mode, gamma and beta exists for every single channel separately.
            # for each channel, calculate mean and std for (mini_batch_size * row * column) elements.
            # then, each channel has own scalar gamma/beta parameters.
            if self.run_mode==0 :
                now_mean = T.mean(input, axis=(0,2,3))
                now_var = T.var(input, axis=(0,2,3))
                # mean, var update
                self.mean = self.momentum * self.mean + (1.0-self.momentum) * now_mean
                self.var = self.momentum * self.var + (1.0-self.momentum) * (self.input_shape[0]/(self.input_shape[0]-1)*now_var)
            else :
                now_mean = self.mean
                now_var = self.var
            # change shape to fit input shape
            now_mean = self.change_shape(now_mean)
            now_var = self.change_shape(now_var)
            now_gamma = self.change_shape(self.gamma)
            now_beta = self.change_shape(self.beta)
            
            output = now_gamma * (input - now_mean) / T.sqrt(now_var+epsilon) + now_beta
            
        return output

    # changing shape for CNN mode
    def change_shape(self, vec) :
        return T.repeat(
            vec, 
            self.input_shape[2]*self.input_shape[3]
        ).reshape((self.input_shape[1],self.input_shape[2],self.input_shape[3]))
        
        
        
    
class ConvLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, W=None, bias=True, padding='valid', activation=T.nnet.relu):

        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        if W==None:
            W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        self.W =W

        conv_out = conv2d(
            input,
            self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode=padding
        )
        
        if bias==True:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
            self.output = self.output = T.clip(activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')), 0.001, 0.999)
            self.params = [self.W, self.b]
        else:
            self.output = T.clip(activation(conv_out), 0.001, 0.999)
            self.params = [self.W]
        self.input = input

        
class DeConvLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, bias=True, W=None, padding='valid',activation=T.nnet.relu):

        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        # initialize weights with random weights
        if W is None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        self.W=W

        conv_out = conv2d(
            input,
            self.W.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1],
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode=padding
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        self.output = T.clip(activation(conv_out), 0.001, 0.999)
        self.params = [self.W, self.b]
        self.input = input
        

class MaxPooling(object):
    """Pool Layer of a convolutional network """

    def __init__(self, input, poolsize=(2, 2), ignore_border=False):
        
        self.input = input
        self.poolsize = poolsize
        self.ignore_border = ignore_border

        pooled_out = pool.pool_2d(
            input=self.input,
            ds=self.poolsize,
            ignore_border=self.ignore_border,
            mode='max'
        )
        self.output = pooled_out
        #self.mask = T.grad(None, wrt=self.input, known_grads={pooled_out: T.ones_like(pooled_out)})

        
        
class ReverseMaxPooling(object):
    '''
    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights
    
    :type input: theano.tensor.dtensor4
    :param input: symbolic variable that describes the input of the
    architecture (one minibatch)
    
    :type mask: theano.tensor.dtensor4
    :param mask: symbolic variable that describes the input of the
    architecture (one minibatch)
    
    :type poolsize: tuple(int,int)
    :param poolsize: pooling size of max pooling
    
    :type ignore_border: boolean
    :param ignore_border: ignore border pixels when doing max pooling
    '''
    def __init__(self, input, mask, poolsize=(2, 2), ignore_border=True):
        self.input = input
        self.poolsize = poolsize
        self.ignore_border = ignore_border
        
        mask_pooled_out = pool.pool_2d(
            input=mask,
            ds=self.poolsize,
            ignore_border=self.ignore_border,
            mode='max'
        )
        
        #Use symbolic programming property to reshape layers with mask 
        #and maxpooled  layer
        self.output = T.grad(None, wrt=mask, known_grads={mask_pooled_out: input})
        self.input = input
        
        
        
        
        
'''
class ConvLayer(object):
    """Conv Layer of a convolutional network without Pooling """

    def __init__(self, rng, input, filter_shape, image_shape):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        self.output = conv_out

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class PoolLayer(object):
    """Pool Layer after a convolutional network """

    def __init__(self, rng, input, filter_shape=(3, 3), poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        self.input = input
   
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=input,
            ds=poolsize,
            ignore_border=True
        )
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.relu(pooled_out)

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
'''
'''        
class DeconvLayer(object):
    deconv_op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
        imshp=output_shape, 
        kshp=filter_shape, 
        border_mode='valid', 
        subsample=(32,32), 
        filter_flip = False
    )
    output = deconv_op(filters, input, output_shape[2:])
'''         
class Unpooling2D(object):
    def __init__(self, input, poolsize=(2, 2), ignore_border=True):

        self.input = input
        self.poolsize = poolsize
        self.ignore_border = ignore_border
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        self.output = T.repeat(T.repeat(input, s1, axis=2), s2, axis=3)

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output
       


def train_nn_restore(train_model, validate_model, test_model,
                     n_train_batches, n_valid_batches, n_test_batches, n_epochs,
                     verbose = True):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, MSE %f '% 
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best MSE of %f  obtained at iteration %i, '%
          (best_validation_loss, best_iter + 1))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    #global restored
    # restored = test_model(0)  # batch_size*3*32*32
    #print(test_model(0).shape)
    # print(restored.shape)
    

    '''
    plt.figure()
    restored = test_model(0)[8]
    print restored.shape
    for i in range(8):
        plt.subplot(3,8,3*i+1)
        img_original = (np.reshape(test_set_x[i],(3,32,32))).transpose(1,2,0)
        plt.imshow(img_original)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3,8,3*i+2)
        img_drop = (np.reshape(test_set_x_drop[i],(3,32,32))).transpose(1,2,0)
        plt.imshow(img_drop)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3,8,3*i+3)
        img_restored = (restored[i,:,:,:]).transpose(1,2,0)
        plt.imshow(img_restored)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('%s'%class_[test_set_y[i]])
        
    #plt.suptitle('Translated pictures')    
    '''
