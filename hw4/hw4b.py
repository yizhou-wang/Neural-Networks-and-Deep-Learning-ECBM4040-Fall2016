"""
Source Code for Homework 4.b of ECBM E4040, Fall 2016, Columbia University

"""

import os
import timeit
import inspect
import sys
import numpy
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from hw4_utils import contextwin, shared_dataset, load_data, shuffle, conlleval, check_dir
from hw4_nn import myMLP, train_nn

def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(num, nbit))
    Y = numpy.mod(numpy.sum(X, axis=1), 2)
    
    return X, Y

def gen_parity_pair_2(nbit, num, flag):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = np.random.randint(2, size=(num,nbit)).astype('float32')
    if flag == True:
        Y = np.zeros((num,nbit)).astype('int64')
        for index in range(X.shape[1]):
            Y[:,index] = np.mod(np.sum(X[:, :index+1], axis=1), 2).astype('int64')
    else:
        Y = np.mod(np.sum(X, axis=1), 2)
        
    return X, Y

#TODO: implement RNN class to learn parity function
class RNN(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, n_in, nh, n_out, normal=True):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        
        # parameters of the model
        self.wx = theano.shared(name='wx',
                                value=numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=numpy.random.uniform(-1.0, 1.0,
                               (nh, n_out))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(n_out,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        # bundle
        self.params = [self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0]

        x = T.matrix()
        y = T.ivector()
        #x = input

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            #s_t = T.dot(h_t, self.w) + self.b
            return [h_t, s_t]
        
        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps = x.shape[0])
        p_y_given_x = s[:, 0, :]
        '''
        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None])
        print (s.shape)
        p_y_given_x = s
        '''
        #self.p_y_given_x = T.nnet.softmax(p_y_given_x)
        #self.p_y_given_x = self.p_y
        self.y_pred = T.argmax(p_y_given_x[-1,:])

        #self.L1 = abs(self.w.sum()) + abs(self.wx.sum()) + abs(self.wh.sum())                                  
        #self.L2_sq = (self.w ** 2).sum() + (self.wx ** 2).sum() + (self.wh ** 2).sum()

        lr = T.scalar('lr')

        nll = -T.mean(T.log(p_y_given_x)[T.arange(x.shape[0]), y])
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p - lr*g)
                              for p, g in
                              zip(self.params, gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=self.y_pred)
        self.train = theano.function(inputs=[x, y, lr],
                                              outputs=nll,
                                              updates=updates)
        
        self.normal = normal 
    

#TODO: implement LSTM class to learn parity function
class LSTM(object):

    def __init__(self, n_in, nh, n_out, normal=True):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        
        # parameters of the model
        self.ui = theano.shared(name='ui',
                                value=numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.uf = theano.shared(name='uf',
                                value=numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.uo = theano.shared(name='uo',
                                value=numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.ug = theano.shared(name='ug',
                                value=numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.wi = theano.shared(name='wi',
                                value=numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wf = theano.shared(name='wf',
                                value=numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wo = theano.shared(name='wo',
                                value=numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wg = theano.shared(name='wg',
                                value=numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.bi = theano.shared(name='bi',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bf = theano.shared(name='bf',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo = theano.shared(name='bo',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bg = theano.shared(name='bg',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.c0 = theano.shared(name='c0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
#        self.sigma = theano.shared(name='sigma',
#                                   value=numpy.ones(1,
#                                   dtype=theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=numpy.random.uniform(-1.0, 1.0,
                               (nh, n_out))
                               .astype(theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(n_out,
                               dtype=theano.config.floatX))
        
        # bundle
        self.params = [self.ui, self.uf, self.uo, self.ug,
                       self.wi, self.wf, self.wo, self.wg, 
                       self.bi, self.bf, self.bo, self.bg, 
                       self.c0, self.h0, self.w, self.b]

        x = T.matrix()
        y = T.ivector()
        #x = input

        def recurrence(x_t, c_tm1, h_tm1):
            i = T.nnet.sigmoid(T.dot(x_t, self.ui) + T.dot(h_tm1, self.wi) + self.bi)
            f = T.nnet.sigmoid(T.dot(x_t, self.uf) + T.dot(h_tm1, self.wf) + self.bf)
            o = T.nnet.sigmoid(T.dot(x_t, self.uo) + T.dot(h_tm1, self.wo) + self.bo)
            g = T.tanh(T.dot(x_t, self.ug) + T.dot(h_tm1, self.wg) + self.bg)
            c_t = c_tm1 * f + g * i
            h_t = T.tanh(c_t) * o
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [c_t, h_t, s_t]

        [c, h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.c0, self.h0, None],
                                n_steps = x.shape[0])
        
        p_y_given_x = s[:, 0, :]
        
        #self.p_y_given_x = T.nnet.softmax(p_y_given_x)
        #self.p_y_given_x = self.p_y
        self.y_pred = T.argmax(p_y_given_x[-1,:])

        #self.L1 = abs(self.w.sum()) + abs(self.wx.sum()) + abs(self.wh.sum())                                  
        #self.L2_sq = (self.w ** 2).sum() + (self.wx ** 2).sum() + (self.wh ** 2).sum()

        lr = T.scalar('lr')

        nll = -T.mean(T.log(p_y_given_x)[T.arange(x.shape[0]), y])
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p - lr*g)
                              for p, g in
                              zip(self.params, gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=self.y_pred)
        self.train = theano.function(inputs=[x, y, lr],
                                              outputs=nll,
                                              updates=updates)
        
        self.normal = normal 



#TODO: build and train a MLP to learn parity function
def test_mlp_parity(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
                    batch_size=20, n_hidden=500, verbose=False, n_hl=1, n_bit=8):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)
    

    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=n_bit,
        n_hidden=n_hidden,
        n_out=2,
        n_hiddenLayers=n_hl,
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)    

    
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
                    batch_size=20, n_hidden=500, verbose=False, n_bit=8):
    
    # generate datasets
    #train_set = gen_parity_pair(n_bit, 1000)
    #valid_set = gen_parity_pair(n_bit, 500)
    #test_set  = gen_parity_pair(n_bit, 100)
    train_set = gen_parity_pair_2(n_bit, 1000, True)
    valid_set = gen_parity_pair_2(n_bit, 500, True)
    test_set = gen_parity_pair_2(n_bit, 100, True)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set
    #train_set_x, train_set_y = shared_dataset(train_set)
    #valid_set_x, valid_set_y = shared_dataset(valid_set)
    #test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x = numpy.asarray(train_set_x, dtype=theano.config.floatX)
    valid_set_x = numpy.asarray(valid_set_x, dtype=theano.config.floatX)
    test_set_x = numpy.asarray(test_set_x, dtype=theano.config.floatX)
    

    # process input arguments
    param = {
        'fold': 3,
        'lr': learning_rate,
        'verbose': True,
        'decay': True,
        'nhidden': n_hidden,
        'seed': 345,
        'nepochs': n_epochs,
        'savemodel': False,
        'normal': True,
        'folder':'../result'}

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

    # instanciate the model
    numpy.random.seed(param['seed'])

    print('... building the model')
    rnn = RNN(n_in=n_bit, nh=n_hidden, n_out=2)

    # train with early stopping on validation set
    print('... training')
    best_f1 = -np.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):

        #print train_set_x.shape
        #print train_set_y.shape
      
        for i, (x, y) in enumerate(zip(train_set_x, train_set_y)):
            #print x.shape
            #print y.shape
            rnn.train(x.reshape((n_bit,1)), y.astype('int32'), param['clr'])
            
        # evaluation // back into the real world : idx -> words
        predictions_test = np.array([rnn.classify(x.reshape((n_bit,1))) for x in test_set_x])
        validations_test = np.array([rnn.classify(x.reshape((n_bit,1))) for x in valid_set_x])
        
        # evaluation // compute the accuracy using conlleval.pl
        test_accuracy = ((predictions_test.reshape(1,test_set_x.shape[0])==test_set_y[:,-1]).sum()*100.0)/test_set_x.shape[0]
        valid_accuracy = ((validations_test.reshape(1,valid_set_x.shape[0])==valid_set_y[:,-1]).sum()*100.0)/valid_set_x.shape[0]
        
        best_val_acc = -  (np.inf)
        best_test_acc = -  (np.inf)
        if (best_val_acc < valid_accuracy):
            best_val_acc = valid_accuracy
            best_test_acc = test_accuracy

            
        if param['verbose']:
            print('NEW BEST: epoch', e,
                      'valid F1', valid_accuracy,
                      'best test F1', test_accuracy)

            
    print('BEST RESULT: epoch', e,
           'valid F1', best_val_acc,
           'best test F1', best_test_acc)
    
    
#TODO: build and train a LSTM to learn parity function
def test_lstm_parity(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
                     batch_size=20, n_hidden=500, verbose=False, n_bit=8):
    
    # generate datasets
    train_set = gen_parity_pair_2(n_bit, 1000, True)
    valid_set = gen_parity_pair_2(n_bit, 500, True)
    test_set = gen_parity_pair_2(n_bit, 100, True)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set
    train_set_x = numpy.asarray(train_set_x, dtype=theano.config.floatX)
    valid_set_x = numpy.asarray(valid_set_x, dtype=theano.config.floatX)
    test_set_x = numpy.asarray(test_set_x, dtype=theano.config.floatX)   

    # process input arguments
    param = {
        'fold': 3,
        'lr': learning_rate,
        'verbose': True,
        'decay': True,
        'nhidden': n_hidden,
        'seed': 345,
        'nepochs': n_epochs,
        'savemodel': False,
        'normal': True,
        'folder':'../result'}

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

    # instanciate the model
    numpy.random.seed(param['seed'])

    print('... building the model')
    rnn = LSTM(n_in=n_bit, nh=n_hidden, n_out=2)

    # train with early stopping on validation set
    print('... training')
    best_f1 = -np.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):

      
        for i, (x, y) in enumerate(zip(train_set_x, train_set_y)):
            #print x.shape
            #print y.shape
            rnn.train(x.reshape((n_bit,1)), y.astype('int32'), param['clr'])
            
        # evaluation // back into the real world : idx -> words
        predictions_test = np.array([rnn.classify(x.reshape((n_bit,1))) for x in test_set_x])
        validations_test = np.array([rnn.classify(x.reshape((n_bit,1))) for x in valid_set_x])
        
        # evaluation // compute the accuracy using conlleval.pl
        test_accuracy = ((predictions_test.reshape(1,test_set_x.shape[0])==test_set_y[:,-1]).sum()*100.0)/test_set_x.shape[0]
        valid_accuracy = ((validations_test.reshape(1,valid_set_x.shape[0])==valid_set_y[:,-1]).sum()*100.0)/valid_set_x.shape[0]
        
        best_val_acc = -  (np.inf)
        best_test_acc = -  (np.inf)
        if (best_val_acc < valid_accuracy):
            best_val_acc = valid_accuracy
            best_test_acc = test_accuracy

            
        if param['verbose']:
            print('NEW BEST: epoch', e,
                      'valid F1', valid_accuracy,
                      'best test F1', test_accuracy)

            
    print('BEST RESULT: epoch', e,
           'valid F1', best_val_acc,
           'best test F1', best_test_acc)
    

    
if __name__ == '__main__':
    test_mlp_parity()
