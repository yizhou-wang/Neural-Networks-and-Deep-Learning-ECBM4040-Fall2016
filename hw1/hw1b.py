import os
os.environ["THEANO_FLAGS"] = "floatX=float32"
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg

from PIL import Image

import theano
import theano.tensor as T
THEANO_FLAGS = 'floatX=float32'
print theano.config.floatX


'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,im_num):
    '''
    This function reconstructs an image given the number of
    coefficients for each image specified by num_coeffs
    '''
    
    '''
        Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mean: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Integer
        an integer that specifies the number of top components to be
        considered while reconstructing
    '''
    
    c_im = c[:num_coeffs,im_num]
    D_im = D[:,:num_coeffs]
    
    #TODO: Enter code below for reconstructing the image

    X_recon = (np.dot(D_im,c_im)).T
    
    X_recon_img = X_recon.reshape((height,width)) + X_mean

    # print '!!!!!'
    # print X_recon_img.shape
    # print '!!!!!'

    return X_recon_img

def plot_reconstructions(D,c,num_coeff_array,X_mean,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number_of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,im_num), cmap=cm.Greys_r)
            
    f.savefig('output/hw1b_{0}.png'.format(im_num))
    plt.close(f)
    
    
    
def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    #TODO: Obtain top 16 components of D and plot them
    
    D_top_16 = D[:,0:16]
    # print D_top_16.shape
    # print D_top_16

    f, axarr = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            plt.axes(axarr[i,j])
            pc = D_top_16[:,i*4+j].reshape((sz,sz))
            # print pc.shape
            plt.imshow(pc, cmap=cm.Greys_r)
    
    f.savefig(imname)
    plt.close(f)



import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

   
def main():
    '''
    Read here all images(grayscale) from Fei_256 folder and collapse 
    each image to get an numpy array Ims with size (no_images, height*width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Write a code snippet that performs as indicated in the above comment

    print '* Checking files...'   
    for root, dirs, files in walk('Fei_256'):
        print 'INFO: There are totally', len(files), 'files in the directory.'
        
    files = [ fi for fi in files if fi.endswith(".jpg") ]
    files = natural_sort(files)

    global no_images
    no_images = len(files)
    im1 = np.array(Image.open("Fei_256/image0.jpg"))
    global height, width
    (height, width) = im1.shape

    print 'INFO: There are totally', no_images, 'images in the directory.'
    print 'INFO: The size of each image is', height, 'X', width, 'pixels.'
    print '* Loading images...'

    im_names = ["" for x in range(no_images)]
    Ims = np.zeros((no_images, height*width))
    for no in range(no_images):
        im_names[no] = 'Fei_256/' + files[no] 
        im = np.array(Image.open(im_names[no]))
        Ims[no,:] = im.reshape((1,height*width))



    
    Ims = Ims.astype(np.float32)
    X_mean = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mean.reshape(1, -1), Ims.shape[0], 0)
    X = np.array(X, dtype=np.float32)
    X_init = X

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''
    
    #TODO: Write a code snippet that performs as indicated in the above comment

    num_coeffs = 16
    learning_rate = 0.1
    max_iterations = 200
    tol = 1e-8

    X_evals = np.zeros((num_coeffs,1))
    X_evecs = np.zeros((height*width,num_coeffs))


    # ran_v = np.random.rand(height*width)
    # global init_v
    # init_v = ran_v / linalg.norm(ran_v) 

    print 'Calculating PCs...'

    for i in xrange(num_coeffs):

        print '- Number of PC:', i+1, '-'

        # ... initialize variables and expressions ...

        ran_v = np.random.rand(height*width)
        init_v = ran_v / linalg.norm(ran_v)
        init_v = np.array(init_v, dtype=np.float32)

        #print '\n$$$$$$$$$$$$$$$$$$$$'
        #print X.shape
        print init_v.shape

        Xv = np.squeeze(np.asarray(np.dot(X,init_v)))

        print Xv.shape
        print '$$$$$$$$$$$$$$$$$$$$\n'

        X_evals[i] = np.dot(Xv.T, Xv)
        X_evecs[:,i] = init_v


        v = theano.shared(init_v, name="v")
        X_ = T.matrix('X_', dtype='float32')
        X_v = T.dot(X_, v)

        # eigenvalues & eigenvectors!!!!!!!! of what???????????
        # X_var = T.dot(X_.T, X_)
        # (evals,evecs) = T.nlinalg.eigh(X_var)
        # idx = evals.argsort()[::-1]   
        # evals = evals[idx]
        # evecs = evecs[:,idx]

        # d_cost = np.sum(X_evals[j]*T.dot(X_evecs[:,j], v)*T.dot(X_evecs[:,j], v) for j in xrange(i))
        # print type(d_cost)
        cost = T.dot(X_v.T, X_v)
        # print type(cost)
        gv = T.grad(cost, v)
        y = v + learning_rate*gv
        update_expression = y / y.norm(2)

        evals = T.dot(X_v.T, X_v)
        evecs = v
        d_v = (update_expression - v).norm(2)


        # ... initialize theano train function ...
        train = theano.function(inputs=[X_], outputs=[evals,evecs,d_v], updates=((v, update_expression),) )

        print '-- * Iterating...'
        t = 1
        while t < max_iterations:
            
            # print 'Iteration times:', t
            # print X
            (X_evals[i],X_evecs[:,i],d_v) = train(X)
            # print d_v
            if d_v < tol:
                break
            t = t + 1

        print '-- * Iteration finished!'

        # ... return eigenvalues and eigenvectors ...
        # print X_evals
        # print X_evecs

        temp1 = np.matrix(np.dot(X,X_evecs[:,i]))
        temp2 = np.matrix(X_evecs[:,i].T)
        X = X - np.dot(temp1.T,temp2)
        X = np.array(X, dtype=np.float32)

    D = X_evecs
    # print D
    # print X_evals
    c = np.dot(D.T, X_init.T)


    print '\n-- @ Reconstructing images...'

        
    for i in range(0, 200, 10):
        plot_reconstructions(D=D, c=c, num_coeff_array=[1, 2, 4, 6, 8, 10, 12, 14, 16], X_mean=X_mean.reshape((256, 256)), im_num=i)
        print 'No.', i, 'image reconstruction succeeded!'


    print '-- @ Images Reconstructed!'

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')
    print '-- @ The top 16 components saved!\n'

print 'Mission Completed!'


if __name__ == '__main__':
    main()
    
    