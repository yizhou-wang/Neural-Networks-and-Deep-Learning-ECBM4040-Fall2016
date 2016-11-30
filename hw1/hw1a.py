import os
from os import walk

import numpy as np
import numpy.linalg as linalg

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs
from theano.tensor.nnet.neighbours import neibs2images



'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,n_blocks,im_num):
    '''
    This function reconstructs an image X_recon_img given the number of
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

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    
    # print '--- @ PCs =', num_coeffs

    # print '!!!!!'
    # print 'SHAPE(c) =', c.shape
    # print 'SHAPE(D) =', D.shape
    # print '!!!!!'
    # print sz
    # print height, width

    c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    D_im = D[:,:num_coeffs]
    
    # print '!!!!!'
    # print 'SHAPE(c_im) =', c_im.shape
    # print 'SHAPE(D_im) =', D_im.shape
    # print '!!!!!'

    #TODO: Enter code below for reconstructing the image X_recon_img

    X_recon = (np.dot(D_im,c_im)).T
    
    # Defining variables
    neibs = T.matrix('neibs')
    im_new = neibs2images(neibs, (sz, sz), (height, width))
    # Theano function definition
    inv_window = theano.function([neibs], im_new)

    # Function application
    X_mean_ex = np.tile(X_mean, (n_blocks, n_blocks))
    X_recon_img = inv_window(X_recon) + X_mean_ex

    # print '!!!!!'
    # print X_recon_img.shape
    # print '!!!!!'

    return X_recon_img

    

def plot_reconstructions(D,c,num_coeff_array,X_mean,n_blocks,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,n_blocks,im_num), cmap=cm.Greys_r)
    
    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
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
    Read here all images(grayscale) from Fei_256 folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Read all images into a numpy array of size (no_images, height, width)
    
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
    im = np.zeros((no_images, 1, height, width))
    for no in range(no_images):
        im_names[no] = 'Fei_256/' + files[no] 
        im[no,0,:,:] = np.array(Image.open(im_names[no]))

    # plt.imshow(im[0,0,:,:], cmap=cm.Greys_r)
    # plt.show()

    print no+1, 'images sucessfully Loaded!'

    szs = [8, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)] 

    print '* Dividing the image array...\n'
    global sz
    for sz, nc in zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        ''' 
        
        #TODO: Write a code snippet that performs as indicated in the above comment
        
        print '- Block Size:', sz, 'X', sz, '-'


        # Defining variables
        images = T.tensor4('images')
        neibs = images2neibs(images, neib_shape=(sz, sz))
        # Constructing theano function
        window_function = theano.function([images], neibs)

        # Function application
        X = window_function(im)
        print '-- The image array has been transfered into a', X.shape, 'matrix.'

        X_mean = np.mean(X, 0)
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)
        # plt.imshow(X, cmap=cm.Greys_r)
        # plt.show()
        # print X.shape

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        
        #TODO: Write a code snippet that performs as indicated in the above comment
        
        print '-- * Performing eigendecomposition on X^T X...'
        X_var = np.dot(X.T, X)
        # print X_var

        (X_eigval,X_eigvec) = linalg.eigh(X_var)

        idx = X_eigval.argsort()[::-1]   
        X_eigval = X_eigval[idx]
        X_eigvec = X_eigvec[:,idx]

        D = X_eigvec
        print D
        print X_eigval
        c = np.dot(D.T, X.T)
        # print D.shape
        
        print '-- * Eigendecomposition finished!'

        print '-- @ Reconstructing images...'

        for i in range(0, 200, 10):
            plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean.reshape((sz, sz)), n_blocks=int(256/sz), im_num=i)
            print 'No.', i, 'image reconstruction succeeded!'


        print '-- @ Images Reconstructed!'

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))
        print '-- @ The top 16 components saved!\n'

    print 'Mission Completed!'


if __name__ == '__main__':
    main()
    
    