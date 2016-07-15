# simple script to test how to project data into the high dimensional space (i.e., neural)


# must specify where the vilds code is
PATH_TO_VLDS = 'submodules/vilds/'

# import the data
import sys
import file_utils

import numpy as np

import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tla
import lasagne       # the library we're using for NN's
# import the nonlinearities we might use
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify, sigmoid
from theano.tensor.shared_randomstreams import RandomStreams
from numpy.random import *
from matplotlib import pyplot as plt

import cPickle

# I always initialize random variables here.
msrng = RandomStreams(seed=20150503)
mnrng = np.random.RandomState(20150503)


theano.config.optimizer = 'fast_compile'

# Load our code

# Add all the paths that should matter right now
sys.path.append(PATH_TO_VLDS + 'code')  # support files (mathematical tools, mostly)
sys.path.append(PATH_TO_VLDS + '/code/lib')  # support files (mathematical tools, mostly)

from GenerativeModel import *       # Class file for generative models.
from RecognitionModel import *      # Class file for recognition models
from SGVB import *                  # The meat of the algorithm - define the ELBO and initialize Gen/Rec model



spikefile = '/mnt/cache/chethan/16lfads/rnn_no_thits/vilds/data/vilds_spikes.h5'
n_latent = 10;
print('loading data from: ' + spikefile)
data_in = file_utils.load(spikefile)

print 'training dimensions:' + str(data_in['train'].shape)
print 'validation dimensions:' + str(data_in['valid'].shape)

n_train_trials = data_in['train'].shape[0]
n_valid_trials = data_in['valid'].shape[0]
n_neurons = data_in['train'].shape[2]
if n_neurons != data_in['valid'].shape[2]:
    raise('training and validation do not match')

n_timebins = data_in['train'].shape[1]
if n_timebins != data_in['valid'].shape[1]:
    raise('training and validation do not match')

print 'training trials: ' + str(n_train_trials)
print 'validation trials: ' + str(n_valid_trials)
print 'num timebins: ' + str(n_timebins)
print 'num neurons: ' + str(n_neurons)
print 'latent dimensionality: ' + str(n_latent)

# the variables 'yDim' and 'xDim' are used later in the code.
yDim = n_neurons
xDim = n_latent

import numpy as np
# the full data in the "y_data" variable is used to set the network means
y_data_train = np.copy(data_in['train']);
y_data_valid = np.copy(data_in['valid']);
y_data = np.concatenate( ( y_data_train.reshape(n_train_trials * n_timebins, n_neurons),
                           y_data_valid.reshape(n_valid_trials * n_timebins, n_neurons)),
                         axis=0)


saved_model = '/mnt/cache/chethan/16lfads/rnn_no_thits/vilds/model/model.p'
sgvb = file_utils.depickle_sgvb(saved_model)

print sgvb


#posterior_means_train=[]
#for itr in np.arange(y_data_train.shape[0]):
#    posterior_means_train.append(sgvb.mrec.postX.eval({sgvb.Y: y_data_train[itr]}))

posterior_means_valid=[]
for itr in np.arange(y_data_valid.shape[0]):
    posterior_means_valid.append(sgvb.mrec.postX.eval({sgvb.Y: y_data_valid[itr]}))

y_data_train_recap = []
for itr in np.arange(len(posterior_means_valid)):
    y_data_train_recap.append(sgvb.mprior.rate.eval({sgvb.mprior.Xsamp: posterior_means_valid[itr]}))


print y_data_train_recap[0].shape
