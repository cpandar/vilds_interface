#!/usr/bin/python

# code to run vilds. based off of the vilds tutorial in their codepack
# (must pull the codepack from https://github.com/earcher/vilds)

# calling syntax:
#   run_vilds infile.hf5 outfile.hf5 [num_latents] 
# num_latents defaults to 3 if not specified

# must specify where the vilds code is
PATH_TO_VLDS = 'submodules/vilds/'

# import the data
import sys
import file_utils

# handle command line args
# caller should pass in a filename for data to be loaded
assert len(sys.argv)>=3, 'need to pass in a filename to load, an output filename for model, and output filename for estimates'
spikefile = sys.argv[1]
model_outfile = sys.argv[2]
estimates_outfile = sys.argv[3]

print "Writing model out to: " + model_outfile
print "Writing estimates out to: " + estimates_outfile

if len(sys.argv)>4:
    n_latent = int(sys.argv[4])
else:
    n_latent = 3;
    print "Assuming latent dimensionality of " + str(n_latent)

print('loading data from: ' + spikefile)
data_in = file_utils.load(spikefile)

# data comes in with as data_in['train'] and data_in['valid']
# each is nTrials x nTimebins x nNeurons

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

print y_data.shape

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



## network definition stuff

########################################
# Describe network for mapping into means
NN_Mu = lasagne.layers.InputLayer((None, yDim))
NN_Mu = lasagne.layers.DenseLayer(NN_Mu, 25, nonlinearity=tanh, W=lasagne.init.Orthogonal())
#--------------------------------------
# let's initialize the first layer to have 0 mean wrt our training data
W0 = np.asarray(NN_Mu.W.get_value(), dtype=theano.config.floatX)
NN_Mu.W.set_value( (W0 / np.dot(y_data, W0).std(axis=0)).astype(theano.config.floatX) )
W0 = np.asarray(NN_Mu.W.get_value(), dtype=theano.config.floatX)
b0 = (-np.dot(y_data, W0).mean(axis=0)).astype(theano.config.floatX)
NN_Mu.b.set_value(b0)
#--------------------------------------
NN_Mu = lasagne.layers.DenseLayer(NN_Mu, xDim, nonlinearity=linear, W=lasagne.init.Normal())
NN_Mu.W.set_value(NN_Mu.W.get_value()*10)
NN_Mu = dict([('network', NN_Mu)])

########################################
# Describe network for mapping into Covariances
NN_Lambda = lasagne.layers.InputLayer((None, yDim))
NN_Lambda = lasagne.layers.DenseLayer(NN_Lambda, 25, nonlinearity=tanh, W=lasagne.init.Orthogonal())
#--------------------------------------
# let's initialize the first layer to have 0 mean wrt our training data
W0 = np.asarray(NN_Lambda.W.get_value(), dtype=theano.config.floatX)
NN_Lambda.W.set_value( (W0 / np.dot(y_data, W0).std(axis=0)).astype(theano.config.floatX) )
W0 = np.asarray(NN_Lambda.W.get_value(), dtype=theano.config.floatX)
b0 = (-np.dot(y_data, W0).mean(axis=0)).astype(theano.config.floatX)
NN_Lambda.b.set_value(b0)
#--------------------------------------
NN_Lambda = lasagne.layers.DenseLayer(NN_Lambda, xDim*xDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
NN_Lambda.W.set_value(NN_Lambda.W.get_value()*10)
NN_Lambda = dict([('network', NN_Lambda)])

########################################
# define dictionary of recognition model parameters
recdict = dict([('A'     , .9*np.eye(xDim)),
                ('QinvChol',  np.eye(xDim)), #np.linalg.cholesky(np.linalg.inv(np.array(tQ)))),
                                ('Q0invChol', np.eye(xDim)), #np.linalg.cholesky(np.linalg.inv(np.array(tQ0)))),
                                ('NN_Mu' ,NN_Mu),
                                ('NN_Lambda',NN_Lambda),
                                ])

########################################
# We can instantiate a recognition model alone and sample from it.
# First, we have to define a Theano dummy variable for the input observations the posterior expects:
Y = T.matrix()

rec_model = SmoothingLDSTimeSeries(recdict, Y, xDim, yDim, srng = msrng, nrng = mnrng)


# initialize training with a random generative model (that we haven't generated data from):
initGenDict = dict([
             ('output_nlin' , 'softplus')
                 ])
# Instantiate an SGVB class:
sgvb = SGVB(initGenDict, PLDS, recdict, SmoothingLDSTimeSeries, xDim = xDim, yDim = yDim)

########################################
# Define a bare-bones thenao training function
batch_y = T.matrix('batch_y')

########################################
# choose learning rate and batch size
learning_rate = 1e-2
batch_size = n_timebins

########################################
# use lasagne to get adam updates
updates = lasagne.updates.adam(-sgvb.cost(), sgvb.getParams(), learning_rate=learning_rate)

########################################
# Finally, compile the function that will actually take gradient steps.
train_fn = theano.function(
    outputs=sgvb.cost(),
    inputs=[theano.In(batch_y)],
    updates=updates,
    givens={sgvb.Y: batch_y},
)


import math
## CP - modifying the original iterator class to work with trial-ized data
class DatasetMiniBatchIndexIterator(object):
    """ Mini-batch iterator to iterate over trials"""
    def __init__(self, y):
        self.y = y
        from sklearn.utils import check_random_state
        self.rng = np.random.RandomState(np.random.randint(12039210))

    def __iter__(self):
        n_trials = self.y.shape[0]
        for _ in xrange(n_trials):
            i = int(math.floor(self.rng.rand(1) * n_trials))
            # flatten the data for output
            yield [self.y[i], i]


# initialize the iterator with training data
yiter = DatasetMiniBatchIndexIterator(y_data_train)

########################################
# Iterate over the training data for the specified number of epochs
n_epochs = 20
cost = []
for ie in np.arange(n_epochs):
    print('--> entering epoch %d' % ie)
    for y, _ in yiter:
        cost.append(train_fn(y))
    print cost[-1]
#    cost.append(train_fn(y_data_train[0]))



model = sgvb
# now that model is trained, we should be able to calculate some posterior mean estimates
posterior_means_train=[]
for itr in np.arange(y_data_train.shape[0]):
    posterior_means_train.append(sgvb.mrec.postX.eval({sgvb.Y: y_data_train[itr]}))

posterior_means_valid=[]
for itr in np.arange(y_data_valid.shape[0]):
    posterior_means_valid.append(sgvb.mrec.postX.eval({sgvb.Y: y_data_valid[itr]}))

posterior_means_y_train=[]
for itr in np.arange(len(posterior_means_train)):
    posterior_means_y_train.append(sgvb.mprior.rate.eval({sgvb.mprior.Xsamp: posterior_means_train[itr]}))

posterior_means_y_valid=[]
for itr in np.arange(len(posterior_means_valid)):
    posterior_means_y_valid.append(sgvb.mprior.rate.eval({sgvb.mprior.Xsamp: posterior_means_valid[itr]}))







# save the model via cpickle
file_utils.pickle_sgvb(sgvb, model_outfile)

# save the posterior estimates 
dict_out = {'posterior_means_train' : posterior_means_train,
            'posterior_means_valid' : posterior_means_valid,
            'posterior_means_y_train' : posterior_means_y_train,
            'posterior_means_y_valid' : posterior_means_y_valid}

file_utils.save(dict_out, estimates_outfile)

