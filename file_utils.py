## NOTE (CP) - this code is adapted directly from the vLGP library. if there are any issues there, of course the lines below are easy to re-write...

"""
File loading/saving functions
"""
import warnings

import h5py
import numpy as np
from numpy import exp, column_stack, roll, sum, dot
from numpy import zeros, ones, diag, arange, eye, asarray, atleast_3d, rollaxis
from scipy.linalg import svd, lstsq, toeplitz


def save(obj, fname):
    """
    Save data in HDF5
    Args:
        obj: data dict
        fname: absolute path and filename

    Returns:

    """
    with h5py.File(fname, 'w') as hf:
        for k, v in obj.items():
            try:
                hf.create_dataset(k, data=v, compression="gzip")
            except TypeError:
                msg = 'Discard unsupported type ({})'.format(k)
                warnings.warn(msg)


def load(fname):
    with h5py.File(fname, 'r') as hf:
        obj = {k: np.array(v) for k, v in hf.items()}
    return obj



import cPickle as pickle

# serialize and deserialize the sgvb models to file
def pickle_sgvb(model, fname):
    # make sure fname has a .p at the end
    if ~fname.endswith('.p')==0:
        fname = fname + '.p'
    pickle.dump(model, open(fname, "wb"))


def depickle_sgvb(fname):
    # make sure fname has a .p at the end
    if fname.endswith('.p')==0:
        fname = fname + '.p'
    return pickle.load(open(fname, "rb"))
