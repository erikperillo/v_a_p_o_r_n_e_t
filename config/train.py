import os
import glob
import numpy as np

from . import datapreproc

_data_dir_path = "/home/erik/proj/att/att/deep/data"

#if not None, uses weights of a pre-trained model from path
pre_trained_model_fp = "/home/erik/data/vgg16.pkl"

#directory where dir with train info/model will be stored
output_dir_basedir = _data_dir_path

_dataset_filepaths = None

#filepaths of train batches
dataset_train_filepaths = None
#filepaths of validation batches, can be None
dataset_val_filepaths = None
#if not None, ignores dataset_{train,val}_filepaths and uses this as source
dataset_filepath = "/home/erik/proj/att/att/deep/config/wow/data.gz"
#validation fraction. ignored if dataset_filepath is not used
val_frac = 0.1

#number of epochs to use in train
n_epochs = 10
#batch size
batch_size = 10
#maximum number of iterations
max_its = None
#0 for nothing, 1 for only warnings, 2 for everything
verbose = 2
#validation function value tolerance
val_f_val_tol = None
#data types
x_dtype = np.float32
y_dtype = np.float32
