#!/usr/bin/env python3

"""
This script trains a model with architecture/data specified in configuration.
It produces a directory with trained model and other stats.
"""

import theano.tensor as T
import theano
import numpy as np
import shutil
import sys
import os

import trloop
import util
import config.train as cfg
import config.model as model

def load_dataset(filepath):
    """
    Loads dataset.
    """
    return util.unpkl(filepath)

def tr_val_split(X, y, val_frac):
    """
    Splits X, y into train and validation sets.
    """
    val = int(val_frac*len(y))
    tr_slc = slice(0, -val if val > 0 else None)
    val_slc = slice(-val, None if val > 0 else 0)

    X_tr, y_tr = X[tr_slc], y[tr_slc]
    X_val, y_val = X[val_slc], y[val_slc]

    return X_tr, y_tr, X_val, y_val

def load_formatted_dataset(filepath, val_frac):
    """
    Loads dataset and splits it.
    """
    print("\tloading data...")
    X, y = load_dataset(filepath)
    print("\tX shape: {} | y shape: {}".format(X.shape, y.shape))

    print("\tsplitting...")
    X_tr, y_tr, X_val, y_val = tr_val_split(X, y, val_frac)
    print("\tX_tr shape: {} | y_tr shape: {}".format(X_tr.shape, y_tr.shape))
    print("\tX_val shape: {} | y_val shape: {}".format(X_val.shape,
        y_val.shape))

    print("\treshaping...")
    X_tr = X_tr.reshape((X_tr.shape[0],) + model.Model.INPUT_SHAPE)
    X_val = X_val.reshape((X_val.shape[0],) + model.Model.INPUT_SHAPE)
    y_tr = y_tr.reshape((y_tr.shape[0],) + model.Model.OUTPUT_SHAPE)
    y_val = y_val.reshape((y_val.shape[0],) + model.Model.OUTPUT_SHAPE)
    print("\tX_tr shape: {} | y_tr shape: {}".format(X_tr.shape, y_tr.shape))
    print("\tX_val shape: {} | y_val shape: {}".format(X_val.shape,
        y_val.shape))

    return X_tr, y_tr, X_val, y_val

def mk_output_dir(base_dir, pattern="train"):
    """
    Creates dir to store model.
    """
    #creating dir
    out_dir = util.uniq_filepath(base_dir, pattern)
    os.makedirs(out_dir)
    return out_dir

def populate_output_dir(out_dir):
    """
    Populates outout dir with info files.
    """
    #copying model generator file to dir
    shutil.copy(model.__file__, os.path.join(out_dir, "model.py"))
    #copying this file to dir
    shutil.copy(cfg.__file__, os.path.join(out_dir, "config.py"))
    #info file
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        print("date created (y-m-d):", util.date_str(), file=f)
        print("time created:", util.time_str(), file=f)
        print("git commit hash:", util.git_hash(), file=f)

def main():
    #theano variables for inputs and targets
    input_var = T.tensor4("inputs", dtype="floatX")
    target_var = T.tensor4("targets", dtype="floatX")

    out_dir = mk_output_dir(cfg.output_dir_basedir)
    print("created output dir '%s'..." % out_dir)
    populate_output_dir(out_dir)

    #neural network model
    print("building network...", flush=True)
    if cfg.pre_trained_model_fp is not None:
        print("loading pre-trained model from '%s'" % cfg.pre_trained_model_fp,
            flush=True)
    net_model = model.Model(input_var, target_var, cfg.pre_trained_model_fp)

    print("compiling functions...", flush=True)
    #compiling function performing a training step on a mini-batch (by giving
    #the updates dictionary) and returning the corresponding training loss
    train_fn = theano.function([input_var, target_var],
        net_model.train_loss, updates=net_model.updates)
    #second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var],
        [net_model.test_loss, net_model.mae])

    print("getting data filepaths...", flush=True)
    #iterative loading of dataset from disk
    if cfg.dataset_filepath is None:
        print("using iterative loading of data from disk")
        tr_set = cfg.dataset_train_filepaths
        val_set = cfg.dataset_val_filepaths
        print("train set:", tr_set)
        print("validation set:", val_set)
    #single-time loading of dataset from disk
    else:
        print("using single-time loading of data from disk")
        X_tr, y_tr, X_val, y_val = load_formatted_dataset(cfg.dataset_filepath,
            cfg.val_frac)
        tr_set = X_tr, y_tr
        if X_val.shape[0] > 0 and y_val.shape[0] > 0:
            val_set = X_val, y_val
        else:
            val_set = None

    print("calling train loop")
    #creating logging object
    log = util.Tee([sys.stdout, open(os.path.join(out_dir, "train.log"), "w")])
    try:
        trloop.train_loop(
            tr_set=tr_set, tr_f=train_fn,
            n_epochs=cfg.n_epochs, batch_size=cfg.batch_size,
            val_set=val_set, val_f=val_fn, val_f_val_tol=cfg.val_f_val_tol,
            max_its=cfg.max_its,
            verbose=cfg.verbose, print_f=log.print)
    except KeyboardInterrupt:
        print("Keyboard Interrupt event.")

    print("end.")

    model_path = os.path.join(out_dir, "model.npz")
    print("saving model to '{}'".format(model_path))
    net_model.save_net(model_path)

    print("\ndone.")

if __name__ == '__main__':
    main()

