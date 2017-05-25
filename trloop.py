import numpy as np
import util
from config import model
import config.train as cfg
import time
import theano

def _str_fmt_time(seconds):
    int_seconds = int(seconds)
    hours = int_seconds//3600
    minutes = (int_seconds%3600)//60
    seconds = int_seconds%60 + (seconds - int_seconds)
    return "%.2dh:%.2dm:%.3fs" % (hours, minutes, seconds)

def _silence(*args, **kwargs):
    pass

def batches_gen(X, y, batch_size, shuffle=False):
    n_samples = len(y)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, max(n_samples-batch_size+1, 1), batch_size):
        excerpt = indices[start_idx:start_idx+batch_size]
        yield X[excerpt], y[excerpt]

def batches_gen_iter(filepaths, batch_size, shuffle=False, print_f=_silence):
    for fp in filepaths:
        msg = "    [loading file '{}'...]".format(fp)
        print_f(msg, end="\r", flush=True)
        X, y = util.unpkl(fp)
        X = X.astype(cfg.x_dtype, casting="same_kind")
        X = X.reshape((X.shape[0],) + model.Model.INPUT_SHAPE)
        y = y.astype(cfg.y_dtype, casting="same_kind")
        y = y.reshape((y.shape[0],) + model.Model.OUTPUT_SHAPE)
        print_f(len(msg)*" ", end="\r")
        #print("\nX: dtype={}, shape={}, isnanc={}, isinfc={}".format(
        #    X.dtype, X.shape, np.isnan(X).sum(), np.isinf(X).sum()))
        #print(min(x.std() for x in X))
        #print("y: dtype={}, shape={}, isnanc={}, isinfc={}".format(
        #    y.dtype, y.shape, np.isnan(y).sum(), np.isinf(y).sum()))
        #print(min(x.std() for x in y))
        for batch_X, batch_y in batches_gen(X, y, batch_size, shuffle):
            yield batch_X, batch_y

def _inf_gen():
    n = 0
    while True:
        yield n
        n += 1

def _str(obj):
    return str(obj) if obj is not None else "?"

def train_loop(
    tr_set, tr_f,
    n_epochs=10, batch_size=1,
    val_set=None, val_f=None, val_f_val_tol=None,
    max_its=None,
    verbose=2, print_f=print):
    """
    General Training loop.
    Parameters:
    tr_set: 2-tuple of type numpy ndarray or list of str
        Training set containing X and y.
    tr_f : callable
        Training function giving loss.
    n_epochs : int or None
        Number of epochs. If None, is infinite.
    batch_size : int
        Batch size.
    tr_set: None or 2-tuple of type numpy ndarray or list of str
        Validation set containing X and y.
    val_f : callable or None
        Validation function giving a tuple of (loss, mae).
    val_f_val_tol : float or None
        If difference of curr/last validations < val_f_val_tol, stop.
    max_its : int or None
        Maximum number of iterations.
    verbose : int
        Prints nothing if 0, only warnings if 1 and everything if >= 2.
    """

    #info/warning functions
    info = print_f if verbose >= 2 else _silence
    warn = print_f if verbose >= 1 else _silence

    #checking whether train set is iterative or not
    if isinstance(tr_set[0], str):
        tr_iter = False
        X_tr, y_tr = None, None
        n_tr_batches = None
        tr_batch_gen = lambda: batches_gen_iter(tr_set, batch_size, True, info)
    elif len(tr_set) == 2:
        tr_iter = True
        X_tr, y_tr = tr_set
        n_tr_batches = len(y_tr)//batch_size
        tr_batch_gen = lambda: batches_gen(X_tr, y_tr, batch_size, True)
    else:
        raise ValueError("tr_set must be either list of str or size-2-tuple") 

    #checking whether validation set is iterative or not
    if val_set is None:
        validation = False
        val_iter = False
        n_val_batches = None
    elif isinstance(val_set[0], str):
        val_iter = False
        validation = True
        X_val, y_val = None, None
        n_val_batches = None
        val_batch_gen = lambda: batches_gen_iter(val_set, batch_size, True,
            info)
    elif len(val_set) == 2:
        validation = True
        val_iter = True
        X_val, y_val = val_set
        n_val_batches = max(len(y_val)//batch_size, 1)
        val_batch_gen = lambda: batches_gen(X_val, y_val, batch_size, True)
    else:
        raise ValueError("val_set must be either None, list of str or "
            "size-2-tuple") 

    #maybe it'll run forever...
    if not val_f_val_tol and n_epochs is None and max_its is None:
        warn("WARNING: training_loop will never stop since"
            " val_f_val_tol, n_epochs and max_its are all None")

    #initial values for some variables
    last_val_mae = None
    its = 0
    start_time = time.time()

    #main loop
    info("starting training loop...")
    for epoch in _inf_gen():
        if n_epochs is not None and epoch >= n_epochs:
            warn("\nWARNING: maximum number of epochs reached")
            end_reason = "n_epochs"
            return end_reason

        info("epoch %d/%s:" % (epoch+1, _str(n_epochs)))

        tr_err = 0
        tr_batch_n = 0
        for batch in tr_batch_gen():
            inputs, targets = batch
            err = tr_f(inputs, targets)
            tr_err += err
            tr_batch_n += 1
            info("    [train batch %d/%s] err: %.4g    %s" %\
                (tr_batch_n, _str(n_tr_batches), err, 32*" "), end="\r")

            its += 1
            if max_its is not None and its > max_its:
                warn("\nWARNING: maximum number of iterations reached")
                done_looping = True
                end_reason = "max_its"
                return end_reason
        n_tr_batches = tr_batch_n
        tr_err /= n_tr_batches

        val_err = 0
        val_mae = 0
        val_batch_n = 0
        if validation:
            for batch in val_batch_gen():
                inputs, targets = batch
                err, val_f_val = val_f(inputs, targets)
                val_err += err
                val_mae += val_f_val
                val_batch_n += 1
                info("    [val batch %d/%s] err: %.4g | val_f_val: %f   %s" %\
                    (val_batch_n, _str(n_val_batches), err, val_f_val, 32*" "),
                    end="\r")
            n_val_batches = val_batch_n
            val_err /= n_val_batches
            val_mae /= n_val_batches

            if last_val_mae is not None:
                val_mae_diff = abs(val_mae - last_val_mae)
                if val_f_val_tol is not None and val_mae_diff < val_f_val_tol:
                    warn("\nWARNING: val_mae_diff=%f < val_f_val_tol=%f" %\
                        (val_mae_diff, val_f_val_tol))
                    done_looping = True
                    end_reason = "val_f_val_tol"
                    return end_reason

            last_val_mae = val_mae

        info("    elapsed time so far: %s" %\
            _str_fmt_time(time.time() - start_time), end=32*" " + "\r")
        info()
        info("    train err: %.4g" % (tr_err))
        if validation:
            info("    val err: %.4g" % val_err, end="")
            info(" | val mae: %.4g" % val_mae)
