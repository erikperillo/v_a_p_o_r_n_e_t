#!/usr/bin/env python3

from skimage import io, transform as tf
import numpy as np
import pickle
import gzip
import os
import glob
from matplotlib import pyplot as plt
import random

IMGS_DIR = "/home/erik/code/wow/vw"
SHAPE = (224, 224)
ROT_RANGE_DEG = (10, 20)
SHOW = False
SAVE = False
TO_SAVE_DIR = "./negatives"
VGG16_WEIGHTS_FP = "/home/erik/data/vgg16.pkl"

def deg_to_rad(deg):
    return (deg/180)*3.1415926536

def rand_angle(rng):
    return deg_to_rad(random.uniform(*rng))

def mirror(img):
    return img[:, ::-1]

def rotate(img, radians, translation=None, scale=None):
    if scale is not None:
        scale = (scale, scale)
    at = tf.AffineTransform(rotation=radians, translation=translation,
        scale=scale)
    return tf.warp(img, at)

def mk_square(img):
    h, w = img.shape[:2]
    mid_y, mid_x = h//2, w//2
    shift = min(h, w)//2
    return img[mid_y-shift:mid_y+shift, mid_x-shift:mid_x+shift]

def _rotate_resizing_cw(img, theta):
    if theta < 0:
        raise ValueError("theta must be >= 0")

    h, w = img.shape[:2]

    a = h*(1 - np.cos(theta))
    b = (w - h*np.tan(theta))*np.tan(theta)
    c = (b + a)*np.cos(theta)
    scale = (1 + c/h)*1.1
    xs = h*np.tan(theta)

    scale = 2*(1/scale, )
    xs = np.ceil(xs)
    at = tf.AffineTransform(rotation=theta, translation=(xs, 0), scale=scale)

    return tf.warp(img, at)

def _rotate_resizing_ccw(img, theta):
    if theta >= 0:
        raise ValueError("theta must be < 0")

    img = np.rot90(mirror(img))
    img = _rotate_resizing_cw(img, -theta)
    img = mirror(np.rot90(img, 3))
    return img

def rotate_resizing(img, theta):
    if theta >= 0:
        return _rotate_resizing_cw(img, theta)
    else:
        return _rotate_resizing_ccw(img, theta)

def resize_to_shape(img, shape):
    return tf.resize(img, shape)

def unit_norm(x, const=1):
    return const*((x - x.min())/(x.max() - x.min()))

def to_mtx(data_dir, model):
    error = []
    mtx = []

    for i, fp in enumerate(glob.glob(os.path.join(data_dir, "*"))):
        print("[counter = {}] in file '{}'...".format(i, fp))
        img = io.imread(fp)
        if img.shape != (224, 224, 3):
            print("ERROR")
            error.append(fp)
            continue
        img = prep_image(img, model)
        mtx.append(img.flatten().astype("uint8"))

    return np.array(mtx, dtype="uint8")

def load_x_y_concat_and_save():
    print("loading model...")
    with open(VGG16_WEIGHTS_FP, "rb") as f:
        model = pickle.load(f, encoding="latin1")

    x_pos = to_mtx("pos", model)
    x_neg = to_mtx("neg", model)
    X = np.vstack((x_pos, x_neg))
    print("x_pos shape, dtype:", x_pos.shape, x_pos.dtype)
    print("x_neg shape, dtype:", x_neg.shape, x_neg.dtype)
    print("X shape, dtype:", X.shape, X.dtype)

    y = np.array(x_pos.shape[0]*[1] + x_neg.shape[0]*[0], dtype="uint8")
    print("y shape, dtype:", y.shape, y.dtype)

    print("shuffling...")
    indexes = list(range(X.shape[0]))
    random.shuffle(indexes)
    X = X[indexes]
    y = y[indexes]

    print("saving...")
    with gzip.open("data.gz", "wb") as f:
        pickle.dump((X, y), f)

    print("done")

def prep_image(im, model):
    im = unit_norm(im, 255)
    #shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    #convert to BGR
    im = im[::-1, :, :]
    #mean-centering
    for c, m in enumerate(model["mean value"]):
        im[c, :, :] -= m
    return im

def test():
    print("loading...")
    with gzip.open("data.gz", "rb") as f:
        X, y = pickle.load(f)

    print("converting...")
    X = X.astype("float32")
    y = y.astype("float32")

def norm_saving(data_dir, stats_fp, out_dir):
    with open(stats_fp, "rb") as f:
        stats = pickle.load(f)

    error = []

    for i, fp in enumerate(glob.glob(os.path.join(data_dir, "*"))):
        print("[counter = {}] in file '{}'...".format(i, fp))

        img = io.imread(fp)
        print(img.dtype, img.shape)
        if img.shape != SHAPE + (3, ):
            print("ERROR LOADING")
            error.append(fp)
            continue

        img = unit_norm(img, 255)
        for c, cn in zip(range(3), ("r", "g", "b")):
            img[:, :, c] = (img[:, :, c] - stats[cn]["mean"])/stats[cn]["std"]

        img = unit_norm(img)
        io.imsave(os.path.join(out_dir, os.path.basename(fp)), img)

def get_stats(data_dir):
    error = []

    r = {
        "mins": [],
        "maxs": [],
        "means": [],
        "stds": [],
    }
    g = {
        "mins": [],
        "maxs": [],
        "means": [],
        "stds": [],
    }
    b = {
        "mins": [],
        "maxs": [],
        "means": [],
        "stds": [],
    }

    for i, fp in enumerate(glob.glob(os.path.join(data_dir, "*"))):
        print("[counter = {}] in file '{}'...".format(i, fp))

        img = io.imread(fp)
        if img.shape != SHAPE + (3, ):
            print("ERROR LOADING")
            error.append(fp)
            continue

        img = unit_norm(img, 255).astype(np.uint8)

        for c, d in zip(range(3), (r, g, b)):
            ch = img[:, :, c]
            d["mins"].append(ch.min())
            d["maxs"].append(ch.max())
            d["means"].append(ch.mean())
            d["stds"].append(ch.std())

    stats = {}
    for c, d in zip(("r", "g", "b"), (r, g, b)):
        stats[c] = {
            "min": min(d["mins"]),
            "max": max(d["maxs"]),
            "mean": sum(d["means"])/len(d["means"]),
            "std": np.sqrt(sum(x**2 for x in d["stds"])/len(d["stds"]))
        }

    print("stats:", stats)

    return stats

def augment():
    error = []
    for i, fp in enumerate(glob.glob(os.path.join(IMGS_DIR, "*"))):
        print("in '{}'".format(fp))

        try:
            img = io.imread(fp)
            img = mk_square(img)
            img = resize_to_shape(img, SHAPE)
            #a = np.load("ey7.npz")["arr_0"]
            #print(a.shape, a.dtype)
            #a = np.reshape(a, (224, 224, 3)).astype("float32")
            #a = np.array((a - a.min())/(a.max() - a.min()))
            #io.imsave("a.jpg", a)
            #break
            #img = np.array(255*(img - img.min())/(img.max() - img.min()),
            #    dtype=np.uint8)
            #io.imsave("ey2.jpg", img)
            #a = img.flatten()
            #print(a.shape, a.dtype)
            #np.savez_compressed("ey7", a)
            #break
            rot_l_img = rotate_resizing(img, rand_angle(ROT_RANGE_DEG))
            rot_r_img = rotate_resizing(img, -rand_angle(ROT_RANGE_DEG))
            mir_img = mirror(img)
            mir_rot_l_img = rotate_resizing(mir_img, rand_angle(ROT_RANGE_DEG))
            mir_rot_r_img = rotate_resizing(mir_img, -rand_angle(ROT_RANGE_DEG))
        except:
            print("ERROR ON IMAGE")
            error.append(fp)
            continue


        if SHOW:
            plt.subplot(2, 3, 1)
            plt.imshow(img)
            plt.axis("off")
            plt.subplot(2, 3, 2)
            plt.imshow(rot_l_img)
            plt.axis("off")
            plt.subplot(2, 3, 3)
            plt.imshow(rot_r_img)
            plt.axis("off")
            plt.subplot(2, 3, 4)
            plt.imshow(mir_img)
            plt.axis("off")
            plt.subplot(2, 3, 5)
            plt.imshow(mir_rot_l_img)
            plt.axis("off")
            plt.subplot(2, 3, 6)
            plt.imshow(mir_rot_r_img)
            plt.axis("off")
            plt.show()

        if SAVE:
            print("\tsaving...")
            for j, im in enumerate([img, rot_l_img, rot_r_img,
                mir_img, mir_rot_l_img, mir_rot_r_img]):
                to_save = os.path.join(TO_SAVE_DIR, "{}_{}.jpg".format(i, j))
                io.imsave(to_save, im)

    if error:
        print("errors ocurred on: {}".format(", ".join(error)))

if __name__ == "__main__":
    #load_x_y_concat_and_save()
    #to_mtx("pos")
    test()
