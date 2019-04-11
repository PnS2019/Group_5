"""Utility functions.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import

import os
try:
    import subprocess32 as sp
except ImportError:
    import subprocess as sp

import gzip
import numpy as np

import pnslib


def speak(msg_str, options=None):
    """Text-to-Speech via espeak and mplayer.

    # Parameters
    msg_str : str
        The candidate string to speak
    options : list
        espeak program options, use default config if None
        default config
        ["-ven+f3", "-k5", "-s150"]

    # Returns
    cmd_out : str
        command output as string
    """
    speak_base = ["/usr/bin/espeak"]
    if options is not None and isinstance(options, list):
        speak_config = options
    else:
        speak_config = ["-ven+f3", "-k5", "-s150"]
    mplayer_base = ["/usr/bin/mplayer", "/tmp/pns_speak.wav"]

    with open("/tmp/pns_speak.wav", "w") as wav_file:
        sp.call(speak_base+speak_config+[msg_str, "--stdout"],
                stdout=wav_file)
        wav_file.close()

    return sp.call(mplayer_base, stderr=sp.PIPE)


def number_mnist_download(train=True, test=True, labels=True):
    """Download Fashion MNIST.

    The original resource of Fashion MNIST is here:
    https://github.com/zalandoresearch/fashion-mnist

    The function uses wget to download

    # Parameters
    train : bool
        if download training dataset
    test : bool
        if download testing dataset
    labels : bool
        if download labels

    # Returns
    downloaded datasets in ~/.pnslibres
    """
    # define datasets URL
    base_url = "http://yann.lecun.com/exdb/mnist/"
    train_x_name = "train-images-idx3-ubyte.gz"
    train_y_name = "train-labels-idx1-ubyte.gz"
    test_x_name = "t10k-images-idx3-ubyte.gz"
    test_y_name = "t10k-labels-idx1-ubyte.gz"

    # construct save path
    save_path = os.path.join(pnslib.PNSLIB_DATA, "number-mnist")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if train is True:
        train_x_path = os.path.join(save_path, train_x_name)
        sp.check_output(["wget", base_url+train_x_name,
                         "-O", train_x_path],
                        stderr=sp.PIPE).decode("UTF-8")
        if labels is True:
            train_y_path = os.path.join(save_path, train_y_name)
            sp.check_output(["wget", base_url+train_y_name,
                             "-O", train_y_path],
                            stderr=sp.PIPE).decode("UTF-8")

    if test is True:
        test_x_path = os.path.join(save_path, test_x_name)
        sp.check_output(["wget", base_url+test_x_name,
                         "-O", test_x_path],
                        stderr=sp.PIPE).decode("UTF-8")
        if labels is True:
            test_y_path = os.path.join(save_path, test_y_name)
            sp.check_output(["wget", base_url+test_y_name,
                             "-O", test_y_path],
                            stderr=sp.PIPE).decode("UTF-8")


def number_mnist_load(data_type="full", flatten=False):
    """Loading Fashion MNIST dataset.

    # Parameters
    data_type : str
        full : training, testing and labels
        train : training and its labels
        test : testing and its labels
        train-only : only training dataset
        test-only : only testing dataset
    flatten : bool
        reshape to 4D tensor if True
        2D tensor if False

    # Returns
    dataset : tuple
        a tuple of requested data
        full: (train_x, train_y, test_x, test_y)
        train: (train_x, train_y)
        test: (test_x, test_y)
        train_only: (train)
        test_only: (test)
    """
    save_path = os.path.join(pnslib.PNSLIB_DATA, "number-mnist")
    if not os.path.isdir(save_path):
        print ("Downloading dataset.")
        number_mnist_download(train=True, test=True, labels=True)

    train_x_name = "train-images-idx3-ubyte.gz"
    train_y_name = "train-labels-idx1-ubyte.gz"
    test_x_name = "t10k-images-idx3-ubyte.gz"
    test_y_name = "t10k-labels-idx1-ubyte.gz"

    dataset = ()
    if data_type in ["train", "train_only", "full"]:
        train_x_path = os.path.join(save_path, train_x_name)
        with gzip.open(train_x_path, "rb") as x_file:
            if flatten is False:
                train_x = np.frombuffer(x_file.read(), dtype=np.uint8,
                                        offset=16).reshape(60000, 28, 28, 1)
            else:
                train_x = np.frombuffer(x_file.read(), dtype=np.uint8,
                                        offset=16).reshape(60000, 784)
            x_file.close()

        dataset = dataset+(train_x,) if train_x is not None else dataset
        if data_type != "train_only":
            train_y_path = os.path.join(save_path, train_y_name)
            with gzip.open(train_y_path, "rb") as y_file:
                train_y = np.frombuffer(y_file.read(), dtype=np.uint8,
                                        offset=8)
                y_file.close()
            dataset = dataset+(train_y,) if train_y is not None else dataset

    if data_type in ["test", "test_only", "full"]:
        test_x_path = os.path.join(save_path, test_x_name)
        with gzip.open(test_x_path, "rb") as x_file:
            if flatten is False:
                test_x = np.frombuffer(x_file.read(), dtype=np.uint8,
                                       offset=16).reshape(10000, 28, 28, 1)
            else:
                test_x = np.frombuffer(x_file.read(), dtype=np.uint8,
                                       offset=16).reshape(10000, 784)
            x_file.close()
        dataset = dataset+(test_x,) if test_x is not None else dataset
        if data_type != "test_only":
            test_y_path = os.path.join(save_path, test_y_name)
            with gzip.open(test_y_path, "rb") as y_file:
                test_y = np.frombuffer(y_file.read(), dtype=np.uint8,
                                       offset=8)
                y_file.close()
            dataset = dataset+(test_y,) if test_y is not None else dataset

    return dataset


def binary_number_mnist_load(class_list=[0, 1], flatten=False):
    """Select binary fashion MNIST dataset.

    # Parameters
    class_list : list
        a list of two values between 0 to 9
    flatten : bool
        reshape to 4D tensor if True
        2D tensor if False

    # Returns
    dataset : tuple
        a tuple of requested data
        (train_x, train_y, test_x, test_y)
    """
    assert len(class_list) == 2
    train_x, train_y, test_x, test_y = number_mnist_load("full",
                                                          flatten=flatten)

    # build up idx
    train_idx = np.logical_or(
            train_y == class_list[0], train_y == class_list[1])
    test_idx = np.logical_or(
            test_y[:] == class_list[0], test_y[:] == class_list[1])
    # select elements
    train_x = train_x[train_idx]
    train_y = train_y[train_idx]
    test_x = test_x[test_idx]
    test_y = test_y[test_idx]

    # post processing
    train_y[train_y == class_list[0]] = 0
    train_y[train_y == class_list[1]] = 1
    test_y[test_y == class_list[0]] = 0
    test_y[test_y == class_list[1]] = 1

    return (train_x, train_y, test_x, test_y)


def generate_lr_data(num_data=10000, x_dim=1, x_coeff=[0, 2.5], function=None):
    """Generate data for linear regression.

    # Parameters
    num_data : int
        number of data point
    x_dim : int
        number of dimensions for each sample x
    x_coeff : list
        [mean, std]
        generate samples in a specified normal distribution
    function : function
        A function that receives a vector x and produce y

    Returns
    -------
    (train_x, train_y, test_x, test_y) : tuple
        output datasets, assume 70/30 split
    """
    if function is None:
        def f(x):
            return 2*x+1

        function = f

    # generate random samples
    X = np.random.randn(num_data, x_dim)*x_coeff[1]+x_coeff[0]
    no_noise_Y = function(X)

    # add noise to the data, fixed noise Normal(0, 0.09)
    noise = np.random.randn(num_data, 1)*0.3
    Y = no_noise_Y+noise

    # split dataset
    train_x = X[:int(0.7*num_data)]
    train_y = Y[:int(0.7*num_data)]
    test_x = X[int(0.7*num_data):]
    test_y = Y[int(0.7*num_data):]

    return (train_x, train_y, test_x, test_y)


def generate_binary_data(
        num_data=10000, x_dim=2,
        c1_coeff=[0, 2.5], c2_coeff=[10, 2.5]):
    """Generate two normal distributions that has different class label.

    # Parameters
    num_data : int
        number of data points, split evenly for two classes
    c1_coeff : list
        [mean, std]
        the distribution for the first distribution
    c2_coeff : list
        [mean, std]
        the distribution for the second distribution

    Returns
    -------
    (train_x, train_y, test_x, test_y) : tuple
        output datasets, assume 70/30 split
    """
    # get num of samples for each class
    num_c1_data = num_data // 2
    num_c2_data = num_data-num_c1_data

    # generate data and label
    X_c1 = np.random.randn(num_c1_data, x_dim)*c1_coeff[1]+c1_coeff[0]
    Y_c1 = np.zeros((num_c1_data, 1))

    X_c2 = np.random.randn(num_c2_data, x_dim)*c2_coeff[1]+c2_coeff[0]
    Y_c2 = np.ones((num_c2_data, 1))

    # put some noise on Xs
    X_c1 += np.random.randn(num_c1_data, x_dim)*0.3
    X_c2 += np.random.randn(num_c2_data, x_dim)*0.3

    # combine and shuffle
    X = np.vstack((X_c1, X_c2))
    Y = np.vstack((Y_c1, Y_c2))
    shuffle_idx = np.random.permutation(num_data)
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]

    # split dataset
    train_x = X[:int(0.7*num_data)]
    train_y = Y[:int(0.7*num_data)]
    test_x = X[int(0.7*num_data):]
    test_y = Y[int(0.7*num_data):]

    return (train_x, train_y, test_x, test_y)


def get_haarcascade_path(xml_name):
    """Get haar cascade path according to the given xml file."""
    return os.path.join(pnslib.HAAR_FEATURE_PATH, xml_name)
