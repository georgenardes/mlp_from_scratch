import tensorflow as tf
import cupy as cp

def load_mnist():
        
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    # normalize
    x_train = tf.cast(tf.reshape(x_train, (60000, 28, 28, 1)), tf.float32) / 255.
    x_test = tf.cast(tf.reshape(x_test, (10000, 28, 28, 1)), tf.float32) / 255.

    # to one-hot
    y_train = tf.cast(tf.one_hot(y_train, 10, 1., 0.), tf.float32)
    y_test = tf.cast(tf.one_hot(y_test, 10, 1., 0.), tf.float32)

    return x_train, y_train, x_test, y_test


def load_fashion_mnist():
        
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    # normalize
    x_train = x_train.reshape(60000, -1).astype(cp.float32) / 255.
    x_test = x_test.reshape(10000, -1).astype(cp.float32) / 255.

    # to one-hot
    y_train = cp.array(tf.one_hot(y_train, 10, 1., 0.))
    y_test = cp.array(tf.one_hot(y_test, 10, 1., 0.))

    return x_train, x_test, y_train, y_test

def load_cifar10():
        
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)

    print(y_train.shape)
    assert y_train.shape == (50000,1)
    assert y_test.shape == (10000,1)

    # normalize
    x_train = x_train.reshape(50000, -1).astype(cp.float32) / 255.
    x_test = x_test.reshape(10000, -1).astype(cp.float32) / 255.

    # to one-hot
    y_train = cp.array(tf.one_hot(y_train[:, 0], 10, 1., 0.))
    y_test = cp.array(tf.one_hot(y_test[:, 0], 10, 1., 0.))

    return x_train, x_test, y_train, y_test
