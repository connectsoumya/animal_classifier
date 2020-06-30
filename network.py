import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import InputLayer
from tensorlayer.layers import Conv2d
from tensorlayer.layers import BatchNormLayer
from tensorlayer.layers import ElementwiseLayer
from tensorlayer.layers import FlattenLayer
from tensorlayer.layers import DenseLayer
from tensorlayer.layers import PReluLayer


def classifier(t_image, is_train=False, reuse=False):
    """
    The classifier network
    :param t_image:
    :param is_train:
    :param reuse:
    :return:
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope('classifier', reuse=reuse):
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (2, 2), act=parametric_relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n
        # residual blocks
        for i in range(8):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=parametric_relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # residual blocks end
        n = Conv2d(n, 128, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = FlattenLayer(n)
        feat = DenseLayer(n, n_units=64, name='dense64')
        n = InputLayer(parametric_relu(feat.outputs), name='feat_act')
        n = DenseLayer(n, n_units=3, name='dense3')
    return n, feat


def parametric_relu(_x):
    """
    Tensorflow implementation of Prelu
    :param _x:
    :return:
    """
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg
