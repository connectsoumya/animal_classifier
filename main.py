#!/usr/bin/env python3

from datagenerator import DataGenerator
from network import classifier
import tensorflow as tf
import os
from misc import save
from misc import load
from configparser import ConfigParser
import logging
import logging.config
import numpy as np

# GENERAL CONFIG
parser = ConfigParser()
parser.read('config.ini')
log_file_path = 'logging.ini'
logging.config.fileConfig(log_file_path)
# TF CONFIG
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

logging.info('Preparing for training')


def train(data_generator, learning_rate, model_path, epochs):
    """
    train the network
    :param data_generator:
    :param learning_rate:
    :param model_path:
    :param epochs:
    :return:
    """
    train_data = data_generator.train_generator()
    # build graph
    data_batch = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name='data_batch')
    label_batch = tf.placeholder(dtype=tf.float32, shape=(None, 3), name='label_batch')
    predicted_logits = classifier(data_batch, is_train=True)
    predicted_logits = predicted_logits.outputs
    predicted_logits_test = classifier(data_batch, is_train=False, reuse=True)
    predicted_logits_test = predicted_logits_test.outputs
    predicted_labels = tf.nn.softmax(predicted_logits_test)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tr_loss = tf.losses.softmax_cross_entropy(label_batch, predicted_logits, label_smoothing=0.1)
    trainable_vars = tf.trainable_variables()
    grad = optimizer.compute_gradients(tr_loss, var_list=trainable_vars)
    opt = optimizer.apply_gradients(grad)
    acc = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # load weights
        try:
            load(sess, model_path)
            logging.info('Pre-trained weights loaded')
            print('Pre-trained weights loaded')
        except ValueError:
            logging.info('Training from scratch')
            print('Training from scratch')
        current_epoch = 0
        # train
        while current_epoch < epochs:
            img_data, label_data, epoch_new, iters = next(train_data)
            _, loss = sess.run(fetches=[opt, tr_loss], feed_dict={data_batch: img_data,
                                                                  label_batch: label_data})
            print('\r iters: {} epochs: {} train loss: {} test accuracy: {}'.format(iters, epoch_new, loss, acc),
                  end="")
            # testing and logging
            if epoch_new != current_epoch:
                test_data = data_generator.test_generator()
                accuracy = []
                for img_data, label_data in test_data:
                    pred = sess.run(fetches=predicted_labels, feed_dict={data_batch: img_data})
                    pred = np.rint(pred)
                    accuracy.append(1 - (np.mean(np.abs(label_data - pred))))
                accuracy = sum(accuracy) / len(accuracy)
                acc = accuracy
                logging.info('iters: %5d, epochs: %3d, train loss: %2.3f, test accuracy: %2.3f' % (
                    iters, epoch_new, loss, accuracy))
                current_epoch = epoch_new
                save(sess, model_path)


def predict():
    pass


if __name__ == '__main__':
    data_file = parser.get('DATA', 'datafile')
    data_path = parser.get('DATA', 'datapath')
    model_path = parser.get('ARGS', 'modelpath')
    batch_size = int(parser.get('ARGS', 'batchsize'))
    epochs = int(parser.get('ARGS', 'epochs'))
    lrate = float(parser.get('ARGS', 'lrate'))
    dg = DataGenerator(data_file, data_path, batch_size, epochs)
    train(data_generator=dg, learning_rate=lrate, model_path=model_path, epochs=epochs)
