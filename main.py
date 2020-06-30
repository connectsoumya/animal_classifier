#!/usr/bin/env python3

from datagenerator import DataGenerator
from datagenerator import preprocess_image
from network import classifier
import tensorflow as tf
from misc import save
from misc import load
from configparser import ConfigParser
import logging
import logging.config
import numpy as np
from misc import center_loss

# GENERAL CONFIG
cparser = ConfigParser()
cparser.read('config.ini')
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
    predicted_logits, features = classifier(data_batch, is_train=True)
    predicted_logits = predicted_logits.outputs
    features = features.outputs
    predicted_logits_test, _ = classifier(data_batch, is_train=False, reuse=True)
    predicted_logits_test = predicted_logits_test.outputs
    predicted_labels = tf.nn.softmax(predicted_logits_test)
    # get losses
    # crossentropy_loss = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_batch, logits=predicted_logits))
    crossentropy_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(label_batch, axis=1),
                                                       logits=predicted_logits))
    c_loss, _ = center_loss(features, label_batch, 0.95, 3)
    tr_loss = crossentropy_loss + 0.2 * c_loss
    # optimize
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
            noise = np.random.randint(-10, 10, size=label_data.shape) / 100.0
            label_data += noise
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
                    correct_pred = np.equal(np.argmax(label_data, axis=1), np.argmax(pred, axis=1)).tolist()
                    accuracy += correct_pred
                accuracy = sum(accuracy) / len(accuracy)
                acc = accuracy
                logging.info('iters: %5d, epochs: %3d, train loss: %2.3f, test accuracy: %2.3f' % (
                    iters, epoch_new, loss, accuracy))
                current_epoch = epoch_new
                save(sess, model_path)


def predict(args):
    image_path = args.image_
    image_batch = preprocess_image(image_path)
    data_batch = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name='data_batch')
    predicted_logits_test, _ = classifier(data_batch, is_train=False)
    predicted_logits_test = predicted_logits_test.outputs
    predicted_labels = tf.nn.softmax(predicted_logits_test)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # load weights
        load(sess, model_path)
        logging.info('Pre-trained weights loaded')
        pred = sess.run(fetches=predicted_labels, feed_dict={data_batch: image_batch})
        pred = np.rint(pred)
    pred_labels = np.argmax(pred, axis=1).tolist()
    label_set = ['cat', 'horse', 'squirrel']
    labels = [label_set[i] for i in pred_labels]
    return labels


if __name__ == '__main__':
    data_file = cparser.get('DATA', 'datafile')
    data_path = cparser.get('DATA', 'datapath')
    model_path = cparser.get('ARGS', 'modelpath')
    batch_size = int(cparser.get('ARGS', 'batchsize'))
    epochs = int(cparser.get('ARGS', 'epochs'))
    lrate = float(cparser.get('ARGS', 'lrate'))
    dg = DataGenerator(data_file, data_path, batch_size, epochs)
    train(data_generator=dg, learning_rate=lrate, model_path=model_path, epochs=epochs)
