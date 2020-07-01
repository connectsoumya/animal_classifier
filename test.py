#!/usr/bin/env python3

from datagenerator import preprocess_image
from network import classifier
import tensorflow as tf
import argparse
import sys
from misc import load
from configparser import ConfigParser
import logging
import logging.config
import numpy as np

# GENERAL CONFIG
cparser = ConfigParser()
cparser.read('config.ini')
log_file_path = 'logging.ini'
logging.config.fileConfig(log_file_path)
# TF CONFIG
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


def predict(args):
    image_path = args.image_path
    image = preprocess_image(image_path)
    image_batch = np.expand_dims(image, axis=0)
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
    pred_labels = np.argmax(pred, axis=1).tolist()
    label_set = ['cat', 'horse', 'squirrel']
    labels = [label_set[i] for i in pred_labels]
    return labels, pred.tolist()[0]


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Absolute path of test image', default='test.jpg')
    return parser.parse_args(argv)


if __name__ == '__main__':
    model_path = cparser.get('ARGS', 'modelpath')
    label, pred_probs = predict(parse_arguments(sys.argv[1:]))  # comment this line for training
    print('Probabilities:- cat: %1.3f, horse: %1.3f, squirrel: %1.3f' % (pred_probs[0], pred_probs[1], pred_probs[2]))
    print('Outcome %s' % label[0])
