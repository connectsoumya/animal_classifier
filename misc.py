import tensorflow as tf
import os
import logging
import logging.config
from tensorflow.python.tools import inspect_checkpoint

log_file_path = 'logging.ini'
logging.config.fileConfig(log_file_path)


def save(sess, filepath='../tmp/tfmodel.mdl', global_step=None):
    """
    Save a TensorFlow model.
    :param sess:
    :param filepath:
    :param global_step:
    :return:
    """
    saver = tf.train.Saver()
    saver.save(sess, filepath, global_step=global_step)
    logging.info('Model saved at ' + filepath)


def load(sess, filepath):
    """
    Load/Restore a TensorFlow model.
    :param sess: The session
    :param filepath:
    :return:
    """
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)
    saver = tf.train.Saver()
    inspect_checkpoint.print_tensors_in_checkpoint_file(filepath, tensor_name=None, all_tensors=False)
    saver.restore(sess, filepath)
    logging.info('Model restored from ' + filepath)
    return sess
