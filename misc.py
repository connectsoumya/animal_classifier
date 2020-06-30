import tensorflow as tf
from PIL import Image
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


def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    label = tf.argmax(label, axis=1, name=None)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers
