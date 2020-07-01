#!/usr/bin/env python3

from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
from random import shuffle
from random import seed
from configparser import ConfigParser

# GENERAL CONFIG
parser = ConfigParser()
parser.read('config.ini')
data_file = parser.get('DATA', 'datafile')
data_path = parser.get('DATA', 'datapath')


def add_margin(pil_img, top, right, bottom, left):
    """
    pad rectangular images with 0 to make it square
    :param pil_img:
    :param top:
    :param right:
    :param bottom:
    :param left:
    :return:
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), (0, 0, 0))
    result.paste(pil_img, (left, top))
    return result


def preprocess_image(image_path, angle=0, zero_padding=True):
    """
    preprocess the images
    :param image_path:
    :param angle:
    :return:
    """
    image = Image.open(image_path)
    if zero_padding:
        if angle != 0:
            image = image.rotate((360 - angle), expand=True)
        width, height = image.size
        if width != height:
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
            if width > height:
                diff = width - height
                pad_top = int(diff / 2)
                pad_bottom = diff - pad_top
            else:
                diff = height - width
                pad_left = int(diff / 2)
                pad_right = diff - pad_left
            image = add_margin(image, pad_top, pad_right, pad_bottom, pad_left)
    image = image.resize((128, 128))
    scaled = (np.array(image) / 127.5) - 1
    # scaled = np.array(image) / 255.0
    return scaled


class DataGenerator(object):
    def __init__(self, data_file, data_path, batch_size=64, epochs=100):
        self.data_file = data_file
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_list = self.dataparser()
        seed(0)
        shuffle(self.data_list)
        self.train_list = self.data_list[:int(0.9 * len(self.data_list))]
        self.validate_list = self.data_list[int(0.9 * len(self.data_list)):]
        self.labels = ['cat', 'horse', 'squirrel']
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(self.labels)
        self.encoded = to_categorical(integer_encoded)

    def train_generator(self):
        """
        Create the data generator for training
        :return:
        """
        epochs = -1
        iters = 0
        image_data = []
        label_data = []
        batch_counter = 0
        while epochs < 100:
            epochs += 1
            shuffle(self.train_list)
            for label_name, fname, angle in self.train_list:
                label = self.encoded[self.labels.index(label_name), :]
                image_path = os.path.join(self.data_path, label_name, fname)
                image = preprocess_image(image_path, int(angle), zero_padding=True)
                image_data.append(image)
                label_data.append(label)
                batch_counter += 1
                if batch_counter == self.batch_size:
                    image_array = np.array(image_data)
                    label_array = np.array(label_data)
                    iters += 1
                    batch_counter = 0
                    image_data = []
                    label_data = []
                    yield image_array, label_array, epochs, iters

    def test_generator(self):
        """
        Create the data generator for validation
        :return:
        """
        image_data = []
        label_data = []
        batch_counter = 0
        for label_name, fname, angle in self.validate_list:
            label = self.encoded[self.labels.index(label_name), :]
            image_path = os.path.join(self.data_path, label_name, fname)
            image = preprocess_image(image_path, int(angle), zero_padding=True)
            image_data.append(image)
            label_data.append(label)
            batch_counter += 1
            if batch_counter == self.batch_size:
                image_array = np.array(image_data)
                label_array = np.array(label_data)
                batch_counter = 0
                image_data = []
                label_data = []
                yield image_array, label_array
        if len(image_data) > 0:
            yield image_array, label_array

    def dataparser(self):
        """
        parse data info from file
        :return:
        """
        datalist = []
        with open(self.data_file, 'r') as f:
            for line in f:
                line = line.strip()
                filepath, rotation = line.split(' ')
                label, filename = filepath.split('/')
                datalist.append((label, filename, rotation))
        return datalist


if __name__ == '__main__':
    # dataparser(data_file)
    # preprocess_image('100.jpeg', angle=270)
    dg = DataGenerator(data_file, data_path, 8)
    a = dg.train_generator()
