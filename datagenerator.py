#!/usr/bin/env python3

from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
from random import shuffle
from configparser import ConfigParser

# GENERAL CONFIG
parser = ConfigParser()
parser.read('config.ini')
data_file = parser.get('DATA', 'datafile')
data_path = parser.get('DATA', 'datapath')


class DataGenerator(object):
    def __init__(self, data_file, data_path, batch_size=64, epochs=100):
        self.data_file = data_file
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_list = self.dataparser()

    def generate(self):
        labels = ['cat', 'horse', 'squirrel']
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        encoded = to_categorical(integer_encoded)
        epochs = 0
        iter = 0
        image_data = []
        label_data = []
        batch_counter = 0
        while epochs < 100:
            epochs += 1
            shuffle(self.data_list)
            for label_name, fname, angle in self.data_list:
                label = encoded[labels.index(label_name), :]
                image_path = os.path.join(self.data_path, label_name, fname)
                image = self.preprocess_image(image_path, angle)
                image_data.append(image)
                label_data.append(label)
                batch_counter += 1
                iter += 1
                if batch_counter == self.batch_size:
                    image_array = np.array(image_data)
                    label_array = np.array(label_data)
                    batch_counter = 0
                    yield image_array, label_array


    def dataparser(self):
        datalist = []
        with open(self.data_file, 'r') as f:
            for line in f:
                line = line.strip()
                filepath, rotation = line.split(' ')
                label, filename = filepath.split('/')
                datalist.append((label, filename, rotation))
        return datalist

    def preprocess_image(self, image_path, angle=0):
        image = Image.open(image_path)
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
            image = self.add_margin(image, pad_top, pad_right, pad_bottom, pad_left)
        image = image.resize((128, 128))
        scaled = (np.array(image) / 127.5) - 1
        return scaled

    @staticmethod
    def add_margin(pil_img, top, right, bottom, left):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), (0, 0, 0))
        result.paste(pil_img, (left, top))
        return result


if __name__ == '__main__':
    # dataparser(data_file)
    # preprocess_image('100.jpeg', angle=270)
    dg = DataGenerator(data_file, data_path, 8)
    dg.generate()
