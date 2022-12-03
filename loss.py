import tensorflow as tf
import os
import numpy as np
import pickle
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.utils import Sequence
import cv2
import tensorflow as tf
import json
import random

class DataGenerator(Sequence):
    def __init__(self, error_final, vel_final,
                 batch_size=32,
                 shuffle=True):
        self.error_final = error_final
        self.vel_final = vel_final
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            temp = [i for i in range(len(self.error_final))]
            random.shuffle(temp)
            #       self.flow_videos = [self.flow_videos[i] for i in temp]
            self.error_final = [self.error_final[i] for i in temp]
            self.vel_final = [self.vel_final[i] for i in temp]

    def __len__(self):
        return int(np.floor(len(self.error_final) / self.batch_size))


    def __getitem__(self, index):
        start_index = index * self.batch_size
#         input_1 = []
        error_reading = []
        vel_reading = []
        i = start_index - 1
        while len(error_reading) < self.batch_size:
            try:
                i += 1
                error = self.error_final[i%len(self.error_final)]
                vel = self.vel_final[i%len(self.vel_final)]
                error_reading.append(error)
                vel_reading.append(vel)

            except Exception as err:
                print(err)
                continue
            
        vel_reading = np.array(vel_reading, dtype = np.float32)
        error_reading = np.array(error_reading, dtype = np.float32)
            
        return error_reading, vel_reading