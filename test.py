import tensorflow as tf
import os
import numpy as np
import pickle
import json
import random
from tensorflow.keras.utils import Sequence
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

input_layer = tf.keras.layers.Input((32, 3))
lstm_1 = tf.keras.layers.LSTM(16, activation="tanh", recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(input_layer)
lstm_2 = tf.keras.layers.LSTM(64, activation="tanh", recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_1[0])
lstm_3 = tf.keras.layers.LSTM(256, activation="tanh", recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_2[0])
lstm_4 = tf.keras.layers.LSTM(512, activation="tanh", recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_3[0])
lstm_5 = tf.keras.layers.LSTM(256, activation="tanh", recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_4[0])
lstm_6 = tf.keras.layers.LSTM(64, activation="tanh", recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_5[0])
lstm_7 = tf.keras.layers.LSTM(16, activation="tanh", recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_6[0])
lstm_8 = tf.keras.layers.LSTM(2, activation="tanh", recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_7[0])
print(lstm_8[0])

model = tf.keras.models.Model(inputs = input_layer, outputs = lstm_8[0], name = "DLVS_Model")

error_final = []
vel_final = []
path = "/home/uas-dtu/Documents/DL_IBVS/Data/"
for a in os.listdir(path):
    file = open(path + "/" + a, "r")
    for i in [file.readlines()]:
        for j in range(len(i) - 32):
#             print(j)
            error_data = []
            vel_data = []
            for k in range(32):
                error_data.append([float(i[j + k].split(",")[0]), 
                                   float(i[j + k].split(",")[1]),
                                   float(i[j + k].split(",")[2])])
                vel_data.append([float(i[j + k].split(",")[3]), 
                                 float(i[j + k].split(",")[4])])
            error_final.append(error_data)
            vel_final.append(vel_data)


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

train_data = DataGenerator(error_final, vel_final, batch_size=1, shuffle=False)

model.load_weights("./NILU_TRAIN/train_70.tf")

pred = model.predict(train_data.__getitem__(0)[0])
print(train_data.__getitem__(0)[1])
print(pred)
