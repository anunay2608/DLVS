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
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def activation_fn(x):
    value = (5*(tf.math.exp(x/5) - tf.math.exp(-x/5))/(tf.math.exp(x/5) + tf.math.exp(-x/5)))
    return value


get_custom_objects().update({'modified_tanh': Activation(activation_fn)})

input_layer = tf.keras.layers.Input((32, 5))
lstm_1 = tf.keras.layers.LSTM(16, activation='modified_tanh', recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(input_layer)
lstm_2 = tf.keras.layers.LSTM(64, activation='modified_tanh', recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_1[0])
lstm_3 = tf.keras.layers.LSTM(256, activation='modified_tanh', recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_2[0])
lstm_4 = tf.keras.layers.LSTM(512, activation='modified_tanh', recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_3[0])
lstm_5 = tf.keras.layers.LSTM(256, activation='modified_tanh', recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_4[0])
lstm_6 = tf.keras.layers.LSTM(64, activation='modified_tanh', recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_5[0])
lstm_7 = tf.keras.layers.LSTM(16, activation='modified_tanh', recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_6[0])
lstm_8 = tf.keras.layers.LSTM(2, activation='modified_tanh', recurrent_activation='sigmoid', dropout = 0.15, return_sequences = True, return_state = True)(lstm_7[0])
print(lstm_8[0])

model = tf.keras.models.Model(inputs = input_layer, outputs = lstm_8[0], name = "DLVS_Model")

model.summary()


def loss(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype = tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype = tf.float32)
    l1 = tf.reduce_mean(tf.math.abs((y_true - y_pred)))
    l2 = tf.reduce_mean(tf.math.pow((y_true, y_pred), 2))
    loss = l1 + l2
    return loss

error_final = []
vel_final = []
path = "/home/uas-dtu/Documents/DL_IBVS/data_2/"
for a in os.listdir(path):
    file = open(path + "/" + a, "r")
    for i in [file.readlines()]:
        for j in range(len(i) - 32):
#             print(j)
            error_data = []
            vel_data = []
            for k in range(32):
                #print(i[j + k].split(",")[3].split('[')[1].split("]")[0])
                #rint(float(i[j + k].split(",")[5]))
                #print(i[j + k].split(",")[4].split("\n"))
                error_data.append([float(i[j + k].split(",")[0]), 
                                   float(i[j + k].split(",")[1]),
                                   float(i[j + k].split(",")[2]), 
                                   float(i[j + k].split(",")[3]), 
                                   float(i[j + k].split(",")[4])])
                vel_data.append([float(i[j + k].split(",")[5].split('[')[1].split("]")[0]),
                                 float(i[j + k].split(",")[6].split('[')[1].split("]")[0])])
                
                #print(error_data)
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

train_data = DataGenerator(error_final, vel_final, batch_size=32, shuffle=True)


callbacks = [tf.keras.callbacks.ModelCheckpoint("/home/uas-dtu/Documents/DL_IBVS/data_script_drone_vel_2/train_{epoch}.tf", verbose = 1,
                                                save_weights_only=True),
             TensorBoard("/home/uas-dtu/Documents/DL_IBVS/data_script_drone_vel_2/combined_logs")


def mse(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype = tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    l1 = tf.reduce_mean(tf.math.pow((y_pred - y_true), 2))
    l2 = tf.reduce_mean(tf.math.abs((y_pred - y_true)))
    loss = l1 + l2
    return loss


def mse(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype = tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    l1 = tf.reduce_mean(tf.math.pow((y_pred - y_true), 2))
    l2 = tf.reduce_mean(tf.math.abs((y_pred - y_true)))
    loss = l1 + l2
    return loss


optimizer=Adam(learning_rate=0.0005)


model.compile(optimizer=optimizer,loss=mse, metrics = [tf.keras.metrics.RootMeanSquaredError()])


model.fit(train_data, epochs=2000, verbose=1, callbacks=callbacks, initial_epoch = 33)


