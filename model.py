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