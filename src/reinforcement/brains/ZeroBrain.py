import numpy as np
import tensorflow as tf
import os
import shutil
from tensorflow.keras.layers import Input,Dense, Flatten, Conv2D,LeakyReLU
from tensorflow.keras import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model
from .BrainComponent import *

class ZeroBrain:
    def __init__(self, iteration , isConv = True):
        self.name = iteration
        file_path = 'Models/{}'.format(iteration)
        if os.path.isdir(file_path):
            self.model = load_model(file_path)
        else:
            self.model = self.build_model(isConv)
        self.isConv = isConv
            

    def build_model(self,isConv):
        if isConv: # this part will be implemented by using convolution network
            input_layer,x = input_conv_layer()
            x = residual_tower(x)
            out_actions_prob = policy_head(x)
            out_value = value_head(x)

        else: # Test with small fully connected network 
            
            input_layer = Input(shape = (6*7,))
            x = Dense(512,activation = 'relu')(input_layer)
            x = Dense(256,activation = 'relu')(x)
            x = Dense(128,activation = 'relu')(x)
            x = Dense(64,activation = 'relu')(x)

            out_value = Dense(1,activation = 'tanh',name = 'value_head')(x)
            out_actions_prob = Dense(7,activation = 'softmax',name = 'policy_head')(x)

        model = Model(inputs=[input_layer], outputs=[out_actions_prob, out_value])
        # model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
        #               loss_weights={'value_head': 0.5, 'policy_head': 0.5},
        #               optimizer=Adam())
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5},
                      optimizer=SGD(lr=0.1, momentum = 0.9))
        # model.summary()
        return model

    def predict(self, s , isConv = True):
        if isConv:
            state = s.reshape((1,3,6,7))
            with tf.device('/gpu:0'):
                P, V = self.model.predict(state)

        else:
            state = s.flatten()
            state = state.reshape((1,42))
            P, V = self.model.predict(state)
        return P[0], V[0][0]

    def train(self, memory):
        S,P,V = list(zip(*memory))
        S = np.asarray(S)
        P = np.asarray(P)
        V = np.asarray(V)
        self.model.fit(x = S, y = [P, V], batch_size = 16, 
                       epochs = 10, verbose=2)

    def saveModel(self):
        self.model.save('Models/{}'.format(self.name))

    def deleteModelFile(self):
        shutil.rmtree('Models/{}'.format(self.name))