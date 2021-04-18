import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense, Flatten, Conv2D,LeakyReLU
from tensorflow.keras import Model
from keras.optimizers import Adam

class ZeroBrain:
    def __init__(self, iteration):
        self.name = iteration
        self.model = self.build_model()

    def build_model(self,conv_net = False):
        if conv_net: # this part will be implemented by using convolution network
            pass 
        else: # Test with small fully connected network 
            
            input_layer = Input(shape = (6*7,))
            x = Dense(512,activation = 'relu')(input_layer)
            x = Dense(256,activation = 'relu')(x)
            x = Dense(128,activation = 'relu')(x)
            x = Dense(64,activation = 'relu')(x)

            out_value = Dense(1,activation = 'tanh',name = 'value')(x)
            out_actions_prob = Dense(7,activation = 'softmax',name = 'policy')(x)

        model = Model(inputs=[input_layer], outputs=[out_actions_prob, out_value])
        model.compile(loss={'value': 'mean_squared_error', 'policy': 'categorical_crossentropy'},
                      loss_weights={'value': 0.5, 'policy': 0.5},
                      optimizer=Adam())
        model.summary()
        return model

    def predict(self, s):
        state = s.flatten()
        state = state.reshape((1,42))
        P, V = self.model.predict(state)
        return P[0], V[0][0]

    def train(self, memory):
        pass

    def saveModel(self):
        self.model.save('../Models/{}'.format(self.name))

