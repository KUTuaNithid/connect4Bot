import os
import pathlib
import tensorflow as tf
import numpy as np
#from pycoral.utils import edgetpu
#from pycoral.utils import dataset
#from pycoral.adapters import common
#from pycoral.adapters import classify

class EmbeddedZeroBrain:
    def __init__(self,model_name):
        model_dir = 'Models/'
        model_file = os.path.join(model_dir, model_name)
        # Initialize the TF interpreter
        #self.interpreter = edgetpu.make_interpreter(model_file)
        self.interpreter =  tf.lite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

    def predict(self,input_state):
        input_details = self.interpreter.get_input_details()
        #print(input_details)
        output_details = self.interpreter.get_output_details()
        #print(output_details)
        input_shape = input_details[0]['shape']

        state = input_state.reshape((1,3,6,7)).astype('float32')
        self.interpreter.set_tensor(input_details[0]['index'], state)
        
        #common.set_input(self.interpreter, input_state)
        self.interpreter.invoke()
        P = self.interpreter.get_tensor(output_details[1]['index'])
        V = self.interpreter.get_tensor(output_details[0]['index'])
        #P = common.tensor(interpreter, 1)
        #V = common.tensor(interpreter, 0)
        return P[0],V[0][0]
