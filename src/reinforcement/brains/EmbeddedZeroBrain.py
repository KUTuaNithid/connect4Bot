import os
import pathlib
import tensorflow as tf
import numpy as np
#from pycoral.utils import edgetpu
#from pycoral.adapters import common

class EmbeddedZeroBrain:
    def __init__(self,model_name):
        model_dir = 'Models/'
        #model_dir = 'reinforcement/Models/'
        model_file = os.path.join(model_dir, model_name)
        # Initialize the TF interpreter
        #self.interpreter = edgetpu.make_interpreter(model_file)
        self.interpreter =  tf.lite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

    def predict(self,input_state):
        input_details = self.interpreter.get_input_details()
        input_scale, input_zero_point = input_details[0]["quantization"]

        output_details = self.interpreter.get_output_details()
        output_scale = [0.0,0.0]
        output_zero_point = [0.0,0.0]
        input_shape = input_details[0]['shape']
        output_scale[0],output_zero_point[0] = output_details[0]['quantization']
        output_scale[1],output_zero_point[1] = output_details[1]['quantization']
        state = ((input_state.reshape((1,3,6,7))/input_scale)+input_zero_point).astype(np.int8)
        self.interpreter.set_tensor(input_details[0]['index'], state)
        
        #common.set_input(self.interpreter, input_state)
        self.interpreter.invoke()
        P = self.interpreter.get_tensor(output_details[1]['index'])
        V = self.interpreter.get_tensor(output_details[0]['index'])

        P_dequantized = (P.astype(np.float32)-output_zero_point[1]).astype(np.float32)*(output_scale[1])
        V_dequantized = (V.astype(np.float32)-output_zero_point[0]).astype(np.float32)*(output_scale[0])
        print(P_dequantized)
        print(V_dequantized)
        #P = common.tensor(interpreter, 1)
        #V = common.tensor(interpreter, 0)
        return P_dequantized[0],V_dequantized[0][0]
