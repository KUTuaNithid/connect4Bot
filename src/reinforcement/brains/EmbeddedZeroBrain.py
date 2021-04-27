import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

class EmbeddedZeroBrain:
    def __init__(self,model_name):
        model_dir = 'Models/{}'.format(iteration)
        model_file = os.path.join(model_dir, 'Zero_quant_edgetpu.tflite')
        # Initialize the TF interpreter
        self.interpreter = edgetpu.make_interpreter(model_file)
        self.interpreter.allocate_tensors()

    def predict(self,input_state):
        common.set_input(self.interpreter, input_state)
        self.interpreter.invoke()
        P = common.tensor(interpreter, 0)
        V = common.tensor(interpreter, 1)
