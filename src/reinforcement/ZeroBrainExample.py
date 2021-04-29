from brains.ZeroBrain import ZeroBrain
import numpy as np
import tensorflow as tf

# brain = ZeroBrain(26,isConv=True)
# input_s = np.random.randint(2, size=(3,6,7))
# P,V = brain.predict(input_s)
# print(P)
# print(V)
# brain.saveModel()

def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 3, 6, 7)
      yield [data.astype(np.float32)]
# Convert the model.
# converter = tf.contrib.lite.toco_convert.from_keras_model_file('AlphaTest.h5')
converter = tf.lite.TFLiteConverter.from_saved_model("./Models/24")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_quantized_model = converter.convert()

# Save the model.
with open('./Models/saiV2.tflite', 'wb') as f:
    f.write(tflite_quantized_model)