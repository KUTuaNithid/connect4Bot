from brains.ZeroBrain import ZeroBrain
import numpy as np

brain = ZeroBrain(1)
input_s = np.random.randint(3, size=(1,42))
P,V = brain.predict(input_s)
brain.saveModel()