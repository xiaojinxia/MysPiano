import json
from utils import *
import numpy as np
from keras.models import load_model
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


noise = -1 + 2 * np.random.random((1, 100))
generator=load_model("./model/g_100_3.h5",compile=False)
predictions = generator.predict(noise)

with open("./data_preparation/created_mapping_0.25", 'r') as fp:
        mappings = json.load(fp)

l = len(mappings)
pred_notes = [x*float(l/2)+float(l/2) for x in predictions[0]]

prediction_output = ['7.10']


for i in pred_notes:
    prediction_output.append([ k for k , v in mappings.items() if v == int(i)][0])

prediction_output.append('72')
prediction_output.append('72')
print(prediction_output)


create_midi(prediction_output,'./predicted_midi/prediction7')

