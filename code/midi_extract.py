import os
import statistics

import json

import music21 as m21
from music21 import converter, instrument, note, chord, stream
import numpy as np
from keras.models import load_model
from utils import *
import tensorflow as tf
# warnings.filterwarnings('ignore')

class Accuracy:
    def extract(self, filename,sigma):
        songs = read_midi_from_file(filename)
        time_step = 0.25      
        encoded_song = []
        notes_in_song = None
        try:
          s2 = instrument.partitionByInstrument(songs)
          notes_in_song = s2.parts[0].recurse()
        except:
          notes_in_song = songs.flat.notes
        for element in notes_in_song:
          if isinstance(element, note.Note):
              m_event = element.pitch.midi
              time = element.duration.quarterLength
              for step in range(0,int(time/ time_step)):
                  if step == 0:
                      encoded_song.append(str(m_event))
                  else:
                      encoded_song.append("_")        
          elif isinstance(element, note.Rest):
              m_event = "r"
              time = element.duration.quarterLength                     
              for step in range(0,int(time/time_step)):
                  if step == 0:
                      encoded_song.append(str(m_event))
                  else:
                      encoded_song.append("_")
          elif isinstance(element, chord.Chord):
              m_event = '.'.join(str(n) for n in element.normalOrder)
              time = element.duration.quarterLength
              #print(m_event)
              for step in range(0,int(time/time_step)):
                  if step == 0:
                      encoded_song.append(str(m_event))
                  else:
                      encoded_song.append("_")
        encoded_song = " ".join(map(str, encoded_song))
        notes_ = encoded_song
        print(notes_)
        with open("./data_preparation/created_mapping_0.25", 'r') as fp:
          mappings = json.load(fp)
        l = len(mappings)
        mid_int_right = notes_to_int(notes_)           
        mid_float_right = [(x - float(l/2)) / float(l/ 2) for x in mid_int_right]
        del(mid_float_right[0])
        mid_float_right.pop()
        mid_float_right_ = np.reshape(mid_float_right, (1, 100))
        decoder = load_model("./model/dec.h5", compile=False)
        noise_right = decoder(mid_float_right_).numpy()  
        origin_right = get_data_ARI(noise_right,sigma)
        return origin_right

if __name__ == '__main__':


    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    i = 0
    ac = Accuracy()              
    delta=0.46
    sigma=1
    origin_right = ac.extract('./generate_output_secret/prediction.mid', sigma)
    b = origin_right.reshape(100*sigma).tolist()
    
    
    secret = DecodeSecret(b)
    print(secret)
