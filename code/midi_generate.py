import os
import statistics

import numpy as np
from keras.models import load_model
from utils import *
import tensorflow as tf
# warnings.filterwarnings('ignore')

class Accuracy:
    # def __init__(self):

    #     self.notes = np.load("notes.npy").tolist()
    #     self.notes_set = sorted(set(item for item in self.notes))
    #     self.vocab = len(self.notes_set)
    #     self.int_to_note = dict((number, note) for number, note in enumerate(self.notes_set))


    def generate(self, msg,sigma,delta, filename="2022_04_17_5_0.25"):
        
        noise_left = data_map_ARI(msg,sigma,delta)
        encoder = load_model("./model/gen.h5", compile=False)


        mid_float_left = encoder.predict(noise_left)

        with open("./data_preparation/created_mapping_0.25", 'r') as fp:
          mappings = json.load(fp)

        l = len(mappings)
        pred_notes = [x*float(l/2)+float(l/2) for x in mid_float_left[0]]

        # prediction_output = []
        prediction_output = ['7.10']



        for i in pred_notes:
          prediction_output.append([ k for k , v in mappings.items() if v == int(i)][0])

        prediction_output.append('72')
        prediction_output.append('72')
       
               

        print(prediction_output)
        os.makedirs(filename,exist_ok=True)
        #create_midi(pred_notes, filename)
        create_midi(prediction_output,filename)



if __name__ == '__main__':


    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    ac = Accuracy()                 
    delta=0.46
    sigma=1    
    f=input("请输入秘密信息：")
    b = EncodeSecret(f)
    c = b.replace(" ", "")
    list =[]
    for i in c:
        if(i == "0" or i =="1"):
            e = int(i)
            #print(e)
            # print(type(e))
            list.append(e)
    list.extend([0,0])
    list2 =[list]
    print(list2)   
    origin_left = np.array(list2)
    ac.generate(origin_left,sigma ,delta ,filename='./generate_output_secret/'+"prediction")
   