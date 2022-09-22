# 1 隐写信息入midi文件
# 2 midi文件中提取
# 3 计算准确率
import os
import statistics
import json
import numpy as np
from keras.models import load_model
from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
# warnings.filterwarnings('ignore')
import music21 as m21
from music21 import converter, instrument, note, chord, stream


class Accuracy:
    #def __init__(self):
      #self.notes
      #self.notes = np.load("notes.npy").tolist()
      #self.notes_set = sorted(set(item for item in self.notes))
      #self.vocab = len(self.notes_set)
      #print(self.vocab)
      #self.int_to_note = dict((number, note) for number, note in enumerate(self.notes_set))
      #self.note_to_int = dict((str(val), key) for key, val in self.int_to_note.items())
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
       

    # 从指定midi文件中提取出01序列
    def extract(self, filename,sigma):
        songs = read_midi_from_file(filename)
        #print(songs)
        time_step = 0.25
        
        encoded_song = []
        notes_in_song = None
        try:
          s2 = instrument.partitionByInstrument(songs)
          notes_in_song = s2.parts[0].recurse()
        except:
          notes_in_song = songs.flat.notes

        #print(notes_in_song)

        for element in notes_in_song:
          #print(type(element))

          if isinstance(element, note.Note):
              m_event = element.pitch.midi
              time = element.duration.quarterLength
              #print(m_event)
              for step in range(0,int(time/ time_step)):
                  if step == 0:
                      encoded_song.append(str(m_event))
                  else:
                      encoded_song.append("_")
            

          elif isinstance(element, note.Rest):
              m_event = "r"
              time = element.duration.quarterLength
              #print(m_event)
            
            
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
           
              
    

          #steps = int(element.duration.quarterLength / time_step)
        

        encoded_song = " ".join(map(str, encoded_song))
        notes_ = encoded_song



        '''
        for song in songs:
          notes_ = encode_song(song)
        '''

        
        print(notes_)

        with open("./data_preparation/created_mapping_0.25", 'r') as fp:
          mappings = json.load(fp)

        l = len(mappings)
       
        #notes_ = read_notes_from_file(filename)
        mid_int_right = notes_to_int(notes_)

        
        '''
        #mid_int_right = [self.notes_set.index(val) for val in notes_]
        mid_int_right = []
        for i in notes_:

          mid_int_right.append([v for k, v in mappings.items() if str(i) == k])

        #print(mid_int_right)
        
        '''
        
        mid_float_right = [(x - float(l/2)) / float(l/ 2) for x in mid_int_right]

        del(mid_float_right[0])
        mid_float_right.pop()
        # del(mid_float_right[100])
        mid_float_right_ = np.reshape(mid_float_right, (1, 100))
       
        decoder = load_model("./model/dec.h5", compile=False)
        noise_right = decoder(mid_float_right_).numpy()
     
        origin_right = get_data_ARI(noise_right,sigma)
      
        return origin_right

    # 计算准确率
    def precision_score(self, np1, np2,sigma):
        c = np1 - np2
        num = 0
        for x, value in np.ndenumerate(c):
            if value == 0:
                num += 1
        return num / (100.0*sigma)

def get_plot():
  delta1 = [0,0.1,0.2,0.3,0.4,0.45,0.46,0.47,0.48,0.49]
  acc1 = [0.89,0.95,0.85,0.97,0.87,0.9,0.86,0.99,1.0,0.99]

  delta2 = [0,0.1,0.2,0.21,0.22,0.23,0.24]
  acc2 = [0.9,0.73,0.745,0.94,0.745,0.675,0.71]

  plt.xlabel('delta')
  plt.ylabel('accuracy')

  plt.plot(delta1,acc1,c='green', linestyle='--',marker="+")
  plt.plot(delta2,acc2,c='blue', linestyle=':',marker="d")

  plt.legend(['100','200',])

  plt.savefig('plotfig.png')
  plt.show()




if __name__ == '__main__':

  

    tf.get_logger().setLevel('ERROR')
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    i = 0
    ac = Accuracy()

    delta_list = [[0,0.1,0.2,0.3,0.4,0.45,0.46,0.47,0.48,0.49],
                  [0,0.1,0.2,0.21,0.22,0.23,0.24]]
    dict = {}
    for sigma in range(1,3,1):

      
            for delta in delta_list[sigma-1]:
                acc=[]
                # for i in range(0, 150):
                origin_left = np.random.randint(low=0, high=2, size=(1, 100*sigma))
                ac.generate(origin_left,sigma ,delta ,filename='./generate_output_test/sigma_{}_delta_{}/prediction'.format(sigma,delta)+str(i))
                origin_right = ac.extract('./generate_output_test/sigma_{}_delta_{}/prediction'.format(sigma,delta)+str(i)+ '.mid', sigma)
                a = origin_left.reshape(100*sigma).tolist()
                b = origin_right.reshape(100*sigma).tolist()
   
                acc.append(ac.precision_score(origin_right, origin_left,sigma))

                txt='./data/secret.txt'
                with open(txt, 'a') as f:
                    f.write('sigma_{}_delta_{}'.format(sigma,delta) +'  ')
                    f.write(str(a))
                    #print(f.write(str(a)))
                    f.write('\r\n')
                f.close() 
    
                txt2='./data/secret2.txt'
                with open(txt2, 'a') as f:
                    f.write('sigma_{}_delta_{}'.format(sigma,delta) + ' ')
                    f.write(str(b))
                    #print(f.write(str(b)))
                    f.write('\r\n')
                f.close() 

                print('origin_right')
                print(origin_right)
                
                

                print("输出准确率")
                for i in acc:
                    print(i)
                print("Average: " + str(statistics.mean(acc)))
                dict.update({'sigma_{}_delta_{}'.format(sigma,delta):statistics.mean(acc)})
                print(dict)

                  
                # plt.plot(acc, c='red')
                # # plt.plot(sigma=2, c='blue')
                # # plt.plot(self.dec_loss,c='green')
                # plt.title("The  test of sigma  and delta")
                # # plt.legend(['sigma=1', 'sigma=2'])
                # plt.xlabel('sigma','dalta')
                # plt.ylabel('acc')
                # plt.savefig('./pic/test.png', transparent=True)
                # # plt.plot(self.decoder,to_file = 'decoder_model.png',show_shapes = True)
                # plt.close()

