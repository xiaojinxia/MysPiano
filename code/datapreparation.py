from utils import *
#from utils import *
import numpy as np

preprocess()
songs = create_dataset()
create_mapping(songs)
#notes = load("created_dataset")
#notes=np.array(notes)
#np.save('notes_2022_04_13_1.npy',notes)
print("Create_mapping over")



X_train_2 = np.array(generate_input_output())
np.save('./data_preparation/X_train.npy',X_train_2)
print("Save X_train over")
