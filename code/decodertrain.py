from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from keras.models import load_model
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import keras.backend as K
from utils import read_notes_from_files,get_input_output,create_midi
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


decoder = load_model("./model/dec_100.h5", compile=True)
encoder = load_model("./model/g_100.h5", compile=False)

dec_loss=[]
epochs = 10000
batch_size = 64
sample_interval = 10
noise_dim=100
sequence_length=100
# notes = read_notes_from_files()
# n_vocab = len(set(notes))
# X_train, y_train = get_input_output(notes,n_vocab)
X_train = np.load('./data_preparation/X_train.npy')
# real = np.ones((batch_size, 1))
# fake = np.zeros((batch_size, 1))
K.set_value(decoder.optimizer.lr,0.00008)
for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_seqs = X_train[idx]

    # noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
    noise = -1 + 2 * np.random.random((batch_size, noise_dim))
    gen_seqs = encoder.predict(noise)
    gen_seqs_for_dec = gen_seqs.reshape(batch_size, sequence_length)
    dec_loss_v = decoder.train_on_batch(gen_seqs_for_dec, noise)
    # noise = -1 + 2 * np.random.random((batch_size, noise_dim))
    # gen_seqs = encoder.predict(noise)
    # dec_seqs =decoder.predict(gen_seqs_for_dec)
    # e_loss1 = encoder.train_on_batch(dec_seqs, gen_seqs)
    if epoch % sample_interval == 0:
        print("%d  [Dec loss: %f]" % (
            epoch, dec_loss_v))
        # print("%d [Dec loss: %f]" % (epoch,  dec_loss))

        dec_loss.append(dec_loss_v)

# self.generate(notes)

#generator.save("./model/g_RNN_1.h5")
decoder.save("./model/dec_RNN_1_trainagain.h5")

# plt.plot(self.disc_loss, c='red')
# plt.plot(self.gen_loss, c='blue')
plt.plot(dec_loss, c='green')
plt.title("GAN Loss per Epoch")
#plt.legend(['Dis', 'Gen', 'Dec'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_RNN_1_trainagain.png', transparent=True)
plt.close()


