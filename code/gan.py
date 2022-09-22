from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from utils import *
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class GAN():
  def __init__(self, rows):
    self.sequence_length = rows
    self.seq_shape = (self.sequence_length, 1)
    self.noise_dim = 100
    self.disc_loss = []
    self.gen_loss =[]
    self.dec_loss =[]
    # self.notes = read_notes_from_files()
    # self.vocab = len(set(self.notes))
    # print(self.vocab)

    optimizer = Adam(0.0006, 0.5)
    # optimizer = Adam(0.0006)
    optimizer2 = Adam(0.0006)
    self.decoder = self.build_decoder()
    self.decoder.compile(loss='mse', optimizer=optimizer2)
    self.generator = self.build_generator()
    self.generator.compile(loss='mse', optimizer=optimizer)
    self.discriminator = self.build_discriminator()
    self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #self.generator = self.build_generator()
    z = Input(shape=(self.noise_dim,))
    generated_seq = self.generator(z)

    self.discriminator.trainable = False
    validity = self.discriminator(generated_seq)
    self.combined = Model(z, validity)
    self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    z_1 = Input(shape=(self.noise_dim,))
    generated_seq = self.generator(z_1)

    #self.discriminator.trainable = False
    validity_1 = self.decoder(generated_seq)
    self.combined_1 = Model(z_1, validity_1)
    self.combined_1.compile(loss='mse', optimizer='adam')
  def build_discriminator(self):

    model = Sequential()
    model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(512))
    # model.add(Dropout(0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    # model.add(Dropout(0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    seq = Input(shape=self.seq_shape)
    validity = model(seq)

    return Model(seq, validity)

  def build_generator(self):
    model = Sequential()
    model.add(Dense(256, input_dim=self.noise_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
    model.add(Reshape(self.seq_shape))
    model.summary()

    noise = Input(shape=(self.noise_dim,))
    seq = model(noise)

    return Model(noise, seq)



  def build_decoder(self):
      model = Sequential()

      model.add(Dense(1024, input_dim=self.sequence_length, activation='relu'))
      model.add(Dense(512, activation='relu'))
      model.add(Dense(256, activation='relu'))
      model.add(Dense(self.noise_dim, activation='tanh'))
      model.summary()
      seq = Input(shape=(self.sequence_length,))

      validity = model(seq)
      return Model(seq, validity)
 
  def train(self, epochs, batch_size=128, sample_interval=50):
    X_train = np.load('./data_preparation/X_train.npy')
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
      idx = np.random.randint(0, X_train.shape[0], batch_size)
      # print(X_train.shape[0]) #158373
      real_seqs = X_train[idx]

      #noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
      noise = -1 + 2 * np.random.random((batch_size, self.noise_dim))
      gen_seqs = self.generator.predict(noise)
      # gen_seqs_for_dec=gen_seqs.reshape(batch_size,self.sequence_length)
      # dec_loss = self.decoder.train_on_batch(gen_seqs_for_dec,noise)
      dec_loss = self.combined_1.train_on_batch(noise, noise)
      if epoch % 20 ==0:
          d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
          d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

      #noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
      #dec_seqs = self.decoder.predict(gen_seqs_for_dec)
      #dec_seqs = self.decoder.predict(gen_seqs)
      # encoder的第一部分loss是看他能否顺利将decoder的产生的序列还原
      #e_loss1 = self.generator.train_on_batch(dec_seqs, gen_seqs)
      noise = -1 + 2 * np.random.random((batch_size, self.noise_dim))
      g_loss = self.combined.train_on_batch(noise, real)
      #e_loss = 1.0 / 4 * np.add(np.dot(e_loss1, 3), g_loss)

      if epoch % sample_interval == 0:
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [Dec loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss,dec_loss))
        #print("%d [Dec loss: %f]" % (epoch,  dec_loss))
        self.disc_loss.append(d_loss[0])
        self.gen_loss.append(g_loss)
        self.dec_loss.append(dec_loss)



    #self.generate(notes)
        if epoch % 500 == 0:
          print('saving...')
          self.save()
          print('saving done')
          self.plot_loss()
    # self.generator.save("./model/generator.h5")
    # self.discriminator.save("./model/discriminator.h5")
    # self.decoder.save("./model/decoder.h5")
          



  def save(self):
    if not os.path.exists('./model/'):
      os.makedirs('./model/')

    self.discriminator.save('./model/discriminator2.h5')
    self.decoder.save('./model/decoder2.h5')
    self.generator.save('./model/generator2.h5')
    print("The trained GAN model (generator and discriminator) have been saved in the Model folder.")

  def generate(self, input_notes):
    notes = input_notes
    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    noise = np.random.normal(0, 1, (1, self.noise_dim))
    predictions = self.generator.predict(noise)

    pred_notes = [x*float(self.vocab/2)+float(self.vocab/2) for x in predictions[0]]
    pred_notes = [int_to_note[int(x)] for x in pred_notes]

    #create_midi(pred_notes, 'predicted_4_RNN')

  

  def plot_loss(self):
    plt.plot(self.disc_loss, c='red')
    plt.plot(self.gen_loss, c='blue')
    plt.plot(self.dec_loss,c='green')
    plt.title("GAN Loss per Epoch")
    plt.legend(['Dis', 'Gen','Dec'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./pic/loss.png', transparent=True)
    # plt.plot(self.decoder,to_file = 'decoder_model.png',show_shapes = True)
    plt.close()
  
    
gan = GAN(rows=100)
gan.train(epochs=20000, batch_size=128, sample_interval=1)
