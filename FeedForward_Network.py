#!/usr/bin/env python
# coding: utf-8

# In[30]:


import keras

import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import optimizers 

import matplotlib.pyplot as plt # package for plotting

from sklearn import metrics # package to for MAE and MSE

from sklearn.model_selection import train_test_split


# In[31]:


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


# In[32]:


output = np.loadtxt('Absorption.txt')
input = np.loadtxt('Thickness.txt')

output= np.array(output)
input= np.array(input)
train_input, val_input, train_output, val_output = train_test_split(input, output, test_size=0.33, shuffle= True)


# In[33]:


# structure of Network

model = Sequential()

# Dropout can be used for some hidden layer or all hidden layers. You can comment some of them to avoid dropout for specific layers
# A small dropout like 20-50% is more effective
model.add(Dense(400, activation='relu',init='random_uniform', input_shape=(7,), name="dense_1"))
model.add(Dropout(0.2)) # drops 20% of the neuron for dense_2 layer
model.add(Dense(400, activation='relu', init='random_uniform',input_shape=(400,), name="dense_2"))
model.add(Dropout(0.2)) # drops 20% of the neuron for dense_3 layer
model.add(Dense(200, activation='relu',init='random_uniform', input_shape=(400,), name="dense_3"))
model.add(Dropout(0.2)) # drops 20% of the neuron for dense_4 layer
model.add(Dense(200, activation='relu', init='random_uniform',input_shape=(200,),name="dense_4"))
model.add(Dropout(0.2)) # drops 20% of the neuron for dense_5 layer
model.add(Dense(500,activation='tanh',init='random_uniform',input_shape=(200,),name="dense_5"))

model.summary()


# In[41]:


adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

model.compile(loss='mse',optimizer='adam' ,metrics=['mae'])

history = LossHistory()

model.fit(train_input, train_output, batch_size=64, epochs=1000, shuffle=True,verbose=2,validation_data=(val_input, val_output), callbacks=[history])

model.save('Weight_ff.h5')

model.save_weights('Weight_ff.h5')

history.loss_plot('epoch') 


# In[42]:


predicted_output = model.predict(test_input) # get prediction for test_input
errors = list()
for true_val, pred_val in zip(test_output, predicted_output):
    temp_error = metrics.mean_absolute_error(true_val, pred_val) 
    errors.append(temp_error)
errors = np.asarray(errors)


# In[43]:


plt.figure()
x = range(len(errors))
plt.plot(x, errors)
plt.xlabel('Test Samples')
plt.ylabel('Prediction Error')
plt.show()


# In[44]:


structure=(0, 10e-9, 450e-9, 100e-9, 66.07e-9, 5e-9, 0) # Proposed Multilayer Structure Layer Thicknesess
structure=np.array(structure).reshape(1,7)
spectrum_predict=model.predict(structure)
spectrum_predict=np.array(spectrum_predict).reshape(500)


# In[45]:


with open('Wavelength.txt') as f:
    lines = f.readlines()
    x1 = [line.split()[0] for line in lines]

for i in range(0, len(x1)): 
    x1[i] = float(x1[i]) 

x1 = np.reshape(x1,(500,1)) 
x1 = x1.flatten() 

plt.figure()
plt.plot(x1, spectrum_predict) # Absorption Spectrum
plt.grid(True)
plt.xlabel('wavelength in um')
plt.ylabel('spectrum_predict')
plt.show()
# plt.savefig('Spectrum_predicted_vs_wavelength.jpg')
# plt.close()


# In[ ]:




