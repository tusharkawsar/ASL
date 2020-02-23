
# coding: utf-8

# In[1]:


from __future__ import print_function,division,absolute_import
import numpy as np
np.random.seed(1337)


# In[2]:


#Now Let's define the model
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,MaxPool2D,Activation,Flatten,BatchNormalization
from keras.optimizers import Adam,Adadelta,RMSprop
from keras.losses import categorical_crossentropy
from keras import utils


# In[7]:


# define two groups of layers: feature (convolutions) and classification (dense)
feature_layers = [
    Conv2D(64,3,
           input_shape=(274,250,1)),
    Activation('elu'),
    Dropout(0.25),
    Conv2D(64,3),
    Activation('elu'),
    MaxPool2D(pool_size=2),
  
    #BatchNormalization(),
    
    Conv2D(128,3),
    Activation('elu'),
    Dropout(0.25),
    Conv2D(128,3),
    Activation('elu'),
    MaxPool2D(pool_size=2),
    
    #BatchNormalization(),
    

    Conv2D(256,3),
    Activation('elu'),
    Dropout(0.25),
    Conv2D(256,3),
    Activation('elu'),
    MaxPool2D(pool_size=2),
    
    #BatchNormalization(),
    
    
    Conv2D(512,3),
    Activation('elu'),
    Dropout(0.25),
    Conv2D(512,3),
    Activation('elu'),
    MaxPool2D(pool_size=2),
    
    #BatchNormalization(),
    
    
    Flatten(),
]

classification_layers = [
    Dense(512),
    Activation('elu'),
    Dropout(0.5),
    
    
    Dense(256),
    Activation('elu'),
    Dropout(0.5),
    
    Dense(10),
    Activation('softmax')
]

#model building
model = Sequential(feature_layers + classification_layers)
model.summary()
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])


# In[8]:


dat = np.load('Hand_data_train.npz')
trainX,TrainY = dat['arr_0'],dat['arr_1']
trainY = utils.np_utils.to_categorical(TrainY,10)
trainX = trainX/255
trainX = trainX.astype('float32')
trainX = trainX.reshape((trainX.shape[0],274,250,1)).astype('float32')
print(trainX.shape,trainY.shape)


# In[9]:


dat = np.load('Hand_data_test.npz')
testX,TestY = dat['arr_0'],dat['arr_1']
testY = utils.np_utils.to_categorical(TestY,10)
testX = testX/255
testX = testX.astype('float32')
testX = testX.reshape((testX.shape[0],274,250,1)).astype('float32')
print(testX.shape,testY.shape)


# In[10]:


#now let's make the data Augmentation
from keras.callbacks import TensorBoard,ModelCheckpoint
from os.path import isfile
data_aug_weight_file = 'Hand_new-weights-original-25jul.h5'

if (isfile(data_aug_weight_file)):
    model.load_weights(data_aug_weight_file)

checkpoint = ModelCheckpoint(data_aug_weight_file, monitor='acc', verbose=1, save_best_only=True, mode='max')
#tensorboard = TensorBoard(log_dir='./logs-dataAug-25-jul', histogram_freq=0,write_graph=True, write_images=True)
callbacks_list=[checkpoint]

model.fit(trainX, trainY, batch_size=20,epochs=100,verbose=1, validation_data=(testX, testY),callbacks=callbacks_list)

