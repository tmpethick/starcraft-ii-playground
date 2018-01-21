#%%

import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, Flatten

NUM_ACTIONS = 2

x = Input(shape=(84, 84, 1))

y = Conv2D(16, 5, strides=(1, 1), padding='same')(x)
y = Conv2D(32, 3, strides=(1, 1), padding='same')(y)

# Action id
a = Conv2D(16, 8, strides=(4, 4), padding='valid')(y)
a = Conv2D(32, 4, strides=(2, 2), padding='valid')(a)
a = Activation('relu')(a)
a = Flatten()(a)
a = Dense(256, activation='relu')(a)
a = Dense(NUM_ACTIONS, name='action_id', activation='softmax')(a)

# Position
p = Conv2D(1, 1, name="action_corr", strides=(1, 1), padding='same')(y)
p = Flatten()(p) # flatten since `categorical_crossentropy` loss requires it.
p = Activation('softmax')(p)

model = Model(inputs=x, outputs=[a, p])

adam = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#%%

X = np.genfromtxt('X.csv', delimiter=",")
X = X.reshape(-1, 84, 84, 1)

# flatten since `categorical_crossentropy` loss requires it.
Yspacial = np.genfromtxt('Y-spacial.csv', delimiter=",")
Yspacial = Yspacial.reshape(-1, 84 * 84) 

#%%

Y = np.genfromtxt('Y.csv', delimiter=",")
Y_hot = keras.utils.to_categorical(Y, num_classes=NUM_ACTIONS)

#%%

model.fit(
        X, 
        [Y_hot, Yspacial], 
        shuffle=True, 
        epochs=100,
        batch_size=64)

#%%

