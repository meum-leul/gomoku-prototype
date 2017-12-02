from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, SeparableConv2D, Concatenate, Add, AveragePooling2D, BatchNormalization, Activation, Input, GlobalAveragePooling2D
import numpy as np
from sklearn.model_selection import train_test_split

def rb(tensor):
    return Activation('relu')(BatchNormalization()(tensor))

def id(tensor):
    return rb(Conv2D(64, (1, 1), padding='same')(tensor))

def normal_nas(pre, now):
    return Concatenate()([
        Add()([
            rb(Conv2D(64, 3, padding='same')(rb(Conv2D(64, 3, padding='same')(now)))),
            id(now)
        ]),
        Add()([
            rb(Conv2D(64, 3, padding='same')(rb(Conv2D(64, 3, padding='same')(pre)))),
            rb(Conv2D(64, 5, padding='same')(rb(Conv2D(64, 5, padding='same')(now)))),
        ]),
        Add()([
            id(AveragePooling2D((3, 3), strides=1, padding='same')(now)),
            id(pre),
        ]),
        Add()([
            id(AveragePooling2D((3, 3), strides=1, padding='same')(pre)),
            id(AveragePooling2D((3, 3), strides=1, padding='same')(pre)),
        ]),
        Add()([
            rb(Conv2D(64, 5, padding='same')(rb(Conv2D(64, 5, padding='same')(pre)))),
            rb(Conv2D(64, 3, padding='same')(rb(Conv2D(64, 3, padding='same')(pre)))),
        ]),
    ])

def reduction_nas(tensor):
    return rb(Conv2D(64, (1, 1), padding='same')(AveragePooling2D((2, 2), padding='valid')(tensor)))

# Based on NASNet-A
input_tensor = Input(shape=(16, 16, 1))

# Base
b0 = Conv2D(64, (3, 3), padding='same')(input_tensor)
b1 = Conv2D(64, (3, 3), padding='same')(b0)

# NAS0
n00 = normal_nas(b0, b1)
n01 = normal_nas(b1, n00)
n02 = normal_nas(n00, n01)
n03 = reduction_nas(n02)
n04 = id(n03)

# NAS1
n10 = normal_nas(n03, n04)
n11 = normal_nas(n04, n10)
n12 = normal_nas(n10, n11)
n13 = reduction_nas(n12)
n14 = id(n13)

# NAS2
n20 = normal_nas(n13, n14)
n21 = normal_nas(n14, n20)
n22 = normal_nas(n20, n21)
# n23 = reduction_nas(n22)
# n24 = id(n23)

# Softmax
output_tensor = GlobalAveragePooling2D()(n22)
output_tensor = Dense(16*16, activation='softmax')(output_tensor)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

x = np.load('x.npy')
x.resize((x.shape[0], 16, 16, 1))
y = np.load('y.npy')
y.resize((y.shape[0], 16*16))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

checkpoint = ModelCheckpoint(filepath='gomoku_nas.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=2, batch_size=32, epochs=20, callbacks=[checkpoint])
