from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
import numpy as np
from sklearn.model_selection import train_test_split


# Based on VGG16
model = Sequential()

model.add(Conv2D(8, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(Conv2D(8, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(16, 16, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Flatten())
model.add(Dense(800, activation='relu'))
model.add(Dense(16*16, activation='softmax'))

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

x = np.load('x.npy')
x.resize((x.shape[0], 16, 16, 1))
y = np.load('y.npy')
y.resize((y.shape[0], 16*16))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

checkpoint = ModelCheckpoint(filepath='gomoku.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=512, epochs=30, callbacks=[checkpoint])