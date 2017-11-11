from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
import numpy as np
from sklearn.model_selection import train_test_split


# Based on VGG16
model = Sequential()

model.add(Conv2D(8, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(Conv2D(8, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1), padding="same", input_shape=(17, 17, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Flatten())
model.add(Dense(800, activation='relu'))
#model.add(Dense(800, activation='relu'))
model.add(Dense(289, activation='relu'))

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

x = np.load('x_0.npy')
print(x.shape)
x.resize((x.shape[0], 17, 17, 1))
y = np.load('y_0.npy')
print(y.shape)
y.resize((y.shape[0], 289))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

checkpoint = ModelCheckpoint(filepath='gomoku.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=512, epochs=10, callbacks=[checkpoint])