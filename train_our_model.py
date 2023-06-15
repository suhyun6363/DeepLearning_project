import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

# dataset load
with open('dataset3D.pickle', 'rb') as f:
    dataset_load = pickle.load(f)
x_train = dataset_load['x_train']
t_train = dataset_load['t_train']
x_test = dataset_load['x_test']
t_test = dataset_load['t_test']

# 이미지 데이터 크기 (28, 28, 1)로 reshape
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)
# 픽셀값을 0~1 범위로 정규화
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (5, 5), padding='same', kernel_initializer='he_normal', activation='relu', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(64, (5, 5), padding='same', kernel_initializer='he_normal', activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    #keras.layers.Dropout(0.0),
    keras.layers.Flatten(),
    keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, kernel_initializer='he_uniform', activation='softmax')
])

# Compile 
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train
model.fit(x_train, t_train, epochs=15, batch_size=10)

_, accuracy = model.evaluate(x_test, t_test)
print("Test Accuracy:", accuracy)

# parameter 값 .h5 file로 저장
model.save('model_parameters.h5')

