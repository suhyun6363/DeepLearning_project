import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

# test data 로드
with open('/content/drive/MyDrive/testdata3D_N.pkl', 'rb') as f:
	datasetL = pickle.load(f)
x_test, t_test = datasetL

# 이미지 데이터 크기 (28, 28, 1)로 reshape
x_test = x_test.reshape(len(x_test), 28, 28, 1)
# 픽셀값을 0~1 범위로 정규화

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (5, 5), padding='same', kernel_initializer='he_normal', activation='relu', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    #keras.layers.Dropout(0.0),
    keras.layers.Flatten(),
    keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
    keras.layers.Dense(10, kernel_initializer='he_uniform', activation='softmax')
])

# h5 file로 model load
loaded_model = keras.models.load_model('/content/drive/MyDrive/model_epochs_15_batch_10_kernels_55_33.h5')


# 평가
test_loss, test_accuracy = loaded_model.evaluate(x_test, t_test)

predictions = loaded_model.predict(x_test)

predicted_labels = np.argmax(predictions, axis=1)


print("Test Accuracy:", test_accuracy)