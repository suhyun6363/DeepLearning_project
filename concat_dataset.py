import os
import tensorflow as tf #framework 안 쓸 수가 없었음.. tensorflow install 미리 해주세요.
from tensorflow import keras
import numpy as np
import pickle

# Load MNIST dataset
# 수업 시간에 사용한 MNIST dataset 사용시 경로 설정이 필요하므로 keras에서 load
mnist = keras.datasets.mnist
(x_train, t_train), (x_test, t_test) = mnist.load_data()

# 추가할 dataset의 경로
# 본인의 경로에 맞게 변경해주세요.
own_dataset_path = 'C:\\Users\\UNA\\deep-learning-from-scratch-master\\deep_learning_project\\team_aug'

# dataset folders 회전
own_images = []
own_labels = []
for folder_name in os.listdir(own_dataset_path):
    folder_path = os.path.join(own_dataset_path, folder_name)
    if os.path.isdir(folder_path):
        label = int(folder_name.split("_")[0])  # folder 명으로부터 label 추출
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=(28, 28))
            image = keras.preprocessing.image.img_to_array(image)
            own_images.append(image)
            own_labels.append(label)

# NumPy array로 데이터형 변환
own_images = np.array(own_images) # own_images의 shape은 (N, 28, 28, 1)
own_labels = np.array(own_labels)

#먼저 mlp에서 사용하는 2D shpae으로 진행하고 cnn에서 사용하는 3D shape은 마지막에 처리하고 다른 이름으로 저장
own_images = np.squeeze(own_images, axis=3)

#6:1 비율로 나누기
rate_into_indexing = int(own_images.shape(0) / 7) * 6
x_kr_train = own_images[:rate_into_indexing]
t_kr_train = own_images[:rate_into_indexing] 
x_kr_test = own_images[rate_into_indexing:]
t_kr_test = own_labels[rate_into_indexing:]

# Concatenate the MNIST dataset with your own dataset
x_train = np.concatenate((x_train, x_kr_train), axis=0)
t_train = np.concatenate((t_train, t_kr_train), axis=0)
x_test = np.concatenate((x_test, x_kr_test), axis=0)
t_test = np.concatenate((t_test, t_kr_test), axis=0)

# Shuffle the combined dataset
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
t_train = t_train[indices]

indices = np.arange(x_test.shape[0])
np.random.shuffle(indices)
x_test = x_test[indices]
t_test = t_test[indices]

###1. pickle file로 저장_2D_for MLP

# dataset 저장할 dictionary 생성
dataset = {
    'x_train': x_train,
    't_train': t_train,
    'x_test': x_test,
    't_test': t_test
}
# pickle file로 저장하기 위한 file_path
file_path = 'dataset2D.pickle'

#pickle file로 저장
with open(file_path, 'wb') as file:
    pickle.dump(dataset, file)
    
###2. pickle file로 저장_3D_for CNN
#(N, 28, 28)을 (N, 1, 28, 28)로 변환
x_train = np.expand_dims(x_train, axis=1)

# dataset 저장할 dictionary 생성
dataset = {
    'x_train': x_train,
    't_train': t_train,
    'x_test': x_test,
    't_test': t_test
}
# pickle file로 저장하기 위한 file_path
file_path = 'dataset3D.pickle'

#pickle file로 저장
with open(file_path, 'wb') as file:
    pickle.dump(dataset, file)







