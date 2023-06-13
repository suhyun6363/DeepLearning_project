#NETWORK weight INITIALIZATION1
#ReLu -> he 적합
#softmax -> 다 해보기	
#init_mode_1 : ['he_normal', 'he_uniform']
#init_mode_2 : ['he_normal', 'he_uniform']
#init_mode_3 : ['he_normal', 'he_uniform']
#init_mode_4 : ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
import numpy as np
import pickle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

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

def create_model(optimizer,
                 learn_rate,
                 pool_size):
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(metrics=['acc'],
                  optimizer='Adadelta',
                  loss="sparse_categorical_crossentropy")
    
    return model

# 재현성을 위해 random seed 고정
seed = 0
np.random.seed(seed)

# 아래 param 모든 조합 test
param_grid = {'epochs': [5],
              'batch_size': [8],
              'optimizer' : ['Adadelta'],
              'learn_rate': [0.001],
              'pool_size' : [[2, 2], [4, 4]]
              }

my_classifier = KerasClassifier(create_model)

grid = GridSearchCV(my_classifier, param_grid, cv=5, n_jobs=1, verbose=1)

grid_result = grid.fit(x_train, t_train)

# 결과 요약
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# 결과 출력
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Test accuracy 출력
print("Test Accuracy", grid_result.score(x_test, t_test))