# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
import pickle

# 데이터 읽기
with open('dataset2D.pickle', 'rb') as file:
    dataset = pickle.load(file)
    
x_train = dataset['x_train']
t_train = dataset['t_train']
x_test = dataset['x_test']
t_test = dataset['t_test']

#flatten
#우리 x_데이터 형상 (N, 28, 28) -> (N, 784)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

#우리 t_데이터는 기본이 정답 숫자 한개만으로 이루어진 one-hot-label, 형상 (N, )
#(N,) -> (N, 10) 형태 데이터셋으로 변경
def convert_to_one_hot(labels, num_classes):
    # Create an identity matrix of shape (num_classes, num_classes)
    identity_matrix = np.eye(num_classes)
    
    # Use the labels as indices to extract the corresponding rows from the identity matrix
    one_hot_labels = identity_matrix[labels]
    
    return one_hot_labels


t_train = convert_to_one_hot(t_train, 10)
t_test = convert_to_one_hot(t_test, 10)

# Print the shapes of the arrays
print('x_train shape:', x_train.shape)
print('t_train shape:', t_train.shape)
print('x_test shape:', x_test.shape)
print('t_test shape:', t_test.shape)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % 622 == 0:    #숫자가 딱 안 나눠져서 나누어 떨어지는 숫자로 나눠줬다.
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

# loss function 추이
iters = np.arange(0, iters_num)
loss = train_loss_list
plt.plot(iters, loss, label="loss")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("loss function trend")
plt.show()

# acccuracy 추이
epochs = np.arange(1, len(train_acc_list)+1)
train_acc_x = np.array(train_acc_list)
test_acc_x = np.array(test_acc_list)
plt.plot(epochs, train_acc_x, label="train acc")
plt.plot(epochs, test_acc_x, linestyle="--", label="test acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("accuracy trend")
plt.legend()
plt.show()
