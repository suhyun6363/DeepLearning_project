import tensorflow as tf
import numpy as np
import cv2

path = '0_arg/' # 데이터 증폭할 폴더의 경로 (숫자 0 데이터 폴더)

# 이미지 증강 클래스
generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=45,  # -45~45도 중 랜덤으로 데이터 증폭
#    width_shift_range=0.2, # 가로로 랜덤으로 데이터 증폭
#    height_shift_range=0.2,    #세로로 랜덤으로 데이터 증폭
    rescale=1. / 255) # 값을 0과 1 사이로 변경

batch_size = 41 # 증폭할 폴더 안의 파일 개수
iterations = 5
images = []

obj = generator.flow_from_directory(    # 폴더 구성 그대로 데이터를 처리하여 로드
    path,   # 경로 설정
    target_size=(28, 28),   # resizing
    batch_size=batch_size,
    class_mode='binary')    # 폴더명을 기반으로 'binary' 방식에 맞춰 labelling 진행

for i, (img, label) in enumerate(obj):
    n_img = len(label)

    base = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)  # keras는 RGB, openCV는 BGR이라 변경
    for idx in range(n_img - 1):
        img2 = cv2.cvtColor(img[idx + 1], cv2.COLOR_RGB2BGR)
        base = np.hstack((base, img2))  # 이미지를 옆으로 결합
    images.append(base)

    if i is iterations - 1: # for문의 반복 횟수를 제한
        break

img = images[0]

for idx in range(len(images) - 1):  # 데이터 증폭 랜덤으로 4번 반복
    for i in range(41): # 파일 41개
        img1 = images[idx + 1][0:28,i*28:(i+1)*28]  # 41개의 데이터가 합쳐진 이미지를 28 크기씩 크롭해서 따로 저장
        cv2.imwrite('./0_arg/0_arg/img0_' + str(idx) + '_' + str(i) + '.png', 255 * img1)   # 파일 저장
    #    cv2.imshow('result', images[idx + 1])

#cv2.waitKey(0)
#cv2.destroyAllWindows()



