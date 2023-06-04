import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = 'data1.jpg'

#1
src = cv2.imread(filename,cv2.IMREAD_GRAYSCALE) #그레이 스케일로 변환
#cv2.imshow('gray',src)

#2
ret , binary = cv2.threshold(src,170,255,cv2.THRESH_BINARY_INV) #영상 이진화

#cv2.imshow('binary',binary)

#3 이진화 이미지
binary = cv2.morphologyEx(binary , cv2.MORPH_OPEN , cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations = 2)
#cv2.imshow('binary',binary)

#4 열기연산을 통해 노이즈 제거
contours , hierarchy = cv2.findContours(binary , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
#외곽선 검출
color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) #이진화 이미지를 color이미지로 복사(확인용)
cv2.drawContours(color , contours , -1 , (0,255,0),3) #초록색으로 외곽선을 그려준다.

#리스트연산을 위해 초기변수 선언
bR_arr = []
digit_arr = []
digit_arr2 = []
count = 0

#검출한 외곽선에 사각형을 그려서 배열에 추가
for i in range(len(contours)) :
    bin_tmp = binary.copy()
    x,y,w,h = cv2.boundingRect(contours[i])
    bR_arr.append([x,y,w,h])

#5
print(bR_arr[:5])
#x값을 기준으로 배열을 정렬
bR_arr = sorted(bR_arr, key=lambda num : num[0], reverse = False)
print(bR_arr[:5])
print(len(bR_arr))

#6  box를 그림으로 그려주면서 일정크기 이하의 box를 버려 노이즈를 제거하고 나머지는 이미지로 잘라내어 새로운 배열에 담아주기
# 작은 노이즈데이터 버림,사각형그리기,2개씩 리스트로 다시 묶어서 저장
for x, y, w, h in bR_arr:
    tmp_y = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[0]
    tmp_x = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[1]
    if tmp_x and tmp_y > 10:
        count += 1
        cv2.rectangle(color, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 1)
        digit_arr.append(bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2])
        if count == 2:
            digit_arr2.append(digit_arr)
            digit_arr = []
            count = 0

cv2.imshow('contours', color)
print(len(digit_arr2[0]))

#7
#리스트에 저장된 이미지를 32x32의 크기로 리사이즈해서 순서대로 저장
for i in range(0,len(digit_arr2)) :
    for j in range(len(digit_arr2[i])) :
        count += 1
        if i == 1 :         #1일 경우 비율 유지를 위해 마스크를 만들어 그위에 얹어줌
            width = digit_arr2[i][j].shape[1]
            height = digit_arr2[i][j].shape[0]
            tmp = (height - width)/2
            mask = np.zeros((height,height))
            mask[0:height,int(tmp):int(tmp)+width] = digit_arr2[i][j]
            digit_arr2[i][j] = cv2.resize(mask,(32,32))
        else:
            digit_arr2[i][j] = cv2.resize(digit_arr2[i][j],(32,32))
#        if i == 9 : i = -1
        cv2.imwrite('./data/'+str(i)+'_'+str(j)+'.png',digit_arr2[i][j])



k = cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show()
