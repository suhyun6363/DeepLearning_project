import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = 'img_ys2.PNG'

#1
img = cv2.imread('img_ys2.png')
src = cv2.imread(filename,cv2.IMREAD_GRAYSCALE) #그레이 스케일로 변환
cv2.imshow('gray',src)

#2
ret , binary = cv2.threshold(src,170,255,cv2.THRESH_BINARY_INV) #영상 이진화

#cv2.imshow('binary',binary)

#3 이진화 이미지
binary = cv2.morphologyEx(binary , cv2.MORPH_OPEN , cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations = 2)
#cv2.imshow('binary',binary)
#k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#dst = cv2.dilate(binary, k)

#4 열기연산을 통해 노이즈 제거
contours , hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
#외곽선 검출
color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) #이진화 이미지를 color이미지로 복사(확인용)
#cv2.drawContours(color , contours , -1 , (0,255,0),3) #초록색으로 외곽선을 그려준다.

#리스트연산을 위해 초기변수 선언
rects = []
digit_arr = []
digit_arr2 = []
count = 0
row = 2

#검출한 외곽선에 사각형을 그려서 배열에 추가
for i in range(len(contours)) :
    bin_tmp = binary.copy()
    x,y,w,h = cv2.boundingRect(contours[i])
    rects.append([x,y,w,h])

#5
#x값을 기준으로 배열을 정렬
rects = sorted(rects, key=lambda num : num[0], reverse = False)
print(len(rects))


# 내부에도 인식된 사각형이 있는 경우
if len(rects) != row*10:
    # 내부 인식된 사각형 제거하기 위해 인식된 사각형들 숫자별로 묶어주기
    number_dict = {}  # 숫자별 사각형 리스트의 딕셔너리
    num = 0
    prev_rect = []  # 다음에 올 사각형과 비교하기 위해 필요
    rect_list = []  # 숫자별 나올 사각형들의 리스트

    for i in range(len(rects)):  # 모든 사각형을 loop
        if i == 0:
            prev_rect = rects[i]
            rect_list.append(prev_rect)
            continue
        if i == len(rects) - 1:  # 마지막 사각형인 경우
            number_dict[num] = rect_list
        if rects[i][0] >= prev_rect[0] + 60:
            # 이전 사각형의 x좌표와 현재 사각형의 x좌표의 차이가 30이면 이를 경계로 숫자가 달라짐
            number_dict[num] = rect_list  # 숫자가 달라졌으므로 이전 숫자들의 사각형 리스트를 딕셔너리에 넣음
            num += 1
            rect_list = []  # 숫자가 달라져 현재 숫자의 사각형 리스트를 모으기 위해 초기화
        prev_rect = rects[i]  # 다음에 올 사각형과 비교하기 위해 저장
        rect_list.append(prev_rect)

    # 내부 인식되는 사각형 제거
    h_list = []

    for i in range(len(number_dict)):  # 0~9
        if len(number_dict[i]) == row:  # 열 개수대로 외부 사각형만 인식되는 경우
            pass
        else:  # 내부 사각형도 같이 인식되는 경우
            while (len(number_dict[i]) != row):
                for j in range(len(number_dict[i])):
                    h = number_dict[i][j][3]
                    h_list.append(h)  # 모두 인식된 사각형들의 높이를 h_list에 저장
                min_index = h_list.index(min(h_list))  # h_list 중 최소값을 찾아 인덱스 저장
                # (내부 사각형이므로 외부 사각형보단 높이가 작을테니까)
                number_dict[i].remove(number_dict[i][min_index])  # 내부 사각형 제거
                h_list = []

    # 묶어둔 숫자별 리스트들을 다시 해체하는 작업
    rects = []
    for i in range(len(number_dict)):
        for j in range(len(number_dict[i])):
            rects.append(number_dict[i][j])

print(len(rects))
#6  box를 그림으로 그려주면서 일정크기 이하의 box를 버려 노이즈를 제거하고 나머지는 이미지로 잘라내어 새로운 배열에 담아주기
# 작은 노이즈데이터 버림,사각형그리기,row개씩 리스트로 다시 묶어서 저장
for x, y, w, h in rects:
    tmp_y = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[0]
    tmp_x = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[1]
    if tmp_x and tmp_y > 10:
        count += 1
        cv2.rectangle(color, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 1)
        digit_arr.append(bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2])
        if count == row:
            digit_arr2.append(digit_arr)
            digit_arr = []
            count = 0

#7
#리스트에 저장된 이미지를 28x28의 크기로 리사이즈해서 순서대로 저장
for i in range(0,len(digit_arr2)) :
    for j in range(len(digit_arr2[i])) :
        count += 1
        if i == 1 :         #1일 경우 비율 유지를 위해 마스크를 만들어 그위에 얹어줌
            width = digit_arr2[i][j].shape[1]
            height = digit_arr2[i][j].shape[0]
            tmp = (height - width)/2
            mask = np.zeros((height,height))
            mask[0:height,int(tmp):int(tmp)+width] = digit_arr2[i][j]
            digit_arr2[i][j] = cv2.resize(mask,(28,28))
        else:
            digit_arr2[i][j] = cv2.resize(digit_arr2[i][j],(28,28))
        #if i == 9 : i = -1
        cv2.imwrite('./dataset/'+str(i)+'/'+ 'img_ys2_'+ str(i)+'_'+str(j)+'.png',digit_arr2[i][j])



k = cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show()
