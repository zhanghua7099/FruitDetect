import cv2
import numpy as np

fruit = cv2.imread("fruit.jpg", 0)
background = cv2.imread("background.jpg", 0)
img = fruit - background
#目标分割
fruit1 = cv2.resize(fruit, (520, 340), interpolation=cv2.INTER_CUBIC)
res = cv2.resize(img, (520, 340), interpolation=cv2.INTER_CUBIC)
#图像分辨率压缩
blur = cv2.GaussianBlur(res,(5,5),0)
ret1, th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#二值化图像
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
#开运算去除轮廓外噪点
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#闭运算去除轮廓内噪点
image, contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#findContours()函数：输入（输入图像，层次类型，轮廓逼近方法），输出（修改后的图像，图像的轮廓，他们的层次）
#轮廓存储为坐标的列表，编号0为第一个轮廓，依次类推
#cv2.RETR_EXTERNAL为只想得到最外层轮廓，cv2.RETR_TREE为得到图像中轮廓的整体层次结构
'''
通过轮廓计算相应数值，将信息存储于列表当中
'''
information = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    # 面积
    perimeter = cv2.arcLength(cnt, True)
    # 周长
    e = 4 * 3.14 * area / perimeter / perimeter
    # 圆形度
    M = cv2.moments(cnt)
    # 图像的矩
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # 图像重心
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    # 图像边界矩形的宽高比
    information.append([area, e, aspect_ratio, cx, cy])

'''
输出圆形度，长宽比，质心位置，并在图像中标记其位置
'''
font=cv2.FONT_HERSHEY_SIMPLEX
for inf in information:
    str1 = '('+str(inf[3])+',' + str(inf[4])+')'
    if inf[1] >= 0.75:
        print('苹果' + ',' + '质心位置：' + str1 + ',' + '圆形度:' + str(inf[1]) + ',' + '长宽比：' + str(inf[2]))
        cv2.putText(fruit1, 'Apple', (inf[3], inf[4]), font, 1, (255, 0, 0), 2)
    else:
        print('香蕉' + ',' + '质心位置：' + str1 + ',' + '圆形度:' + str(inf[1]) + ',' + '长宽比：' + str(inf[2]))
        cv2.putText(fruit1, 'Banana', (inf[3], inf[4]), font, 1, (255, 0, 0), 2)
'''
输出苹果与圆所占像素个数
'''
AreaApple = 0.0
AreaBanana = 0.0
for inf1 in information:
    if inf1[1] >= 0.75:
        AreaApple += inf1[0]
    else:
        AreaBanana += inf1[0]
print('苹果所占像素：'+str(AreaApple))
print('香蕉所占像素：'+str(AreaBanana))
cv2.imshow("canny", fruit1)
cv2.waitKey()
cv2.destroyAllWindows()
