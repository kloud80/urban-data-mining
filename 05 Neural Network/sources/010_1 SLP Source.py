# -*- coding: utf-8 -*-
"""
@author: Bigvalue_Bigdata Lab

작성자 : 구름
목적 : SLP 단 퍼셉트론의 학습방법에 대한 이해를 위해 작성
단층퍼셉트론은 AND연산, OR연산은 학습이 가능
XOR에 대한 학습 불가능
"""

# ds = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]  # AND
# ds=[[0,0,0],[0,1,1],[1,0,1],[1,1,1]] #OR
# ds=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]  #NAND
ds=[[0,0,0],[0,1,1],[1,0,1],[1,1,0]] #XOR

w0, w1, w2 = 0.3, 0.4, 0.1
x0 = -1
threshold = 0
learning_rate = 0.05

t = 0

while t < 100:
    print('-----------------------------------' + str(t + 1) + ' 번째 루프')
    bLearn = False

    for x1, x2, y in ds:
        # x1, x2, y = ds[2]
        if x0 * w0 + x1 * w1 + x2 * w2 >= threshold:
            y_hat = 1
        else:
            y_hat = 0

        print("x1:{0:.0f} x2:{1:.0f} y:{2:.0f} y^:{3:.0f}".format(x1, x2, y, y_hat))

        if y != y_hat:
            print('학습 필요 w0:{0:.2f}, w1:{1:.2f}, w2:{2:.2f}'.format(w0, w1, w2))
            print('학습량  w0:{0:.2f}, w1:{1:.2f}, w2:{2:.2f}'.format(
                x0 * (y - y_hat), x1 * (y - y_hat), x2 * (y - y_hat)))

            w0 = w0 + learning_rate * x0 * (y - y_hat)
            w1 = w1 + learning_rate * x1 * (y - y_hat)
            w2 = w2 + learning_rate * x2 * (y - y_hat)

            print('학습 결과 w0:{0:.2f} w1:{1:.2f} w2:{2:.2f}'.format(w0, w1, w2))
            bLearn = True
    t += 1
    if not bLearn: break
