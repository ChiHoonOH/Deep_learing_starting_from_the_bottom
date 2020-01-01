import sys

# sys.path.append()
# sys.path.insert(0, '../')
sys.path.insert(0,'c:\\Users\\rngdr\\OneDrive\\바탕 화면\\밑바닥부터 시작하는 딥러닝2')

import numpy as np
from common.layers import MatMul

# 입력 값
c0 = np.array([[1,0,0,0,0,0,0]])
c1 = np.array([[0,0,1,0,0,0,0]])


# 가중치 초기화
W_in = np.random.randn(7,3)# 입력에도 가중치 
W_out = np.random.randn(3,7)# 출력에도 가중치

'''
# 입력에도 가중치 
# 출력에도 가중치?
왜?
'''
# 계층 생성(matmul 선언)
input_layer0 = MatMul(W_in)
input_layer1 = MatMul(W_in) # 왜 층이 두개인가?

'''
# 왜 층이 두개인가? -> 일단 반항중
'''
output_layer = MatMul(W_out)

# 순전파
h0 = input_layer0.forward(c0)
h1 = input_layer0.forward(c1)
h = (h0+h1)/2
s = output_layer.forward(h)
print(s)


# 출력 
## 이 경우엔 동일 가중치

