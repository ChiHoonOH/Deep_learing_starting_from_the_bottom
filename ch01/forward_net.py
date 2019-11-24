import numpy as np

class Sigmoid:
    def __init__(self):
        self.params = []
    def foward(self, x):
        return 1/(1 + np.exp(-x))

class Affine: # 아핀이라 읽는다. matmul이 가능한 것 자체가 완전연결계층임(fully connected layers)을 뜻한다.
    def __init__(self, W, b):
        self.params = [W, b]

    def foward(self, x):
        W, b = self.params
        out = np.matmul(x,W) + b
        return out

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화

        W1 = np.random.randn(I,H) # 표준정규분포 난수 생성, 근데 왜 표준정규분포로 생성 했는지는 모르겠음.
        b1 = np.random.randn(H)
        W2 = np.random.randn(H,O)
        b2 = np.random.randn(O)

        # 계층 생성

        self.layers=[
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2,b2)
        ]

        self.params = []

        for layer in self.layers:
            self.params+=layer.params # sigmoid = []  affine #1차원 리스트가 될 듯하다.
        print(self.params)
        print(len(self.params))

    def predict(self,x):
        for layer in self.layers:
            layer.forward(x)
        return x

TwoLayerNet(2,4,2)
