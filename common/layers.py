import numpy as np
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)] # W와 같은 크기의 0행렬 생성
        self.x = None
    def forward(self,x):
        W, = self.params
        out = np.matmul(x,W)
        self.x = x
        return out
    def backward(self,dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T,dout)
        # a[...] = b 의 경우 a의 주소는 그대로고 b의 값이 복사 된다.
        self.grads[0][...] = dW
        return dx
        '''의문점
        # parmas를 W 하나만 할거면 뭐하러 list를 쓴건가
        self.grads를 따로 저장하는 이유가 뭔가? 그리고 이 역시 리스트를 쓰는 이유는 뭔가?

        '''
        
