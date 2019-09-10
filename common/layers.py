import numpy as np
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)] # W와 같은 크기의 0행렬 생성
        self.x = None
    def foward(self,x):
        W, = self.params
        out = np.matmul(x,W)
        self.x = x
        return out
    def backward(self,dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T,dout)
        self.grads[0][...] = dW
        return dx
