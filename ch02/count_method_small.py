# 그 동안 신뢰는 충분히 쌓아 왔다 생각, ai 과제를 받을 수 있도록 해야함.
# svd에서 usv 에서 u,v는 직교 행렬
# s 는 대각행렬 -> 중요도를 담고 있다.
# 이유는 기억이 나지 않는다.
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size, window_size=1)
W = ppmi(C)
U,S,V = np.linalg.svd(W)

# 차원 감소를 시키는데 :2를 사용했다. 중요도가 :2 순서로 내림차순이라는
# 보장이 있어야 적절한 차원감소기법이라고 생각한다. 
# 그리고 이 원소가 어느정도 비중을 차지하는지에 대한 값도 알아야한다.
# 주성분이 떠 오르긴 하나 확실하지는 않다.

print('W[0]....',W[0])
print('U[0]...',U[0])
print('S....',S)
# 놀랍게도 S가 내림차순이다.

for word, word_id in word_to_id.items():
    plt.annotate(word,(U[word_id,0],U[word_id,1]))

plt.scatter(U[:,0],U[:,1], alpha=0.5)
plt.show()
