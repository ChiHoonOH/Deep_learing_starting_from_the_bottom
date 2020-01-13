# chapter 3 
## p120
import sys
# sys.path[0]= ('..')
import numpy as np
from common.layers import MatMul

# c = np.array([[1,0,0,0,0,0,0]])
# W = np.random.randn(7,3)
# layer = MatMul(W)
# h = layer.foward(c)
# print(W)
# print(h)

# conver_one_hot 입력데이터 확인

from common.util import preprocess, create_contexts_target, convert_one_hot
text = 'You say goodbye and I say hello'
corpus, word_id, id_to_word = preprocess(text)
print('corpus...',corpus)
print(corpus.ndim)
print(corpus.shape)
print('word_id...',word_id)
print('id_to_word....',id_to_word)
one_hot = convert_one_hot(corpus,len(word_id))
print(one_hot)



