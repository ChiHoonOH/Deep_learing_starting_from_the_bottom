import numpy as np


def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([[word_to_id[w]] for w in words])
    return corpus, word_to_id, id_to_word

text = 'You say goodbye and I say hello.'
corpus, word_to_id , id_to_word = preprocess(text) 

print("corpus:",corpus)
print(word_to_id)
print(id_to_word)

# x가 행렬이냐? 벡터냐? x**2 는 원소 각각을 제곱한다. np.sum을 통해서 합을 구하고 np.sqrt로 제곱근을 구해주는 것이다.
def cos_similarity(x,y,eps=1e-8):
    # 내적은 이용한 cos, 유사할 수록 1, 유사하지 않을 수록 0
     return np.dot(x,y)/((np.sqrt(np.sum(x**2))+eps) *(np.sqrt(np.sum(y**2))+eps))
    