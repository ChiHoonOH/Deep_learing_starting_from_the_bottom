import sys
sys.path.append('..')

from common.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
print(word_to_id)
print(id_to_word)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

