import sys
sys.path.append('..')

from common.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print('courpus...',corpus)
print('word_to_id....',word_to_id)
print('id_to_word....',id_to_word)
print(word_to_id.keys())
vocab_size = len(word_to_id)
print('vocab_size...',vocab_size)
C = create_co_matrix(corpus, vocab_size)
print('C.....',C)
c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print(cos_similarity(c0,c1))