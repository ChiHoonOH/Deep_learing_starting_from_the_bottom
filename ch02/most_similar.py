import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, most_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print(word_to_id, id_to_word)
most_similarity('you', word_to_id, id_to_word, C, top=5)