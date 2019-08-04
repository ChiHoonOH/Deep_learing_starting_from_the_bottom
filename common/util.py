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

# text = 'You say goodbye and I say hello.'
# corpus, word_to_id , id_to_word = preprocess(text) 
'''
print("corpus:",corpus)
print(word_to_id)
print(id_to_word)
'''
# create co matrix는 corpus를 이용해서 단어 행렬을 만든다.

'''

크기는 vocab_size vocab_size는 코퍼스의 크기를 의미한다. 

# 1. vocabsize를 이용하여 영행렬을 만든다.

# 2. 1을 넣을지점은 해당 번째 단어 주변에 다른 해당 단어가 있냐 없냐라는 것을 묻는 것이다. 
# 주변을 정의 할 주변 단어 갯수를 지정해줘야 할 것이다.
해당 단어 corpus 번호의 +1, -1을 하여 그 값이 존재하는 갯수를 적어준다. 근데 단어는 하나 밖에 없지 않나?
'''
def create_co_matrix(corpus, vocab_size, window_size=1):    
    corpus_size = len(corpus)    
    # co_matrix에서 행과 열의 정체?
    # corpus는 중복제거를 하지 않은 날 것
    # vocab_size는 중복 제거 한 것    
    # print(corpus_size, vocab_size)
    '''
    vocab_size를 외부에서 받아 올 필요가 있을까?
    그럴 필요는 없을 것 같은데... 
    word_to_id or id_to_word를 사용한다면 corpus 의 unique 연산보다 효율적일 것이다.
    '''
    co_matrix = np.zeros((vocab_size,vocab_size), dtype=np.int32)    
    for idx, word_id in enumerate(corpus):
        left_idx = idx - window_size
        right_idx = idx + window_size
        # print(left_idx, right_idx)
        # idx는 행 -> 각각 단어들의 idx(단어들은 유일값이다.)
        # corpus[idx]는 열 -> 특성 
        # window_size를 bound point 로 지정 하는게 낫지 않을까?
        if left_idx>=0:
            left_word_id = corpus[left_idx]
            co_matrix[word_id, left_word_id] += 1
            ''' 
            # word_idx 자리에 idx를 쓰면 에러가 나는 이유는 corpus is not unique이고 corpus element 값이 같으면 그건 동일 숫자이기 때문에 element를 지칭하는 
            # word_id를 행 식별자로 내세운다.
            '''
        if right_idx<corpus_size:
            right_word_id = corpus[right_idx]
            co_matrix[word_id, right_word_id] += 1
    return co_matrix

# x가 행렬이냐? 벡터냐? x**2 는 원소 각각을 제곱한다. np.sum을 통해서 합을 구하고 np.sqrt로 제곱근을 구해주는 것이다.
def cos_similarity(x,y,eps=1e-8):
    # 내적은 이용한 cos, 유사할 수록 1, 유사하지 않을 수록 0
    return np.dot(x,y)/((np.sqrt(np.sum(x**2))+eps) *(np.sqrt(np.sum(y**2))+eps))

def most_similarity(standard_word, word_to_id, id_to_word, word_matrix, top=5):
    '''
    내 예상 : 단어 간의 모든 유사도를 구한 후에 정렬
    '''
    if standard_word not in word_to_id:
        print('찾을 수 없는 단어입니다.')
        return 
        '''    
        a = {'바보':1,'호구':2}
        print([element for element in a])
        key 값만 나온다.
        '''
        
    query_id = word_to_id[standard_word]
    query_vec = word_matrix[query_id]
    # 모든 단어와 유사도 계산
    # Ordereddict를 사용 햇을 것 같다. 
    # 근데 ordereddict는 입력순서를 유지 한다. 값 기준으로 유지되는 collection은 아니다.
    similarity_dict ={}
    for i in id_to_word:
        similarity_dict[id_to_word[i]] = cos_similarity(query_vec,word_matrix[i])
        
    vocab_size = len(word_to_id)    
    similarity = np.zeros(vocab_size)    
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    count=0        
    for i in (-1 * similarity).argsort(): # argsort() 각 원소들의 순위 반환
        if id_to_word[i] == standard_word:
            continue
        print('%s:%s' % (id_to_word[i], similarity[i]))
        
        count+=1
        
        if count >= top:
            return

def ppmi(C, verbose=False, eps=1e-8):
    # 동시 발생 갯수 * N / x 발생갯수 , y 발생 갯수
    # 각각 단어당 pmi 값을 구한다.
    # N 총 발생 횟수 
    N = np.sum(C)
    # x 발생횟수, y 발생횟수
    M = np.sum(C, axis=0) # (vocab_size,1)
    total = C.shape[0] * C.shape[1]
    
    # plate = np.zeros(C.shape[0],C.shape[1])
    ppmi_plate = np.zeros_like(C, dtype=np.float32) # C 와 같은 크기를 가지는 plate생성
    '''
    # i = 0; i<C.shape[0]; i++ 
    # j = 0; j<C.shape[0]; j++ 
    i의 빈도 M[i]
    j의 빈도 M[j]
    동시 발생 빈도 C[i,j]
    '''
    cnt = 0 
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):            
            # pmi = np.log2(C[i,j] * N/((M[i]+eps)*(M[j]+eps)))
            '''
            한 번도 출연하지 않았던 단어가 동시 발생 행렬에 포함 될리가 없기 때문에, 분모가 0이 되는 경우의 수는 없다.
            하지만 분자가 0이 되는 경우의 수는 존재 하고, 이것이 log가 씌워지면 -inf  값을 출력하기 때문에 eps를 더해주는 것이다.
            0이 되면 안되는 경우의 수가 있는지 없는지를 먼저 체크하고 eps를 더해준다. 막연하게 더해줘버리면 이런 오류가 발생한다.
            '''
            pmi = np.log2(C[i,j] * N/(M[i]*M[j])+eps)
            ppmi_plate[i,j] = max(0,pmi)
            if verbose:
                cnt +=1
                if cnt % (total//100) == 0: # 요 if 문의 의미가 이해가 가진 않는다. 
                    print('%.1f%% 완료' % (100*cnt/total))

    return ppmi_plate
    
        
        
        
