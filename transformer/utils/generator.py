import random
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from translate import Translator


def load_embedding_model():
    import gensim.downloader as api
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin


# pprint.pprint(wv_from_bin.most_similar(positive=['buy']))

def calculate_similarity(text1, text2, word_vectors):
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()

    vectors1 = [word_vectors[word] for word in tokens1 if word in word_vectors]
    vectors2 = [word_vectors[word] for word in tokens2 if word in word_vectors]

    if len(vectors1) == 0 or len(vectors2) == 0:
        return 0.0  # 如果任意一个向量列表为空，则相似度为0

    # 计算文本的平均词向量
    vector1 = np.mean(vectors1, axis=0)
    vector2 = np.mean(vectors2, axis=0)

    # print(f"vector1 shape: {vector1.shape}, vector2 shape: {vector2.shape}")

    vector1 = np.array(vector1).reshape(1, -1)
    vector2 = np.array(vector2).reshape(1, -1)
    # print(f"Reshaped vector1 shape: {vector1.shape}, Reshaped vector2 shape: {vector2.shape}")

    similarity = cosine_similarity(vector1, vector2)[0][0]

    return similarity


def generate(button_text, word_vectors, threshold=0.4):
    base_terms = ["accept", "confirm", "submit", "ok", "yes", "next", "okay"]
    button_text = button_text.lower()
    if not word_vectors:
        # print("Word vectors not provided.")
        return

    similarities = []
    for term in base_terms:
        similarity = calculate_similarity(term, button_text, word_vectors)
        similarities.append(similarity)

    similarity_score = max(similarities)
    if similarity_score > threshold:
        return 1
    return 0


def embedding(text, count, children, word_vectors):
    translator = Translator(from_lang='chinese', to_lang='english')
    english_pattern = re.compile(r'^[A-Za-z]+$')
    chinese_pattern = re.compile(r'^[\u4e00-\u9fa5]+$')
    if text.strip() and chinese_pattern.match(text):
        result = translator.translate(text)
        # print("result" + result)
        text_similar = generate(result, word_vectors)
    elif english_pattern.match(text):
        # print("english")
        text_similar = generate(text, word_vectors)
    else:
        text_similar = random.uniform(0, 0.5)
    combined_vector = np.concatenate((np.array([text_similar]), np.array([count]), np.array(children)))
    return combined_vector



# if __name__ == '__main__':
#     wv_from_bin = load_embedding_model()
#     translator = Translator(from_lang='chinese', to_lang='english')
#     # result = translator.translate("")
#     result1 = embedding("确认", 1, [1, 2, 3, 4, 5, 6], wv_from_bin)
#     result1 = embedding("okay", 1, [1, 2, 3, 4, 5, 6], wv_from_bin)
#     result = embedding("dimeshift", 1, [1,2,3,4,5,6], wv_from_bin)
#     print(result)