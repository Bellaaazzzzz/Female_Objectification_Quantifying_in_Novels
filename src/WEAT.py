from inspect import getclosurevars
import numpy as np
from typing import Iterable, Dict, List, Tuple
from gensim.models import Word2Vec
from scipy.stats import permutation_test
import itertools
from random import sample


class Embeddings:
    """
    This class represents a container that holds a collection of words
    and their corresponding word embeddings.
    """

    def __init__(self, words: Iterable[str], vectors: np.ndarray):
        """
        Initializes an Embeddings object directly from a list of words
        and their embeddings.

        :param words: A list of words
        :param vectors: A 2D array of shape (len(words), embedding_size)
            where for each i, vectors[i] is the embedding for words[i]
        """
        self.words = list(words)
        self.indices = {w: i for i, w in enumerate(words)}
        self.vectors = vectors

    def __len__(self):
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def __getitem__(self, words: Iterable[str]) -> np.ndarray:
        """
        Retrieves embeddings for a list of words.

        :param words: A list of words
        :return: A 2D array of shape (len(words), embedding_size) where
            for each i, the ith row is the embedding for words[i]
        """
        return np.array(self.vectors[list(map(self.indices.get, words))])

    @classmethod
    def from_file(cls, filename: str) -> "Embeddings":
        """
        Initializes an Embeddings object from a .txt file containing
        word embeddings in GloVe format.

        :param filename: The name of the file containing the embeddings
        :return: An Embeddings object containing the loaded embeddings
        """
        words = np.genfromtxt(filename,dtype='str',comments=None)[:,0]
        vectors = np.genfromtxt(filename,comments=None)[:,1:]
        return cls(words, vectors)

def cosine_sim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the cosine similarity between two matrices of row vectors.

    :param x: A 2D array of shape (m, embedding_size)
    :param y: A 2D array of shape (n, embedding_size)
    :return: An array of shape (m, n), where the entry in row i and
        column j is the cosine similarity between x[i] and y[j]
    """
    res = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            res[i][j] = np.dot(x[i], y[j])/(np.linalg.norm(x[i])*np.linalg.norm(y[j]))
    return res

def filter_words(word_list, word_embedding):
    res = []
    for w in word_list:
        if w in word_embedding:
            res.append(w)
    return res

def attribute(word_list, word_embedding):
    words_lower = [w.lower() for w in word_list]
    attribute_vectors = word_embedding[words_lower]
    return attribute_vectors

#Scipy permutation test
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def scipy_test(f_attribute, m_attribute, target, n_resamples):
    res = permutation_test((cosine_sim(f_attribute,target), cosine_sim(m_attribute,target)),
                       statistic,
                       n_resamples=n_resamples,
                       vectorized=True,
                       axis = -1,
                       alternative="greater")
    effect_size = np.mean(res.statistic)/np.std(res.statistic)
    p_value = np.sum(np.mean(res.null_distribution, axis=-1) > np.mean(res.statistic)) / n_resamples
    return effect_size, p_value

def weat_test(model_dir, novel_name, names, target_words, baseline_model, n_resamples):
    f_typical = ["woman","female","girl","she","her","herself","wife","mother","aunt","sister","miss","lady"]
    m_typical = ["man","male","boy","he","him","himself","husband","father","uncle","brother","mr","sir"]

    embeddings = Word2Vec.load(model_dir+novel_name+".model").wv

    f_name = filter_words(filter_words(f_typical, embeddings) + filter_words(names[novel_name]['Female'], embeddings), baseline_model)
    m_name = filter_words(filter_words(m_typical, embeddings) + filter_words(names[novel_name]['Male'], embeddings), baseline_model)
    if len(f_name) > len(m_name):
        f_name = f_name[:len(m_name)]
    else:
        m_name = m_name[:len(f_name)]
    target_words_filtered = filter_words(filter_words(target_words, embeddings), baseline_model)

    if len(f_name) < 2 or len(m_name) < 2 or len(target_words_filtered) < 2:
        effect_size = p_value = effect_size_base = p_value_base = np.nan
    else:
        # Novel specific
        f_attribute = attribute(f_name, embeddings)
        m_attribute = attribute(m_name, embeddings)
        target = attribute(target_words_filtered, embeddings)
        effect_size, p_value = scipy_test(f_attribute, m_attribute, target, n_resamples)

        # Baseline 
        f_attribute_base = attribute(f_name, baseline_model)
        m_attribute_base = attribute(m_name, baseline_model)
        target_base = attribute(target_words_filtered, baseline_model)
        effect_size_base, p_value_base = scipy_test(f_attribute_base, m_attribute_base, target_base, n_resamples)
    return effect_size, p_value, effect_size_base, p_value_base

# Step by step calculation
# def weat_diff(f_attribute, m_attribute, target):
#     s = np.mean(cosine_sim(f_attribute,target), axis=-1) - np.mean(cosine_sim(m_attribute,target), axis=-1)
#     return s

# def weat_effect_size(f_attribute, m_attribute, target):
#     s = np.mean(cosine_sim(f_attribute,target), axis=-1) - np.mean(cosine_sim(m_attribute,target), axis=-1)
#     return np.mean(s)/np.std(s)

# def step_test(model_dir, novel_name, names, target_words, n_resamples):
#     f_typical = ["woman","female","girl","she","her","herself","wife","mother","miss","lady"]
#     m_typical = ["man","male","boy","he","him","himself","husband","father","mr","sir"]
#     embeddings = Word2Vec.load(model_dir+novel_name+".model").wv
#     f_name = filter_words(f_typical, embeddings) + names[novel_name]['Female']
#     m_name = filter_words(m_typical, embeddings) + names[novel_name]['Male']
#     if len(f_name) > len(m_name):
#         f_name = f_name[:len(m_name)]
#     else:
#         m_name = m_name[:len(f_name)]
#     target_words_filtered = filter_words(target_words, embeddings)
#     f_attribute = attribute(f_name, embeddings)
#     m_attribute = attribute(m_name, embeddings)
#     target = attribute(target_words_filtered, embeddings)

#     observed = weat_diff(f_attribute, m_attribute, target)
#     effect_size = weat_effect_size(f_attribute, m_attribute, target)

#     fm = np.concatenate((f_attribute, m_attribute))
#     n = len(fm)
#     size = len(f_attribute)
#     samples = []
#     for idx in sample(list(itertools.combinations(range(n), r=size)), n_resamples):
#         Fi = fm[list(idx)]
#         Mi = fm[list(set(range(n)).difference(idx))]
#         si = weat_diff(Fi, Mi, target)
#         samples.append(si)
#     p_value = np.sum(samples > observed) / len(samples)

#     print(f'Effect size: {effect_size}, p-value: {p_value}')

