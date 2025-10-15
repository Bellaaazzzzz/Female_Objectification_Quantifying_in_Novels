import nltk
nltk.download('punkt')
import re
from gensim.models import Word2Vec
import gensim.downloader as api

glove_model = api.load('glove-wiki-gigaword-50')

def cleanWord(word):
        clean_word = re.sub(r'[^a-zA-Z]', '', word)
        return clean_word.lower() # Convert to lowercase and return

def train_embeddings(doc_name:str, novels_dir:str, model_dir:str):
    with open(novels_dir+doc_name+'.txt', 'r', encoding='utf-8-sig') as f:
        text = f.read()
    sentences = nltk.sent_tokenize(text)

    # Tokenize each sentence into words
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    # Convert all words to lowercase and remove any leading/trailing white spaces
    novel = [[word.lower().strip() for word in sentence] for sentence in tokenized_sentences]

    # define a regular expression to match non-alphabetic characters
    pattern = re.compile('[^a-z]+')

    # remove non-alphabetic characters and convert to lowercase
    clean_novel = []
    for sentence in novel:
        clean_sentence = []
        for word in sentence:
            if bool(re.search('[a-zA-Z]', word)) and bool(re.search('[^a-z]+', word)): # if word contains alphabets & non-alphabetic characters
                clean_word = pattern.sub(' ', word).split() # substitute non-alphabetic characters with space
                for word in clean_word:
                    clean_sentence.append(word)
            else:
                clean_sentence.append(word)
        clean_novel.append(clean_sentence)

    new_model = Word2Vec(vector_size=50, window=20, min_count=1, workers=4)
    new_model.build_vocab(clean_novel)

    for key in new_model.wv.index_to_key:
        if key in glove_model:
            new_model.wv[key] = glove_model[key]
        elif cleanWord(key) in glove_model:
            new_model.wv[cleanWord(key)] = glove_model[cleanWord(key)]

    new_model.train(novel,total_examples=new_model.corpus_count, epochs=new_model.epochs)

    new_model.save(model_dir+doc_name+".model")