import torch
from collections import Counter
import re
from tqdm.notebook import tqdm
import spacy
from spacy import displacy
from spacy import tokenizer
from spacy.tokens.doc import Doc
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 1500000
from nltk.tokenize import sent_tokenize
import allennlp_models.tagging
from allennlp.predictors.predictor import Predictor
predictor_cor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz", cuda_device=torch.cuda.current_device())



def identify_name_gender(filename, novels_dir):
    '''
    Extract character names and identify gender 
    ----
    filename: List of female names
    novels_dir: List of male names
    --
    return: mix_gen_surname: List of mix-gendered surnames
            f_names_cleaned: List of female names
            m_names_cleaned: List of male names
    '''
    novel_text, novel_model = read_process(filename, novels_dir)
    names = get_names(novel_model, 3)
    mix_gen_surname, single_gen_name = filter_surnames(names, novel_text)
    coref_cluster = allen_coref(filename, novels_dir)
    f_names, m_names, non_gender_names = group_gender(coref_cluster, single_gen_name)
    f_names_cleaned, m_names_cleaned = extract_name(f_names, m_names, mix_gen_surname)
    return mix_gen_surname, f_names_cleaned, m_names_cleaned

 

def read_process(doc_name, novels_dir):
    '''
    Read in a novel text and run it through the nlp pipeline to generate spacy doc
    ----
    doc_name: Novel name
    novels_dir: File directory of the novels
    --
    return: NLP model spacy.tokens.doc.Doc
    '''
    doc_dir = novels_dir + doc_name
    with open(doc_dir) as f:
        text = f.read().replace('\n', ' ')
    if len(text) > 1500000:
        doc1 = nlp(text[:1500000])
        doc2 = nlp(text[1500000:])
        return text, [doc1, doc2]        
    else:
        doc = nlp(text)
        return text, [doc]

def get_names(doc_list, n):
    '''
    NER tagging and select frequent name entities
    ----
    doc_list: List of NLP model spacy.tokens.doc.Doc or list of two NLP model spacy.tokens.doc.Doc
    n: Threshold of name frequency, only names which appear equal to or more than n times 
        are included in results
    --
    return: Dictionary of identified and filtered names by frequency {[name]:frequency}
    '''
    names = []
    for doc in doc_list:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                names.append(ent.text)
    name_list = dict(Counter(names))
    name_freq = dict(sorted(name_list.items(), key=lambda item: item[1],reverse=True))
    filtered_names = {k:v for k,v in name_freq.items() if v >= n}
    return filtered_names

def filter_surnames(names, text):
    '''
    Seperate out mix-gendered surnames and clean name entities to exclude honorifics
    ----
    names: Dictionary of identified and filtered names by frequency {[name]:frequency}
    models: NLP model spacy.tokens.doc.Doc 
    --
    return: mix_gen_surname: List of surnames 
            single_gen_name: List of single-gendered name entities
    '''
    mix_gen_surname = []
    single_gen_name = []
    for name in list(names.keys()):
        if ('Mr. '+name in text and 'Mrs. '+name in text)\
            or ('Mr. '+name in text and 'Miss '+name in text)\
            or ('Mr. '+name in text and 'Madam '+name in text)\
            or ('Sir '+name in text and 'Mrs. '+name in text)\
            or ('Sir '+name in text and 'Miss '+name in text)\
            or ('Sir '+name in text and 'Madam '+name in text)\
            or ('Monsieur '+name in text and 'Madame '+name in text)\
            or ('Monsieur '+name in text and 'Mademoiselle '+name in text)\
            or ('M. '+name in text and 'Mme '+name in text)\
            or ('M. '+name in text and 'Mlle '+name in text):
            mix_gen_surname.append(name)
        else: single_gen_name.append(name)
    return mix_gen_surname, single_gen_name

def chunkstring(sentences, length):
    '''
    Put sentences in chunks 
    ----
    sentences: Tokenized sentences
    length: Number of sentences in each chunk
    --
    return: Chunck of text in pre-defined length
    '''
    return (' '.join(sentences[0+i:length+i]) for i in range(0, len(sentences), length))

def allen_coref(novel_name, novels_dir):
    '''
    Run coreference resolutions and put resolution clusters together
    ----
    doc_name: Filename of input novel (without file extension)
    --
    return: Coreference resolution clusters
    '''
    with open(novels_dir+novel_name) as f:
        text_whole = f.read().replace('\n', ' ')
    sentences = sent_tokenize(text_whole)
    text_split = list(chunkstring(sentences, 10))
    clus_a = []
    for i in range(len(text_split)):
        pred = predictor_cor.predict(text_split[i])
        clusters = pred['clusters']
        document = pred['document']
        for cluster in clusters:
            chain = []
            for obj in cluster:
                item = ' '.join(document[obj[0]:obj[1]+1])
                chain.append(item)
            clus_a.append(chain)
        torch.cuda.empty_cache()
    return clus_a

def group_gender(coref_cluster, name_list):
    '''
    Determine gender from coreference resolution clusters
    ----
    coref_cluster: Coreference resolution clusters
    name_list：List of names to be put in gender groups
    --
    return: Gender groups of names: female, male, non-gender
    '''
    f_names = []
    m_names = []
    non_gender = []

    for name in name_list:
        if name == "God" or name == "Jesus":
            non_gender.append(name)
        else:
            f_chain = 0
            m_chain = 0
            for chain in coref_cluster:
                f_pronoun = 0
                m_pronoun = 0
                if name in ' '.join(chain):
                    for i in chain:
                        words = i.lower().split(' ')
                        for word in words:
                            if word.lower() in ["she","her","herself"]:
                                f_pronoun += 1
                            if word.lower() in ["he", "him", "himself"]:
                                m_pronoun += 1
                    if f_pronoun > m_pronoun:
                        f_chain += 1
                    elif m_pronoun > f_pronoun:
                        m_chain += 1
        if f_chain > m_chain:
            f_names.append(name)
        elif m_chain > f_chain:
            m_names.append(name)
        else:
            non_gender.append(name)
    
    return f_names, m_names, non_gender

def extract_name(f_names, m_names, mix_gen_surname):
    '''
    Clean grouped name entities
    ----
    f_name: List of female names
    m_names: List of male names
    mix_gen_surname: List of mix-gender surnames
    --
    return: Cleaned female and male names
    '''
    female_cleaned = []
    male_cleaned = []

    for name in f_names:
        if ' ' not in name:
            female_cleaned.append(name)
        if ' ' in name:
            if name.split(' ')[0] != 'Miss' and name.split(' ')[0] != 'Mademoiselle':
                female_cleaned.append(name.split(' ')[0])
                female_cleaned.append(name.split(' ')[-1])
            if name.split(' ')[0] == 'Miss' or name.split(' ')[0] == 'Mademoiselle':
                female_cleaned.append(name.split(' ')[-1])

    for name in m_names:
        if ' ' not in name:
            male_cleaned.append(name)
        if ' ' in name:
            if name.split(' ')[0] != 'Monsieur' and name.split(' ')[0] != 'M.':
                male_cleaned.append(name.split(' ')[0])
                male_cleaned.append(name.split(' ')[-1])
            if name.split(' ')[0] == 'Monsieur' or name.split(' ')[0] == 'M.':
                male_cleaned.append(name.split(' ')[-1])

    female_cleaned = list(dict.fromkeys(female_cleaned))
    male_cleaned = list(dict.fromkeys(male_cleaned))

    for name in female_cleaned:
        if name in male_cleaned or name in mix_gen_surname:
            female_cleaned.remove(name)
    for name in male_cleaned:
        if name in female_cleaned or name in mix_gen_surname:
            male_cleaned.remove(name)
    return female_cleaned, male_cleaned


