import pandas as pd
import os
import re
# import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import spacy
from spacy.symbols import ORTH
from multiprocessing import Pool
import time

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import numpy as np

Dir_github = '/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github'
project = 'FactorAnalysis_fMRI'

def lemmatize_spacy(i,row,nlp):
    
    doc = nlp(row['Word'])
    lemmas = [(token.text, token.lemma_, token.pos_) for token in doc]
    df_tmp = pd.DataFrame(lemmas,columns=['Word','Lemma_spacy','PoS_tag_spacy'])
    df_tmp['idx'] = i
    
    return df_tmp
    
def lemmatize_US(i,row,nlp,lemmatizer):

    PoS_nltk_map = {'Adjective':wordnet.ADJ,'Adverb':wordnet.ADV,'Verb':wordnet.VERB,'Noun':wordnet.NOUN}
    word = row['Word']
    tag_raw = row['Dom_PoS_US']
    
    # spacy
    doc = nlp(word)
    lemma_spacy = [(token.text, token.lemma_, token.pos_) for token in doc]
    
    # NLTK
    if tag_raw in PoS_nltk_map.keys():
        tag = PoS_nltk_map.get(tag_raw)
    else:
        tag = wordnet.NOUN
    lemma_nltk = lemmatizer.lemmatize(word, tag)

    df_tmp = pd.DataFrame([[word,tag_raw,lemma_nltk,lemma_spacy[0][1],lemma_spacy[0][2]]],
                          columns=['Word','Dom_PoS_US','Lemma_US_nltk','Lemma_US_spacy','PoS_US_spacy'])
    df_tmp['idx'] = i
    # df_tmp['Lemma_US_nltk'] = lemma_nltk
    # df_tmp.rename(columns = {'Lemma_spacy':'Lemma_US_spacy','PoS_tag_spacy':'PoS_US_spacy'})
    return df_tmp


def lemmatize_step1(df_SCOPE):
    # Step1: Get the most frequently used lemmas from Subtlex-UK
    onset = time.time()
    df_UK = pd.read_excel(os.path.join(Dir_github,'data','SUBTLEX-UK.xlsx'))
    
    df_SCOPE = df_SCOPE.set_index('Word').join(df_UK.set_index('Spelling')[['DomPoSLemma']], how='left').reset_index()
    df_SCOPE.rename(columns = {'DomPoSLemma':'Lemma_UK'},inplace=True)
    print(f"Step1:\n---> {df_SCOPE['Lemma_UK'].dropna().shape[0]}/{df_SCOPE.shape[0]} ({round(df_SCOPE['Lemma_UK'].dropna().shape[0]/df_SCOPE.shape[0]*100,2)}%) words have available lemmas.")
    offset = time.time()
    print(f"---> {round(offset-onset,2)}s")
    return(df_SCOPE)

def lemmatize_step2(df_SCOPE,nlp,lemmatizer):
    onset = time.time()

    
    df_US = pd.read_excel(os.path.join(Dir_github,'data','SUBTLEX-US frequency list with PoS and Zipf information.xlsx'))
    df_SCOPE = df_SCOPE.set_index('Word').join(df_US.set_index('Word')[['Dom_PoS_SUBTLEX','All_PoS_SUBTLEX']], how='left').reset_index()
    df_SCOPE.rename(columns = {'Dom_PoS_SUBTLEX':'Dom_PoS_US','All_PoS_SUBTLEX':'ALL_PoS_US'},inplace=True)

    
    PoS_nltk_map = {'Adjective':wordnet.ADJ,'Adverb':wordnet.ADV,'Verb':wordnet.VERB,'Noun':wordnet.NOUN}
    
    df_step2 = df_SCOPE.loc[df_SCOPE['Lemma_UK'].isna()].loc[~df_SCOPE['Dom_PoS_US'].isna()]

    with Pool(processes=48) as pool:  
        # results = pool.map(do_process, [row for _, row in df_spacy.iterrows()])
        results = pool.starmap(lemmatize_US, [(i,row,nlp,lemmatizer) for i, row in df_step2.iterrows()])
    offset = time.time()
    df_step2 = pd.concat(results, ignore_index=True)

    # I also used Spacy to lemmatize those words without the PoS tags input. It seems that NLTK worked better for this surpervised lemmatization task.
    df_SCOPE = df_SCOPE.set_index('Word').join(df_step2.set_index('Word')[['Lemma_US_nltk']], how='left').reset_index()
    
    df_SCOPE['Lemma_UKUS'] = df_SCOPE['Lemma_UK']
    df_SCOPE['PoS_UKUS'] = df_SCOPE['DPoS_VanH']
    df_SCOPE.loc[~df_SCOPE['Lemma_US_nltk'].isna(),'Lemma_UKUS'] = df_SCOPE.loc[~df_SCOPE['Lemma_US_nltk'].isna(),'Lemma_US_nltk'] 
    df_SCOPE.loc[~df_SCOPE['Dom_PoS_US'].isna(),'PoS_UKUS'] = df_SCOPE.loc[~df_SCOPE['Dom_PoS_US'].isna(),'Dom_PoS_US'] 
    print(f"Step2:\n---> {df_SCOPE['Lemma_UKUS'].dropna().shape[0]}/{df_SCOPE.shape[0]} ({round(df_SCOPE['Lemma_UKUS'].dropna().shape[0]/df_SCOPE.shape[0]*100,2)}%) words have available lemmas.")
    
    offset = time.time()
    print(f"---> {round(offset-onset,2)}s")
    return(df_SCOPE)
    

    
def lemmatize_step3(df_SCOPE,nlp):
    
    df_step3 = df_SCOPE.loc[df_SCOPE['Lemma_UKUS'].isna()]
    onset = time.time()
    with Pool(processes=48) as pool:  
        # results = pool.map(do_process, [row for _, row in df_spacy.iterrows()])
        results = pool.starmap(lemmatize_spacy, [(i,row,nlp) for i, row in df_step3.iterrows()])
    
    
    df_spacy = pd.concat(results, ignore_index=True)
    # df_SCOPE.loc[:,['Lemma_spacy','PoS_tag_spacy']] = df_final[['Lemma_spacy','PoS_tag_spacy']]
    
    df_SCOPE = df_SCOPE.set_index('Word').join(df_spacy.set_index('Word')[['Lemma_spacy','PoS_tag_spacy']], how='left').reset_index()
    
    df_SCOPE['Lemma'] = df_SCOPE['Lemma_UKUS']
    df_SCOPE['PoS_tag'] = df_SCOPE['PoS_UKUS']
    df_SCOPE.loc[~df_SCOPE['Lemma_spacy'].isna(),'Lemma'] = df_SCOPE.loc[~df_SCOPE['Lemma_spacy'].isna(),'Lemma_spacy'] 
    df_SCOPE.loc[~df_SCOPE['PoS_tag_spacy'].isna(),'PoS_tag'] = df_SCOPE.loc[~df_SCOPE['PoS_tag_spacy'].isna(),'PoS_tag_spacy']
    offset = time.time()
    print(f"Step3:\n---> {round(offset-onset,2)}s")
    return(df_SCOPE)


def lemmatize_step4(df_SCOPE,special_cases):
    onset = time.time()
    df_SCOPE['idx'] = df_SCOPE.index
    # df3 = df_lemmas.set_index('Word')
    df_SCOPE.loc[df_SCOPE.set_index('Word').loc[special_cases].reset_index()['idx'],'Lemma'] = df_SCOPE.loc[df_SCOPE.set_index('Word').loc[special_cases].reset_index()['idx'],'Word']
    offset = time.time()
    print(f"Step4:\n")
    print(f'---> {len(special_cases)} special cases.')
    print(f"---> {round(offset-onset,2)}s")
    return(df_SCOPE)
    
def lemmatize_batch(df_SCOPE):
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 1013000
    # Add specials case to keep as single tokens
    List_punct = ["'","_","-","."]
    special_cases = list(df_SCOPE.loc[df_SCOPE['Word'].apply(lambda x: any(punct in x for punct in List_punct)),'Word'].values)
    special_cases = special_cases+["cant","couldnt","wont","gonna","wanna","kinda",
                                  # "woulda","coulda","shoulda","outta","sorta","oughta","dunno",
                                  ]
    special_cases = special_cases + ['cannot', 'gotta', 'hes', 'id', 'Id', 'shes', 'thats', 'theres',
           'theyre', 'wed', 'whats', 'whos', 'whys']
    
    for case in special_cases:
        nlp.tokenizer.add_special_case(case, [{ORTH: case}])

    lemmatizer = WordNetLemmatizer()
    
    df1 = lemmatize_step1(df_SCOPE)
    df2 = lemmatize_step2(df1,nlp,lemmatizer)
    df3 = lemmatize_step3(df2,nlp)
    df4 = lemmatize_step4(df3,special_cases)
    
    df_final = df4.copy()
    return(df_final)