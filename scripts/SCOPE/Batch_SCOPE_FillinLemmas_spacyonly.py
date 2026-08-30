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
from gBatch_lemmatize import lemmatize_batch,lemmatize_step1,lemmatize_step2,lemmatize_step3,lemmatize_step4,lemmatize_spacy
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')  # For POS tagging
# Dir_github = '/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github'
Dir_github = '/work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI'
repo = 'FactorAnalysis_fMRI'

# # skip if already done
# df = pd.read_excel(os.path.join(Dir_github,'data','SCOPE','data_with_metadata.xlsx'))
# df2 = pd.read_excel(os.path.join(Dir_github,'data','SCOPE','data_with_metadata.xlsx'),usecols=["Word"],keep_default_na=False, na_values=[])
# df['Word'] = df2['Word']
# for label in ['Word','NonWord']:
#     file = os.path.join(Dir_github,'data',f"SCOPE_{label.lower()}.csv")
#     df.loc[df['Status']==label].to_csv(file,index=False)
scope_dir = os.path.join(Dir_github, 'data', 'SCOPE')
scope_word_file = os.path.join(scope_dir, 'SCOPE_word.csv')

if not os.path.isfile(scope_word_file):
    print(f'Generating {scope_word_file}')

    metadata_file = os.path.join(scope_dir, 'data_with_metadata.xlsx')

    df = pd.read_excel(metadata_file)
    df_word = pd.read_excel(
        metadata_file,
        usecols=['Word'],
        keep_default_na=False,
        na_values=[],
    )
    df['Word'] = df_word['Word']

    df.loc[df['Status'] == 'Word'].to_csv(
        scope_word_file,
        index=False,
    )
else:
    print(f'Using existing SCOPE word file: {scope_word_file}')
    
# Load the file downloaded from the SCOPE database, which only contains real words. 
# read the "Word" column separately to avoid a problem caused by some special words like "NA", 'nan'
df_SCOPE_raw = pd.read_csv(os.path.join(Dir_github,'data','SCOPE',f"SCOPE_word.csv"))
# df = pd.read_csv(os.path.join(Dir_github,'data','SCOPE',f"SCOPE_word.csv"), usecols=["Word"], keep_default_na=False, na_values=[],encoding="latin-1")
# df_SCOPE_raw['Word'] = df['Word'] 
df_SCOPE_raw = pd.read_csv(scope_word_file)

df_word = pd.read_csv(
    scope_word_file,
    usecols=['Word'],
    keep_default_na=False,
    na_values=[],
    encoding='latin-1',
)

df_SCOPE_raw['Word'] = df_word['Word']

df_SCOPE_raw = df_SCOPE_raw.loc[~df_SCOPE_raw['Word'].duplicated()].reset_index() # the word "used" is duplicated
# # df_SCOPE.drop(columns=['Unnamed: 246'],inplace=True) # there is an extra column contains nothing 
print(f'The original SCOPE database has {df_SCOPE_raw.shape[0]} words, {df_SCOPE_raw.shape[1]-3} variables.')


def lemmatize_step3only(df_SCOPE,nlp):
    onset = time.time()
    with Pool(processes=48) as pool:  
        # results = pool.map(do_process, [row for _, row in df_spacy.iterrows()])
        results = pool.starmap(lemmatize_spacy, [(i,row,nlp) for i, row in df_SCOPE.iterrows()])
    
    
    df_spacy = pd.concat(results, ignore_index=True)
    # df_SCOPE.loc[:,['Lemma_spacy','PoS_tag_spacy']] = df_final[['Lemma_spacy','PoS_tag_spacy']]
    
    df_SCOPE = df_SCOPE.set_index('Word').join(df_spacy.set_index('Word')[['Lemma_spacy','PoS_tag_spacy']], how='left').reset_index()
    
#     df_SCOPE['Lemma'] = df_SCOPE['Lemma_UKUS']
#     df_SCOPE['PoS_tag'] = df_SCOPE['PoS_UKUS']
#     df_SCOPE.loc[~df_SCOPE['Lemma_spacy'].isna(),'Lemma'] = df_SCOPE.loc[~df_SCOPE['Lemma_spacy'].isna(),'Lemma_spacy'] 
#     df_SCOPE.loc[~df_SCOPE['PoS_tag_spacy'].isna(),'PoS_tag'] = df_SCOPE.loc[~df_SCOPE['PoS_tag_spacy'].isna(),'PoS_tag_spacy']
    
    df_SCOPE['Lemma'] = df_SCOPE['Lemma_spacy']
    df_SCOPE['PoS_tag'] = df_SCOPE['PoS_tag_spacy']

    offset = time.time()
    print(f"Step3:\n---> {round(offset-onset,2)}s")
    return(df_SCOPE)



nlp = spacy.load('en_core_web_sm')
nlp.max_length = 1013000
# Add specials case to keep as single tokens
List_punct = ["'","_","-","."]

df_SCOPE = df_SCOPE_raw[['Word','DPoS_VanH']].copy()
special_cases = list(df_SCOPE.loc[df_SCOPE['Word'].apply(lambda x: any(punct in x for punct in List_punct)),'Word'].values)
special_cases = special_cases+["cant","couldnt","wont","gonna","wanna","kinda",
                              # "woulda","coulda","shoulda","outta","sorta","oughta","dunno",
                              ]
special_cases = special_cases + ['cannot', 'gotta', 'hes', 'id', 'Id', 'shes', 'thats', 'theres',
       'theyre', 'wed', 'whats', 'whos', 'whys']

for case in special_cases:
    nlp.tokenizer.add_special_case(case, [{ORTH: case}])

df_spacyonly = lemmatize_step3only(df_SCOPE,nlp)

df_lemmas = lemmatize_step4(df_spacyonly,special_cases)


df_lemmas.to_csv(os.path.join(Dir_github,'data','SCOPE','SCOPE_lemmatization_output_spacyonly.csv'),index=False)

