from st_pages import show_pages_from_config, add_page_title
import streamlit as st
import pandas as pd
import file_checkpoint as fc
import numpy as np
import utility as ut
import sys, os, nltk, words as w
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

file = None
file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

def RunSlang(listFile: list[str], dataframe: pd.DataFrame) -> pd.DataFrame:
    for val in listFile:
        if val.endswith(".csv"):
            slang_obj.ReadCsv(val)
        if val.endswith(".xlsx"):
            slang_obj.ReadExcel(val)
        st.write(val)
        dataframe['slang_word'] = dataframe['Case_Folding'].apply(slang_obj.Slangwords)
        st.write(dataframe['slang_word'].head())
    return dataframe

if file is not None:
    read_excel = pd.read_excel(file)
    fc.checkpoint.SetDataframe(read_excel)
    df = fc.checkpoint.GetDataframe()
    # Data Cleansing
    df2 = df.copy()
    df2['Text_Clean'] = df['review'].drop_duplicates()
    df2 = df2.dropna()
    df2 = df2.reset_index(drop=True)
    df2['Data_Cleansing'] = df2['Text_Clean'].apply(ut.Data_Cleansing)

    # Case Folding
    df2['Case_Folding'] = df2['Data_Cleansing'].apply(ut.Case_Folding)
    df2 = df2.replace(r'^\s*$', np.nan, regex=True)
    df2.isna().sum()
    df2 = df2.dropna()
    df2 = df2.reset_index(drop=True)

    # Slang Word
    SlangFileList_Path = os.getcwd()+"dataset/slang_word"
    dir = os.listdir(SlangFileList_Path)
    slang_obj = ut.SlangWords()
    df2 = RunSlang(dir, df2)

    # Tokenizing
    df2['Tokenizing'] = df2['slang_word'].apply(ut.split_word)

    # Stopword
    nltk.download('words')
    nltk.download('stopwords')
    listStopword =  list(stopwords.words('indonesian'))
    list_stopword = listStopword
    list_stopword.extend(w.LIST_HAPUS)
    for word in w.LIST_HAPUS_CORPUS:
        list_stopword.remove(word)

    list_stopword = set(list_stopword)
    stopword_obj = ut.Stopword(list_stopword)
    df2['Stopword'] = df2['Tokenizing'].apply(stopword_obj.execute)

    # Stemming sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()


    
    


    