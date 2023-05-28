import streamlit as st
import pandas as pd
import file_checkpoint as fc
import numpy as np
import utility as ut
import sys, nltk, helper.words as w
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class Prepocessing:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def DataCleansing(self):
        self.dataframe['Text_Clean'] = self.dataframe['review'].drop_duplicates()
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe.reset_index(drop=True)
        self.dataframe['Data_Cleansing'] = self.dataframe['Text_Clean'].apply(ut.Data_Cleansing)

    def CaseFolding(self):
        self.dataframe['Case_Folding'] = self.dataframe['Data_Cleansing'].apply(ut.Case_Folding)
        self.dataframe = self.dataframe.replace(r'^\s*$', np.nan, regex=True)
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe.reset_index(drop=True)

    def SlangWord(self):
        self.dataframe = ut.RunSlang("dataset/slang_word", self.dataframe)
    
    def Tokenizing(self):
        self.dataframe['Tokenizing'] = self.dataframe['slang_word'].apply(ut.split_word)
    
    def Stopword(self):
        nltk.download('words')
        nltk.download('stopwords')
        listStopword =  list(stopwords.words('indonesian'))
        list_stopword = listStopword
        list_stopword.extend(w.LIST_HAPUS)
        for word in w.LIST_HAPUS_CORPUS:
            list_stopword.remove(word)

        list_stopword = set(list_stopword)
        stopword_obj = ut.Stopword(list_stopword)
        self.dataframe['Stopword'] = self.dataframe['Tokenizing'].apply(stopword_obj.execute)
    
    def Stemming(self):
        stemmer = ut.Stemmer()
        self.dataframe['Stemming'] = self.dataframe['Stopword'].apply(lambda x: stemmer.stem(x))
        self.dataframe['teks_remove'] = self.dataframe['Stemming'].apply(ut.satu)
        self.dataframe['teks_remove'] = self.dataframe['teks_remove'].str.findall('\w{2,}').str.join(' ').apply(ut.split_word)

    def GetDataframe(self) -> pd.DataFrame:
        return self.dataframe
    
    def MethodStep(self) -> list[tuple[str, callable]]:
        return [
            ("Data Cleansing", self.DataCleansing),
            ("Case Folding", self.CaseFolding),
            ("Slang Word", self.SlangWord),
            ("Tokenizing", self.Tokenizing),
            ("Stopword", self.Stopword),
            ("Stemming", self.Stemming),
            ("Done!", self.GetDataframe)
        ]

file = None 
file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
if file is not None:
    read_excel = pd.read_excel(file)
    file = None
    fc.checkpoint.SetDataframe(read_excel)
    prep = Prepocessing(fc.checkpoint.GetDataframe().copy())
    res = None
    placeholder = st.empty()
    for step in prep.MethodStep():
        with placeholder.container():
            st.write(step[0])
            res = step[1]()
    st.dataframe(res)
    fc.checkpoint.SetDataframe(res)