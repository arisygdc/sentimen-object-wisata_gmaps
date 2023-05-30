import streamlit as st
import pandas as pd
import file_checkpoint as fc
import numpy as np
import utility as ut
import sys, nltk, helper.words as w
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import concurrent.futures

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
        self.dataframe['Tokenizing'] = self.dataframe['slang_word'].apply(w.split_word)
    
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

    def _stem(self, start: int, end: int):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return self.dataframe['Stopword'][start:end].apply(
        lambda x: [
            stemmer.stem(word) 
            for word in x
        ])
        
    
    def Stemming(self):
        MAX_WORKERS = 4
        data_amount = self.dataframe['Stopword'].count()
        MaxPerThread = data_amount // MAX_WORKERS
        loopRange = data_amount // MaxPerThread
        if data_amount % MaxPerThread != 0:
            loopRange += 1
        res = [None] * loopRange
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)

        self.dataframe['Stemming'] = ""
        for i in range(loopRange):
            start = i * MaxPerThread
            end = start + MaxPerThread
            if end > data_amount:
                end = data_amount
            res[i]=pool.submit(self._stem, start, end)

        for i in range(loopRange):
            start = i * MaxPerThread
            end = start + MaxPerThread
            if end > data_amount:
                end = data_amount
            self.dataframe['Stemming'][start:end] = res[i].result()
        pool.shutdown(wait=True)
        self.dataframe['teks_remove'] = self.dataframe['Stemming'].apply(ut.satu)
        self.dataframe['teks_remove'] = self.dataframe['teks_remove'].str.findall('\w{2,}').str.join(' ').apply(w.split_word)
        
    def GetDataframe(self) -> pd.DataFrame:
        return self.dataframe

file = None 
file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
if file is not None:
    read_excel = pd.read_excel(file)
    file = None
    fc.checkpoint.SetDataframe(read_excel)
    prep = Prepocessing(fc.checkpoint.GetDataframe().copy())
    res = None
    placeholder = st.empty()
    method_step: list[tuple[str, callable]] = [
        ("Data Cleansing", prep.DataCleansing),
        ("Case Folding", prep.CaseFolding),
        ("Slang Word", prep.SlangWord),
        ("Tokenizing", prep.Tokenizing),
        ("Stopword", prep.Stopword),
        ("Stemming", prep.Stemming),
        ("Done!", prep.GetDataframe)
    ]
    for step in method_step:
        with placeholder.container():
            st.write(step[0])
            res = step[1]()
    fc.checkpoint.SetDataframe(res.copy())
    st.dataframe(res)
