import streamlit as st
import pandas as pd
import file_checkpoint as fc
import numpy as np
import utility as ut
import sys, nltk, helper.words as w
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import concurrent.futures
import time

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

    def _stem(dataSeries: pd.Series):
        bench = time.perf_counter()
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return [dataSeries.apply(
        lambda x: [
            stemmer.stem(word) 
            for word in x
        ]), time.perf_counter()-bench]
    
    def Stemming(self):
        MAX_WORKERS = 4
        DataAmount = self.dataframe['Stopword'].count()
        MaxPerProccess = (DataAmount // MAX_WORKERS)
        loopRange = DataAmount // MaxPerProccess
        loopRange = DataAmount % MaxPerProccess != 0 and loopRange+1 or loopRange
        res = [None] * loopRange
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)

        self.dataframe['Stemming'] = ""
        stemProcessVerbose = st.empty()
        bench = time.perf_counter()
        for i in range(loopRange):
            start = i * MaxPerProccess
            end = start + MaxPerProccess
            EndIndexGTDataAmount = end > DataAmount
            end = DataAmount if EndIndexGTDataAmount else end
            res[i]=pool.submit(w.stem, self.dataframe['Stopword'][start:end])

        with stemProcessVerbose.container():
            for i in range(loopRange):
                start = i * MaxPerProccess
                end = start + MaxPerProccess
                EndIndexGTDataAmount = end > DataAmount
                end = DataAmount if EndIndexGTDataAmount else end
            
                self.dataframe['Stemming'][start:end] = res[i].result()[0]
                st.write(f"Time taken for {i+1} is {res[i].result()[1]}")
                st.write(f"Start: {start}, End: {end}, Data Carry: {end-start}")

        timesTaken = time.perf_counter()-bench
        stemProcessVerbose.empty()
        st.write(f"Total time taken: {timesTaken}")
        st.write("Hasil Preprocessing")
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
    prep = Prepocessing(read_excel)
    del read_excel
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
    del res
