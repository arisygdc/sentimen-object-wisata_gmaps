import streamlit as st
import pandas as pd
import file_checkpoint as fc
import numpy as np
import utility as ut
import sys, helper.words as w
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

    def Tokenizing(self):
        self.dataframe['Tokenizing'] = self.dataframe['Case_Folding'].apply(w.split_word)

    def Translate(self):
        en_ds = ut.endict('dataset/en_to_id/kamus inggris.xlsx')
        self.dataframe['eng_word'] = self.dataframe['Tokenizing'].apply(en_ds.engwords)

    def SlangWord(self):
        slangers = ut.SlangWords()
        slangers.ReadLexicon()
        self.dataframe['slang_word'] = slangers.SlangWords_Series(self.dataframe['eng_word'])
        slangers.ReadKamus()
        self.dataframe['slang_word'] = slangers.SlangWords_Series(self.dataframe['slang_word'])
    
    def Stopword(self):
        stopword_obj = ut.Stopword()
        self.dataframe['Stopword'] = self.dataframe['slang_word'].apply(stopword_obj.execute)

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

        stemProcessVerbose.empty()
        del stemProcessVerbose
        pool.shutdown(wait=True)
        del pool
        self.dataframe['teks_remove'] = self.dataframe['Stemming'].apply(ut.satu)
        self.dataframe['teks_remove'] = self.dataframe['teks_remove'].str.findall('\w{2,}').str.join(' ').apply(w.split_word)
    
    def FinalCheck(self):
        self.dataframe['Final_Cek'] = self.dataframe['teks_remove'].apply(ut.satu).drop_duplicates().apply(w.split_word)
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe.reset_index(drop=True)
        
    def Untokenizing(self):
        self.dataframe['Untokenizing'] = self.dataframe['teks_remove'].apply(ut.satu)

    def GetDataframe(self) -> pd.DataFrame:
        return self.dataframe
    

file = None 
file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
if file is not None:
    read_excel = pd.read_excel(file)
    file = None
    prep = Prepocessing(read_excel)
    del read_excel
    with st.empty().container():
        st.write("Uploaded Dataset")
        st.write(prep.GetDataframe())
    bench = time.perf_counter()
    with st.empty().container():
        st.write("Data Cleansing")
        prep.DataCleansing()
        st.write(prep.GetDataframe()['Data_Cleansing'])
    with st.empty().container():
        st.write("Case Folding")
        prep.CaseFolding()
        st.write(prep.GetDataframe()['Case_Folding'])
    with st.empty().container():
        st.write("Tokenizing")
        prep.Tokenizing()
        st.write(prep.GetDataframe()["Tokenizing"])
    with st.empty().container():
        st.write("Translate")
        prep.Translate()
        st.write(prep.GetDataframe()["eng_word"])
    with st.empty().container():
        st.write("Slang Word")
        prep.SlangWord()
        st.write(prep.GetDataframe()['slang_word'])
    with st.empty().container():
        st.write("Stopword")
        prep.Stopword()
        st.write(prep.GetDataframe()['Stopword'])
    with st.empty().container():
        st.write("Stemming")
        prep.Stemming()
        st.write(prep.GetDataframe()["Stemming"])
    with st.empty().container():
        st.write("Final Check")
        prep.FinalCheck()
        st.write(prep.GetDataframe()['Final_Cek'])
    with st.empty().container():
        st.write("Untokenizing")
        prep.Untokenizing()
        st.write(prep.GetDataframe()['Untokenizing'])
    with st.empty().container():
        st.write("Done!")
    
    with st.empty().container():
        timesTaken = time.perf_counter() - bench
        st.write(f"Total time taken: {timesTaken}")
        st.write("Hasil Preprocessing")
        st.dataframe(prep.GetDataframe())
    
    fc.checkpoint.SetDataframe(prep.GetDataframe().copy())
    del prep
