import unicodedata, re, os
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def Data_Cleansing(text):
    # Hapus non-ascii
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Menghapus Tanda Baca
    text = re.sub(r'[^\w]|_',' ', text)
    
    # Menghapus Angka
    text = re.sub("\S*\d\S*", "", text).strip()
    text = re.sub(r"\b\d+\b", " ", text)

    # Menghapus white space
    text = re.sub('[\s]+', ' ', text)

    return text

def Case_Folding(text):
    # Mengubah text menjadi lowercase
    text = text.lower()  
    return text

def satu(teks):
  text = (" ").join(teks)
  return text

class SlangWords:
    def __init__(self):
        self.dict = {}

    def ReadCsv(self, path):
        slang_word = pd.read_csv(path)
        slang_dict = pd.Series(slang_word['formal'].values,index=slang_word['slang']).to_dict()
        self.setDict(slang_dict)

    def ReadExcel(self, path):
        slang_word = pd.read_excel(path)
        slang_dict = pd.Series(slang_word['formal'].values,index=slang_word['slang']).to_dict()
        self.setDict(slang_dict)
        
    def setDict(self, dict):
        self.dict = dict

    def Slangwords(self, text):
        for word in text.split():
            if word in self.dict.keys():
                text = text.replace(word, self.dict[word])
        return text

def split_word(teks):
    list_teks = []
    for txt in teks.split(" "):
        list_teks.append(txt)
    return list_teks

class Stopword:
    def __init__(self, list_stopword: set):
        self.list_stopword = list_stopword
    
    def execute(self, words):
        return [word for word in words if word not in self.list_stopword]

def RunSlang(path, dataframe: pd.DataFrame) -> pd.DataFrame:
    slang_obj = SlangWords()
    SlangFileList_Path = os.getcwd()+"/"+path
    dir = os.listdir(SlangFileList_Path)
    for val in reversed(dir):
        file = SlangFileList_Path+"/"+val
        if val.endswith(".csv"):
            slang_obj.ReadCsv(file)
        if val.endswith(".xlsx"):
            slang_obj.ReadExcel(file)
        dataframe['slang_word'] = dataframe['Case_Folding'].apply(slang_obj.Slangwords)
    return dataframe

class Stemmer:
    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
    
    def stem(self, words):
        return [self.stemmer.stem(word) for word in words]