import unicodedata, re, os, math
import pandas as pd, numpy as np
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nlp_id.stopword import StopWord
import helper.words as w, nltk

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

def PrepareDataframe(df: pd.DataFrame):
    df['teks_remove'] = df['teks_remove'].apply(satu)
    _=df[df['teks_remove'].str.isspace()==True].index
    return df.drop(df.index[[25, 59, 211, 212, 220, 268, 301, 312, 325, 360]])

class endict:
    def __init__(self, ds_path) -> None:
        path = os.getcwd()+"/"+ds_path
        eng = pd.read_excel(path)
        self.dict = pd.Series(eng['indo'].values,index=eng['inggris']).to_dict()
    
    def engwords(self, text):
        for word in text:
            if word in self.dict.keys():
                text = [s.replace(word, self.dict[word]) for s in text]
        return text

class SlangWords:
    def __init__(self):
        SlangFileList_Path = os.getcwd()+"/dataset/slang_word/"
        self.files = [
            'colloquial-indonesian-lexicon.csv',
            'kamus.xlsx'
        ]
        for i in range(len(self.files)):
            self.files[i] = SlangFileList_Path+self.files[i]
        self.dict = {}

    def ReadLexicon(self):
        slang_word = pd.read_csv(self.files[0])
        self.dict = pd.Series(slang_word['formal'].values,index=slang_word['slang']).to_dict()

    def ReadKamus(self):
        slang_word = pd.read_excel(self.files[1])
        self.dict = pd.Series(slang_word['formal'].values,index=slang_word['slang']).to_dict()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    def Slangwords(self, text: str):
        for word in text:
            if word in self.dict.keys():
                text = [s.replace(word, self.dict[word]) for s in text]
        return text
    
    def SlangWords_Series(self, series: pd.Series):
        return series.apply(self.Slangwords)
        

class Stopword:
    def __init__(self, list_stopword: set=()):
        self.list_stopword = list_stopword
        if len(list_stopword)==0:
            self.list_stopword = self._dataset()

    def _dataset(self):
        nltk.download('words')
        stopword = StopWord()
        list_stopword = stopword.get_stopword()
        list_stopword.extend(w.LIST_HAPUS)
        for word in w.LIST_HAPUS_CORPUS:
            list_stopword.remove(word)
        return set(list_stopword)
    
    def execute(self, words):
        return [word for word in words if word not in self.list_stopword]   

class Stemmer:
    def __init__(self) -> None:
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
    
    def stem(self, teks):
        text = [self.stemmer.stem(word) for word in teks]
        return text

def TF_RF(df_teks):
    # Membuat list kosa kata dari seluruh teks input
    kosa_kata = set()
    for i in df_teks:
        token_kal=set(word_tokenize(i.lower()))
        kosa_kata=set.union(kosa_kata,token_kal)
    kosa_kata = sorted(kosa_kata)
    # Membuat dictionary kata-kata
    indeks_kata = {}
    for kata in kosa_kata:
        indeks_kata[kata] = 0
    # Jumlah kosa kata(untuk kolom)
    jumlah_kosa_kata = len(kosa_kata)
    # Membangun matriks kata TF
    vektor_TF=np.array([])
    for baris_teks in df_teks:
        baris_kata = indeks_kata.copy()
        split_baris = baris_teks.lower().split()
        for kata in baris_teks.lower().split():
            baris_kata[kata]=+1
        baris_kata = np.array(list(baris_kata.values()))
        baris_kata = baris_kata/len(split_baris)
        vektor_TF = np.concatenate((vektor_TF,baris_kata), axis=0)
    matriks_TF = vektor_TF[:1230341].reshape(df_teks.shape[0], jumlah_kosa_kata)
    # Membagun dataframe kata TF
    df_TF = pd.DataFrame(matriks_TF, columns=kosa_kata, index=df_teks.tolist())
    # Menghitung baris RF
    baris_RF=[]
    for kolom in df_TF.columns :
        b = len(df_TF[df_TF[kolom]>0])
        c = len(df_TF)-b
        baris_rf = (math.log10(2+(b/c)))  
        baris_RF.append(baris_rf)
    # Membuat matriks TF-RF
    vektor_TFRF=np.array([])
    for baris in matriks_TF:
        baris_tfrf=np.multiply(baris,baris_RF)
        vektor_TFRF=np.concatenate((vektor_TFRF,baris_tfrf), axis=0)
    matriks_TFRF = vektor_TFRF[:1230341].reshape(df_teks.shape[0], jumlah_kosa_kata)
    # Membuat dataframe TF-RF
    df_TF_RF = pd.DataFrame(matriks_TFRF, columns=kosa_kata, index=df_teks.tolist())
    return df_TF_RF