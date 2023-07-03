import streamlit as st
import file_checkpoint as fc
import pandas as pd
import utility as ut
import helper.words as wrd

tmp=""
teks = st.text_input('Prediksi Data Baru', 'Teks untuk diprediksi')

df2 = fc.checkpoint.GetDataframe().copy()
X=df2['Untokenizing']
y=df2['label']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(df2['Untokenizing'])
Train_X_Tfidf = Tfidf_vect.transform(X_train)
Test_X_Tfidf = Tfidf_vect.transform(X_test)

df_idf = pd.DataFrame(Train_X_Tfidf.toarray(), columns=Tfidf_vect.get_feature_names_out(), index=X_train)
df_idf2 = pd.DataFrame(Test_X_Tfidf.toarray(), columns=Tfidf_vect.get_feature_names_out(), index=X_test)

# Modelling
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Training
#klasifikasi naive bayes
clf = MultinomialNB().fit(Train_X_Tfidf.toarray(), y_train)
predicted = clf.predict(Train_X_Tfidf.toarray())

# Naive bayes
#klasifikasi naive bayes
clf = MultinomialNB().fit(Train_X_Tfidf.toarray(), y_train)
predicted = clf.predict(Test_X_Tfidf.toarray())

from sklearn.model_selection import cross_validate, KFold
cv_train = cross_validate(estimator=MultinomialNB(),
                          X=Train_X_Tfidf.toarray(),
                          y=y_train,
                          cv=KFold(n_splits=10),
                          scoring=(('accuracy','precision_weighted', 'recall_weighted', 'f1_weighted')))

import nltk
from nltk.corpus import stopwords
nltk.download('words')
nltk.download('stopwords')

slangers = ut.SlangWords()
stopper = ut.Stopword()
stemmer = ut.Stemmer()
dict = ut.endict('dataset/en_to_id/kamus inggris.xlsx')

def slang(teks):
  slangers.ReadLexicon()
  teks = slangers.Slangwords(teks)
  slangers.ReadKamus()
  return slangers.Slangwords(teks)

def model_terbaik(text):
  teks = ut.Case_Folding(text)
  teks = wrd.split_word(teks)
  teks = dict.engwords(teks)
  teks = slang(teks)
  teks = stopper.execute(teks)
  teks = stemmer.stem(teks)
  teks = ut.satu(teks)
  teks = teks.replace(r'\W*\b(?!no)\w{1,2}\b', '')

  tf_baru = Tfidf_vect.transform([teks])
  predicted = clf.predict(tf_baru.toarray())

  if predicted:
    return 'Positive'
  return 'Negative'

if teks != tmp:
  st.write(teks)
  pred = model_terbaik(teks)
  st.write(pred)
  updated=False
  updated=True
  tmp = teks


    
    