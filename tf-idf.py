import file_checkpoint as fc
import pandas as pd
import streamlit as st
import utility as ut
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

if fc.checkpoint.CheckDataframe():
    df = fc.checkpoint.GetDataframe().copy()
    st.write(df)
    df['teks_remove'] = df['teks_remove'].apply(ut.satu)
    _ = df[df['teks_remove'].str.isspace()==True].index
    df = df.drop(df.index[[25, 59, 211, 212, 220, 268, 301, 312, 325, 360]])
    
    # Train Test Split
    X=df['teks_remove']
    y=df['label']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.5,random_state = 110)
    
    # Label Encoder TF-IDF
    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)
    y_test = Encoder.fit_transform(y_test)

    Tfidf_vect = TfidfVectorizer()
    Tfidf_vect.fit(df['teks_remove'])
    Train_X_Tfidf = Tfidf_vect.transform(X_train)
    Test_X_Tfidf = Tfidf_vect.transform(X_test)

    # Data Training
    df_idf = pd.DataFrame(Train_X_Tfidf.toarray(), columns=Tfidf_vect.get_feature_names_out(), index=X_train)
    st.write("Data Training")
    st.write(df_idf)

    # Data Testing
    df_idf2 = pd.DataFrame(Test_X_Tfidf.toarray(), columns=Tfidf_vect.get_feature_names_out(), index=X_test)
    st.write("Data Testing")
    st.write(df_idf2)

    #klasifikasi naive bayes
    clf = MultinomialNB().fit(Train_X_Tfidf.toarray(), y_train)
    predicted = clf.predict(Train_X_Tfidf.toarray())

    sns.heatmap(confusion_matrix(y_train, predicted), annot=True,cmap='Blues')
    st.write("MultinomialNB Accuracy:" , accuracy_score(y_train,predicted))

    cv_train = cross_validate(estimator=MultinomialNB(),
                          X=Train_X_Tfidf.toarray(),
                          y=y_train,
                          cv=KFold(n_splits=10),
                          scoring=(('accuracy','precision_weighted', 'recall_weighted', 'f1_weighted')))
    
    df_cv_train = pd.DataFrame({'akurasi':cv_train['test_accuracy'],
                            'presisi':cv_train['test_precision_weighted'],
                            'recall':cv_train['test_recall_weighted'],
                            'f1_score':cv_train['test_f1_weighted']})