import streamlit as st
import file_checkpoint as fc, utility as ut
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def tf_idf(selected_key: str, df: pd.DataFrame):
    st.write("Hasil Preprocessing")
    st.write(df)
    df['teks_remove'] = df['teks_remove'].apply(ut.satu)
    _ = df[df['teks_remove'].str.isspace()==True].index
    df = df.drop(df.index[[25, 59, 211, 212, 220, 268, 301, 312, 325, 360]])
    
    # Train Test Split
    X=df['teks_remove']
    y=df['label']
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size=selected_value[selected_key][0],
        random_state=selected_value[selected_key][1]
    )
    
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

    #  Training
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_train, predicted), annot=True,cmap='Blues', ax=ax)
    st.write("MultinomialNB Accuracy:" , accuracy_score(y_train,predicted))
    st.write(fig)

    # Testing
    clf = MultinomialNB().fit(Train_X_Tfidf.toarray(), y_train)
    predicted = clf.predict(Test_X_Tfidf.toarray())
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, predicted), annot=True,cmap='Blues', ax=ax)
    st.write("MultinomialNB Accuracy:" , accuracy_score(y_test,predicted))
    st.write(fig)

    # Training
    columns = ['accuracy', 'precision', 'recall', 'f1_score']
    cv_train = cross_validate(
        estimator=MultinomialNB(),
        X=Train_X_Tfidf.toarray(),
        y=y_train,
        cv=KFold(n_splits=10),
        scoring=(('accuracy','precision_weighted', 'recall_weighted', 'f1_weighted'))
    )
    
    # Testing
    df_cv_train = pd.DataFrame({
            columns[0]:cv_train['test_accuracy'],
            columns[1]:cv_train['test_precision_weighted'],
            columns[2]:cv_train['test_recall_weighted'],
            columns[3]:cv_train['test_f1_weighted']
    })
    
    st.write("Training overview")
    st.write(df_cv_train*100)

    cv_test = cross_validate(
        estimator=MultinomialNB(),
        X=Test_X_Tfidf.toarray(),
        y=y_test,
        cv=KFold(n_splits=10),
        scoring=(('accuracy','precision_weighted', 'recall_weighted', 'f1_weighted'))
    )
    
    df_cv_test = pd.DataFrame({
        columns[0]:cv_test['test_accuracy'],
        columns[1]:cv_test['test_precision_weighted'],
        columns[2]:cv_test['test_recall_weighted'],
        columns[3]:cv_test['test_f1_weighted']
    })
    
    st.write("Testing overview")
    st.write(df_cv_test*100)

    index = (df_cv_test.mean()*100).index.tolist()
    training_series = (df_cv_train.mean()*100).tolist()
    testing_series = (df_cv_test.mean()*100).tolist()
    reader = pd.DataFrame([training_series, testing_series], index=['Training', 'Testing'], columns=index)
    st.write("Mean of Training and Testing")
    st.write(reader)

selected_value = {
    "50_50":(0.5,110), 
    "60_40":(0.4, 42), 
    "70_30":(0.3, 42), 
    "80_20":(0.2, 42)
}

select_state = None
selected = st.selectbox("TFIDF",selected_value.keys())
placeholder = st.empty()
init_df = True

if not fc.checkpoint.CheckDataframe():
    init_df = False
    with placeholder.container():
        st.write("Dataframe not initialized")

if init_df:
    if select_state != selected:
        select_state = selected
        with placeholder.container():
            tf_idf(selected, fc.checkpoint.GetDataframe().copy())