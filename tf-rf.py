import streamlit as st
import pandas as pd
import file_checkpoint as fc, utility as ut
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import cross_validate, KFold


selected_value = {
    "50_50":(0.5, 0), 
    "60_40":(0.4, 0), 
    "70_30":(0.3, 0), 
    "80_20":(0.2, 0)
}

select_state = None
selected = st.selectbox("TFRF",selected_value.keys())
placeholder = st.empty()
init_df = True

def tf_rf(selected, df):
    st.write("Hasil Preprocessing")
    st.write(df)
    
    df = ut.PrepareDataframe(df)
    # Label Encoder TF-RF
    nltk.download('punkt')
    df_TF_RF=ut.TF_RF(df['teks_remove'])
    
    iterator = 0
    invalid_rows = []
    for index, row in df_TF_RF.iterrows():
        if row.isnull().any():
            invalid_rows.append((index, iterator))
        iterator += 1

    for i in range(len(invalid_rows)):
        df_TF_RF.drop(invalid_rows[i][0], inplace=True)
        df.drop(invalid_rows[i][1], inplace=True)

    st.write("Hasil TF-RF")
    st.write(df_TF_RF)

    le = LabelEncoder()
    X = df_TF_RF
    y = le.fit_transform(df['label'])
    st.write("Label Encoder")
    st.write(y)

    Train_X_Tfrf,Test_X_Tfrf,y_train,y_test = train_test_split(
        X,y,
        test_size=selected_value[selected][0], 
        random_state=selected_value[selected][1]
    )

    # Training
    clf = MultinomialNB().fit(Train_X_Tfrf, y_train)
    predicted = clf.predict(Train_X_Tfrf)

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_train, predicted), annot=True,cmap='Blues', ax=ax)
    st.write("MultinomialNB Training Accuracy:" , accuracy_score(y_train,predicted))
    st.write(fig)

    # Testing
    clf = MultinomialNB().fit(Train_X_Tfrf, y_train)
    predicted = clf.predict(Test_X_Tfrf)

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, predicted), annot=True,cmap='Blues', ax=ax)
    st.write("MultinomialNB Testing Accuracy:" , accuracy_score(y_test,predicted))
    st.write(fig)

    # K-Fold
    cv_train = cross_validate(estimator=MultinomialNB(),
                          X=Train_X_Tfrf,
                          y=y_train,
                          cv=KFold(n_splits=10),
                          scoring=(('accuracy','precision_weighted', 'recall_weighted', 'f1_weighted')))
    
    df_cv_train = pd.DataFrame({'akurasi':cv_train['test_accuracy'],
                            'presisi':cv_train['test_precision_weighted'],
                            'recall':cv_train['test_recall_weighted'],
                            'f1_score':cv_train['test_f1_weighted']})
    
    
    st.write("Training overview")
    st.write(df_cv_train*100)

    cv_test = cross_validate(estimator=MultinomialNB(),
                         X=Test_X_Tfrf,
                         y=y_test,
                         cv=KFold(n_splits=10),
                         scoring=(('accuracy','precision_weighted', 'recall_weighted', 'f1_weighted')))
    
    df_cv_test = pd.DataFrame({'akurasi':cv_test['test_accuracy'],
                            'presisi':cv_test['test_precision_weighted'],
                            'recall':cv_test['test_recall_weighted'],
                            'f1_score':cv_test['test_f1_weighted']})
    
    st.write("Testing overview")
    st.write(df_cv_test*100)

    index = (df_cv_test.mean()*100).index.tolist()
    training_series = (df_cv_train.mean()*100).tolist()
    testing_series = (df_cv_test.mean()*100).tolist()
    reader = pd.DataFrame([training_series, testing_series], index=['Training', 'Testing'], columns=index)
    st.write("Mean of Training and Testing")
    st.write(reader)
    


init_df = True
if not fc.checkpoint.CheckDataframe():
    init_df = False
    with placeholder.container():
        st.write("Dataframe not initialized")

if init_df:
    if select_state != selected:
        select_state = selected
        with placeholder.container():
            tf_rf(selected, fc.checkpoint.GetDataframe().copy())