import streamlit as st
import file_checkpoint as fc, utility as ut
import pandas as pd, matplotlib.pyplot as plt
from wordcloud import WordCloud
import regex as re

def satu(teks):
  # Menghapus Tanda Baca
  text = re.sub(r"[^\w]|_"," ", teks)
  return text

def PieChart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize = (6, 6))
    sizes = [count for count in df['label'].value_counts()]
    labels = list(df['label'].value_counts().index)
    explode = (0, 0)
    colors = ['#ffcc99', '#ff9999']
    ax.pie(x = sizes, labels = labels, colors=colors, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
    ax.set_title('Sentiment Polarity Pada Data Ulasan Bukit Pelalangan Arosbaya', fontsize = 16, pad = 20)
    return fig

def WordCloud_TextProcessed(df: pd.DataFrame):
    list_words=''
    for review in df['teks_remove']:
        for word in review:
            list_words += word
            
    wordcloud = WordCloud(width = 600, height = 400, background_color = 'white', min_font_size = 10).generate(list_words)
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.set_title('Word Cloud pada Pada Data Ulasan Bukit Pelalangan Arosbaya', fontsize = 18)
    ax.grid(False)
    ax.imshow((wordcloud))
    fig.tight_layout(pad=0)
    ax.axis('off')
    return fig

def WordCloud_NegativeVsPositive(df: pd.DataFrame):
    positive_review = df[df['label'] == 'P']
    positive_words = positive_review['teks_remove']

    fig, ax = plt.subplots(1,2, figsize = (15, 10))
    list_words_postive=''
    for row_word in positive_words:
        for word in row_word:
            list_words_postive +=(word)
    wordcloud_positive = WordCloud(width = 800, height = 600, background_color = 'black', colormap = 'Greens'
                                , min_font_size = 10).generate(list_words_postive)
    ax[0].set_title('Word Cloud dari Kata Positif Pada Data Ulasan Bukit Pelalangan Arosbaya', fontsize = 14)
    ax[0].grid(False)
    ax[0].imshow((wordcloud_positive))
    fig.tight_layout(pad=0)
    ax[0].axis('off')

    negative_review = df[df['label'] == 'N']
    negative_words = negative_review['teks_remove']

    list_words_negative=''
    for row_word in negative_words:
        for word in row_word:
            list_words_negative +=(word)
    wordcloud_negative = WordCloud(width = 800, height = 600, background_color = 'black', colormap = 'Reds'
                                , min_font_size = 10).generate(list_words_negative)
    ax[1].set_title('Word Cloud dari Kata Negatif Pada Data Ulasan Bukit Pelalangan Arosbaya', fontsize = 14)
    ax[1].grid(False)
    ax[1].imshow((wordcloud_negative))
    fig.tight_layout(pad=0)
    ax[1].axis('off')
    return fig

def wordOnPositiveSentiment(kata_positif: pd.Series):
    positif_kata = kata_positif.value_counts().nlargest(10)

    positif_x = positif_kata.index
    positif_y = positif_kata.values

    fig = plt.figure(figsize = (12, 10))
    plt.bar(positif_x, positif_y)
    plt.xlabel("Kata", fontweight ='bold')
    plt.ylabel("Frekuensi", fontweight ='bold')
    plt.title("Kata pada Sentimen positif", fontweight ='bold', fontsize = 14)
    plt.xticks(positif_x, rotation = 30)
    return fig

def wordOnNegativeSentiment(kata_negatif: pd.Series):
    negatif_kata = kata_negatif.value_counts().nlargest(10)
    negatif_x = negatif_kata.index
    negatif_y = negatif_kata.values

    fig = plt.figure(figsize = (12, 10))
    plt.bar(negatif_x, negatif_y)
    plt.xlabel("Kata", fontweight ='bold')
    plt.ylabel("Frekuensi", fontweight ='bold')
    plt.title("Kata pada Sentimen Negatif", fontweight ='bold', fontsize = 14)
    plt.xticks(negatif_x, rotation = 30)
    return fig

placeholder = st.empty()
init_df = True

if not fc.checkpoint.CheckDataframe():
    init_df = False
    with placeholder.container():
        st.write("Dataframe not initialized")

if init_df:
    df = fc.checkpoint.GetDataframe().copy()
    df = ut.PrepareDataframe(df)
    with placeholder.container():
        st.write("Hasil Preprocessing")
        st.write(df)
        
        fig = PieChart(df)
        st.write(fig)

        fig = WordCloud_TextProcessed(df)
        st.write(fig)

        fig = WordCloud_NegativeVsPositive(df)
        st.write(fig)

        # WORDCLOUD CONTENT SENTIMEN NEGATIF & POSITIVE
        kata_positif = pd.Series(" ".join(df[df["label"] == 'P']["teks_remove"].apply(satu).astype("str")).split())
        kata_negatif = pd.Series(" ".join(df[df["label"] == 'N']["teks_remove"].apply(satu).astype("str")).split())

        fig = wordOnPositiveSentiment(kata_positif)
        st.write(fig)

        fig = wordOnNegativeSentiment(kata_negatif)
        st.write(fig)
        del kata_positif, kata_negatif
        del fig
        del df


