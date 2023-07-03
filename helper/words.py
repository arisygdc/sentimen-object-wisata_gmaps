LIST_HAPUS = [
    'https', 'google','diterjemahkan','ga', 'gue', 'lo', 'lu',
    'us', 'kee', 'n', 'pool', 'broo', 'haha', 'wau', 'hehe',
    'ghaesss', 'pad', 'and', 'lg', 'kna', 'geng', 'wkwk',
    'hmmm', 'deh', 'guys', 'wkwkwkk', 'dii', 'huufff', 'lho',
    'hehehe', 'eh', 'btw', 'sieh', 'dian', 'dir', 'pet', 'an',
    'ituu', 'mak', 'gan', 'bosque', 'ks', 'ig', 'oya', 'up',
    'hahaha', 'ehe', 'ha', 'ha', 'fyi', 'spf', 'bebs', 'mbak',
    'dll', 'vroff', 'bozzqu', 'ar','instagramable', 'instagamable',
    'instagrameble', 'adul', 'abdul', 'job'
]

LIST_HAPUS_CORPUS = [
    'gua', 'ada', 'layaknya', 'layaklah', 'dekat', 'datang', 'akurat',
    'adanya', 'adapun', 'arah', 'baik', 'benar-benar', 'benarlah',
    'bener-benar','beri', 'berikan', 'betul', 'betulkah', 'bisa',
    'boleh', 'boleh-boleh', 'bolehkah', 'bolehnya', 'cukup',
    'cukup-cukup', 'cukupkah', 'cukuplah', 'cukuuppp', 'diingat',
    'dijamu', 'diperlukan', 'diperlihatkan', 'diperlukannya', 'lebih',
    'lebih-lebih', 'mendapat', 'mendapatkan', 'meyakinkan', 'penting',
    'pentingnya', 'sangat', 'sangat-lah', 'sangat-sangat', 'sangatlah',
    'sungguh', 'sungguh-sungguh', 'tambah', 'tambahnya', 'tepat',
    'terbanyak', 'teringat', 'teringat-ingat', 'tertarik',
    'tidak', 'kurang', 'kecil', 'deg', 'deg-degan', 'apesnya', 'belum',
    'belumlah', 'sedikit', 'bukan', 'bukankah', 'bukanlah', 'bukannya',
    'agak', 'agak-agak', 'agaknya', 'enyahlah', 'jangan', 'jangankan',
    'janganlah', 'jarang', 'jarang-jarang', 'jauh', 'keterlaluan',
    'mirisnya', 'masalah', 'masalahnya', 'wow', 'luar', 'biasa','tempat'
    , 'oke'
]

import pandas as pd
import time
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def split_word2(teks):
    temp = teks.split() # split words
    temp = " ".join(word for word in temp) # join all words

    return temp

def split_word(teks):
    list_teks = []
    for txt in teks.split(" "):
        list_teks.append(txt)
    return list_teks

def stem(dataSeries: pd.Series):
    bench = time.perf_counter()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [dataSeries.apply(
    lambda x: [
        stemmer.stem(word) 
        for word in x
    ]), time.perf_counter()-bench]