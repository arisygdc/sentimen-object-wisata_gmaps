LIST_HAPUS = [
    'https', 'google','diterjemahkan','ga', 'gue', 'lo', 'lu',
    'us', 'kee', 'n', 'pool', 'broo', 'haha', 'wau', 'hehe',
    'ghaesss', 'pad', 'and', 'lg', 'kna', 'geng', 'wkwk', 
    'hmmm', 'deh', 'guys', 'wkwkwkk', 'dii', 'huufff', 'lho',
    'hehehe', 'eh', 'btw', 'sieh', 'dian', 'dir', 'pet', 'an',
    'ituu', 'mak', 'gan', 'bosque', 'ks', 'ig', 'oya', 'up',
    'hahaha', 'ehe', 'ha', 'ha', 'fyi', 'spf', 'bebs', 'mbak',
    'dll', 'vroff', 'bozzqu', 'ar'
]

LIST_HAPUS_CORPUS = [
    'ada', 'adanya', 'agak', 'agaknya',
    'amat', 'amatlah', 'belum', 'belumlah', 'benar',
    'benarkah', 'benarlah', 'berlebihan', 'betul',
    'betulkah', 'besar', 'boleh', 'biasa', 'bolehkah',
    'bolehlah',  'bukan', 'bukankah', 'bukanlah','bukannya',
    'cukup', 'cukupkah','cukuplah', 'dekat', 'diberi','diberikan',
    'diberikannya', 'dibuat', 'dibuatnya', 'didapat','digunakan',
    'diingat','diingatkan', 'ingat','ingat-ingat','ingin','inginkah',
    'inginkan', 'jangan','jangankan','janganlah','jauh','kecil', 
    'kelihatan','kelihatannya','kurang','lama','lamanya','lebih',
    'makin','mampu', 'masalah','masalahnya','masih','mirip','paling',
    'panjang','penting','pentingnya', 'percuma','perlu','perlukah',
    'pihak','pihaknya','punya','rasa','rasanya','sangat','sangatlah',
    'sebaik','sebaik-baiknya','sebaiknya','sebanyak','sebesar',
    'sedikit','sedikitnya','sekecil','sekurang-kurangnya','sekurangnya',
    'semakin', 'semua','semuanya','sering','seringnya','tambah','tambahnya',
    'tampak','tampaknya','tentu','tentulah','tentunya','tepat','terasa','terbanyak',
    'teringat','teringat-ingat', 'tidak','tidakkah','tidaklah','tinggi'
]

def split_word2(teks):
    temp = teks.split() # split words
    temp = " ".join(word for word in temp) # join all words

    return temp

def split_word(teks):
    list_teks = []
    for txt in teks.split(" "):
        list_teks.append(txt)
    return list_teks