import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sqlalchemy import create_engine, text
import json
import random

def main():
    # Download stopwords NLTK jika belum ada
    nltk.download('stopwords')

    # Buat koneksi ke database MySQL
    engine = create_engine("mysql+mysqlconnector://root:@localhost/ikn-app")

    # Ambil semua data dari tabel dataset
    query = "SELECT id, full_text FROM dataset ORDER BY id"
    df = pd.read_sql(query, con=engine)

    # Filter data: hanya yang mengandung kata kunci
    keywords = ['ikn', 'nusantara', 'ibu kota', 'ibukota', 'pemindahan', 'perpindahan']
    filtered_df = df[df['full_text'].str.contains('|'.join(keywords), case=False, na=False)].copy()
    filtered_df = filtered_df[filtered_df['full_text'].str.strip() != '']
    filtered_df = filtered_df.drop_duplicates(subset='full_text')

    # Persiapkan stemmer dan kamus kata dasar
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    kamus_path = r'C:\xampp\htdocs\ikn-preprocessing\sastrawi\kata-dasar.txt'
    with open(kamus_path, 'r', encoding='utf-8') as f:
        kamus_sastrawi = set(word.strip() for word in f.readlines())

    # Kata penting yang harus ada dan kata-kata positif/negatif untuk analisis sentimen
    kata_penting = {'ikn', 'nusantara', 'ibu', 'kota', 'ibukota', 'pemindahan', 'perpindahan'}
    stop_words = set(stopwords.words('indonesian'))
    tambahan_stop = {
        'gw', 'gue', 'gua', 'lu', 'loe', 'lo', 'elu', 'nya', 'ya', 'aja', 'sih', 'lah', 'deh', 'dong',
        'kok', 'nih', 'tuh', 'lagi', 'kayak', 'gak', 'ga', 'nggak', 'ngga', 'yg', 'yang', 'saya', 'kamu'
    }
    stop_words.update(tambahan_stop)

    kata_positif = {
        'baik', 'bagus', 'maju', 'dukung', 'setuju', 'positif', 'indah', 'hebat',
        'sejahtera', 'aman', 'nyaman', 'modern', 'teratur', 'subur', 'makmur',
        'berhasil', 'mantap', 'sukses', 'optimal', 'unggul', 'ceria', 'produktif',
        'stabil', 'harmonis', 'adil', 'bersih', 'ramah', 'berkah', 'amanah',
        'visioner', 'cerdas', 'terdepan', 'efisien', 'ekonomis', 'peduli',
        'inovatif', 'terpercaya', 'terkendali', 'berdaya', 'kompeten', 'kreatif',
        'menarik', 'cerah', 'bersemangat', 'bangga', 'luar biasa', 'keren',
        'top', 'wow', 'sip', 'asik', 'mantul', 'terbaik', 'juara', 'smart',
        'jempol', 'bagus banget', 'puas', 'solutif', 'terlaksana', 'tercapai',
        'terstruktur', 'progresif', 'cepat', 'mudah', 'lancar', 'efektif',
        'inspiratif', 'semangat', 'aman banget', 'makin bagus', 'memuaskan',
        'bijak', 'inilah', 'bagus sekali', 'oke', 'yes', 'supportif',
        'semakin baik', 'stabil', 'terang', 'mewah', 'damai', 'terpuji',
        'unggulan', 'top banget', 'luar biasa', 'berfaedah', 'gokil', 'terpercaya',
        'senang', 'bahagia', 'terobosan', 'nyata', 'menawan', 'paten', 'berkelas',
        'menyentuh', 'bernilai', 'istimewa', 'mujarab', 'tokcer', 'tepat',
        'berhasil banget', 'cerdas banget', 'mantep', 'jos', 'cuan', 'aman sentosa',
        'makin keren', 'relate', 'tepat guna', 'pas', 'efisien banget',
        'berfungsi', 'sukses besar', 'berpengaruh', 'super', 'wow banget',
        'efektif banget', 'brilian', 'visioner banget', 'cepat tanggap',
        'peduli rakyat', 'merakyat', 'hati-hati', 'rapi', 'lugas', 'terukur',
        'dipercaya', 'luwes', 'terorganisir', 'tenang', 'wajar', 'tidak panik',
        'bijaksana', 'legit', 'recommended', 'makin maju', 'tanggung jawab',
        'support', 'cukup baik', 'solid', 'progres', 'adaptif', 'relevan',
        'fleksibel', 'membangun', 'mencerahkan', 'kompak', 'simpel', 'enak dilihat',
        'konsisten', 'solusi', 'tuntas', 'jujur', 'senyum', 'senyum rakyat',
        'mudah dimengerti', 'pasti', 'paham', 'positif thinking'
    }

    kata_negatif = {
        'tidak', 'buruk', 'tolak', 'negatif', 'korup', 'jelek', 'hancur',
        'bencana', 'rusak', 'gagal', 'macet', 'rawan', 'ancam', 'bahaya',
        'rugi', 'protes', 'kritik', 'sesat', 'merugikan', 'sengketa', 'sulit',
        'gelap', 'curang', 'cacat', 'terbelakang', 'parah', 'lemah', 'krisis',
        'konflik', 'tidak adil', 'semrawut', 'terbengkalai', 'merosot',
        'miskin', 'terancam', 'tercela', 'tidak layak', 'bising', 'polusi',
        'biaya tinggi', 'tidak aman', 'tidak nyaman', 'tidak efektif',
        'melemah', 'lambat', 'malas', 'ragu', 'basi', 'bobrok', 'menyedihkan',
        'frustrasi', 'nggak jelas', 'mengecewakan', 'marah', 'kesal',
        'tidak suka', 'susah', 'ribet', 'bikin pusing', 'nggak beres',
        'kacau', 'curiga', 'resah', 'malapetaka', 'kritis', 'mencekam',
        'mengerikan', 'menakutkan', 'palsu', 'provokatif', 'salah kaprah',
        'tidak setuju', 'nggak suka', 'tidak puas', 'overpriced',
        'tidak efisien', 'tidak mendidik', 'tidak layak', 'menakutkan',
        'ancaman', 'negatif banget', 'sakit', 'murka',
        'menyimpang', 'ribut', 'kontra', 'bubar', 'amburadul', 'zonk', 'ancur',
        'lelet', 'telat', 'ribet banget', 'malas banget', 'serem', 'nanggung',
        'menjengkelkan', 'ganggu', 'ga nyambung', 'bikin kesel', 'bikin emosi',
        'salah total', 'misinformasi', 'hoax', 'cacat logika', 'fitnah', 'musibah',
        'kekerasan', 'konspirasi', 'adu domba', 'penghasut', 'provokasi', 'berantakan',
        'terbelit', 'overload', 'dibully', 'disalahkan', 'dilecehkan', 'difitnah',
        'tidak konsisten', 'tidak jelas', 'menyesatkan', 'omong kosong', 'asal-asalan',
        'disalahgunakan', 'salah paham', 'menurun', 'melemahkan', 'tidak berguna',
        'tidak solutif', 'ngaco', 'murahan', 'sepele', 'gak niat', 'ga peduli',
        'ga mikir', 'ga penting', 'ga masuk akal', 'ga relevan', 'ga mendukung',
        'kecewa berat', 'bikin rugi', 'bikin repot', 'tidak siap', 'setengah hati',
        'bermasalah', 'dipersulit', 'kebohongan', 'keliru', 'tidak kompeten',
        'tidak bertanggung jawab', 'tidak transparan', 'tanpa hasil', 'salah arah'
    }

    def preprocess_steps(text):
        # Hapus URL, hashtag, mention, angka
        data_clean = re.sub(r'http\S+|#\w+|@\w+|\d+', '', text)
        lower = data_clean.lower()
        no_punct = re.sub(r'[^\w\s]', '', lower)

        # Ganti kata gaul menjadi baku
        replaced = re.sub(r'\bgw\b|\bgue\b|\bgua\b', 'saya', no_punct)
        replaced = re.sub(r'\blu\b|\bloe\b|\belo\b|\belu\b', 'kamu', replaced)
        replaced = re.sub(r'\bnggak\b|\bngga\b|\bga\b|\bgak\b', 'tidak', replaced)

        # Tokenisasi
        tokens = replaced.split()

        # Hapus stopword
        tokens_stop_removed = [w for w in tokens if w not in stop_words]

        # Stemming
        joined_for_stem = ' '.join(tokens_stop_removed)
        stemmed_text = stemmer.stem(joined_for_stem)
        stemmed_tokens = stemmed_text.split()

        # Filter tokens: hanya yang ada di kamus Sastrawi atau kata penting
        tokens_filtered = [w for w in stemmed_tokens if w in kamus_sastrawi or w in kata_penting]
        tokens_sorted = sorted(tokens_filtered)
        final_cleaned = ' '.join(tokens_sorted)

        # Analisis sentimen: jika ada kata penting, sentimen dianggap positif
        sentimen = 'positif' if any(k in tokens_sorted for k in kata_positif) else (
            'negatif' if any(k in tokens_sorted for k in kata_negatif) else 'negatif'
        )

        return [
            final_cleaned,
            lower,
            no_punct,
            json.dumps(tokens, ensure_ascii=False),
            json.dumps(tokens_stop_removed, ensure_ascii=False),
            json.dumps(tokens_sorted, ensure_ascii=False),
            sentimen,
            len(tokens_sorted)
        ]

    # Proses seluruh data
    filtered_df = filtered_df.reset_index()
    processed = filtered_df['full_text'].apply(preprocess_steps)
    processed = pd.DataFrame(processed.tolist(), columns=[
        'data_clean',
        'lowercasing',
        'remove_punctuation',
        'tokenizing',
        'stopword',
        'stemming',
        'sentiment',
        'jumlah_kata'
    ])

    # Cocokkan kembali index asli
    processed['original_index'] = filtered_df['index']
    filtered_df = filtered_df.set_index('index')

    # Hapus data dengan kata terlalu sedikit
    processed = processed[processed['jumlah_kata'] >= 3].drop(columns=['jumlah_kata'])
    filtered_df = filtered_df.loc[processed['original_index']].reset_index(drop=True)
    processed = processed.drop(columns=['original_index']).reset_index(drop=True)

    # AUGMENTASI DATA

    def augment_text(text):
        # Duplikasi satu kata secara acak
        words = text.split()
        if len(words) < 3:
            return text
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, words[idx])
        return ' '.join(words)

    def augment_swap_text(text):
        # Tukar posisi dua kata acak
        words = text.split()
        if len(words) < 2:
            return text
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def augment_data(df):
        # Tambahkan data sampai seimbang jumlah positif-negatif
        pos_df = df[df['sentiment'] == 'positif']
        neg_df = df[df['sentiment'] == 'negatif']

        if len(pos_df) > len(neg_df):
            # Jumlah data positif lebih banyak → augmentasi data NEGATIF
            target = len(pos_df)
            aug_needed = target - len(neg_df)
            aug_samples = neg_df.sample(aug_needed, replace=True).copy()
            # Terapkan augmentasi kombinasi
            aug_samples['data_clean'] = aug_samples['data_clean'].apply(
                lambda x: augment_swap_text(augment_text(x))
            )
            aug_samples['sentiment'] = 'negatif'
            result = pd.concat([df, aug_samples], ignore_index=True)
        elif len(neg_df) > len(pos_df):
            # Jumlah data negatif lebih banyak → augmentasi data POSITIF
            target = len(neg_df)
            aug_needed = target - len(pos_df)
            aug_samples = pos_df.sample(aug_needed, replace=True).copy()
            # Terapkan augmentasi kombinasi
            aug_samples['data_clean'] = aug_samples['data_clean'].apply(
                lambda x: augment_swap_text(augment_text(x))
            )
            aug_samples['sentiment'] = 'positif'
            result = pd.concat([df, aug_samples], ignore_index=True)
        else:
            # Jumlah sudah seimbang → tidak ada augmentasi
            result = df
        return result

    processed_aug = augment_data(processed)

    # SIMPAN KE DATABASE
    processed_aug.to_sql(name='preprocessing', con=engine, if_exists='append', index=False)

    print("Data berhasil disimpan ke tabel MySQL: 'preprocessing'")
    print("Contoh data hasil pembersihan, augmentasi, dan sentimen:")
    print(processed_aug[['stemming', 'sentiment']].head())

if __name__ == "__main__":
    main()
