import streamlit as st
from streamlit_option_menu import option_menu
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd
import time
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

from nltk.tokenize.treebank import TreebankWordDetokenizer


# Calc tfidf and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize.treebank import TreebankWordDetokenizer

dfCosine, dfTFIDF = pd.DataFrame([[0,0],[0,1]], columns=[0,0])

stop_factory = StopWordRemoverFactory()
# Tala, F. Z. (2003). A Study of Stemming Effects on Information Retrieval in Bahasa Indonesia. M.Sc. Thesis. Master of Logic Project. Institute for Logic, Language and Computation. Universiteit van Amsterdam, The Netherlands.
more_stopword = ['ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya',
                 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara', 'antaranya', 'apa', 'apaan', 'apabila',
                 'apakah', 'apalagi', 'apatah', 'artinya', 'asal', 'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya',
                 'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya',
                 'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini', 'beginian', 'beginikah',
                 'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun', 'bekerja', 'belakang', 'belakangan', 'belum', 'belumlah',
                 'benar', 'benarkah', 'benarlah', 'berada', 'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah',
                 'berapalah', 'berapapun', 'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya',
                 'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan', 'berlalu',
                 'berlangsung', 'berlebihan', 'bermacam', 'bermacam-macam', 'bermaksud', 'bermula', 'bersama', 'bersama-sama',
                 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 'berturut', 'berturut-turut', 'bertutur', 'berujar', 'berupa',
                 'besar', 'betul', 'betulkah', 'biasa', 'biasanya', 'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah',
                 'bolehlah', 'buat', 'bukan', 'bukankah', 'bukanlah', 'bukannya', 'bulan', 'bung', 'cara', 'caranya', 'cukup',
                 'cukupkah', 'cukuplah', 'cuma', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang', 'dekat', 'demi',
                 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia', 'diakhiri', 'diakhirinya', 'dialah', 'diantara',
                 'diantaranya', 'diberi', 'diberikan', 'diberikannya', 'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'digunakan',
                 'diibaratkan', 'diibaratkannya', 'diingat', 'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya',
                 'dikarenakan', 'dikatakan', 'dikatakannya', 'dikerjakan', 'diketahui', 'diketahuinya', 'dikira', 'dilakukan',
                 'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan', 'dimaksudkannya', 'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan',
                 'dimulai', 'dimulailah', 'dimulainya', 'dimungkinkan', 'dini', 'dipastikan', 'diperbuat', 'diperbuatnya',
                 'dipergunakan', 'diperkirakan', 'diperlihatkan', 'diperlukan', 'diperlukannya', 'dipersoalkan', 'dipertanyakan',
                 'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan', 'disebutkannya', 'disini', 'disinilah',
                 'ditambahkan', 'ditandaskan', 'ditanya', 'ditanyai', 'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjuk', 'ditunjuki',
                 'ditunjukkan', 'ditunjukkannya', 'ditunjuknya', 'dituturkan', 'dituturkannya', 'diucapkan', 'diucapkannya',
                 'diungkapkan', 'dong', 'dua', 'dulu', 'empat', 'enggak', 'enggaknya', 'entah', 'entahlah', 'guna', 'gunakan', 'hal',
                 'hampir', 'hanya', 'hanyalah', 'hari', 'harus', 'haruslah', 'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga',
                 'ia', 'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat', 'ingin', 'inginkah',
                 'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah', 'jadinya', 'jangan', 'jangankan',
                 'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas', 'jelaskan', 'jelaslah', 'jelasnya', 'jika', 'jikalau',
                 'juga', 'jumlah', 'jumlahnya', 'justru', 'kala', 'kalau', 'kalaulah', 'kalaupun', 'kalian', 'kami', 'kamilah', 'kamu',
                 'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun', 'karena', 'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah',
                 'katanya', 'ke', 'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan', 'kelihatan',
                 'kelihatannya', 'kelima', 'keluar', 'kembali', 'kemudian', 'kemungkinan', 'kemungkinannya', 'kenapa', 'kepada',
                 'kepadanya', 'kesampaian', 'keseluruhan', 'keseluruhannya', 'keterlaluan', 'ketika', 'khususnya', 'kini', 'kinilah',
                 'kira', 'kira-kira', 'kiranya', 'kita', 'kitalah', 'kok', 'kurang', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu',
                 'lama', 'lamanya', 'lanjut', 'lanjutnya', 'lebih', 'lewat', 'lima', 'luar', 'macam', 'maka', 'makanya', 'makin',
                 'malah', 'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'masa', 'masalah', 'masalahnya', 'masih',
                 'masihkah', 'masing', 'masing-masing', 'mau', 'maupun', 'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya',
                 'memang', 'memastikan', 'memberi', 'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta', 'memintakan',
                 'memisalkan', 'memperbuat', 'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan', 'mempersoalkan',
                 'mempertanyakan', 'mempunyai', 'memulai', 'memungkinkan', 'menaiki', 'menambahkan', 'menandaskan', 'menanti',
                 'menanti-nanti', 'menantikan', 'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan', 'mendatang',
                 'mendatangi', 'mendatangkan', 'menegaskan', 'mengakhiri', 'mengapa', 'mengatakan', 'mengatakannya', 'mengenai',
                 'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki', 'mengibaratkan', 'mengibaratkannya', 'mengingat',
                 'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan', 'mengucapkannya', 'mengungkapkan', 'menjadi', 'menjawab',
                 'menjelaskan', 'menuju', 'menunjuk', 'menunjuki', 'menunjukkan', 'menunjuknya', 'menurut', 'menuturkan',
                 'menyampaikan', 'menyangkut', 'menyatakan', 'menyebutkan', 'menyeluruh', 'menyiapkan', 'merasa', 'mereka', 'merekalah',
                 'merupakan', 'meski', 'meskipun', 'meyakini', 'meyakinkan', 'minta', 'mirip', 'misal', 'misalkan', 'misalnya', 'mula',
                 'mulai', 'mulailah', 'mulanya', 'mungkin', 'mungkinkah', 'nah', 'naik', 'namun', 'nanti', 'nantinya', 'nyaris',
                 'nyatanya', 'oleh', 'olehnya', 'pada', 'padahal', 'padanya', 'pak', 'paling', 'panjang', 'pantas', 'para', 'pasti',
                 'pastilah', 'penting', 'pentingnya', 'per', 'percuma', 'perlu', 'perlukah', 'perlunya', 'pernah', 'persoalan',
                 'pertama', 'pertama-tama', 'pertanyaan', 'pertanyakan', 'pihak', 'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'rasa',
                 'rasanya', 'rata', 'rupanya', 'saat', 'saatnya', 'saja', 'sajalah', 'saling', 'sama', 'sama-sama', 'sambil', 'sampai',
                 'sampai-sampai', 'sampaikan', 'sana', 'sangat', 'sangatlah', 'satu', 'saya', 'sayalah', 'se', 'sebab', 'sebabnya',
                 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik', 'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 'sebanyak',
                 'sebegini', 'sebegitu', 'sebelum', 'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar', 'sebetulnya', 'sebisanya',
                 'sebuah', 'sebut', 'sebutlah', 'sebutnya', 'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian', 'sedikit',
                 'sedikitnya', 'seenaknya', 'segala', 'segalanya', 'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh',
                 'sejenak', 'sejumlah', 'sekadar', 'sekadarnya', 'sekali', 'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun',
                 'sekarang', 'sekarang', 'sekecil', 'seketika', 'sekiranya', 'sekitar', 'sekitarnya', 'sekurang-kurangnya',
                 'sekurangnya', 'sela', 'selain', 'selaku', 'selalu', 'selama', 'selama-lamanya', 'selamanya', 'selanjutnya', 'seluruh',
                 'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya', 'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya',
                 'sementara', 'semisal', 'semisalnya', 'sempat', 'semua', 'semuanya', 'semula', 'sendiri', 'sendirian', 'sendirinya',
                 'seolah', 'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti',
                 'sepertinya', 'sepihak', 'sering', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera',
                 'sesekali', 'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah',
                 'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi', 'seusai', 'sewaktu',
                 'siap', 'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'soal', 'soalnya', 'suatu', 'sudah', 'sudahkah',
                 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu', 'tahun', 'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya',
                 'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya', 'tapi', 'tegas', 'tegasnya', 'telah', 'tempat',
                 'tengah', 'tentang', 'tentu', 'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa', 'terbanyak', 'terdahulu',
                 'terdapat', 'terdiri', 'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah', 'terjadinya',
                 'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan', 'tersebut', 'tersebutlah',
                 'tertentu', 'tertuju', 'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba', 'tiba-tiba', 'tidak', 'tidakkah',
                 'tidaklah', 'tiga', 'tinggi', 'toh', 'tunjuk', 'turut', 'tutur', 'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya',
                 'umum', 'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai', 'waduh', 'wah', 'wahai', 'waktu', 'waktunya',
                 'walau', 'walaupun', 'wong', 'yaitu', 'yakin', 'yakni', 'yang']


def create_docContentDict(filePaths):
    rawContentDict = {}
    for filePath in filePaths:
        rawContentDict[filePath.name] = filePath.read()
    return rawContentDict


def dok_low(dok_raw):
    filteredContents = [term.lower() for term in dok_raw]
    return filteredContents


def dok_tok(dok_low):
    tokenized = nltk.tokenize.word_tokenize(dok_low)
    return tokenized


def dok_stop(dok_tok):
    # Menggabungkan stopword dari sastrawi dengan stopword yang dibuat
    data = stop_factory.get_stop_words()+more_stopword
    stopword = stop_factory.create_stop_word_remover()

    filteredContents = [stopword.remove(word)
                        for word in dok_tok if word not in data]
    return filteredContents


def dok_stem(dok_tok):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    filteredContents = [stemmer.stem(word) for word in dok_tok]
    return filteredContents


def dok_punc(dok_tok):
    excludePuncuation = set(string.punctuation)

    # manually add additional punctuation to remove
    doubleSingleQuote = '\'\''
    doubleDash = '--'
    doubleTick = '``'

    excludePuncuation.add(doubleSingleQuote)
    excludePuncuation.add(doubleDash)
    excludePuncuation.add(doubleTick)

    filteredContents = [
        word for word in dok_tok if word not in excludePuncuation]
    return filteredContents


def dok_ang(dok_tok):
    filteredContents = [re.sub(r"[0-9]+", '', word) for word in dok_tok]
    return filteredContents


def dok_spa(dok_tok):
    filteredContents = [re.sub('[\s+]', '', word) for word in dok_tok]
    return filteredContents


def dok_untok(dok_tok):
    untokenized = TreebankWordDetokenizer().detokenize(dok_tok)
    return untokenized


def dok_tok1(dok_tok):
    tokenized1 = nltk.tokenize.word_tokenize(dok_tok)
    return tokenized1


def dok_up(dok_tok):
    filteredContents = [term.upper() for term in dok_tok]
    return filteredContents


def dok_sing(dok_tok):
    pattern = r"(\b[MDCLXVI]+\b)(\.)?"

    # filteredContents = [word for word in dok_tok if word not in pattern]
    filteredContents = [re.sub(pattern, '', word) for word in dok_tok]
    return filteredContents


def dok_low2(dok_tok):
    filteredContents = [term.lower() for term in dok_tok]
    return filteredContents


def dok_untok2(dok_tok):
    untokenized = TreebankWordDetokenizer().detokenize(dok_tok)
    return untokenized


def dok_tok2(dok_tok):
    tokenized1 = nltk.tokenize.word_tokenize(dok_tok)
    return tokenized1


def processData(rawContents):
    cleaned = dok_tok(rawContents)
    cleaned = dok_stop(cleaned)
    cleaned = dok_stem(cleaned)
    cleaned = dok_punc(cleaned)
    cleaned = dok_ang(cleaned)
    cleaned = dok_low(cleaned)
    cleaned = dok_spa(cleaned)
    cleaned = dok_untok(cleaned)
    cleaned = dok_tok1(cleaned)
    cleaned = dok_up(cleaned)
    cleaned = dok_sing(cleaned)
    cleaned = dok_low2(cleaned)
    cleaned = dok_untok2(cleaned)
    cleaned = dok_tok2(cleaned)
    return cleaned


# 1. as sidebar menu
with st.sidebar:
    st.title('PENGKLASIFIKASIAN DOKUMEN')
    st.subheader('PEMERINTAHAN KOTA DEPOK')
    selected = option_menu(None, ["Home", 'Tentang'],
                           icons=['house', 'info-circle'], menu_icon="cast", default_index=1)

if selected == "Home":
    st.header('Home')
    st.subheader('Untuk menggunakan website ini siapkan sebuah dokumen uji dan minimal 1 dokumen pembanding dengan format txt, kemudian masukan data tersebut kedalam form Upload Data. Dokumen yang diupload akan ditampilkan pada tabel untuk menghapus dokumen klik (X) pada file. Untuk mulai melakukan pengklasifikasin dokumen klik tombol Mulai.')

    st.write('<hr/>', unsafe_allow_html=True)

    st.title('Pilih file yang akan diuji')
    uploaded_files = st.file_uploader('Pilih File .txt', type=['txt', 'csv'], accept_multiple_files=True, key=None,
                                      help='Pilih file txt yang akan di-analisa', on_change=None, args=None, kwargs=None, disabled=False, label_visibility="hidden")

    if st.button("Mulai", key=None, help='Memulai proses analisa', on_click=None, type="primary", disabled=False, use_container_width=False):
        rawContentDict = {}
        listFileName = []
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name +
                     ", filetype:", uploaded_file.type)
            listFileName.append(uploaded_file.name)
            rawContentDict[uploaded_file.name] = bytes_data

        with st.spinner('Loading...'):
            tfidf = TfidfVectorizer(sublinear_tf=True, tokenizer=processData)
            tfs = tfidf.fit_transform(rawContentDict.values())
            tfs_Values = tfs.toarray().transpose()
            tfs_Term = tfidf.get_feature_names_out()
            dataraw = {}

            for i in range(len(listFileName)):
                dataraw[listFileName[i]] = tfs.toarray()[i]
            st.write(dataraw)

            # st.subheader('TF-IDF')
            df = pd.DataFrame(
                dataraw,
                tfs_Term)
            dfTFIDF = df

            st.session_state['hasiltfidf'] = df

            # st.dataframe(df, use_container_width=True)
            # st.download_button(
            #     label="Download",
            #     data=dfTFIDF.to_csv().encode('utf-8'),
            #     file_name='TF-IDF.csv',
            #     mime='text/csv',
            # )

            datarawCosin = {}

            for i in range(len(listFileName)):
                dataValue = []
                for j in range(len(listFileName)):
                    arrayValue = cosine_similarity(
                        tfs[i], tfs[j])
                    dataValue.append(arrayValue[0][0])
                datarawCosin[listFileName[i]] = dataValue

            dfCosin = pd.DataFrame(datarawCosin, listFileName)
            dfCosine = dfCosin
            # st.subheader('Cosine Similarity')

            st.session_state['hasilcosine'] = dfCosin

            # st.dataframe(dfCosin, use_container_width=True)

            # st.download_button(
            #     label="Download",
            #     data=dfCosine.to_csv().encode('utf-8'),
            #     file_name='Cosine-Similarity.csv',
            #     mime='text/csv',
            # )
            
            

        st.balloons()


    if 'hasiltfidf' in st.session_state:
        st.subheader('TF-IDF') 
        st.dataframe(st.session_state['hasiltfidf'], use_container_width=True)
        st.download_button(
                label="Download",
                data=st.session_state['hasiltfidf'].to_csv().encode('utf-8'),
                file_name='TF-IDF.csv',
                mime='text/csv',
            )
    
    if 'hasilcosine' in st.session_state:
        st.subheader('Cosine Similarity')
        st.dataframe(st.session_state['hasilcosine'], use_container_width=True)
        st.download_button(
                label="Download",
                data=st.session_state['hasilcosine'].to_csv().encode('utf-8'),
                file_name='Cosine-Similarity.csv',
                mime='text/csv',
            )
        
       

elif selected == "Tentang":
    col1, col2 = st.columns([4, 8])

    with col1:
        st.header('Tentang Saya')
        my_bar = st.progress(0, 'in progress')
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, 'in progress')
            if percent_complete == 99:
                my_bar.progress(100, 'Done!')
                st.subheader(
                    'Perkenalkan nama saya Agus Susanto, saya tinggal di Bekasi. Saat ini saya berkuliah di Universitas Gunadarma Kota Depok dan mengambil jurusan Sistem Informasi.')

    with col2:
        st.header('Tentang Web')
        my_bar = st.progress(0, 'in progress')
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, 'in progress')
            if percent_complete == 99:
                my_bar.progress(100, 'Done!')
                st.subheader(
                    'Website ini merupakan website untuk mengklasifikasikan dokumen Kota Depok. Tujuannya yaitu membantu pemerintah Kota Depok maupun pengguna lainnya dalam pengklasifikasian dokumen berdasarkan kategorinya. Pengklasifikasian website ini menggunakan metode Cosine Similarity dan Term Frequency Inverse Document Frequency (TF-IDF). Untuk framework yang digunakan pada pembuatan website ini, website ini menggunakan framework streamlit. ')

