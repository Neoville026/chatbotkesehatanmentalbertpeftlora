import streamlit as st
import torch
import pandas as pd
import random
import json
import re
import string
import ast  # Import ast untuk evaluasi literal (mengubah string list ke list asli)
import sys  # Import sys untuk pengecekan interaktif

from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemover, ArrayDictionary, StopWordRemoverFactory

# ==============================
# Bagian Preprocessing Teks
# ==============================

# Membuat objek stemmer untuk stemming bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Membuat objek stopword remover
stopWordFactory = StopWordRemoverFactory()

# Fungsi untuk mengubah teks menjadi huruf kecil
def lower(text):
    return text.lower()

# Fungsi untuk menghapus karakter non-ascii
def non_ascii(text):
    return text.encode('ascii', 'replace').decode('ascii')

# Fungsi untuk menghapus tanda baca kecuali tanda tanya
def remove_punctuation(text):
    remove = string.punctuation.replace("?", "")
    pattern = r"[{}]".format(remove)
    return re.sub(pattern, "", text)

# Fungsi untuk menghapus whitespace di awal/akhir teks
def remove_whitespace_LT(text):
    return text.strip()

# Fungsi untuk menghapus whitespace ganda di tengah kalimat
def remove_whitespace_multiple(text):
    return re.sub('\\s+',' ',text)

# Fungsi untuk menghapus karakter tab/escape
def remove_tab(text):
    return text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")

# Fungsi untuk menghapus angka
def remove_angka(text):
    return re.sub(r"\d+", "", text)

# Fungsi untuk menghapus stopwords (kata tidak penting)
def stopwords(text):
    more_stopword = ['sih', 'nya','rt','loh','lah', 'dd', 'mah', 'nye', 'eh', 'ehh', 'ah', 'yang','yg']
    data = stopWordFactory.get_stop_words()
    # Menghapus kata "tidak" dari stopword agar tetap dipertahankan
    if 'tidak' in data:
        data.remove('tidak')
    dictionary = ArrayDictionary(data+more_stopword)
    str_stopwords = StopWordRemover(dictionary)
    return str_stopwords.remove(text)

# Fungsi untuk melakukan stemming (mengembalikan kata ke bentuk dasar)
def stemming(text):
    return stemmer.stem(text)

# Fungsi utama preprocessing: menjalankan semua tahapan di atas
def preprocessing_user_input(text):
    proc = lower(text)
    proc = non_ascii(proc)
    proc = remove_tab(proc)
    proc = remove_punctuation(proc)
    proc = remove_angka(proc)
    proc = remove_whitespace_LT(proc)
    proc = remove_whitespace_multiple(proc)
    proc = stopwords(proc)
    proc = stemming(proc)
    return proc

# ==============================
# Load Dataset Intent & Model
# ==============================

model_path = r"model"
intents_file_path = r'dataset/intent_dataset.json'

# Membaca file intent dataset
with open(intents_file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Mengubah dataset JSON menjadi DataFrame
df_intents = pd.DataFrame(intents['intents'])

# Membuat struktur dataset pola input dan respon
dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df_intents)):
    ptrns = df_intents.loc[i, 'patterns']
    rspns = df_intents.loc[i, 'responses']
    tag = df_intents.loc[i, 'tag']
    for p in ptrns:
        dic['tag'].append(tag)
        dic['patterns'].append(p)
        dic['responses'].append(rspns)

# Dataset final untuk response
df_responses = pd.DataFrame.from_dict(dic)

# Membuat label intent
labels = df_responses['tag'].unique().tolist()
labels = [s.strip() for s in labels]
num_labels = len(labels)
id2label = {id:label for id, label in enumerate(labels)}
label2id = {label:id for id, label in enumerate(labels)}

# Fungsi caching untuk load model BERT dengan PEFT
@st.cache_resource
def load_model(model_path, num_labels, id2label, label2id):
    # Load base model BERT bahasa Indonesia
    base_model_name = 'cahya/bert-base-indonesian-522M'
    base_model = BertForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    # Load model PEFT hasil fine-tuning
    peft_trained_model = PeftModel.from_pretrained(base_model, model_path)
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # Deteksi device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_trained_model.to(device)
    peft_trained_model.eval()
    return peft_trained_model, tokenizer, device

# Load model dan tokenizer
model, tokenizer, device = load_model(model_path, num_labels, id2label, label2id)

# ==============================
# Bagian Utama Aplikasi Streamlit
# ==============================
def main():
    st.set_page_config(page_title="RISELF - Teman Cerita Virtual.", layout="wide")

    # ---- Sidebar dengan motivasi & informasi ----
    motivasi = [
        "🌸 Kamu lebih kuat dari yang kamu kira.",
        "💡 Tidak apa-apa merasa lelah, istirahat juga bagian dari perjalanan.",
        "🌿 Setiap langkah kecil tetap berarti.",
        "🫂 Kamu tidak sendirian. Ada yang peduli."
    ]
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=100)
        st.markdown("### ℹ️ Tentang RISELF")
        st.write("RISELF adalah chatbot konseling kesehatan mental untuk mahasiswa. "
                 "Chatbot ini tidak menggantikan tenaga profesional, namun bisa menjadi teman awal untuk berbagi cerita.")
        st.markdown("### ✨ Motivasi Hari Ini")
        st.success(random.choice(motivasi))
        st.markdown("### ⚠️ Kontak Darurat")
        st.info("Jika kamu merasa dalam kondisi darurat, segera hubungi layanan profesional "
                "atau nomor darurat kesehatan terdekat.")

    # ---- Header utama chatbot ----
    st.title("🤖 RISELF - Rise starts from Yourself")
    st.markdown("Selamat datang di RISELF. Kami percaya bahwa setiap orang berhak memiliki ruang aman untuk mencurahkan isi hatinya. "
                "RISELF hadir sebagai sahabat virtual yang siap mendengarkan keluh kesah Kamu kapan saja, tanpa menghakimi. "
                "Tujuan kami adalah menemanimu menemukan jalan untuk 'Bangkit' dimulai dari 'Diri Sendiri', "
                "karena setiap perasaanmu berharga dan Kamu tidak sendirian.")

    # Inisialisasi session_state untuk menyimpan percakapan
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_name" not in st.session_state:
        st.session_state.user_name = None

    # Pesan awal dari chatbot
    if not st.session_state.messages and st.session_state.user_name is None:
        initial_greeting = "Halo! Saya RISELF. Untuk memulai, bolehkah saya tahu nama Kamu?"
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

    # Menampilkan semua pesan sebelumnya
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input pesan pengguna
    if prompt := st.chat_input("Masukkan pesan Anda di sini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Preprocessing input pengguna
        processed_text = preprocessing_user_input(prompt)
        response_text = ""
        captured_name = None

        # Ekstraksi nama jika pengguna memperkenalkan diri
        name_patterns = r"(?:nama saya|aku dipanggil|panggil aku|nama ku|panggil saja|halo aku|hai aku|nama aku)\s+([\w\s]+)"
        name_match = re.search(name_patterns, prompt, re.IGNORECASE)

        if name_match:
            captured_name = name_match.group(1).strip()
        elif len(st.session_state.messages) == 2 and st.session_state.user_name is None:
            captured_name = prompt.strip()
            if len(captured_name) < 2 or not captured_name[0].isalpha():
                captured_name = None

        # Jika ada nama yang terdeteksi
        if captured_name:
            st.session_state.user_name = string.capwords(captured_name)
            response_text = f"Baik, {st.session_state.user_name}! Senang berkenalan dengan Anda. Ada lagi yang bisa saya bantu?"

        # Jika input kosong
        elif not processed_text:
            response_text = "Maaf, saya belum sepenuhnya mengerti. Jika ada yang ingin Anda ceritakan, saya siap mendengarkan."

        # Jika input valid, lakukan prediksi intent
        else:
            inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                scores = torch.softmax(logits, dim=-1)
                confidence = scores[0][predictions[0]].item()

            # Ambil label intent prediksi
            predicted_label_id = predictions[0].item()
            predicted_label_str = id2label[predicted_label_id]
            confidence_threshold = 0.2

            # Jika confidence rendah → fallback response
            if confidence < confidence_threshold:
                response_text = "Terima kasih sudah berbagi. Saya sedang berusaha memahami. Jika Kamu merasa kesulitan, berbicara dengan profesional bisa sangat membantu 🙏"
            else:
                # Cari respon sesuai intent
                response_row = df_responses[df_responses['tag'] == predicted_label_str]
                if not response_row.empty:
                    response_options = response_row['responses'].iloc[0]
                    if isinstance(response_options, str):
                        try:
                            response_options = ast.literal_eval(response_options)
                        except (ValueError, SyntaxError):
                            response_text = "Maaf, terjadi kesalahan dalam memproses respon."
                            response_options = []

                    if response_options:
                        selected_response = random.choice(response_options)
                        if st.session_state.user_name:
                            response_text = selected_response.replace("{user_name}", st.session_state.user_name)
                        else:
                            response_text = selected_response
                    else:
                        response_text = "Terima kasih sudah berbagi. Saya di sini untuk mendengarkan."
                else:
                    response_text = "Terima kasih sudah berbagi. Saya di sini untuk mendengarkan."

        # Menampilkan respon chatbot
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})


# Jalankan aplikasi
if __name__ == "__main__":
    main()
