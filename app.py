import streamlit as st
import torch
import pandas as pd
import random
import json
import re
import string
import ast # Import ast for literal_eval
import sys # Import sys to check if running interactively

from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemover, ArrayDictionary, StopWordRemoverFactory

# Define preprocessing functions (copy from your notebook)
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopWordFactory = StopWordRemoverFactory()

def lower(text):
    return text.lower()

def non_ascii(text):
    return text.encode('ascii', 'replace').decode('ascii')

def remove_punctuation(text):
    remove = string.punctuation.replace("?", "") # Keep question marks
    pattern = r"[{}]".format(remove) # Fixed the escaped backslash
    return re.sub(pattern, "", text)

def remove_whitespace_LT(text):
    return text.strip()

def remove_whitespace_multiple(text):
    return re.sub('\\s+',' ',text)

def remove_tab(text):
    return text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")

def remove_angka(text):
    return re.sub(r"\d+", "", text) # Fixed the escaped backslash

def stopwords(text):
    more_stopword = ['sih', 'nya','rt','loh','lah', 'dd', 'mah', 'nye', 'eh', 'ehh', 'ah', 'yang','yg']
    data = stopWordFactory.get_stop_words()
    if 'tidak' in data: # Check if 'tidak' exists before removing
        data.remove('tidak')
    # stopwords_sastrawi = stopWordFactory.create_stop_word_remover() # This line is not needed here

    dictionary = ArrayDictionary(data+more_stopword)
    str_stopwords = StopWordRemover(dictionary)
    return str_stopwords.remove(text)

def stemming(text):
    return stemmer.stem(text)

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

# --- Load Model and Data ---
# Define the paths (adjust if necessary)
model_path = r"model"
intents_file_path = r'dataset/intent_dataset.json'

# Load intents data to create df_responses, id2label, label2id
with open(intents_file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

df_intents = pd.DataFrame(intents['intents'])

dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df_intents)):
    ptrns = df_intents.loc[i, 'patterns']
    rspns = df_intents.loc[i, 'responses']
    tag = df_intents.loc[i, 'tag']
    for p in ptrns:
        dic['tag'].append(tag)
        dic['patterns'].append(p)
        dic['responses'].append(rspns)

df_responses = pd.DataFrame.from_dict(dic)

# Create id2label and label2id
labels = df_responses['tag'].unique().tolist()
labels = [s.strip() for s in labels]
num_labels = len(labels)
id2label = {id:label for id, label in enumerate(labels)}
label2id = {label:id for id, label in enumerate(labels)}


@st.cache_resource # Cache the model loading
def load_model(model_path, num_labels, id2label, label2id):
    # Load base model
    # Replace 'cahya/bert-base-indonesian-522M' with the actual base model name
    base_model_name = 'cahya/bert-base-indonesian-522M'
    base_model = BertForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Load the trained PEFT model (adapter weights)
    peft_trained_model = PeftModel.from_pretrained(base_model, model_path)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_trained_model.to(device)
    peft_trained_model.eval() # Set model to evaluation mode

    return peft_trained_model, tokenizer, device

# Load the model and tokenizer
model, tokenizer, device = load_model(model_path, num_labels, id2label, label2id)

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="RISELF - Teman Cerita Virtual.", layout="wide") # Changed title

    st.title("🤖 RISELF - Rise starts from Yourself") # Changed title
    st.markdown("Selamat datang di RISELF. Kami percaya bahwa setiap orang berhak memiliki ruang aman untuk mencurahkan isi hatinya. RISELF hadir sebagai sahabat virtual yang siap mendengarkan keluh kesah Kamu kapan saja, tanpa menghakimi. Tujuan kami adalah menemanimu menemukan jalan untuk 'Bangkit' dimulai dari 'Diri Sendiri', karena setiap perasaanmu berharga dan Kamu tidak sendirian.") # Changed greeting

    # Initialize chat history and user_name in session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_name" not in st.session_state:
        st.session_state.user_name = None

    # Add initial greeting if it's the first interaction and name is not known
    if not st.session_state.messages and st.session_state.user_name is None:
        initial_greeting = "Halo! Saya RISELF. Untuk memulai, bolehkah saya tahu nama Kamu?" # Changed greeting
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Masukkan pesan Anda di sini..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process user input
        processed_text = preprocessing_user_input(prompt)
        response_text = "" # Use response_text to build the final response

        # --- Logic to capture user name ---
        captured_name = None

        # More robust name capture logic using regex
        # Look for patterns like "nama saya [nama]", "aku dipanggil [nama]", "panggil aku [nama]", or just "[nama]" at the beginning if it seems like a name
        name_patterns = r"(?:nama saya|aku dipanggil|panggil aku|nama ku|panggil saja|halo aku|hai aku|nama aku)\s+([\w\s]+)"
        name_match = re.search(name_patterns, prompt, re.IGNORECASE)

        if name_match:
            captured_name = name_match.group(1).strip()
        elif len(st.session_state.messages) == 2 and st.session_state.user_name is None:
             # If it's the very first response after the greeting, assume the whole input is the name
             captured_name = prompt.strip()
             # Add a basic check to see if it looks like a name (e.g., not too short, starts with a letter)
             if len(captured_name) < 2 or not captured_name[0].isalpha():
                 captured_name = None # Discard if it doesn't look like a name

        if captured_name:
            st.session_state.user_name = string.capwords(captured_name)
            # Provide a confirmation message if name was just captured
            response_text = f"Baik, {st.session_state.user_name}! Senang berkenalan dengan Anda. Ada lagi yang bisa saya bantu?"
        elif not processed_text:
            # Updated empathetic response for empty or unprocessable input
            response_text = "Maaf, saya belum sepenuhnya mengerti. Jika ada yang ingin Anda ceritakan, saya siap mendengarkan."
        else:
            # Tokenize input
            inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

            # Move inputs to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get model prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                scores = torch.softmax(logits, dim=-1)
                confidence = scores[0][predictions[0]].item()

            predicted_label_id = predictions[0].item()
            predicted_label_str = id2label[predicted_label_id]

            # Confidence threshold
            confidence_threshold = 0.2 # Adjust as needed

            if confidence < confidence_threshold:
                 # Updated empathetic response for low confidence
                 response_text = f"Terima kasih sudah berbagi. Saya sedang berusaha memahami. Jika Anda merasa kesulitan, mungkin berbicara dengan profesional bisa sangat membantu. Mereka punya keahlian khusus untuk mendampingi Anda. Saya di sini untuk mendengarkan kapan pun Anda butuh."
            else:
                # Get random response from detected intent
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
                         # Replace {user_name} if user_name is available
                         if st.session_state.user_name:
                             response_text = selected_response.replace("{user_name}", st.session_state.user_name)
                         else:
                             # If name is not available, just use the original response
                             response_text = selected_response
                    else:
                        # Updated empathetic response if no specific response found for the intent
                        response_text = "Terima kasih sudah berbagi. Saya di sini untuk mendengarkan. Jika Anda merasa kesulitan, mungkin berbicara dengan profesional bisa sangat membantu. Mereka punya keahlian khusus untuk mendampingi Anda."
                else:
                    # Updated empathetic response if no intent match found (should be rare with confidence threshold)
                    response_text = "Terima kasih sudah berbagi. Saya di sini untuk mendengarkan. Jika Anda merasa kesulitan, mungkin berbicara dengan profesional bisa sangat membantu. Mereka punya keahlian khusus untuk mendampingi Anda."


        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_text)
        # Add chatbot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})


# This allows the script to be run directly from the terminal using `streamlit run app.py`
if __name__ == "__main__":
    main()