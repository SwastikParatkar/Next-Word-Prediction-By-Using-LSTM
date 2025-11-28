import streamlit as st

# 반드시 첫 번째 Streamlit 명령어여야 함
st.set_page_config(page_title="Next Word Predictor", layout="centered")

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# -------------------------
# Load model & tokenizer
# -------------------------
@st.cache_resource
def load_assets():
    model = load_model("nextword_model.h5")
    
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    index_word = {i: w for w, i in tokenizer.word_index.items()}
    seq_len = model.input_shape[1]
    return model, tokenizer, index_word, seq_len


model, tokenizer, index_word, INPUT_SEQ_LEN = load_assets()

# -------------------------
# Text generation function
# -------------------------
def generate_text(seed_text, max_words=10):
    seed_text = seed_text.lower().strip()

    MIN_WORDS = 3   # Force at least these many predictions
    generated = []

    for i in range(max_words):
        seq = tokenizer.texts_to_sequences([seed_text])[0]
        seq = pad_sequences([seq], maxlen=INPUT_SEQ_LEN, padding='pre')

        preds = model.predict(seq, verbose=0)[0]

        next_index = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # Always stop if model predicts padding
        if next_index == 0:
            break

        next_word = index_word.get(next_index, "")

        if next_word == "":
            break

        # AFTER minimum words, apply strict stop rules
        if i >= MIN_WORDS:
            # Stop if confidence low
            if confidence < 0.10:
                break
            
            # Avoid useless repetition
            if generated.count(next_word) >= 2:
                break

        # Append prediction
        generated.append(next_word)
        seed_text += " " + next_word

    return seed_text




# -------------------------
# UI
# -------------------------
st.title("Next Word Predictor (LSTM)")
st.write("Type a sentence based on course or FAQ data and predict next words.")

user_input = st.text_input("Enter starting text:", "what is the duration")
num_words = st.slider("Number of words", 1, 20, 10)

if st.button("Predict"):
    output = generate_text(user_input, num_words)
    st.subheader("Predicted Text:")
    st.write(output)
