import streamlit as st
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

# ------------------------------------------------
# SAME FAQ TEXT USED DURING TRAINING (copy-paste)
# ------------------------------------------------
faqs = """About the Program
What is the course fee for Data Science Mentorship Program (DSMP 2023)
The course follows a monthly subscription model where you have to make monthly payments of Rs 799/month.
What is the total duration of the course?
The total duration of the course is 7 months. So the total course fee becomes 799*7 = Rs 5600(approx.)
What is the syllabus of the mentorship program?
We will be covering the following modules:
Python Fundamentals
Python libraries for Data Science
Data Analysis
SQL for Data Science
Maths for Machine Learning
ML Algorithms
Practical ML
MLOPs
Case studies
You can check the detailed syllabus here - https://learnwith.campusx.in/courses/CampusX-Data-Science-Mentorship-Program-637339a
Will Deep Learning and NLP be a part of this program?
No, NLP and Deep Learning both are not a part of this program’s curriculum.
What if I miss a live session? Will I get a recording of the session?
Yes all our sessions are recorded, so even if you miss a session you can go back and watch the recording.
Where can I find the class schedule?
Checkout this google sheet to see month by month time table of the course - https://docs.google.com/spreadsheets/d/16OoTax_A6OR
What is the time duration of all the live sessions?
Roughly, all the sessions last 2 hours.
What is the language spoken by the instructor during the sessions?
Hinglish
How will I be informed about the upcoming class?
You will get a mail from our side before every paid session once you become a paid user.
Can I do this course if I am from a non-tech background?
Yes, absolutely.
I am late, can I join the program in the middle?
Absolutely, you can join the program anytime.
If I join/pay in the middle, will I be able to see all the past lectures?
Yes, once you make the payment you will be able to see all the past content in your dashboard.
Where do I have to submit the task?
You don’t have to submit the task. We will provide you with the solutions, you have to self evaluate the task yourself.
Will we do case studies in the program?
Yes.
Where can we contact you?
You can mail us at nitish.campusx@gmail.com
Payment/Registration related questions
Where do we have to make our payments? Your YouTube channel or website?
You have to make all your monthly payments on our website. Here is the link for our website - https://learnwith.campusx.in/
Can we pay the entire amount of Rs 5600 all at once?
Unfortunately no, the program follows a monthly subscription model.
What is the validity of monthly subscription? Suppose if I pay on 15th Jan, then do I have to pay again on 1st Feb or 15th Feb
15th Feb. The validity period is 30 days from the day you make the payment. So essentially you can join anytime you don’t have
What if I don’t like the course after making the payment. What is the refund policy?
You get a 7 days refund period from the day you have made the payment.
I am living outside India and I am not able to make the payment on the website, what should I do?
You have to contact us by sending a mail at nitish.campusx@gmail.com
Post registration queries
Till when can I view the paid videos on the website?
This one is tricky, so read carefully. You can watch the videos till your subscription is valid. Suppose you have purchased sub
But once the course is over and you have paid us Rs 5600(or 7 installments of Rs 799) you will be able to watch the paid sessio
Why lifetime validity is not provided?
Because of the low course fee.
Where can I reach out in case of a doubt after the session?
You will have to fill a google form provided in your dashboard and our team will contact you for a 1 on 1 doubt clearance sessi
If I join the program late, can I still ask past week doubts?
Yes, just select past week doubt in the doubt clearance google form.
I am living outside India and I am not able to make the payment on the website, what should I do?
You have to contact us by sending a mail at nitish.campusx@gmai.com
Certificate and Placement Assistance related queries
What is the criteria to get the certificate?
There are 2 criterias:
You have to pay the entire fee of Rs 5600
You have to attempt all the course assessments.
I am joining late. How can I pay payment of the earlier months?
You will get a link to pay fee of earlier months in your dashboard once you pay for the current month.
I have read that Placement assistance is a part of this program. What comes under Placement assistance?
This is to clarify that Placement assistance does not mean Placement guarantee. So we dont guarantee you any jobs or for that m
Portfolio Building sessions
Soft skill sessions
Sessions with industry mentors
Discussion on Job hunting strategies
"""

# -----------------------------
# CONSTANTS (from your notebook)
# -----------------------------
VOCAB_SIZE = 283   # 282 words + 1 for index 0
MAX_LEN = 56       # input sequence length

# -----------------------------
# BUILD TOKENIZER FROM TEXT
# -----------------------------
@st.cache_resource
def build_tokenizer():
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([faqs])
    return tokenizer

tokenizer = build_tokenizer()
index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

# -----------------------------
# BUILD MODEL + LOAD WEIGHTS
# -----------------------------
@st.cache_resource
def build_and_load_model():
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=100, input_length=MAX_LEN))
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(150, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(VOCAB_SIZE, activation="softmax"))

    # Load weights saved in Colab
    model.load_weights("nextword_model.h5")

    # Compile (only needed if you ever evaluate, but harmless)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model

model = build_and_load_model()

# -----------------------------
# TEXT GENERATION FUNCTION
# -----------------------------
def generate_text(seed_text: str, num_words: int) -> str:
    text = seed_text.strip()
    if not text:
        return ""

    for _ in range(num_words):
        # convert to sequence
        token_list = tokenizer.texts_to_sequences([text])[0]

        # keep last MAX_LEN tokens
        token_list = token_list[-MAX_LEN:]

        # pad
        padded = pad_sequences([token_list], maxlen=MAX_LEN, padding="pre")

        # predict next word index
        preds = model.predict(padded, verbose=0)
        next_index = int(np.argmax(preds, axis=-1)[0])

        # convert index → word
        next_word = index_to_word.get(next_index, "")
        if not next_word:
            break

        text += " " + next_word

    return text

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Next Word Predictor Using LSTM")
st.write("Type a starting sentence and the model will predict the next words based on the FAQ dataset.")

seed_text = st.text_input(
    "Enter the starting sentence:",
    value="total duration of the course"
)

num_words = st.slider(
    "How many words to predict?",
    min_value=1,
    max_value=30,
    value=10
)

if st.button("Generate"):
    if not seed_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Predicting..."):
            result = generate_text(seed_text, num_words)
        st.subheader("Generated text:")
        st.write(result)
