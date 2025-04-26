import streamlit as st 
import pickle 
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.stem import PorterStemmer
import string

ps = PorterStemmer()

stop_words = stopwords.words('english')

tfdf = pickle.load(open(r'vectorconverter', 'rb'))
model = pickle.load(open(r'model', 'rb'))

st.title("Email and Message Spam Detection")
st.markdown("""
    This application uses machine learning to classify emails and messages as spam or not spam.
    Please enter your text in the input box below and click on 'Predict'.
""")

user_input = st.text_input('Enter the email or message', placeholder="Type your message here...")

def transform_text(text):
    text = text.lower()  
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

if st.button('Predict'):
    transformed_input = transform_text(user_input)
    vector_input = tfdf.transform([transformed_input])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.success("**Result:** This message is classified as **Spam**!")
    else:
        st.success("**Result:** This message is classified as **Not Spam**.")