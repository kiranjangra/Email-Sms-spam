import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('stopwords')

def transform_word(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # if this code doen't work at any other source then try test=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
model = pickle.load(open('model1.pkl', 'rb'))

# Load the pickle file
# with open('model.pkl', 'rb') as f:
#     data = joblib.load(f)
# # Check the data types of the arrays
# # print(data.dtype)
#
# # Convert data types if needed
# model = data.astype(np.float64)  # Convert to compatible data type

# Use the converted data

# Define the custom CSS directly
custom_css = """
<style>
/* Change the hover color */
.st-expander:hover .st-expander-title {
    background-color: blue !important;
    /* Change other properties as needed */
}
</style>
"""

# Display the custom CSS using st.markdown
st.markdown(custom_css, unsafe_allow_html=True)


st.title("Email/Sms Classifier")
st.image("E:\Spam Project\images.jpg",width=100)

with st.expander("1. Check if your text is spam or not"):
    input_sms = st.text_area("Enter the message")
    if st.button("Predict"):
        # 1. Preprocess
        transformed_msg = transform_word(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_msg])

        # 3. Predict
        result = model.predict(vector_input)

        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
