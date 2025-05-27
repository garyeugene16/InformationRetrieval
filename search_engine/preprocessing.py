import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    # Lowercase, remove non-alphabet
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    # Stopword removal & stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens
