from app import app
from flask import request
from sklearn.feature_extraction.text import TfidfVectorizer
import sumy
import json
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import TruncatedSVD
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
def tokenizeDocument(textDoc):
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(textDoc)
    return word_tokens
def removeStopword(textDoc):
    stop_words = set(stopwords.words('english'))
    wordTokens = tokenizeDocument(textDoc)
    filtered_sentence = [word for word in wordTokens if not word.lower() in stop_words] 
    return filtered_sentence
def stemDocument(textDoc):
    ps =PorterStemmer()
    stemmed_words = []
    filtered_sentence = removeStopword(textDoc) 
    for word in filtered_sentence:
        rootWord=ps.stem(word)
        stemmed_words.append(rootWord)
    return stemmed_words

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    return "Hello, World!"

@app.route('/json-example', methods=['POST'])
def json_example():
    text = request.form.get('text')
    return text



