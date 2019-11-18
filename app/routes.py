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
@app.route('/lsa-summarization', methods=['POST'])
def summarize_document():
    req_data = request.get_json()
    textDocument = req_data['textDocument']
    numberOfSentence = req_data['numberOfSentence']
    result = []
    parser = PlaintextParser.from_string(textDocument,Tokenizer("english"))
    summarizer_lsa = LsaSummarizer()
    summary = summarizer_lsa(parser.document, numberOfSentence)
    for sentence in summary:
        print (sentence)
        result.append(str(sentence))
        print (result)
    return json.dumps(result)

@app.route('/tokenize-document', methods=['POST'])
def tokenize_document():
    req_data = request.get_json()
    textDocument = req_data['textDocument']
    wordTokens = tokenizeDocument(textDocument)
    return json.dumps(wordTokens)

@app.route('/remove-stopword', methods=['POST'])
def remove_stopword():
    req_data = request.get_json()
    textDocument = req_data['textDocument']
    filtered_sentence = removeStopword(textDocument) 
    return json.dumps(filtered_sentence)

@app.route('/stem-document', methods=['POST'])
def stem_document():
    req_data = request.get_json()
    textDocument = req_data['textDocument']
    stemmed_word = stemDocument(textDocument)
    return json.dumps(stemmed_word)

@app.route('/tf-idf', methods =['POST'])
def svd():
    req_data = request.get_json()
    textDocument = req_data['textDocument']
    result = {}
    text = stemDocument(textDocument)
    #TfidfVectorizer().fit_transform allows only raw text document or string as parameter
    textToString = ' '.join(text)
    tfidf = TfidfVectorizer()
    response = tfidf.fit_transform([textToString])
    feature_names = tfidf.get_feature_names()
    for col in response.nonzero()[1]:
        result [feature_names[col]] = response[0, col]
    resultToJson = json.dumps(result)
    return resultToJson
@app.route('/generate-svd', methods =['POST'])
def svd2():
    req_data = request.get_json()
    textDocument = req_data['textDocument']
    text = stemDocument(textDocument)
    count_vectorizer = CountVectorizer()
    data = count_vectorizer.fit_transform(text).toarray()
    count_vectorizer._validate_vocabulary()
    featurenames = count_vectorizer.get_feature_names()
    tfidf = TfidfTransformer()
    tfidfMatrix = tfidf.fit_transform(data) 
    svd = TruncatedSVD(n_components = 30)
    svdMatrix = svd.fit_transform(tfidfMatrix)
    print (svdMatrix)
    return json.dumps(svdMatrix.tolist())


