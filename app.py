from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None, stop_words = None, ngram_range = (1,2),max_features = 10000) 
tfidf = TfidfTransformer()
from flask_restx import reqparse,abort,Api,Resource

app = Flask(__name__)
api = Api(app)


model = pickle.load(open('model.pkl', 'rb'))
vect =  pickle.load(open('vector.pkl', 'rb'))


def clean_text(raw_phrase):
    # Function to convert a raw phrase to a string of words
    
    # Import modules
   
    import re

    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_phrase) 
    
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()   

    # Remove stop words (use of sets makes this faster)
    from nltk.corpus import stopwords
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]                             

    # Reduce word to stem of word
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed_words = [porter.stem(w) for w in meaningful_words]

    # Join the words back into one string separated by space
    joined_words = ( " ".join( stemmed_words ))
    return joined_words

@app.route('/')
def home():
	return "Welcome"

class PredictAilment(Resource):
	def post(self):
		input_text = request.args.get('message')
		input_phrase_clean = [input_text]
		input_phrase_clean = clean_text(input_text)
		input_phrase_clean = [input_phrase_clean]
		input_data_features = vect.transform(input_phrase_clean)
		input_data_features = input_data_features.toarray()
		input_data_tfidf_features = tfidf.fit_transform(input_data_features)
		input_data_tfidf_features = input_data_tfidf_features.toarray()
		my_prediction = model.predict(input_data_tfidf_features).tolist()
		return jsonify({'prediction' : my_prediction})

api.add_resource(PredictAilment,'/predict')


if __name__ == '__main__':
	app.run(debug=True)
