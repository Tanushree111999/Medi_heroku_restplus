# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


medical_data=pd.read_csv("recordings.csv")


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.model_selection import train_test_split
X = list(medical_data['phrase'])
y = list(medical_data['prompt'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

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

#define a function that will apply the cleaning function to a series of records (the clean text function works on one string of text at a time)
def apply_cleaning_function_to_series(X):
    print('Cleaning data')
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    print ('Finished')
    return cleaned_X


#clean the text of both the training and the test data
X_train_clean = apply_cleaning_function_to_series(X_train)
X_test_clean = apply_cleaning_function_to_series(X_test)


def create_bag_of_words(X):
    from sklearn.feature_extraction.text import CountVectorizer
    
    print ('Creating bag of words...')
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    
    # In this example features may be single words or two consecutive words
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 ngram_range = (1,2), \
                                 max_features = 10000) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings. The output is a sparse array
    train_data_features = vectorizer.fit_transform(X)
    
    # Convert to a NumPy array for easy of handling
    train_data_features = train_data_features.toarray()
    
    # tfidf transform
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(train_data_features).toarray()

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
   
    return vectorizer, vocab, train_data_features, tfidf_features, tfidf

# apply our bag_of_words function to our training set
vectorizer, vocab, train_data_features, tfidf_features, tfidf  = (create_bag_of_words(X_train_clean))


def train_logistic_regression(features, label):
    print ("Training the logistic regression model...")
    from sklearn.linear_model import LogisticRegression
    ml_model_lr = LogisticRegression(C = 100,random_state = 0,solver='lbfgs',max_iter=1000)
    ml_model_lr.fit(features, label)
    print ('Finished')
    return ml_model_lr

ml_model_lr = train_logistic_regression(tfidf_features, y_train)

# Saving model to disk
pickle.dump(ml_model_lr, open('model.pkl','wb'))
pickle.dump(vectorizer, open('vector.pkl','wb'))

'''

def classify(input_phrase):
    input_phrase_clean = clean_text(input_text)
    input_phrase_clean = [input_phrase_clean]
    input_data_features = vectorizer.transform(input_phrase_clean)
    input_data_features = input_data_features.toarray()
    input_data_tfidf_features = tfidf.fit_transform(input_data_features)
    input_data_tfidf_features = input_data_tfidf_features.toarray()
    predicted_prompt = ml_model_lr.predict(input_data_tfidf_features)
    return predicted_prompt
	
	

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

x = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

'''
