import sys
import os
import re
import pickle
from sqlalchemy import create_engine
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    
    """
    INPUT:
    database_filepath - 
    
    OUTPUT:
    X - messages (input variable) 
    y - categories of the messages (output variable)
    category_names - category name for y
    
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath.replace(".db","") + "_table"
    df = pd.read_sql_table(table_name, engine)

    X = df['message']   # only column 'message' relevant
    y = df.iloc[:,4:]
    
    

def tokenize(text):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,'urlplaceholder')

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
      """
      Starting Verb Extractor class
    
      This class extract the starting verb of a sentence,
      creating a new feature for the ML classifier
      """

  
      def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

      # Given it is a tranformer we can return the self
        def fit(self, X, y=None):
              return self

        def transform(self, X):
            X_tagged = pd.Series(X).apply(self.starting_verb)
            return pd.DataFrame(X_tagged)


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()