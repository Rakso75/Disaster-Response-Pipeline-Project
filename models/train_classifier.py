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
    
    category_names = y.columns

    return X, y, category_names 


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



def build_model():
    '''
    Build a ML pipeline using Tfidf, Random Forest, and GridSearch
    
    Parameters: None
    
    Returns:
        best model of GridSearchCV        
    '''
    # pipeline1 is best classifier with parameters from GridSearch, which is performed in the notebook
    pipeline1 = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf_transformer', TfidfTransformer())
                ]))

            ])),

            ('classifier', MultiOutputClassifier(AdaBoostClassifier(n_estimators=10, learning_rate=0.01)))
        ])



    pipeline2 = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf_transformer', TfidfTransformer())
                ]))

            ])),

            ('classifier', MultiOutputClassifier(estimator=RandomForestClassifier(random_state=42)))
        ])

    model = pipeline1
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Parameters:
    
    model - ML model
    X_test - test messages (Features)
    y_test - categories for test messages (Labels)
    category_names - category name for y
    
    Returns:
    
    none - print scores (precision, recall, f1-score) for each output category of the dataset.
    """

    # report the f1 score, precision and recall for each output category of the dataset.

    y_pred_test = model.predict(X_test)

    # test accuracy
    print("The test accuracy is: ")
    print(accuracy_score(y_test, y_pred_test)*100)
    print(" ")

    # classification report
    print("--------------------------------classification report TEST--------------------------------")
    print(classification_report(y_test, y_pred_test, target_names=y_test.columns.values))

    
def save_model(model, model_filepath):
    """
    Saves trained model as pickle file.
    
    Parameters:
    
    model - ML model
    model_filepath - location to save the model
    
    Returns: none
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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