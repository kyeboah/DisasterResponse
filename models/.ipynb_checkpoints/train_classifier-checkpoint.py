# import libraries
import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import joblib

import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download(['punkt','wordnet','stopwords'])

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

import sys


def load_data(database_filepath):
    '''
    Load the dataset stored in the database. Returns input and target data and category names
    Argument: database_filepath (str)
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table('DisasterResponse_table', engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):
    '''
    Cleans and prepares raw text, returning a list of lemmatised tokens
    Argument: text (str) which is the raw text
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # normalise and change to lower case
    
    words = word_tokenize(text) # split into tokens of words
    words = [w for w in words if w not in stopwords.words('english')] # remove stopwords
    
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w).strip() for w in words]
    # lemmatise verbs by specifying their part of speech
    cleaned_words = [lemmatizer.lemmatize(w, pos='v') for w in lemmed] 
    
    return cleaned_words


def build_model():
    '''
     ML pipeline takes in the message column as input and output classification results on the
    36 categories in the datase. 
    returns the gridsearch object containing the model
    
    '''
    rf_classifier = RandomForestClassifier(random_state=42)
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([ # text processing pipeline
                ('vect', CountVectorizer(tokenizer=tokenize)), 
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(estimator=rf_classifier))  # classifier
    ])

    params = {
    'clf__estimator__n_estimators': [20], 
    'features__text_pipeline__vect__stop_words': ['english']
    }

    cv = GridSearchCV(pipeline, params, cv=3, verbose=3, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Measures evaluation metrics on the model mainly, precisio, recall, f1
    Arguments: model(trained model), X_test(dataset), Y_test(dataset), category_names
    
    '''

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    
    # for i, col in enumerate(category_names):
        # print("Feature {}: {}".format(i+1, col))
        # print(classification_report(Y_test[col], y_pred[:,i]))

    #return True


def save_model(model, model_filepath):
    '''
    Saves model to a filepath as a pickle file

    Arguments: trained model, filepath to save model
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
        #joblib.dump(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()