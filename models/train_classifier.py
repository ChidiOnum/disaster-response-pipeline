#*****************************************************************************
# import libraries
#*****************************************************************************

# Common packages
import pickle
import sys
import re
import pandas as pd
from sqlalchemy import create_engine

#NLTK
import nltk
nltk.download(['punkt','wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

#Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


#*****************************************************************************
# defining functions
#*****************************************************************************

# load data
def load_data(database_filepath):
    engine = create_engine('sqlite:///' + str (database_filepath))
    df = pd.read_sql ('SELECT * FROM MessagesCategories', engine)
    X = df ['message']
    y = df.iloc[:,4:]
    return X, y


# tokenize data
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words("english")

    #tokenize
    words = word_tokenize (text)

    #stemming
    stemmed = [PorterStemmer().stem(w) for w in words]

    #lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]

    return words_lemmed


# build model
def build_model():
    #setting pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier()))
        ])


    #model parameters for GridSearchCV
    parameters = {  'vect__max_df': (0.75, 1.0),
                    'clf__estimator__n_estimators': [10, 20],
                    'clf__estimator__min_samples_split': [2, 5]
              }
    cv = GridSearchCV (pipeline, param_grid= parameters)

    return cv

# evaluate model
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_tuned = model.predict (X_test)
    y_pred_tuned = pd.DataFrame (y_pred_tuned, columns = Y_test.columns)


# save model
def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


# main module
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