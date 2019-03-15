import sys
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import warnings
warnings.filterwarnings("ignore")


def load_data(db_filepath):
    """
    Load data from SQLite database
    Args:
    database_filepath
    """
    # Load data from database
    engine = create_engine('sqlite:///{}'.format(db_filepath))
    df = pd.read_sql_table('Message', engine)

    # drop columns with null
    df = df[~(df.isnull().any(axis=1))|((df.original.isnull())&~(df.offer.isnull()))]

    # Create X and y datasets    
    X = df['message']
    y = df.iloc[:,4:]
    categories = y.columns

    return X,y,categories

def tokenize(text):
    """
    Remove capitalization and special characters and lemmatize texts
       
    Returns:
    clean_tokens: list of strings
    """  
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def length_of_messages(data):

    return np.array([len(tx) for tx in data]).reshape(-1, 1)

def build_model():
    """
    Build model with a pipeline
       
    Returns:
    cv: gridsearchcv object
    """

    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, min_df = 5)),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 50,
                                                             min_samples_split = 3)))
    ])
    
    # Create parameters dictionary
    parameters = {  'vect__ngram_range': ((1, 1), (1, 2)),
                    'tfidf__use_idf': [True, False],
                    'tfidf__norm': ['l1', 'l2'],
                    'clf__estimator__n_estimators': [50, 100, 150],
                    'clf__estimator__min_samples_split': [2, 4],
                    'clf__estimator__max_depth':[2,4,6]
                    }
    
    
    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10)

    return cv

def evaluate_model(model, X_test, y_test, cat_names):
    """Returns classification report for the model
    
    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.
    
    Returns:
    None
    """ 
    
    # make predictions with model
    y_pred = model.predict(X_test)

    # print scores
    print(classification_report(y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]), 
        target_names=cat_names))


def save_model(model, model_filepath):
    """
    Pickle model to designated file
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        db_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(db_filepath))
        X, y, cat_names = load_data(db_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, cat_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
            )


if __name__ == '__main__':
    main()
