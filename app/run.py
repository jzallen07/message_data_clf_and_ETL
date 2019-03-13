import json
import plotly
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Remove capitalization and special characters and lemmatize texts
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    clean_tokens: list of strings. List containing normalized and stemmed word tokens
    """  
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens

def length_of_messages(data):
    ''' Calculates the length of text in message
    '''
    return np.array([len(text) for text in data]).reshape(-1, 1)
    
db_filepath = 'data/DisasterResponse.db'
# load data
engine = create_engine('sqlite:///{}'.format(database_filepath))
df = pd.read_sql_table('Message', engine)

# load model
model = joblib.load("/home/workspace/models/classifier.pkl")

# msg len
df['text_length'] = length_of_messages(df['message'])

# model viz
@app.route('/')
@app.route('/index')

def index():
    
    # extract data 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract categories
    category_map = df.iloc[:,4:].corr().values
    category_names = list(df.iloc[:,4:].columns)

    # extract length of messages
    length_direct = df.loc[df.genre=='direct','text_length']
    length_social = df.loc[df.genre=='social','text_length']
    length_news = df.loc[df.genre=='news','text_length']
    
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names[::-1],
                    z=category_map
                )    
            ],

            'layout': {
                'title': 'Heatmap of Categories'
            }
        },

        {
            'data': [
                Histogram(
                    y=length_direct,
                    name='Direct',
                    opacity=0.5
                ),
                Histogram(
                    y=length_social,
                    name='Social',
                    opacity=0.5
                ),
                Histogram(
                    y=length_news,
                    name='News',
                    opacity=0.5
                )
            ],

            'layout': {
                'title': 'Distribution of Text Length',
                'yaxis':{
                    'title':'Count'
                },
                'xaxis': {
                    'title':'Text Length'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    clf_labels = model.predict([query])[0]
    clf_results = dict(zip(df.columns[4:], clf_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=clf_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
