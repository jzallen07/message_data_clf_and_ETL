import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(msg_filepath, cat_filepath):
    """Load and merge messages and categories datasets
       
    Returns:
    df: Dataframe containing merged content of messages and categories datasets.
    """
    messages = pd.read_csv(msg_filepath)
    categories = pd.read_csv(cat_filepath)
    # merge datasets
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')
    
    return df
   
    
def clean_data(df):
    """Clean dataframe by removing duplicates and converting categories from strings
    
    Args:
    df: Dataframe containing merged content of messages and categories datasets.
       
    Returns:
    df: Dataframe containing cleaned version of input dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.loc[0]

    cat_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    # rename the columns of `categories`
    categories.columns = cat_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # Drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # Drop duplicates
    df.drop_duplicates(subset = 'id', inplace = True)
    
    return df

def save_data(df, database_filename):
    """Save cleaned data into an SQLite database.
    
    Args:
    df: Dataframe containing message and category data.
    database_filename: string.
       
    Returns:
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Message', engine, index=False, chunksize=500, if_exists='append') 


def main():
    if len(sys.argv) == 4:

        msg_filepath, cat_filepath, db_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(msg_filepath, cat_filepath))
        df = load_data(msg_filepath, cat_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(db_filepath))
        save_data(df, db_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
            )


if __name__ == '__main__':
    main()
