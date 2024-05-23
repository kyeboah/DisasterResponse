'''
This script contains functions that consititute
the ETL Pipeline
'''

# Import libraries
import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input data: messages_filepath for the messages dataset and the categories_filepath
    for the categories filepath

    Output:
    a DataFrame(df) containing merged messages and categories
    '''
    messages = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')

    df = pd.merge(messages,categories, on='id')

    return df


def clean_data(df):
    '''
    Input data: the dataframe (df)

    Output: return cleaned df
    '''
    # create each category from semi-colon separated list
    categories = df['categories'].str.split(';', expand=True) 
    row = categories.iloc[0,:]  # select the first row of the categories dataframe
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values of strings into 0s or 1s
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop('categories', axis=1, inplace=True) 
    # concatenate the original dataframe with the new `categories` dataframe 
    df = pd.concat([df, categories], axis=1) 

    # drop rows with related = 2
    df = df[df['related'] != 2]

    # drop duplicates
    df = df.drop_duplicates()

    return df
    


def save_data(df, database_filename):
    '''
    Input: df and specified database name

    output: saves df in database under 'Combined'
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, con=engine, index=False, if_exists='replace' )


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()