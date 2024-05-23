# Disaster Response Pipeline Project

### Project Description
This project is aimed at using classified text data provided bt Figure 8 to create a machine learning pipeline:
1. ETL Pipeline data/process_data.py: etracts data from two datasets, clean, merge and load to an SQLite database
2. ML Pipeline models/train_classifier.py: Builds the entire ML pipeline and stores in a pickle file
3. Web App in app folder: creates a visual representation of the model

The full pipeline is essential in helping responder professionals easily retrieve messages during disasters
and to be able to categorise, analyse trends which will enable right allocation of resources with the help of 
supervised machine learning models.


### Files
1. data/process_data.py: Python script executing the ETL pipeline
2. data/DisasterResponse.db: the SQLite database
3. data/messages.csv: messages dataset
4. data/categories: categories dataset
5. models/classifier.pkl: pickle file storing the trained model
6. models/train_classifier.py: Python Script that trains and stores the model
7. app/run.py: Python script to launch the web app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. You may need to install SQlAlchemy: pip install sqlalchemy, Joblib: pip install joblib

