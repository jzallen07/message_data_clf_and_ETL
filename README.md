# Disaster Response ETL and Message Classifier Project

Data Engineering: ETL and ML Pipeline preparation, with web app, and data visualisation.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline
        `python data/etl.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_clf.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Important Files:
- `data/process_data.py`: ETL pipeline used to data pre-process.
- `models/train_classifier.py`: Generates the NLP model used to classify the messages.
- `run.py`: Start the Python server and web app and prepare visualizations.
