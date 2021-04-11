# Disaster Response Pipeline Project

## Table of contents
* [Installation](#installation)
* [Project Motivation](#project-motivation)
* [File Description](#file-description)
* [Results](#results)
* [Acknowledgements](#authors-acknowledgements)
* [Setup](#setup)

## Installation
Project is created using:
* Python 3
* Pandas package
* Numpy package
* NLTK

## Project Motivation
In support of disaster relief efforts, I have created in this project, data pipelines which integrate NLP and ML models to analyze and categorize messages sent during disasters for insights.
	
## File Description
1. Data Folder
* process_data.py: Pipeline script to extract, transform and load data into a disaster_msg SQLite database
* DisasterResponse.db: output of process_data.py
* disaster_messages.csv: input dataset containing messages sent by people during disasters.
* disaster_categories.csv: input dataset containing disaster message categories.

2. Models folder
* train_classifier.py: ML pipeline script to train a multi-output classifier
* classifier.pkl: saved model

3. App folder
* run.py: Flask file that runs web app
* master.html: main page of web app
* go.html: classification result page of web app


## Results
* ETL Pipeline (process_data.py): developed to read disaster messages and message categories from the associated csv files: disaster.csv; disaster_categories.csv
* Machine Learning Pipeline (train_classifier.py): developed to use messages data to train a classifier to conduct a multi-output classification on disaster messages into the 36 categories.
* Flask App: Created to visualize results.


## Authors, Acknowledgements
* Udacity: for the templates and starter codes.
* Figure Eight: for the data used for the project.

## Setup
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
