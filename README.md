# Disaster Response Pipeline Project

## Table of contents
* [Installation](#installation)
* [Project Motivation](#project-motivation)
* [File Description](#file-description)
* [Results](#results)
* [Acknowledgements](#acknowledgements)
* [Setup](#setup)

## Installation
Project is created with:
* Lorem version: 12.3
* Ipsum version: 2.33
* Ament library version: 999

## Project Motivation
This project is simple Lorem ipsum dolor generator
	
## File Description
To run this project, install it locally using npm:

## Results
To run this project, install it locally using npm:

## Acknowledgements
To run this project, install it locally using npm:

## Setup
1. Run the following commands in the project's root directory to set up your database and model.

```
$ cd ../lorem
$ npm install
$ npm start
```
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
