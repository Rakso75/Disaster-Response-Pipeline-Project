# Disaster Response Pipeline Project

The project's goal is to classify messages of disaster events from a dataset  in order to sent these messages  the appropriate disaster relief agency.  The dataset is provided by Figure Eight containing real messages sent during disaster events. 
There are 36 pre-defined categories, e.g. Aid Related, Medical Help, Search And Rescue, etc. , which means it is a multi-label classification task, since a message can belong to one or more categories. 

The final outcome is a web app where an emergency worker can enter a new message and get classification results in different categories.
  
  ### Installation

- Python3
-  Machine Learning Libraries: `NumPy`, `Pandas`, `Scikit-Learn`  
-  Natural Language Process Libraries: `nltk`
-  SQLlite Database Libraries: `SQLalchemy`
-  Model Loading and Saving Library: `Pickle`
-  Web App and Data Visualization: `Flask`, `Plotly`

Finally, this project contains a web app where you can input a message and get classification results.

### Instructions:
1. Run the following commands to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database( execute in directory data)
        `python process_data.py disaster_messages.csv disaster_categories.csv ETL_Preparation.db`
    - To run ML pipeline that trains classifier and saves( execute in directory models)
        `python train_classifier.py ETL_Preparation.db classifier.pkl`

2. Run the following command to execute your web app( execute in directory app):

    - `python run.py`
	-  Click the `PREVIEW` button to open the homepage

### File Descriptions
#### Folder: app
run.py - python script to launch web application.
Folder: templates - web dependency files (go.html & master.html) required to run the web application.

#### Folder: data
disaster_messages.csv - real messages sent during disaster events (provided by Figure Eight)
disaster_categories.csv - categories of the messages
process_data.py - ETL pipeline used to load, clean, extract feature and store data in SQLite database
ETL Pipeline Preparation.ipynb - Jupyter Notebook used to prepare ETL pipeline
DisasterResponse.db - cleaned data stored in SQlite database

#### Folder: models
train_classifier.py - ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file for later use
classifier.pkl - pickle file contains trained model
ML Pipeline Preparation.ipynb - Jupyter Notebook used to prepare ML pipeline