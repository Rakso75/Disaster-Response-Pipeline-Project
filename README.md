# Disaster Response Pipeline Project

In this project, we will build a model to classify messages that are sent during disasters. There are 36 pre-defined categories, and examples of these
categories include Aid Related, Medical Help, Search And Rescue, etc. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency. This project will involve the building of a basic ETL and Machine Learning pipeline to facilitate the task. This is
also a multi-label classification task, since a message can belong to one or more categories. We will be working with a data set provided by Figure Eight
containing real messages that were sent during disaster events.  

The project goal is to create a machine learning pipeline to classify disaster events from a dataset provided by Figure Eight containing real messages.  
The final outcome is a web app where an emergency worker can enter a new message and get classification results in different categories.
  
  # Installation

 - Python3
 - Machine Learning Libraries: `NumPy`, `Pandas`, `Scikit-Learn`
-  Natural Language Process Libraries: `nltk`
-  SQLlite Database Libraries: `SQLalchemy`
-  Model Loading and Saving Library: `Pickle`
-  Web App and Data Visualization: `Flask`, `Plotly`

Finally, this project contains a web app where you can input a message and get classification results.
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
