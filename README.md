# yelp_sentiment_analysis
This project aims to build a sentiment analysis model that can classify Yelp reviews as positive, negative, or neutral. The project involves several key steps including data preprocessing, model training, evaluation, and deployment. The final application is deployed as a web service using flask.


Project Steps
Data Preprocessing:

Load the Yelp dataset.
Clean and preprocess the text data by removing non-alphabet characters, tokenizing, removing stopwords, and lemmatizing.
Assign sentiment labels based on star ratings (1-2: negative, 3: neutral, 4-5: positive).
Model Training and Evaluation:

Split the data into training and testing sets.
Train multiple models (Logistic Regression, Naive Bayes, SVM) using different vectorizers (TF-IDF, Count Vectorizer).
Evaluate the models and select the best performing one using accuracy and classification reports.
Perform hyperparameter tuning using GridSearchCV for the best model.
Save the trained model and vectorizer to disk.
API Development:

Develop a Flask API that loads the saved model and vectorizer.
Create an endpoint to receive review text, preprocess it, vectorize it, and return the sentiment prediction.
Deployment:

Create a requirements.txt file to list all dependencies.
Create a Procfile for Heroku deployment.
Test the Flask application locally.
