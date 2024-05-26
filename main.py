import pandas as pd
from preprocessing import load_and_preprocess_data
from model_training import train_and_evaluate_model, perform_grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess data
file_path = '/home/os/great_learn/NLP_Artifacts/yelp.csv'
yelp_data = load_and_preprocess_data(file_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(yelp_data['cleaned_text'], yelp_data['sentiment'], test_size=0.2, random_state=42)

# Initialize vectorizers
vectorizers = {
    'tfidf': TfidfVectorizer(max_features=3000),
    'count': CountVectorizer(max_features=3000)
}

# Initialize models
models = {
    'logistic_regression': LogisticRegression(multi_class='ovr'),
    'naive_bayes': MultinomialNB(),
    'svm': SVC(probability=True)
}

# Train and evaluate models with vectorizers
for vec_name, vectorizer in vectorizers.items():
    for model_name, model in models.items():
        train_and_evaluate_model(X_train, y_train, X_test, y_test, vectorizer, model, model_name, vec_name)

# Perform grid search on the best model (logistic regression with tfidf)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf_vectorizer.fit_transform(X_train)
best_model = perform_grid_search(X_train_vec, y_train)

# Save the best model and vectorizer
joblib.dump(best_model, 'models/best_logistic_regression_model.pkl')
joblib.dump(tfidf_vectorizer, 'models/best_tfidf_vectorizer.pkl')

