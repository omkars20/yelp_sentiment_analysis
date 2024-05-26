from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_evaluate_model(X_train, y_train, X_test, y_test, vectorizer, model, model_name, vectorizer_name):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f'Accuracy for {model_name} with {vectorizer_name}: {accuracy}')
    print(report)
    
    joblib.dump(model, f'models/{model_name}_{vectorizer_name}_model.pkl')
    joblib.dump(vectorizer, f'models/{vectorizer_name}_vectorizer.pkl')

def perform_grid_search(X_train_vec, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    grid = GridSearchCV(LogisticRegression(multi_class='ovr'), param_grid, cv=5)
    grid.fit(X_train_vec, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation score: {grid.best_score_}")
    return grid.best_estimator_
