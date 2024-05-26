import spacy
import logging
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import joblib
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    logging.debug("Original text: %s", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    processed_text = ' '.join(tokens)
    logging.debug("Processed text: %s", processed_text)
    return processed_text

# Load the best model and vectorizer
logging.debug("Loading model and vectorizer")
model = joblib.load('models/best_logistic_regression_model.pkl')
vectorizer = joblib.load('models/best_tfidf_vectorizer.pkl')
logging.debug("Model and vectorizer loaded successfully")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

def process_request(data):
    try:
        logging.debug("Received request data: %s", data)
        text = data.get('text', '')
        if not text:
            logging.error("No text provided in the request")
            return {'error': 'No text provided'}, 400
        
        # Preprocess the input text
        processed_text = preprocess_text(text)
        
        # Vectorize the input text
        vectorized_text = vectorizer.transform([processed_text])
        logging.debug("Vectorized text: %s", vectorized_text)
        
        # Make prediction
        prediction = model.predict(vectorized_text)
        logging.debug("Prediction: %s", prediction[0])
        
        return {'sentiment': prediction[0]}
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return {'error': str(e)}, 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_request, data)
        result = future.result()
        status_code = 200 if 'sentiment' in result else 500
        return jsonify(result), status_code

if __name__ == '__main__':
    app.run(debug=True, threaded=True)





